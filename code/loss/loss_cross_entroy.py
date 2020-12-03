import torch
import torch.nn.functional as func
import torch.nn as nn


# def to_onehot(num_class, label, alpha):
#     return torch.zeros((label.shape[0], num_class)).fill_(alpha).scatter_(1, label.unsqueeze(1), 1 - alpha)


# class naive_cross_entropy_loss(nn.Module):
#     def __init__(self, num_class, alpha):
#         self.num_class = num_class
#         self.alpha = alpha
#         super(naive_cross_entropy_loss, self).__init__()
#
#     def forward(self, inputs, target):
#         target = to_onehot(self.num_class, target, self.alpha)
#         return - (func.log_softmax(inputs, dim=-1) * target).sum(dim=-1).mean()


class multi_cross_entropy_loss(nn.Module):
    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss(size_average=True)
        super(multi_cross_entropy_loss, self).__init__()

    def forward(self, inputs, target):
        '''

            :param inputs: N C S
            :param target: N C
            :return:
            '''
        num = inputs.shape[-1]
        inputs_splits = torch.chunk(inputs, num, dim=-1)
        loss = self.loss(inputs_splits[0].squeeze(-1), target)
        for i in range(1, num):
            loss += self.loss(inputs_splits[i].squeeze(-1), target)
        loss /= num
        return loss


def naive_cross_entropy_loss(inputs, target):
    return - (func.log_softmax(inputs, dim=-1) * target).sum(dim=-1).mean()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# def multi_cross_entropy_loss(inputs, target):
#     '''
#
#     :param inputs: N C S
#     :param target: N C
#     :return:
#     '''
#     num = inputs.shape[-1]
#     inputs_splits = torch.chunk(inputs, num, dim=-1)
#     loss = - (func.log_softmax(inputs_splits[0].squeeze(-1), dim=-1) * target).sum(dim=-1).mean()
#     for i in range(1, num):
#         loss += - (func.log_softmax(inputs_splits[i].squeeze(-1), dim=-1) * target).sum(dim=-1).mean()
#     loss /= num
#     return loss
