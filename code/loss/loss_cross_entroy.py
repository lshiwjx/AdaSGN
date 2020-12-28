import torch
import torch.nn.functional as func
import torch.nn as nn


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


class AdaLoss(nn.Module):
    def __init__(self, classes, alpha=1, label_smoothing_num=0.0, dim=-1, freeze_alpha=0, warm_epoch=0, beta=0,
                 begin_action=1):
        super().__init__()
        self.CE = LabelSmoothingLoss(classes, label_smoothing_num, dim)
        self.alpha = alpha
        self.freeze_alpha = freeze_alpha
        self.warm_epoch = warm_epoch
        self.beta = beta
        self.begin_action = begin_action

    def forward(self, pred, target, flops_vector, actions, epoch):
        """

        :param pred: N, num_cls
        :param target: N
        :param flops_vector: num_act
        :param actions: num_act N M 1 1 T
        :return:
        """
        if epoch < self.freeze_alpha:
            alpha = 0
        else:
            if self.warm_epoch != 0:
                ratio = min(epoch / self.warm_epoch, 1)
            else:
                ratio = 1
            alpha = self.alpha * ratio
        flops_vector = flops_vector.to(target.device)
        # ls_flops = (torch.mean(torch.mean(actions[self.begin_action:], dim=(2, 3, 4)).transpose(1, 0) *
        #                        flops_vector[self.begin_action:].unsqueeze(0)) + flops_vector[0]) * alpha

        ls_flops = torch.mean(
            torch.mean(actions[self.begin_action:], dim=(2, 3, 4, 5)).transpose(1, 0) *
                              flops_vector[self.begin_action:].unsqueeze(0)) * alpha
        ls_cls = self.CE(pred, target)
        avg_action = torch.mean(actions, dim=(1, 2, 3, 4, 5))
        ls_uniform = torch.norm(avg_action - avg_action.mean(), p=2) * self.beta
        return ls_flops, ls_cls, ls_uniform


if __name__ == '__main__':
    cls = 5
    num_a = 3
    l = AdaLoss(cls)
    pred = torch.randn([1, cls])
    target = torch.zeros(1, dtype=torch.long)
    fv = torch.randn(num_a)
    act = torch.randn([num_a, 1, 1, 1, 20])
    l(pred, target, fv, act, 0)
