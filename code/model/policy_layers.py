from model.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class Lstm(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(Lstm, self).__init__()
        self.lstm1 = nn.LSTM(in_c, mid_c, num_layers=1, batch_first=True)  # bc1t -> btc
        self.lstm2 = nn.LSTM(mid_c, out_c, num_layers=1, batch_first=True)  # bc1t -> btc

    def forward(self, x):  # b c 1 t
        x = x.squeeze(2).transpose(2, 1)  # b t c
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.transpose(2, 1).contiguous().unsqueeze(-2)  # b c 1 t
        return x


class Transformer(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(Transformer, self).__init__()
        self.q = nn.Conv2d(in_c, mid_c, 1)
        self.k = nn.Conv2d(in_c, mid_c, 1)
        self.v = nn.Conv2d(in_c, mid_c, 1)
        self.ff = nn.Conv2d(mid_c, out_c, 1)
        self.mid_c = mid_c

    def forward(self, x):  # b c 1 t
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        a = torch.einsum('ncvt,ncvq->ntq', [q, k]) / self.mid_c
        y = torch.einsum('ntq,ncvt->ncvt', [a, v])
        y = self.ff(y)
        return y


class Tconv2(nn.Module):
    def __init__(self, in_c, mid_c, out_c, k=3, d=1, init_type='random'):
        super(Tconv2, self).__init__()
        pad = (k + (d - 1) * (d - 1) - 1) // 2
        self.conv1 = nn.Conv2d(in_c, mid_c, kernel_size=(1, k),
                               dilation=(1, d),
                               padding=(0, pad))
        self.bn = nn.BatchNorm2d(mid_c)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_c, out_c, kernel_size=(1, k),
                               dilation=(1, d),
                               padding=(0, pad))
        if init_type == 'fix':
            nn.init.constant_(self.conv2.weight[-1:], 1 / k)
            nn.init.constant_(self.conv2.weight[:-1], 0)
            nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):  # b c 1 t
        x = self.relu(self.bn(self.conv1(x)))
        x = self.conv2(x)
        return x


class Tconv(nn.Module):
    def __init__(self, in_c, out_c, k=3, d=1, init_type='random'):
        super(Tconv, self).__init__()
        pad = (k + (d - 1) * (d - 1) - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=(1, k),
                              dilation=(1, d),
                              padding=(0, pad))
        if init_type == 'fix':
            nn.init.constant_(self.conv.weight[-1:], 1 / k / out_c)
            nn.init.constant_(self.conv.weight[:-1], 0)
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):  # b c 1 t
        x = self.conv(x)
        return x


class Temconv(nn.Module):
    def __init__(self, in_c, out_c, k=3, d=1, seg=20, bias=True, dim=256, init_type='random'):
        super().__init__()

        tem = one_hot(seg)
        tem = tem.permute(0, 3, 1, 2)
        self.register_buffer('tem', tem)
        self.tem_embed = embed(seg, dim, norm=False, bias=bias)
        pad = (k + (d - 1) * (d - 1) - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=(1, k),
                              dilation=(1, d),
                              padding=(0, pad))

        if init_type == 'fix':
            nn.init.constant_(self.conv.weight[-1:], 1 / k)
            nn.init.constant_(self.conv.weight[:-1], 0)
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, input):
        tem1 = self.tem_embed(self.tem)  # 1 c 1 t
        # Frame-level Module
        input = input + tem1
        output = self.conv(input)  # 3.9%  b c 1 t

        return output


class EmptyNet(nn.Module):
    def __init__(self):
        super(EmptyNet, self).__init__()
        pass

    def forward(self, input):
        return input


def get_random_action(batch_size, num_action, T):
    prob = torch.randn([batch_size, num_action, 1, T])

    action = F.gumbel_softmax(prob, 1e-6, hard=True, dim=1)  # batch_size num_action 1 t

    return prob, action


def get_fixed_action(batch_size, num_action, T):
    prob = None

    action = torch.ones([batch_size, num_action, 1, T])  # batch_size num_action 1 t

    return prob, action


def get_litefeat_and_action(input_feat, tau, policy_net):
    prob = torch.log(F.softmax(policy_net(input_feat), dim=1).clamp(min=1e-8))  # batch_size num_action 1 t

    action = F.gumbel_softmax(prob, tau, hard=True, dim=1)  # batch_size num_action 1 t

    return prob, action


def get_input_feats(input_list, models):
    output_list = []
    for input, model in zip(input_list, models):
        dif = torch.cat(
            [torch.zeros([*input.shape[:3], 1], device=input.device), input[:, :, :, 1:] - input[:, :, :, 0:-1]],
            dim=-1)

        output_list.append(model(input, dif))
    return output_list


def downsample_input(input, transforms, num_models):
    # n c v t
    out_list = []
    for i, t in enumerate(transforms):
        out_list.append(torch.matmul(input.transpose(2, 3), t).transpose(2, 3).contiguous())
    return out_list