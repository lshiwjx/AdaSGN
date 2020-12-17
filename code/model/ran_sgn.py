from torch import nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from ptflops import get_model_complexity_info

from model.init_transforms import Transforms

def one_hot(spa):
    y = torch.arange(spa).unsqueeze(-1)
    y_onehot = torch.FloatTensor(spa, spa)

    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)

    return y_onehot


def get_random_action(batch_size, num_action, T):
    prob = torch.randn([batch_size, num_action, 1, T])

    action = F.gumbel_softmax(prob, 1e-6, hard=True, dim=1)  # batch_size num_action 1 t

    return prob, action


def get_input_feats(input_list, models):
    output_list = []
    for input, model in zip(input_list, models):
        output_list.append(model(input))
    return output_list


def downsample_input(input, transforms):
    # n c v t
    out_list = []
    input = input.transpose(2, 3)  # n c t v
    for t in transforms:
        t = t.to(input.device)
        out_list.append(torch.matmul(input, t).transpose(2, 3).contiguous())
    out_list.append(input.transpose(2, 3).contiguous())
    return out_list


class RANSGN(nn.Module):
    def __init__(self, num_classes, num_joints, seg, transforms, bias=True, dim=256, tau=5):
        super(RANSGN, self).__init__()

        self.seg = seg
        self.transforms = [Transforms[t] for t in transforms]
        self.num_joints = num_joints
        self.seg = seg
        self.dim = dim

        self.spa_nets = nn.ModuleList([SpatialNet(num_joint, seg, bias, dim) for num_joint in num_joints])
        self.tem_net = TempolNet(seg, bias, dim)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(dim * 2, num_classes)

        self.get_gflops_table()

    def get_gflops_table(self):
        kwards = {
            'print_per_layer_stat': False,
            'as_strings': False
        }
        self.gflops_table = {}
        self.gflops_table['spanet'] = [get_model_complexity_info(m, (3, j, self.seg), **kwards)[0] / 1e9 for
                                       m, j in zip(self.spa_nets, self.num_joints)]
        self.gflops_table['temnet'] = \
            get_model_complexity_info(self.tem_net, (self.dim, 1, self.seg), **kwards)[0] / 1e9
        self.gflops_table['policy'] = 0

        print("gflops_table: ")
        for k in self.gflops_table:
            print(k, self.gflops_table[k])

        self.gflops_vector = torch.FloatTensor(self.gflops_table['spanet'])

    def get_policy_usage_str(self, action_list):
        gflops_fix = self.gflops_table['temnet'] + self.gflops_table['policy']

        actions_mean = np.concatenate(action_list, axis=1).mean(-1).squeeze(-1).squeeze(-1).mean(-1)  # num_act

        gflops_sma, gflops_mid, gflops_big = actions_mean * self.gflops_table['spanet']

        printed_str = 'gflops_fix: {}, gflops_sma: {}, gflops_mid: {}, gflops_big: {}, actions: '.format(gflops_fix, gflops_sma, gflops_mid,
                                                                                         gflops_big) + str(actions_mean)
        return printed_str

    def train(self, mode=True):
        super().train(mode)

    def forward(self, input):
        if len(input.shape) == 6:
            bs, s, c, t, num_joint, m = input.shape
            input = input.view(bs * s, c, t, num_joint, m)
        else:
            bs, c, t, num_joint, m = input.shape
            s = 1
        input = input.permute(0, 4, 1, 3, 2).contiguous().view(bs * m * s, c, num_joint, t)  # nctvm->nmcvt

        input_list = downsample_input(input, self.transforms)  # get input list with different size s b c v t

        input_feats = get_input_feats(input_list, self.spa_nets)  # get input features s b c 1 t

        prob, action = get_random_action(bs*m*s, len(self.num_joints), t)  # batch_size num_action 1 t

        action = action.permute(1, 0, 2, 3).unsqueeze(2).to(input.device)  # num_action, b11t

        input = (action * torch.stack(input_feats)).sum(0)  # b c 1 t

        # input = self.spa_net(input)  # b c 1 t
        input = self.tem_net(input)  # b c 1 t
        # Classification
        output = self.maxpool(input)  # b c 1 1
        output = torch.flatten(output, 1)  # b c
        output = self.fc(output)  # b p
        output = output.view(bs, m * s, -1).mean(1)

        return output, action


class GCNBig(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        self.gcn1 = gcn_spa(dim // 2, dim // 2, bias=bias)
        self.gcn2 = gcn_spa(dim // 2, dim, bias=bias)
        self.gcn3 = gcn_spa(dim, dim, bias=bias)

    def forward(self, x, g):
        x = self.gcn1(x, g)
        x = self.gcn2(x, g)
        x = self.gcn3(x, g)
        return x


class GCNMid(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        self.gcn1 = gcn_spa(dim // 2, dim, bias=bias)
        self.gcn2 = gcn_spa(dim, dim, bias=bias)

    def forward(self, x, g):
        x = self.gcn1(x, g)
        x = self.gcn2(x, g)
        return x


class GCNSma(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        self.gcn1 = gcn_spa(dim // 2, dim, bias=bias)

    def forward(self, x, g):
        x = self.gcn1(x, g)
        return x


class SpatialNet(nn.Module):
    def __init__(self, num_joint, seg, bias=True, dim=256):
        super().__init__()

        spa = one_hot(num_joint)
        spa = spa.permute(0, 3, 2, 1)
        self.register_buffer('spa', spa)

        self.spa_embed = embed(num_joint, dim // 4, norm=False, bias=bias, num_joints=num_joint)
        self.joint_embed = embed(3, dim // 4, norm=True, bias=bias, num_joints=num_joint)
        self.dif_embed = embed(3, dim // 4, norm=True, bias=bias, num_joints=num_joint)
        self.compute_g1 = compute_g_spa(dim // 2, dim, bias=bias)
        if num_joint == 25:
            self.gcn = GCNBig(dim, bias)
        elif num_joint == 13:
            self.gcn = GCNMid(dim, bias)
        elif num_joint == 5:
            self.gcn = GCNSma(dim, bias)
        else:
            raise RuntimeError('No such config')

        self.maxpool = nn.AdaptiveMaxPool2d((1, seg))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        for m in self.gcn.modules():
            if isinstance(m, gcn_spa):
                nn.init.constant_(m.w.cnn.weight, 0)
        # nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        # nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        # nn.init.constant_(self.gcn3.w.cnn.weight, 0)

    def forward(self, input):
        bs, c, num_joints, step = input.shape
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)
        pos = self.joint_embed(input)
        spa1 = self.spa_embed(self.spa).repeat(bs, 1, 1, step)
        dif = self.dif_embed(dif)
        dy = pos + dif
        # Joint-level Module
        input = torch.cat([dy, spa1], 1)
        g = self.compute_g1(input)  # compute self-attention graph  19.1%
        input = self.gcn(input, g)  # 128->128  9.6%
        # input = self.gcn2(input, g)  # 128->256  19.3%
        # input = self.gcn3(input, g)  # 256->256  38.3%
        output = self.maxpool(input)

        return output  # bs c 1 t


class TempolNet(nn.Module):
    def __init__(self, seg, bias=True, dim=256):
        super().__init__()

        tem = one_hot(seg)
        tem = tem.permute(0, 3, 1, 2)
        self.register_buffer('tem', tem)
        self.tem_embed = embed(seg, dim, norm=False, bias=bias)
        self.cnn = local(dim, dim * 2, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        tem1 = self.tem_embed(self.tem)  # 1 c 1 t
        # Frame-level Module
        input = input + tem1
        output = self.cnn(input)  # 3.9%  b c 1 t

        return output


class norm_data(nn.Module):
    def __init__(self, dim=64, num_joints=25):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim * num_joints)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x


class embed(nn.Module):
    def __init__(self, dim=3, dim1=128, norm=True, bias=False, num_joints=25):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim, num_joints=num_joints),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x


class cnn1x1(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x


class local(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)

    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x


class compute_g_spa(nn.Module):
    def __init__(self, dim1=64 * 3, dim2=64 * 3, bias=False):
        super(compute_g_spa, self).__init__()
        self.g1 = cnn1x1(dim1, dim2, bias=bias)
        self.g2 = cnn1x1(dim1, dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):
        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g


if __name__ == '__main__':
    import os
    from thop import profile

    num_js = [5, 13, 25]
    num_j = num_js[-1]
    num_t = 20
    dim = 256
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'

    model = RANSGN(60, num_js, num_t, transforms=["M25to5", "M25to13"])
    dummy_data = torch.randn([1, 3, num_t, num_j, 2])
    pro, act = model(dummy_data)
    print(model.get_policy_usage_str([act]))
    # hooks = {}
    # flops, params = profile(model, inputs=(dummy_data,), custom_ops=hooks)
    # gflops = flops / 1e9
    # params = params / 1e6
    #
    # print(gflops)
    # print(params)

    # flops, params = get_model_complexity_info(model, (3, num_t, num_j, 2), as_strings=True)

    # print(flops)  # 0.16 gmac
    # print(params)  # 0.69 m
