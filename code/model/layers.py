from torch import nn
import torch
import math


def one_hot(spa):
    y = torch.arange(spa).unsqueeze(-1)
    y_onehot = torch.FloatTensor(spa, spa)

    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)

    return y_onehot


# class GCNBig(nn.Module):
#     def __init__(self, dim, bias):
#         super().__init__()
#         self.gcn1 = gcn_spa(dim // 2, dim, bias=bias)
#         self.gcn2 = gcn_spa(dim, dim, bias=bias)
#         self.gcn3 = gcn_spa(dim, dim, bias=bias)
#
#     def forward(self, x, g):
#         x = self.gcn1(x, g)
#         x = self.gcn2(x, g)
#         x = self.gcn3(x, g)
#         return x


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


class GCNSma(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        # self.gcn1 = gcn_spa(dim // 2, dim // 2, bias=bias)
        # self.gcn2 = gcn_spa(dim // 2, dim // 2, bias=bias)
        self.gcn = gcn_spa(dim // 2, dim, bias=bias)

    def forward(self, x, g):
        # x = self.gcn1(x, g)
        # x = self.gcn2(x, g)
        x = self.gcn(x, g)
        return x


class SpatialNet(nn.Module):
    def __init__(self, num_joint, bias=True, dim=256, gcn_type='small'):
        super().__init__()

        spa = one_hot(num_joint)  # 1, 1, 25, 25
        spa = spa.permute(0, 3, 2, 1)
        self.register_buffer('spa', spa)

        self.spa_embed = embed(num_joint, dim // 4, norm=False, bias=bias, num_joints=num_joint)
        self.joint_embed = embed(3, dim // 4, norm=True, bias=bias, num_joints=num_joint)
        self.dif_embed = embed(3, dim // 4, norm=True, bias=bias, num_joints=num_joint)
        self.compute_g1 = compute_g_spa(dim // 2, dim, bias=bias)
        if gcn_type == 'big':
            self.gcn = GCNBig(dim, bias)
        # elif gcn_type == 'mid':
        #     self.gcn = GCNMid(dim, bias)
        elif gcn_type == 'small':
            self.gcn = GCNSma(dim, bias)
        else:
            raise RuntimeError('No such config')

        self.maxpool = nn.MaxPool2d((num_joint, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        for m in self.gcn.modules():
            if isinstance(m, gcn_spa):
                nn.init.constant_(m.w.cnn.weight, 0)

    def forward(self, input, dif):
        bs, c, num_joints, step = input.shape
        pos = self.joint_embed(input)
        spa1 = self.spa_embed(self.spa).repeat(bs, 1, 1, step)
        dif = self.dif_embed(dif)
        dy = pos + dif
        input = torch.cat([dy, spa1], 1)
        g = self.compute_g1(input)  # compute self-attention graph  19.1%
        input = self.gcn(input, g)
        output = self.maxpool(input)

        return output  # bs c 1 t


class TempolNet(nn.Module):
    def __init__(self, seg, bias=True, dim=256):
        super().__init__()

        tem = one_hot(seg)  # 1 1 t c
        tem = tem.permute(0, 3, 1, 2).contiguous()  # 1 c 1 t
        self.register_buffer('tem', tem)
        self.tem_embed = embed(seg, dim, norm=False, bias=bias)
        self.cnn = local(dim, dim * 2, bias=bias, seg=seg)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        tem1 = self.tem_embed(self.tem)  # 1 c 1 t
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
    def __init__(self, dim1=3, dim2=3, bias=False, seg=20):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, seg))
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
        return g  # ntvv


if __name__ == '__main__':
    from model.flops_count import get_model_complexity_info
    # temNet: 0.007
    # tconv: 4e-5 2-layer 0.002
    # transformer: 0.0019 or 2e-3
    # lstm: 6e-5  2-layer 0.004
    num_t = 20
    num_j = 25
    Big = SpatialNet(num_j)  # 0.01 0.07 0.15
    Mid = SpatialNet(num_j)  # 0.01 0.07 0.15
    Sma = SpatialNet(num_j)  # 0.01 0.07 0.15
    flops1, params1 = get_model_complexity_info(Big, (3, num_j, num_t), as_strings=True)  # 0.15
    flops2, params2 = get_model_complexity_info(Mid, (3, num_j, num_t), as_strings=True)  # 0.07
    flops3, params3 = get_model_complexity_info(Sma, (3, num_j, num_t), as_strings=True)  # 0.01
    print(flops1, flops2, flops3)
    print(params1, params2, params3)