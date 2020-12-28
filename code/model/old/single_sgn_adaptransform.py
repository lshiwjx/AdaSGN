from torch import nn
import torch
from model.layers import *
from model.init_transforms import Transforms


class GetTransform(nn.Module):
    def __init__(self, in_c, out_joint, num_frame, ori_joint=25):
        super(GetTransform, self).__init__()
        # self.conv = nn.Conv1d(in_c*num_frame, out_joint, kernel_size=1, padding=0)
        self.conv = nn.Linear(in_c*num_frame*ori_joint, ori_joint*out_joint)
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)


    def forward(self, x):
        n, c, v, t = x.shape
        x = x.view(n, -1)
        x = self.conv(x).view(n, v, -1)
        # x = x.permute(0, 1, 3, 2).contiguous().view(n, c*t, v)
        # x = self.conv(x)
        # x = x.permute(0, 2, 1)
        return x


class Single_SGN(nn.Module):
    def __init__(self, num_classes, num_joint, seg, bias=True, dim=256, adaptive_transform=False, num_joint_ori=25,
                 gcn_type='mid'):
        super(Single_SGN, self).__init__()

        self.seg = seg

        self.spa_net = SpatialNet(num_joint, bias, dim, gcn_type)
        self.tem_net = TempolNet(seg, bias, dim)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(dim * 2, num_classes)

        self.transform = nn.Parameter(Transforms['M{}to{}'.format(num_joint_ori, num_joint)], requires_grad=adaptive_transform)
        self.get_transform = GetTransform(3, num_joint, seg)

    def forward(self, input):
        if len(input.shape) == 6:
            bs, s, c, step, num_joints, m = input.shape
            input = input.view(bs * s, c, step, num_joints, m)
        else:
            bs, c, step, num_joints, m = input.shape
            s = 1
        input = input.permute(0, 4, 1, 3, 2).contiguous().view(bs * m * s, c, num_joints, step)  # nctvm->nmcvt

        transform = torch.repeat_interleave(self.transform+self.get_transform(input), repeats=c, dim=0)
        input = torch.bmm(input.view(bs*m*s*c, num_joints, step).transpose(1, 2), transform).transpose(2, 1).contiguous().view(bs*m*s, c, num_joints, step)  # bcvt

        dif = torch.cat([torch.zeros([*input.shape[:3], 1], device=input.device), input[:, :, :, 1:] - input[:, :, :, 0:-1]], dim=-1)

        input = self.spa_net(input, dif)  # b c 1 t
        input = self.tem_net(input)  # b c 1 t
        # Classification
        output = self.maxpool(input)  # b c 1 1
        output = torch.flatten(output, 1)  # b c
        output = self.fc(output)  # b p
        output = output.view(bs, m * s, -1).mean(1)

        return output


if __name__ == '__main__':
    import os
    from model.flops_count import get_model_complexity_info
    from thop import profile

    num_j = 25
    num_t = 20
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'
    model = Single_SGN(60, num_j, num_t, adaptive_transform=False, gcn_type='big')
    # torch.save(model.state_dict(), '../../pretrain_models/single_sgn_jpt{}.state'.format(num_j),)
    dummy_data = torch.randn([2, 3, num_t, 25, 2])
    a = model(dummy_data)
    # a.mean().backward()

    # hooks = {}
    # flops, params = profile(model, inputs=(dummy_data,), custom_ops=hooks)
    # gflops = flops / 1e9
    # params = params / 1e6
    #
    # print(gflops)
    # print(params)

    # flops, params = get_model_complexity_info(model, (3, num_t, 25, 1), as_strings=True)  # not support

    # print(flops)  # 0.16 gmac
    # print(params)  # 0.69 m
