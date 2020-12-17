from torch import nn
import torch
from model.layers import *
from model.init_transforms import Transforms


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
        # self.transform = nn.Parameter(torch.ones([num_joint_ori, num_joint]) / num_joint_ori,
        #                               requires_grad=adaptive_transform)

    def forward(self, input):
        if len(input.shape) == 6:
            bs, s, c, step, num_joints, m = input.shape
            input = input.view(bs * s, c, step, num_joints, m)
        else:
            bs, c, step, num_joints, m = input.shape
            s = 1
        input = input.permute(0, 4, 1, 3, 2).contiguous().view(bs * m * s, c, num_joints, step)  # nctvm->nmcvt

        input = torch.matmul(input.transpose(2, 3), self.transform).transpose(2, 3).contiguous()  # bcvt

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

    num_j = 2
    num_t = 20
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'
    model = Single_SGN(60, num_j, num_t, adaptive_transform=False, gcn_type='small')
    # torch.save(model.state_dict(), '../../pretrain_models/single_sgn_jpt{}.state'.format(num_j),)
    # dummy_data = torch.randn([1, 3, num_t, 25, 2])
    # a = model(dummy_data)
    # a.mean().backward()

    # hooks = {}
    # flops, params = profile(model, inputs=(dummy_data,), custom_ops=hooks)
    # gflops = flops / 1e9
    # params = params / 1e6
    #
    # print(gflops)
    # print(params)

    flops, params = get_model_complexity_info(model, (3, num_t, 25, 1), as_strings=True)  # not support

    print(flops)  # 0.16 gmac
    print(params)  # 0.69 m
