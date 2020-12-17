from torch import nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from model.flops_count import get_model_complexity_info

from model.init_transforms import Transforms, Transformsadap
from model.policy_layers import *


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


class ADASGN(nn.Module):
    def __init__(self, num_classes, num_joints, seg, bias=True, dim=256, tau=5., policy_kernel=3,
                 policy_dilate=1, adaptive_transform=[], policy_type='tconv', args=None, tau_decay=-0.045,
                 pre_trains=None, tau_type='cos', init_type='fix', init_num=5, gcn_types=[], num_joint_ori=25):
        super(ADASGN, self).__init__()

        self.seg = seg
        self.transforms = nn.ParameterList(
            [nn.Parameter(Transforms['M{}to{}'.format(num_joints[-1], i)], requires_grad=adaptive_transform[ind]) for
             ind, i in enumerate(num_joints) for gcn_type in gcn_types])
        # self.transforms = nn.ParameterList(
        #     [nn.Parameter(torch.ones([num_joint_ori, num_joint]) / num_joint_ori,
        #                   requires_grad=adaptive_transform[ind]) for ind, num_joint in enumerate(num_joints)])

        self.num_joints = num_joints
        self.seg = seg
        self.dim = dim
        self.bias = bias
        self.tau_decay = tau_decay
        self.tau_type = tau_type
        self.gcn_types = gcn_types

        self.spa_nets = nn.ModuleList([SpatialNet(num_joint, bias, dim, gcn_type)
                                       for num_joint in num_joints for gcn_type in gcn_types])  # j1m1 j1m2 j2m1 ...
        self.tem_net = TempolNet(seg, bias, dim)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(dim * 2, num_classes)
        self.tau = tau
        self.epoch = 0
        self.args = args
        self.num_gcns = len(gcn_types)
        self.num_jpt = len(num_joints)
        self.num_action = self.num_gcns * self.num_jpt

        if pre_trains is not None:
            self.load(pre_trains, init_num)

        # b c 1 t -> b 3 1 t
        if policy_type == 'tconv':
            self.policy_net = Tconv(dim, self.num_action, k=policy_kernel, d=policy_dilate, init_type=init_type)
        elif policy_type == 'tconv2':
            self.policy_net = Tconv2(dim, dim // 2, self.num_action, k=policy_kernel, d=policy_dilate,
                                     init_type=init_type)
        elif policy_type == 'transformer':
            self.policy_net = Transformer(dim, dim // 2, self.num_action)
        elif policy_type == 'lstm':
            self.policy_net = Lstm(dim, dim // 2, self.num_action)
        elif policy_type == 'tnet':
            self.policy_net = Temconv(dim, self.num_action, k=policy_kernel, d=policy_dilate, seg=seg, dim=dim,
                                      init_type=init_type)
        else:
            raise RuntimeError('No such policy net')

        self.get_gflops_table()

    def get_gflops_table(self):
        kwards = {
            'print_per_layer_stat': False,
            'as_strings': False
        }

        def input_constructor(a):
            return {'input': torch.ones(()).new_empty((1, *a)), 'dif': torch.ones(()).new_empty((1, *a))}

        self.gflops_table = {}
        self.gflops_table['spanet'] = [
            get_model_complexity_info(m, (3, j, self.seg), input_constructor=input_constructor, **kwards)[0] / 1e9 for
            m, j in zip(self.spa_nets, [x for x in self.num_joints for _ in self.gcn_types])]
        self.gflops_table['temnet'] = \
            get_model_complexity_info(self.tem_net, (self.dim, 1, self.seg), **kwards)[0] / 1e9
        self.gflops_table['policy'] = \
            get_model_complexity_info(self.policy_net, (self.dim, 1, self.seg), **kwards)[0] / 1e9

        self.gflops_table['flops_fix'] = self.gflops_table['spanet'][0] + self.gflops_table['temnet'] + \
                                         self.gflops_table['policy']
        self.gflops_vector = torch.FloatTensor(self.gflops_table['spanet'])

        print("gflops_table: ")
        for i in range(self.num_jpt):
            print('spanet', self.num_joints[i], self.gflops_table['spanet'][i * self.num_gcns:(i + 1) * self.num_gcns])

        print('temnet', self.gflops_table['temnet'])
        print('policy', self.gflops_table['policy'])
        print('flops_fix', self.gflops_table['flops_fix'])

        # self.gflops_vector = torch.FloatTensor([self.gflops_table['flops_fix'], *self.gflops_table['spanet'][1:]])

    def get_policy_usage_str(self, action_list):
        actions_mean = np.concatenate(action_list, axis=1).mean(-1).squeeze(-1).squeeze(-1).mean(-1)  # num_act

        gflops = actions_mean[1:] * self.gflops_table['spanet'][1:]

        printed_str = 'Gflops_fix: ' + str(self.gflops_table['flops_fix']) \
                      + '\nGflops: ' + str(gflops) \
                      + '\nALL: ' + str(sum(gflops) + self.gflops_table['flops_fix']) \
                      + '\nActions: ' + str(actions_mean)
        return printed_str

    def train(self, mode=True):
        super(ADASGN, self).train(mode)
        if mode:
            self.epoch += 1
            for freeze_key in self.args.freeze_keys:
                if freeze_key[0] == 'policy_net' and freeze_key[1] >= self.epoch:
                    return
            if self.tau_type == 'linear':
                self.tau = self.tau * np.exp(self.tau_decay)
            elif self.tau_type == 'cos':
                self.tau = 0.01 + 0.5 * (self.tau - 0.01) * (1 + np.cos(np.pi * self.epoch / self.args.max_epoch))
            else:
                raise RuntimeError('no such tau type')
            print('current tau: ', self.tau)

    def forward(self, input):
        if len(input.shape) == 6:
            bs, s, c, step, num_joint, m = input.shape
            input = input.view(bs * s, c, step, num_joint, m)
        else:
            bs, c, step, num_joint, m = input.shape
            s = 1
        # nctvm->nmcvt
        input = input.permute(0, 4, 1, 3, 2).contiguous().view(bs * m * s, c, num_joint, self.seg)
        # get input list with different size s b c v t
        input_list = downsample_input(input, self.transforms, len(self.gcn_types))
        # get input features s b c 1 t
        input_feats = get_input_feats(input_list, self.spa_nets)
        # batch_size num_action 1 t
        prob, action = get_litefeat_and_action(input_feats[0], self.tau, self.policy_net)
        # num_action, b11t
        action = action.permute(1, 0, 2, 3).unsqueeze(2)
        # b c 1 t
        input = (action * torch.stack(input_feats)).sum(0)
        # b c 1 t
        input = self.tem_net(input)
        # b c 1 1
        output = self.maxpool(input)
        # b c
        output = torch.flatten(output, 1)
        # b p
        output = self.fc(output)
        output = output.view(bs, m * s, -1).mean(1)

        return output, action

    def test(self, input):
        if len(input.shape) == 6:
            bs, s, c, step, num_joint, m = input.shape
            input = input.view(bs * s, c, step, num_joint, m)
        else:
            bs, c, step, num_joint, m = input.shape
            s = 1

        input = input.permute(0, 4, 1, 3, 2).contiguous().view(bs * m * s, c, num_joint, self.seg)  # nctvm->nmcvt

        input_list = downsample_input(input, self.transforms)  # get input list with different size s b c v t

        diff_list = [
            torch.cat([torch.zeros(*input.shape[:3], 1).zero_(), input[:, :, :, 1:] - input[:, :, :, 0:-1]], dim=-1) for
            input in input_list]

        policy_fea = self.spa_nets[0](input_list[0], diff_list[0])  # b c 1 t
        print(input_list[5])

        prob = torch.log(F.softmax(self.policy_net(policy_fea), dim=1).clamp(min=1e-8))  # batch_size num_action 1 t

        action = F.gumbel_softmax(prob, 1e-5, hard=True, dim=1)  # batch_size num_action 1 t
        features = []
        for i in range(bs * m * s):
            feats = []
            for j in range(step):
                a = action[i, :, 0, j].detach().cpu().numpy()
                a = int(np.argmax(a))
                if a == 0:
                    feats.append(policy_fea[i:i + 1, :, :, j:j + 1])
                else:
                    feats.append(self.spa_nets[a](input_list[a][i:i + 1, :, :, j:j + 1],
                                                  diff_list[a][i:i + 1, :, :, j:j + 1]))  # b c 1 1
            features.append(torch.cat(feats, dim=-1))

        spa_fea = torch.cat(features, dim=0)
        print(spa_fea.shape)
        input = self.tem_net(spa_fea)  # b c 1 t
        # Classification
        output = self.maxpool(input)  # b c 1 1
        output = torch.flatten(output, 1)  # b c
        output = self.fc(output)  # b p
        output = output.view(bs, m * s, -1).mean(1)

        return output

    def load_part(self, model, pretrained_dict, key):
        model_dict = model.state_dict()
        pretrained_dict = {k[len(key) + 1:]: v for k, v in pretrained_dict.items() if key in k}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    def load(self, checkpoints_paths, init_num=5):
        assert len(checkpoints_paths) == len(self.spa_nets)
        assert len(checkpoints_paths) != 0
        for i, path in enumerate(checkpoints_paths):
            state_dict = torch.load(path, map_location='cpu')['model']
            self.load_part(self.spa_nets[i], state_dict, 'spa_net')
            # self.load_part(self.joint_embeds[i], state_dict, 'joint_embed')
            # self.load_part(self.dif_embeds[i], state_dict, 'dif_embed')
            with torch.no_grad():
                self.transforms[i].data = state_dict['transform']
            if i == init_num:
                self.load_part(self.fc, state_dict, 'fc')
                self.load_part(self.tem_net, state_dict, 'tem_net')
        print(checkpoints_paths)


if __name__ == '__main__':
    import os
    from thop import profile

    num_js = [5, 9, 13, 17, 21, 25]
    num_j = num_js[-1]
    num_t = 20
    dim = 256
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'

    # Big = SpatialNet(5, 20)  # 0.01 0.07 0.15
    # Mid = SpatialNet(5, 20)  # 0.01 0.07 0.15
    # Sma = SpatialNet(5, 20)  # 0.01 0.07 0.15
    # flops1, params1 = get_model_complexity_info(Big, (3, 5, num_t), as_strings=True)  # 0.15
    # flops2, params2 = get_model_complexity_info(Mid, (3, 5, num_t), as_strings=True)  # 0.07
    # flops3, params3 = get_model_complexity_info(Sma, (3, 5, num_t), as_strings=True)  # 0.01
    # print(flops1, flops2, flops3)
    # print(params1, params2, params3)

    # temNet: 0.007
    # tconv: 4e-5 2-layer 0.002
    # transformer: 0.0019 or 2e-3
    # lstm: 6e-5  2-layer 0.004
    pretrained = [
        '../../pretrain_models/single_sgn_jpt{}.state'.format(j) for j in num_js
    ]
    model = ADASGN(60, num_js, num_t,
                   policy_type='tconv', tau=1e-5, pre_trains=None, init_type='fix', init_num=5,
                   adaptive_transform=[True, True, True, True, True, False], gcn_types=['small', 'mid', 'big'])
    model.eval()
    dummy_data = torch.randn([1, 3, num_t, num_j, 2])
    # o2, a2 = model(dummy_data)
    # o2.mean().backward()

    # o1 = model.test(dummy_data)
    o2, a2 = model(dummy_data)
    # print((o1 == o2).all())
    print('finish')
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
