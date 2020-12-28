import numpy as np
from model.flops_count import get_model_complexity_info
from model.policy_layers import *
from model.init_transforms import Transforms


class RANSGN(nn.Module):
    def __init__(self, num_classes, num_joints, seg, adaptive_transform=[], bias=True, dim=256, gcn_types=[], init_num=5, pre_trains=None):
        super(RANSGN, self).__init__()

        self.seg = seg
        self.transforms = nn.ParameterList(
            [nn.Parameter(Transforms['M{}to{}'.format(num_joints[-1], i)], requires_grad=adaptive_transform[ind]) for
             ind, i in enumerate(num_joints) for gcn_type in gcn_types])
        self.num_joints = num_joints
        self.seg = seg
        self.dim = dim

        self.spa_nets = nn.ModuleList([SpatialNet(num_joint, bias, dim, gcn_type)
                                       for num_joint in num_joints for gcn_type in gcn_types])  # j1m1 j1m2 j2m1 ...
        self.tem_net = TempolNet(seg, bias, dim)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(dim * 2, num_classes)
        self.num_gcns = len(gcn_types)
        self.num_jpt = len(num_joints)
        self.num_action = self.num_gcns * self.num_jpt

        self.get_gflops_table()

        if pre_trains is not None:
            self.load(pre_trains, init_num)

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
        self.gflops_table['policy'] = 0

        self.gflops_table['flops_fix'] = self.gflops_table['spanet'][0] + self.gflops_table['temnet'] + \
                                         self.gflops_table['policy']
        self.gflops_vector = torch.FloatTensor(self.gflops_table['spanet'])

        print("gflops_table: ")
        for i in range(self.num_jpt):
            print('spanet', self.num_joints[i], self.gflops_table['spanet'][i * self.num_gcns:(i + 1) * self.num_gcns])

        print('temnet', self.gflops_table['temnet'])
        print('policy', self.gflops_table['policy'])
        print('flops_fix', self.gflops_table['flops_fix'])

    def get_policy_usage_str(self, action_list):
        actions_mean = np.concatenate(action_list, axis=1).mean(-1).squeeze(-1).squeeze(-1).mean(-1)  # num_act

        gflops = actions_mean[1:] * self.gflops_table['spanet'][1:]

        printed_str = 'Gflops_fix: ' + str(self.gflops_table['flops_fix']) \
                      + '\nGflops: ' + str(gflops) \
                      + '\nALL: ' + str(sum(gflops) + self.gflops_table['flops_fix']) \
                      + '\nActions: ' + str(actions_mean)
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
        # get input list with different size s b c v t
        input_list = downsample_input(input, self.transforms, len(self.gcn_types))
        # get input features s b c 1 t
        input_feats = get_input_feats(input_list, self.spa_nets)
        # batch_size num_action 1 t
        prob, action = get_random_action(bs*m*s, self.num_action, t)
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
