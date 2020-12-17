from torch import nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from ptflops import get_model_complexity_info

from model.init_transforms import Transforms, Transformsadap


def one_hot(spa):
    y = torch.arange(spa).unsqueeze(-1)
    y_onehot = torch.FloatTensor(spa, spa)

    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)

    return y_onehot


def get_litefeat_and_action(input_feat, tau, policy_net):
    prob = torch.log(F.softmax(policy_net(input_feat), dim=1).clamp(min=1e-8))  # batch_size num_action 1 t

    action = F.gumbel_softmax(prob, tau, hard=True, dim=1)  # batch_size num_action 1 t

    return prob, action


def get_input_feats(input_list, models):
    output_list = []
    for input, model in zip(input_list, models):
        output_list.append(model(input))
    return output_list


def downsample_input(input, transforms, dif_embeds, joint_embeds):
    # n c v t
    out_list = []
    for i, t in enumerate(transforms):
        # t = t.to(input.device)
        output = torch.matmul(input.transpose(2, 3), t).transpose(2, 3).contiguous()
        bs, c, num_joints, step = output.shape
        dif = output[:, :, :, 1:] - output[:, :, :, 0:-1]
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)
        dif = dif_embeds[i](dif)
        pos = joint_embeds[i](output)
        output = pos + dif
        out_list.append(output)
    # bs, c, num_joints, step = input.shape
    # dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
    # dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)
    # dif = dif_embeds[-1](dif)
    # pos = joint_embeds[-1](input)
    # output = pos + dif
    # out_list.append(output)
    return out_list


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
            nn.init.constant_(self.conv.weight[-1:], 1 / k)
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


class ADASGN(nn.Module):
    def __init__(self, num_classes, num_joints, seg, bias=True, dim=256, tau=5., policy_kernel=3,
                 policy_dilate=1, adaptive_transform=[], policy_type='tconv', args=None, tau_decay=-0.045,
                 pre_trains=None, tau_type='cos', init_type='fix', init_num=5):
        super(ADASGN, self).__init__()

        self.seg = seg
        # if adaptive_transform:
        #     self.transforms = nn.ParameterList(
        #         [nn.Parameter(Transformsadap['M{}to{}'.format(num_joints[-1], i)], requires_grad=True) for i in
        #          num_joints])
        # else:
        self.transforms = nn.ParameterList(
            [nn.Parameter(Transforms['M{}to{}'.format(num_joints[-1], i)], requires_grad=adaptive_transform[ind]) for ind, i in
             enumerate(num_joints)])
            # self.transforms = [Transforms['M{}to{}'.format(num_joints[-1], i)] for i in num_joints]

        self.num_joints = num_joints
        self.seg = seg
        self.dim = dim
        self.bias = bias
        self.tau_decay = tau_decay
        self.tau_type = tau_type

        self.joint_embeds = nn.ModuleList(
            [embed(3, dim // 4, norm=True, bias=bias, num_joints=num_joint) for num_joint in num_joints])
        self.dif_embeds = nn.ModuleList(
            [embed(3, dim // 4, norm=True, bias=bias, num_joints=num_joint) for num_joint in num_joints])
        for m in self.joint_embeds.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        for m in self.dif_embeds.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.spa_nets = nn.ModuleList([SpatialNet(num_joint, bias, dim)
                                       for num_joint in num_joints])
        self.tem_net = TempolNet(seg, bias, dim)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(dim * 2, num_classes)
        self.tau = tau
        self.epoch = 0
        self.args = args

        if pre_trains is not None:
            self.load(pre_trains, init_num)

        # b c 1 t -> b 3 1 t
        if policy_type == 'tconv':
            self.policy_net = Tconv(dim, len(num_joints), k=policy_kernel, d=policy_dilate, init_type=init_type)
        elif policy_type == 'tconv2':
            self.policy_net = Tconv2(dim, dim // 2, len(num_joints), k=policy_kernel, d=policy_dilate,
                                     init_type=init_type)
        elif policy_type == 'transformer':
            self.policy_net = Transformer(dim, dim // 2, len(num_joints))
        elif policy_type == 'lstm':
            self.policy_net = Lstm(dim, dim // 2, len(num_joints))
        elif policy_type == 'tnet':
            self.policy_net = Temconv(dim, len(num_joints), k=policy_kernel, d=policy_dilate, seg=seg, dim=dim,
                                      init_type=init_type)
        else:
            raise RuntimeError('No such policy net')

        self.get_gflops_table()

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
            self.load_part(self.joint_embeds[i], state_dict, 'joint_embed')
            self.load_part(self.dif_embeds[i], state_dict, 'dif_embed')
            with torch.no_grad():
                self.transforms[i].data = state_dict['transform']
            if i == init_num:
                self.load_part(self.fc, state_dict, 'fc')
                self.load_part(self.tem_net, state_dict, 'tem_net')
        print(checkpoints_paths)

    def get_gflops_table(self):
        kwards = {
            'print_per_layer_stat': False,
            'as_strings': False
        }
        self.gflops_table = {}
        self.gflops_table['spanet'] = [get_model_complexity_info(m, (self.dim // 4, j, self.seg), **kwards)[0] / 1e9 for
                                       m, j in zip(self.spa_nets, self.num_joints)]
        self.gflops_table['temnet'] = \
            get_model_complexity_info(self.tem_net, (self.dim, 1, self.seg), **kwards)[0] / 1e9
        self.gflops_table['policy'] = \
            get_model_complexity_info(self.policy_net, (self.dim, 1, self.seg), **kwards)[0] / 1e9

        self.gflops_table['joint_embeds'] = [get_model_complexity_info(m, (3, j, self.seg), **kwards)[0] / 1e9 for
                                             m, j in zip(self.joint_embeds, self.num_joints)]
        self.gflops_table['diff_embeds'] = [get_model_complexity_info(m, (3, j, self.seg), **kwards)[0] / 1e9 for
                                            m, j in zip(self.dif_embeds, self.num_joints)]
        self.gflops_table['flops_fix'] = self.gflops_table['spanet'][0] + self.gflops_table['temnet'] + \
                                         self.gflops_table['policy'] + sum(self.gflops_table['joint_embeds']) \
                                         + sum(self.gflops_table['diff_embeds'])

        print("gflops_table: ")
        for k in self.gflops_table:
            print(k, self.gflops_table[k])

        self.gflops_vector = torch.FloatTensor(self.gflops_table['spanet'])
        # self.gflops_vector = torch.FloatTensor([self.gflops_table['flops_fix'], *self.gflops_table['spanet'][1:]])

    def get_policy_usage_str(self, action_list):
        actions_mean = np.concatenate(action_list, axis=1).mean(-1).squeeze(-1).squeeze(-1).mean(-1)  # num_act

        gflops = actions_mean[1:] * self.gflops_table['spanet'][1:]

        printed_str = 'gflops_fix: ' + str(self.gflops_table['flops_fix']) + ' gflops: ' + str(gflops) + ' actions: ' + str(actions_mean)
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
        input = input.permute(0, 4, 1, 3, 2).contiguous().view(bs * m * s, c, num_joint, self.seg)  # nctvm->nmcvt

        input_list = downsample_input(input, self.transforms, self.dif_embeds,
                                      self.joint_embeds)  # get input list with different size s b c v t

        input_feats = get_input_feats(input_list, self.spa_nets)  # get input features s b c 1 t

        prob, action = get_litefeat_and_action(input_feats[0], self.tau, self.policy_net)  # batch_size num_action 1 t

        action = action.permute(1, 0, 2, 3).unsqueeze(2)  # num_action, b11t

        input = (action * torch.stack(input_feats)).sum(0)  # b c 1 t

        # input = self.spa_net(input)  # b c 1 t
        input = self.tem_net(input)  # b c 1 t
        # Classification
        output = self.maxpool(input)  # b c 1 1
        output = torch.flatten(output, 1)  # b c
        output = self.fc(output)  # b p
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

        input_list = downsample_input(input, self.transforms, self.dif_embeds,
                                      self.joint_embeds)  # get input list with different size s b c v t

        policy_fea = self.spa_nets[0](input_list[0])  # b c 1 t

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
                    feats.append(self.spa_nets[a](input_list[a][i:i + 1, :, :, j:j + 1]))  # b c 1 1
            features.append(torch.cat(feats, dim=-1))

        spa_fea = torch.cat(features, dim=0)

        input = self.tem_net(spa_fea)  # b c 1 t
        # Classification
        output = self.maxpool(input)  # b c 1 1
        output = torch.flatten(output, 1)  # b c
        output = self.fc(output)  # b p
        output = output.view(bs, m * s, -1).mean(1)

        return output


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


class GCNMidS(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        self.gcn1 = gcn_spa(dim // 2, dim // 2, bias=bias)
        self.gcn2 = gcn_spa(dim // 2, dim, bias=bias)

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
    def __init__(self, num_joint, bias=True, dim=256):
        super().__init__()

        spa = one_hot(num_joint)
        spa = spa.permute(0, 3, 2, 1)
        self.register_buffer('spa', spa)

        self.spa_embed = embed(num_joint, dim // 4, norm=False, bias=bias, num_joints=num_joint)
        # self.joint_embed = embed(3, dim // 4, norm=True, bias=bias, num_joints=num_joint)
        # self.dif_embed = embed(3, dim // 4, norm=True, bias=bias, num_joints=num_joint)
        self.compute_g1 = compute_g_spa(dim // 2, dim, bias=bias)
        if num_joint > 20:
            self.gcn = GCNBig(dim, bias)
        elif 15 < num_joint <= 20:
            self.gcn = GCNMid(dim, bias)
        elif 10 < num_joint <= 15:
            self.gcn = GCNMidS(dim, bias)
        elif num_joint < 10:
            self.gcn = GCNSma(dim, bias)
        else:
            raise RuntimeError('No such config')

        # self.maxpool = nn.AdaptiveMaxPool2d((1, seg))
        self.maxpool = nn.MaxPool2d((num_joint, 1))

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
        # dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        # dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)
        # pos = self.joint_embed(input)
        spa1 = self.spa_embed(self.spa).repeat(bs, 1, 1, step)
        # dif = self.dif_embed(dif)
        # dy = pos + dif
        # Joint-level Module
        input = torch.cat([input, spa1], 1)
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
        self.cnn = local(dim, dim * 2, bias=bias, seg=seg)

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
        return g


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
                   policy_type='tconv', tau=1e-5, pre_trains=None, init_type='fix', init_num=5, adaptive_transform=[True, True, True, True, True, False])
    dummy_data = torch.randn([1, 3, num_t, num_j, 2])
    o2, a2 = model(dummy_data)
    o2.mean().backward()

    o1 = model.test(dummy_data)
    o2, a2 = model(dummy_data)
    print((o1 == o2).all())
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
