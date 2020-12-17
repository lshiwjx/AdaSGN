from model.layers import *


class SGN(nn.Module):
    def __init__(self, num_classes, num_joint, seg, bias=True, dim=256):
        super(SGN, self).__init__()

        self.seg = seg

        self.spa_net = SpatialNet(num_joint, bias, dim, gcn_type='mid')
        self.tem_net = TempolNet(seg, bias, dim)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(dim * 2, num_classes)

    def forward(self, input):
        if len(input.shape) == 6:
            bs, s, c, step, num_joints, m = input.shape
            input = input.view(bs * s, c, step, num_joints, m)
        else:
            bs, c, step, num_joints, m = input.shape
            s = 1
        input = input.permute(0, 4, 1, 3, 2).contiguous().view(bs * m * s, c, num_joints, self.seg)  # nctvm->nmcvt

        dif = torch.cat([torch.zeros([*input.shape[:3], 1], device=input.device),
                         input[:, :, :, 1:] - input[:, :, :, 0:-1]], dim=-1)

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
    from ptflops import get_model_complexity_info
    from thop import profile

    num_j = 25
    num_t = 20
    os.environ['CUDA_VISIBLE_DEVICE'] = '6'
    model = SGN(60, num_j, num_t)
    # dummy_data = torch.randn([1, 3, num_t, num_j, 2])
    # hooks = {}
    # flops, params = profile(model, inputs=(dummy_data,), custom_ops=hooks)
    # gflops = flops / 1e9
    # params = params / 1e6
    #
    # print(gflops)
    # print(params)

    flops, params = get_model_complexity_info(model, (3, num_t, num_j, 1), as_strings=True)  # not support

    print(flops)  # 0.16 gmac
    print(params)  # 0.69 m
