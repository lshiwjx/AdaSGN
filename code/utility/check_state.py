import torch
from code.utility.vis_env import *

a = torch.load('/home/lshi/Project/Pytorch/DSTA-Net/train_val_test/work_dir/ntu20/skenet_drop0_glo_sub2_atts11_ffd1357_noalpha-latest.state')
m = a['model']
s = a['model']['attention_layers.0.atts.0.glo_reg_att'].cpu().numpy()
