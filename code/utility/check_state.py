import torch
from utility.vis_env import *

a = torch.load('../../work_dir/ntu60/sgnadapre_alpha4warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001-best.state', map_location='cpu')
# a = torch.load('../../pretrain_models/single_sgn_jpt5.state', map_location='cpu')
m = a['model']
s = abs(m['transforms.5'].numpy())
plt.imshow(s, cmap='gray');plt.show()
s = a['model']['transforms']
print('fds')
