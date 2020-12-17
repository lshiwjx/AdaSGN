import torch
from utility.vis_env import *

a = torch.load('../../work_dir/ntu60/sgnsingle25_transformfix30_valfirst_smallnew-latest.state', map_location='cpu')
# a = torch.load('../../pretrain_models/single_sgn_jpt5.state', map_location='cpu')
m = a['model']
s = abs(m['transform'].numpy())
plt.imshow(s, cmap='gray');plt.show()
s = a['model']['transforms']
print('fds')
