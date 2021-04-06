import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
# parser.add_argument('--label', default='/home/lshi/Project/Pytorch/EfficientVideoNet/data/ntu60/CS/test_label.pkl', help='')
# parser.add_argument('--joint', default='../work_dir/ntu60/sgnadapre_alpha4warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001')
# parser.add_argument('--bone', default='../work_dir/ntu60/sgnadapre_alpha4warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001_bone')
# parser.add_argument('--vel', default='../work_dir/ntu60/sgnadapre_alpha4warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001_vel')
# parser.add_argument('--joint', default='../work_dir/ntu60/sgn_2020_newdata_rot17')
# parser.add_argument('--bone', default='../work_dir/ntu60/sgn_2020_newdata_rot17_bone')
# parser.add_argument('--vel', default='../work_dir/ntu60/sgn_2020_newdata_rot17_vel')
# parser.add_argument('--temporal_fast', default='../work_dir/ntu60/sgnadapre_alpha06warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001_bone')
# parser.add_argument('--other', default='../work_dir/ntu60/sgnadapre_alpha06warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001_bone')

# parser.add_argument('--label', default='/home/lshi/Project/Pytorch/EfficientVideoNet/data/ntu60/CV/test_label.pkl', help='')
# parser.add_argument('--joint', default='../work_dir/ntu60cv/sgnadapre_alpha4warm0_policyran_lineartau5_transformadap_models6_lr0001_rotnorm')
# parser.add_argument('--bone', default='../work_dir/ntu60cv/sgnadapre_alpha4warm0_policyran_lineartau5_transformadap_models6_lr0001_rotnorm_bone')
# parser.add_argument('--vel', default='../work_dir/ntu60cv/sgnadapre_alpha4warm0_policyran_lineartau5_transformadap_models6_lr0001_rotnorm_vel')
# parser.add_argument('--joint', default='../work_dir/ntu60cv/sgn_2020_newdata_rot17_rotnorm')
# parser.add_argument('--bone', default='../work_dir/ntu60cv/sgn_2020_newdata_rot17_rotnorm_bone')
# parser.add_argument('--vel', default='../work_dir/ntu60cv/sgn_2020_newdata_rot17_rotnorm_vel')
# parser.add_argument('--temporal_fast', default='../work_dir/ntu60cv/sgnadapre_alpha06warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001_rotnorm_bone')
# parser.add_argument('--other', default='../work_dir/ntu60cv/sgnadapre_alpha06warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001_rotnorm_bone') #
# parser.add_argument('--alpha', default=[3, 1, 1, 0, 0], help='weighted summation')


# parser.add_argument('--label', default='../data/shrec/val_label_14.pkl', help='')
# parser.add_argument('--joint', default='../work_dir/shrec/sgnadapre_alpha1warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001')
# parser.add_argument('--bone', default='../work_dir/shrec/sgnadapre_alpha1warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001_bone')
# parser.add_argument('--vel', default='../work_dir/shrec/sgnadapre_alpha1warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001_vel')
# # parser.add_argument('--joint', default='../work_dir/shrec/sgn_2020_newdata')
# # parser.add_argument('--bone', default='../work_dir/shrec/sgn_2020_newdata_bone')
# # parser.add_argument('--vel', default='../work_dir/shrec/sgn_2020_newdata_vel')
# parser.add_argument('--temporal_fast', default='../work_dir/shrec/sgn_2020_newdata_vel')
# parser.add_argument('--other', default='../work_dir/shrec/sgn_2020_newdata_vel') #

# parser.add_argument('--label', default='../data/shrec/val_label_28.pkl', help='')
# parser.add_argument('--joint', default='../work_dir/shrec28/sgnadapre_alpha4warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001')
# parser.add_argument('--bone', default='../work_dir/shrec28/sgnadapre_alpha4warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001_bone')
# parser.add_argument('--vel', default='../work_dir/shrec28/sgnadapre_alpha4warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001_vel')
# parser.add_argument('--joint', default='../work_dir/shrec28/sgn_2020_newdata')
# parser.add_argument('--bone', default='../work_dir/shrec28/sgn_2020_newdata_bone')
# parser.add_argument('--vel', default='../work_dir/shrec28/sgn_2020_newdata_vel')
# parser.add_argument('--temporal_fast', default='../work_dir/shrec28/sgn_2020_newdata_vel')
# parser.add_argument('--other', default='../work_dir/shrec28/sgn_2020_newdata_vel') #
# parser.add_argument('--alpha', default=[2, 1, 1, 0, 0], help='weighted summation')

# parser.add_argument('--label', default='/home/lshi/Project/Pytorch/EfficientVideoNet/data/ntu120/CS/test_label.pkl', help='')
# parser.add_argument('--joint', default='../work_dir/ntu120/sgnadapre_alpha4warm5_policyran_lineartau5_transformfix30_models3fix30_lr0001')
# parser.add_argument('--bone', default='../work_dir/ntu120/sgnadapre_alpha4warm5_policyran_lineartau5_transformfix30_models3fix30_lr0001_bone')
# parser.add_argument('--vel', default='../work_dir/ntu120/sgnadapre_alpha4warm5_policyran_lineartau5_transformfix30_models3fix30_lr0001_vel')
# # parser.add_argument('--joint', default='../work_dir/ntu120/sgn_2020_newdata_rot17')
# # parser.add_argument('--bone', default='../work_dir/ntu120/sgn_2020_newdata_rot17_bone')
# # parser.add_argument('--vel', default='../work_dir/ntu120/sgn_2020_newdata_rot17_vel')
# parser.add_argument('--temporal_fast', default='../work_dir/ntu120/sgn_2020_newdata_rot17_vel')
# parser.add_argument('--other', default='../work_dir/ntu120/sgn_2020_newdata_rot17_vel')
# parser.add_argument('--alpha', default=[3, 3, 1, 0, 0], help='weighted summation')

parser.add_argument('--label', default='/home/lshi/Project/Pytorch/EfficientVideoNet/data/ntu120/CE/test_label.pkl', help='')
# parser.add_argument('--joint', default='../work_dir/ntu120ce/sgnadapre_alpha4warm5_policyfix_lineartau5_transformfix30_models6fix30_lr0001')
# parser.add_argument('--bone', default='../work_dir/ntu120ce/sgnadapre_alpha4warm5_policyfix_lineartau5_transformfix30_models6fix30_lr0001_bone')
# parser.add_argument('--vel', default='../work_dir/ntu120ce/sgnadapre_alpha4warm5_policyfix_lineartau5_transformfix30_models6fix30_lr0001_vel')
parser.add_argument('--joint', default='../work_dir/ntu120ce/sgn_2020_newdata_rot17')
parser.add_argument('--bone', default='../work_dir/ntu120ce/sgn_2020_newdata_rot17_bone')
parser.add_argument('--vel', default='../work_dir/ntu120ce/sgn_2020_newdata_rot17_vel')
parser.add_argument('--temporal_fast', default='../work_dir/ntu120ce/sgn_2020_newdata_rot17')
parser.add_argument('--other', default='../work_dir/ntu120ce/sgn_2020_newdata_rot17')
parser.add_argument('--alpha', default=[3, 3, 1, 0, 0], help='weighted summation')
arg = parser.parse_args()

label = open(arg.label, 'rb')
label = np.array(pickle.load(label))
r1 = open('{}/score.pkl'.format(arg.joint), 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('{}/score.pkl'.format(arg.bone), 'rb')
r2 = list(pickle.load(r2).items())
r3 = open('{}/score.pkl'.format(arg.vel), 'rb')
r3 = list(pickle.load(r3).items())
r4 = open('{}/score.pkl'.format(arg.temporal_fast), 'rb')
r4 = list(pickle.load(r4).items())
r5 = open('{}/score.pkl'.format(arg.other), 'rb')
r5 = list(pickle.load(r5).items())
right_num = total_num = right_num_5 = 0
final_scores = []
final_labels = []
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    _, r33 = r3[i]
    _, r44 = r4[i]
    _, r55 = r5[i]
    r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3] + r55 * arg.alpha[4]
    final_scores.append(r/5)
    final_labels.append(int(l))
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)

# np.save('./score_ntu_cs.npy', np.array(final_scores))
# np.save('./label_ntu_cs.npy', np.array(final_labels))