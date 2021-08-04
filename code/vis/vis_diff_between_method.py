import os

os.environ['DISPLAY'] = 'localhost:10.0'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def acc_of_each_cls_from_score(score, label, num_cls):
    right_nums = [0 for _ in range(num_cls)]
    total_nums = [0 for _ in range(num_cls)]
    for i in range(len(label[0])):
        _, l = label[:, i]
        _, s = score[i]
        r = np.argmax(s)
        right_nums[int(l)] += int(r == int(l))
        total_nums[int(l)] += 1
    accs = [x/y for x,y in zip(right_nums, total_nums)]
    print('total acc: ', sum(accs)/num_cls)
    return accs

#
# def get_m(f):
#     lines = f.readlines()
#     pre_list = []
#     true_list = []
#     for line in lines:
#         pre, true = line[:-1].split(',')
#         pre_list.append(int(pre))
#         true_list.append(int(true))
#     m = confusion_matrix(true_list, pre_list)
#     return m

import pickle
import matplotlib

font = {
    # 'family': 'aria',
        'size': 12}
matplotlib.rc('font', **font)
label = open('../../data/shrec/val_label_14.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open(
    '/home/lshi/Project/Pytorch/EfficientVideoNet/work_dir/shrec/sgnadapre_alpha1warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001/score.pkl',
    'rb')
# r1 = open(
#     '/home/lshi/Project/Pytorch/DSTA-Net/train_val_test/work_dir/ntu60msg3d/skenet_drop02_glo_sub4_atts11_ffd1357k5_5123s2_b64_newdata_lr01_noaug_rot15_soft/score.pkl',
#     'rb')
r1 = list(pickle.load(r1).items())
# r2 = open(
#     '/home/lshi/Project/Pytorch/DSTA-Net/train_val_test/work_dir/shrec14/spanet4D_drop02_12832_d1357_512s2/score.pkl',
#     'rb')
# r2 = open(
#     '/home/lshi/Project/Pytorch/DSTA-Net/train_val_test/work_dir/ntu60/spanet4D_drop02_12832_d1357_5124s2_b64_k55_newdata/score.pkl',
#     'rb')
# r2 = list(pickle.load(r2).items())
num_class = 14
# num_class = 60
a1 = acc_of_each_cls_from_score(r1, label, num_class)
# a2 = acc_of_each_cls_from_score(r2, label, num_class)
a2 = np.load('/home/lshi/Project/Pytorch/EfficientVideoNet/work_dir/shrec/sgnadapre_alpha1warm5_policyran_lineartau5_transformfix30_models6fix30_lr0001/action.npy')
a2 /= 22
# total = open('/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/jester_c3d_kpre_32f_valt.txt')
# total_regular = open('/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/jester_nopositiont.txt')
# total_reverse = open('/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/jester_nopositiont.txt')
# total_single = open('/home/lshi/Project/Pytorch/3d-resnet-pytorch/train_val_test/jester_nopositiont.txt')
# m_reg = get_m(total_regular)
# m_rev = get_m(total_reverse)
# m_sin = get_m(total_single)
# plt.figure(figsize=[12.5,7])
plt.figure(figsize=[8,4])
# plt.title('Recognition Accuracy for each classes')
# plt.ylabel('ACC')
# reg = []
# for index, line in enumerate(m_reg):
#     reg.append(line[index] / sum(line))
# rev = []
# for index, line in enumerate(m_rev):
#     rev.append(line[index] / sum(line))
# sin = []
# for index, line in enumerate(m_sin):
#     sin.append(line[index] / sum(line))
index = np.arange(len(a1))
bar_width = 0.3
rects_reg = plt.bar(index - bar_width, a1, bar_width, color='#4bacc6')
rects_rev = plt.bar(index, a2, bar_width, color='#4f81bd')
# rects_sin = plt.bar(index + bar_width, a4, bar_width, color='#8064a2')

# diff_reg_rev = [a2[i] - a1[i] for i in range(num_class)]
# diff_reg_sin = [a3[i] - a4[i] for i in range(num_class)]
# diff_line_rev, = plt.plot(diff_reg_rev, color='orange', linestyle='-.')
# diff_line_sin, = plt.plot(diff_reg_sin, color='#ff007f', linestyle='-.')
label_file = open("../prepare/shrec/label.txt")
# label_file = open("../prepare/ntu_60/label.txt")
classes = label_file.readlines()
classes = [x[:-1] for x in classes]
tick_marks = np.arange(len(classes))
# plt.ylim([-0.2, 1.2])
plt.ylim([0, 1.2])
plt.xticks(tick_marks, classes, rotation=45)
plt.axes().spines['top'].set_visible(False)
plt.axes().spines['right'].set_visible(False)
# plt.legend((rects_reg[0], rects_rev[0], diff_line_rev),
#            ('sem', 'spa',  'sem - spa'), loc=0, ncol=5)
plt.legend((rects_reg[0], rects_rev[0]),
           ('Acc', 'Jpt',), loc=0, ncol=5)
# plt.show()
plt.savefig('../../vis_results/acc_joint.pdf', format='pdf', bbox_inches='tight')
