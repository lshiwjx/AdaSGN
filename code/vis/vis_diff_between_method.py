# show accuracy and the number of joints used for each action.
import os

os.environ['DISPLAY'] = 'localhost:10.0'
import numpy as np
import matplotlib.pyplot as plt


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


import pickle
import matplotlib

font = {
    # 'family': 'aria',
        'size': 12}
matplotlib.rc('font', **font)
label = open('../../data/ntu60/CS/test_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open(
    '../../work_dir/ntu60/sgnadapre/score.pkl',
    'rb')
r1 = list(pickle.load(r1).items())

# num_class = 14
num_class = 60
a1 = acc_of_each_cls_from_score(r1, label, num_class)
a2 = np.load('../../work_dir/ntu60/sgnadapre/action.npy')
a2 /= 25  # the number of original input joints

plt.figure()
index = np.arange(len(a1))
bar_width = 0.3
rects_reg = plt.bar(index, a1, bar_width, color='#4bacc6')
rects_rev = plt.bar(index + bar_width, a2, bar_width, color='#4f81bd')

label_file = open("../prepare/ntu60/statistics/label.txt")
# label_file = open("../prepare/shrec/label.txt")
classes = label_file.readlines()
classes = [x[:-1] for x in classes]
tick_marks = np.arange(len(classes))
plt.ylim([0, 1.2])
# plt.xticks(tick_marks, classes, rotation=90)
# plt.axes().spines['top'].set_visible(False)
# plt.axes().spines['right'].set_visible(False)
plt.legend((rects_reg[0], rects_rev[0]),
           ('Acc', '#Jpt',), loc=0, ncol=5)
# plt.show()
plt.savefig('../../vis_results/acc_joint.png', format='png', bbox_inches='tight')
