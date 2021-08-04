import pickle
from tqdm import tqdm
import sys
from dataset.rotation import *

sys.path.extend(['../../'])

import numpy as np
import os


def normalize_skeletons(skeleton, origin=None, base_bone=None, zaxis=None, xaxis=None):
    '''

    :param skeleton: M, T, V, C(x, y, z)
    :param origin: int
    :param base_bone: [int, int]
    :param zaxis:  [int, int]
    :param xaxis:  [int, int]
    :return:
    '''

    M, T, V, C = skeleton.shape

    # print('move skeleton to begin')
    if skeleton.sum() == 0:
        raise RuntimeError('null skeleton')
    if skeleton[:, 0].sum() == 0:  # pad top null frames
        index = (skeleton.sum(-1).sum(-1).sum(0) != 0)
        tmp = skeleton[:, index].copy()
        skeleton *= 0
        skeleton[:, :tmp.shape[1]] = tmp

    if origin is not None:
        # print('sub the center joint #0 (wrist)')
        main_body_center = skeleton[0, 0, origin].copy()  # c
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)  # only for none zero frames
            skeleton[i_p] = (skeleton[i_p] - main_body_center) * mask

    if base_bone is not None:
        # skeleton /= base_bone
        # div base bone lenghth
        t = 0
        main_body_spine = 0
        while t < T and main_body_spine == 0:
            main_body_spine = np.linalg.norm(skeleton[0, t, base_bone[1]] - skeleton[0, t, base_bone[0]])
            t += 1
        # print(main_body_spine)
        if main_body_spine == 0:
            print('zero bone')
        else:
            skeleton /= main_body_spine

    if zaxis is not None:
        # print('parallel the bone between wrist(jpt 0) and MMCP(jpt 1) of the first person to the z axis')
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    skeleton[i_p, i_f, i_j] = np.dot(matrix_z, joint)

    if xaxis is not None:
        # print('parallel the bone in x plane between wrist(jpt 0) and TMCP(jpt 1) of the first person to the x axis')
        joint_left = skeleton[0, 0, xaxis[0]].copy()
        joint_right = skeleton[0, 0, xaxis[1]].copy()
        # axis = np.cross(joint_right - joint_left, [1, 0, 0])
        joint_left[2] = 0
        joint_right[2] = 0  # rotate by zaxis
        axis = np.cross(joint_right - joint_left, [1, 0, 0])
        angle = angle_between(joint_right - joint_left, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    skeleton[i_p, i_f, i_j] = np.dot(matrix_x, joint)

    # print(skeleton[0, 0, zaxis[0]], skeleton[0, 0, zaxis[1]], skeleton[0, 0, xaxis[0]], skeleton[0, 0, xaxis[1]])
    skeleton = np.transpose(skeleton, [3, 1, 2, 0])  # mtvc - ctvm
    return skeleton


def read_skeleton(ske_txt):
    ske_txt = open(ske_txt, 'r').readlines()
    skeletons = []
    for line in ske_txt:
        nums = line.split(' ')
        # num_frame = int(nums[0]) + 1
        coords_frame = np.array(nums).reshape((22, 3)).astype(np.float32)
        skeletons.append(coords_frame)
    num_frame = len(skeletons)
    skeletons = np.expand_dims(np.array(skeletons).transpose((2, 0, 1)), axis=-1)  # CTVM
    skeletons = np.transpose(skeletons, [3, 1, 2, 0])  # M, T, V, C
    return skeletons, num_frame


def gendata():
    root = '../../../data/raw/shrec_hand/'
    save_path = '../../../data/shrec_hand'
    train_split = open(os.path.join(root, 'train_gestures.txt'), 'r').readlines()
    val_split = open(os.path.join(root, 'test_gestures.txt'), 'r').readlines()

    skeletons_all_train = []
    names_all_train = []
    labels14_all_train = []
    labels28_all_train = []
    skeletons_all_val = []
    names_all_val = []
    labels14_all_val = []
    labels28_all_val = []

    for line in tqdm(train_split):
        line = line.rstrip()
        g_id, f_id, sub_id, e_id, label_14, label_28, size_seq = map(int, line.split(" "))
        src_path = os.path.join(root, "gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt"
                                .format(g_id, f_id, sub_id, e_id))
        skeletons, num_frame = read_skeleton(src_path)
        skeletons = normalize_skeletons(skeletons, origin=0, base_bone=[0, 10])
        # ske_vis(skeletons, view=1, pause=0.1)
        skeletons_all_train.append(skeletons)
        labels14_all_train.append(label_14-1)
        labels28_all_train.append(label_28-1)
        names_all_train.append("{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id))

    pickle.dump(skeletons_all_train, open(os.path.join(save_path, 'train_skeleton.pkl'), 'wb'))
    pickle.dump([names_all_train, labels14_all_train],
                open(os.path.join(save_path, 'train_label_14.pkl'), 'wb'))
    pickle.dump([names_all_train, labels28_all_train],
                open(os.path.join(save_path, 'train_label_28.pkl'), 'wb'))

    for line in tqdm(val_split):
        line = line.rstrip()
        g_id, f_id, sub_id, e_id, label_14, label_28, size_seq = map(int, line.split(" "))
        src_path = os.path.join(root, "gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt"
                                .format(g_id, f_id, sub_id, e_id))
        skeletons, num_frame = read_skeleton(src_path)
        skeletons = normalize_skeletons(skeletons, origin=0, base_bone=[0, 10])

        skeletons_all_val.append(skeletons)
        labels14_all_val.append(label_14-1)
        labels28_all_val.append(label_28-1)
        names_all_val.append("{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id))

    pickle.dump(skeletons_all_val, open(os.path.join(save_path, 'val_skeleton.pkl'), 'wb'))
    pickle.dump([names_all_val, labels14_all_val],
                open(os.path.join(save_path, 'val_label_14.pkl'), 'wb'))
    pickle.dump([names_all_val, labels28_all_val],
                open(os.path.join(save_path, 'val_label_28.pkl'), 'wb'))


if __name__ == '__main__':
    gendata()
