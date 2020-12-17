import cv2
from numpy import random as nprand
import random
import imutils
import numpy as np
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt
import warnings
import math
from dataset.rotation import *

warnings.filterwarnings('ignore', '.*output shape of zoom.*')


def zoom_T(p, target_l=64):
    '''

    :param p: ctv
    :param target_l:
    :return:
    '''
    C, T, V, M = p.shape
    p_new = np.empty([C, target_l, V, M])
    for m in range(M):
        for v in range(V):
            for c in range(C):
                p_new[c, :, v, m] = inter.zoom(p[c, :, v, m], target_l / T)[:target_l]
    return p_new


def filter_T(p, kernel_size=3):
    C, T, V, M = p.shape
    p_new = np.empty([C, T, V, M])
    for m in range(M):
        for v in range(V):
            for c in range(C):
                p_new[c, :, v, m] = medfilt(p[c, :, v, m], kernel_size=kernel_size)
    return p_new


def uniform_sample_np(data_numpy, size):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    interval = T / size
    uniform_list = [int(i * interval) for i in range(size)]
    return data_numpy[:, uniform_list]


def random_sample_np(data_numpy, size):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    interval = T / size
    random_list = [int(i * interval + np.random.randint(interval * 10) / 10) for i in range(size)]
    return data_numpy[:, random_list]


def random_sample_group(data_numpy, size, group):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    interval = T / size
    final = []
    for i in range(group):
        random_list = [int(i * interval + np.random.randint(interval * 10) / 10) for i in range(size)]
        final.append(data_numpy[:, random_list])
    return np.stack(final)


def random_choose_simple(data_numpy, size, center=False):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if size < 0:
        assert 'resize shape is not right'
    if T == size:
        return data_numpy
    elif T < size:
        return data_numpy
    else:
        if center:
            begin = (T - size) // 2
        else:
            begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[0.0],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)  # 需要变换的帧的段数 0, 16, 32
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):  # 使得每一帧的旋转都不一样
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]  # pingyi bianhuan
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def rotate_skeleton(data_numpy, angles=None):
    if angles is None:
        return data_numpy
    agz, agy, agx = angles
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_numpy = data_numpy.transpose((1, 2, 3, 0)).reshape(-1, C)

    agx = math.radians(agx)
    agy = math.radians(agy)
    Rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
    Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
    Rz = np.asarray([[math.cos(agz), math.sin(agz), 0], [-math.sin(agz), math.cos(agz), 0], [0, 0, 1]])

    data_numpy = np.dot(np.reshape(data_numpy, (-1, 3)), np.dot(Rz, np.dot(Ry, Rx)))
    data_numpy = data_numpy.reshape((T, V, M, C)).transpose((3, 0, 1, 2))
    return data_numpy


def scale_skeleton(data_numpy,  s=None):
    if s is None:
        return data_numpy
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_numpy = data_numpy.transpose((1, 2, 3, 0)).reshape(-1, C)

    Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])

    data_numpy = np.dot(np.reshape(data_numpy, (-1, 3)), Ss)
    data_numpy = data_numpy.reshape((T, V, M, C)).transpose((3, 0, 1, 2))
    return data_numpy


def turn_two_to_one(seq):
    # c t v m -> t m v c
    seq = seq.transpose((1, 3, 2, 0))
    new_seq = list()
    for idx, ske in enumerate(seq):
        if ske[0].sum()==0:
            new_seq.append(ske[1:2])
        elif ske[1].sum()==0:
            new_seq.append(ske[0:1])
        else:
            new_seq.append(ske[0:1])
            new_seq.append(ske[1:2])
    new_seq = np.array(new_seq).transpose((3, 0, 2, 1))
    return new_seq


def rot_to_fix_angle_fstframe(skeleton, jpts=[0, 1], axis=[0, 0, 1], frame=0, person=0, fix_dim=None):
    '''
    :param skeleton: c t v m
    :param axis: 001 for z, 100 for x, 010 for y
    '''
    skeleton = np.transpose(skeleton, [3, 1, 2, 0])  # M, T, V, C
    joint_bottom = skeleton[person, frame, jpts[0]].copy()
    joint_top = skeleton[person, frame, jpts[1]].copy()
    if fix_dim is not None:
        joint_bottom[fix_dim] = 0
        joint_top[fix_dim] = 0
    axis_c = np.cross(joint_top - joint_bottom, axis)
    angle = angle_between(joint_top - joint_bottom, axis)
    matrix_z = rotation_matrix(axis_c, angle)
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        for i_f, frame in enumerate(person):
            if frame.sum() == 0:
                continue
            for i_j, joint in enumerate(frame):
                skeleton[i_p, i_f, i_j] = np.dot(matrix_z, joint)
    return skeleton.transpose((3, 1, 2, 0))


def sub_center_jpt_fstframe(skeleton, jpt=0, frame=0, person=0):
    C, T, V, M = skeleton.shape
    skeleton = np.transpose(skeleton, [3, 1, 2, 0])  # M, T, V, C
    main_body_center = skeleton[person, frame, jpt].copy()  # c
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        mask = (person.sum(-1) != 0).reshape(T, V, 1)  # only for none zero frames
        skeleton[i_p] = (skeleton[i_p] - main_body_center) * mask
    return skeleton.transpose((3, 1, 2, 0))


def sub_center_jpt_perframe(skeleton, jpt=0, person=0):
    C, T, V, M = skeleton.shape
    skeleton = np.transpose(skeleton, [3, 1, 2, 0])  # M, T, V, C
    main_body_center = skeleton[person, :, jpt].copy().reshape((T, 1, C))  # tc
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        skeleton[i_p] = (skeleton[i_p] - main_body_center)  # TVC-T1C
    return skeleton.transpose((3, 1, 2, 0))


def decouple_spatial(skeleton, edges=()):
    tmp = np.zeros(skeleton.shape)
    for v1, v2 in edges:
        tmp[:, :, v2, :] = skeleton[:, :, v2] - skeleton[:, :, v1]
    return tmp


def obtain_angle(skeleton, edges=()):
    tmp = skeleton.copy()
    for v1, v2 in edges:
        v1 -= 1
        v2 -= 1
        x = skeleton[0, :, v1, :] - skeleton[0, :, v2, :]
        y = skeleton[1, :, v1, :] - skeleton[1, :, v2, :]
        z = skeleton[2, :, v1, :] - skeleton[2, :, v2, :]
        atan0 = np.arctan2(y, x) / 3.14
        atan1 = np.arctan2(z, x) / 3.14
        atan2 = np.arctan2(z, y) / 3.14
        t = np.stack([atan0, atan1, atan2], 0)
        tmp[:, :, v1, :] = t
    return tmp


def decouple_temporal(skeleton, inter_frame=1):  # CTVM
    skeleton = skeleton[:, ::inter_frame]
    diff = skeleton[:, 1:] - skeleton[:, :-1]
    return diff


def norm_len_fstframe(skeleton, jpts=[0, 1], frame=0, person=0):
    C, T, V, M = skeleton.shape
    skeleton = np.transpose(skeleton, [3, 1, 2, 0])  # M, T, V, C
    main_body_spine = np.linalg.norm(skeleton[person, frame, jpts[0]] - skeleton[person, frame, jpts[1]])
    if main_body_spine == 0:
        print('zero bone')
    else:
        skeleton /= main_body_spine
    return skeleton.transpose((3, 1, 2, 0))


def random_move_joint(data_numpy, sigma=0.1):  # 只随机扰动坐标点
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape

    rand_joint = np.random.randn(C, T, V, M) * sigma

    return data_numpy + rand_joint


def pad_recurrent(data):
    skeleton = np.transpose(data, [3, 1, 2, 0])  # C, T, V, M  to  M, T, V, C
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        if person[0].sum() == 0:  # TVC 去掉头空帧，然后对齐到顶端
            index = (person.sum(-1).sum(-1) != 0)
            tmp = person[index].copy()
            person *= 0
            person[:len(tmp)] = tmp
        for i_f, frame in enumerate(person):
            if frame.sum() == 0:
                if person[i_f:].sum() == 0:  # 循环pad之前的帧
                    rest = len(person) - i_f
                    num = int(np.ceil(rest / i_f))
                    pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                    skeleton[i_p, i_f:] = pad
                    break
    return skeleton.transpose((3, 1, 2, 0))  # ctvm


def pad_recurrent_fix(data, length):  # CTVM
    if data.shape[1] < length:
        num = int(np.ceil(length / data.shape[1]))
        data = np.concatenate([data for _ in range(num)], 1)[:, :length]
    return data


def pad_zero(data, length):
    if data.shape[1] < length:
        new = np.zeros([data.shape[0], length - data.shape[1], data.shape[2], data.shape[3]])
        data = np.concatenate([data, new], 1)
    return data
