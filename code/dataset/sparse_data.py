import random
from dataset.rotation import *
from math import cos as cos
from math import sin as sin
import MinkowskiEngine as ME
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from numpy.lib.format import open_memmap
import os
import pickle


# def normalize(data_numpy, norm_range=[0, 1]):
#     C, T, V, M = data_numpy.shape
#     data_numpy = data_numpy.astype(np.float32)
#     scaler_feature = MinMaxScaler(feature_range=norm_range)
#     for m in range(M):
#         tmp = data_numpy[:, :, :, m].transpose(1, 2, 0).reshape((-1, C))
#         data_numpy[:, :, :, m] = scaler_feature.fit_transform(tmp).reshape((T, V, C)).transpose((2, 0, 1))
#     return data_numpy

def normalize(data_numpy, norm_range=[0, 1]):
    C, T, V, M = data_numpy.shape
    data_numpy = data_numpy.astype(np.float32)
    scaler_feature = MinMaxScaler(feature_range=norm_range)
    tmp = data_numpy
    non_zeros = M
    for m in range(M)[::-1]:
        if data_numpy[:, :, :, m].sum()==0:
            tmp = data_numpy[:, :, :, :m]
            non_zeros = m
        else:
            break
    tmp = tmp.transpose(1, 2, 3, 0).reshape((-1, C))
    data_numpy[:, :, :, :non_zeros] = scaler_feature.fit_transform(tmp).reshape((T, V, non_zeros, C)).transpose((3, 0, 1, 2))
    return data_numpy

def coor_to_volume(data, size):
    '''

    :param data: CTVM
    :param size: [D, H, W]
    :return: CTDHW
    '''
    C, T, V, M = data.shape
    volume = np.zeros([V * M, T, size[0], size[1], size[2]], dtype=np.float32)
    fst_ind = np.indices([T, V, M])[0]  # T, V, M
    # one_hots = np.concatenate([np.tile(np.eye(V), [M, 1]), np.repeat(np.eye(M), V, axis=0)], axis=1).reshape(
    #     (V, M, V + M)).transpose((2, 0, 1))
    one_hots = np.eye(V * M).reshape((M, V, V * M)).transpose((2, 1, 0))  # C, V, M
    scd_inds = (data[::-1, :, :, :] * (np.array(size) - 1)[:, np.newaxis, np.newaxis, np.newaxis]).astype(
        np.long)  # 3, T, V, M
    scd_inds = np.split(scd_inds, 3, axis=0)
    volume[:, fst_ind, scd_inds[0][0], scd_inds[1][0], scd_inds[2][0]] = one_hots[:, np.newaxis, :, :]
    return volume


class ConvertToSparse:
    def __init__(self, shape, size, dilate_value=0, edges=None, interval=None):
        """
        shape: [C, T, V, M]
        size: [T, D, H, W] 量化到多大的晶格上去
        dilate_value: 2->[-2,-1,1,2] 除时间维外都扩张
        edges: [ [1,2], [] ]
        interval: [1, 2, 3] 每根骨头的插值数目
        """
        C, T, V, M = shape  # T is the volume size
        size = np.array(size[1:]) - 1
        self.features = np.eye(V * M).reshape((M, V, V * M)).transpose((1, 0, 2))
        # self.T_tensor = np.repeat(np.linspace(0, 1, T), V * M).reshape([1, T, V, M])
        self.norm_tensor = size[:, np.newaxis, np.newaxis, np.newaxis]
        self.edges = edges
        self.interval = interval
        if edges is not None and interval is not None:
            self.use_edge = True
            self.weights = [np.linspace(0, 1, interval[i]).reshape([1, interval[i], 1, 1]) for i in range(len(edges))]
            print('use_edge')
        else:
            self.use_edge = False

        self.dilate_value = dilate_value
        if dilate_value != 0:
            self.use_dilate = True
            # [-1, 1] 点乘 CxC单位阵
            dilates = np.stack([np.eye(C) * i for i in range(-dilate_value, dilate_value + 1) if i != 0])
            # 时间维度pad0,也就是不加减
            dilates = np.concatenate([np.zeros([dilate_value * 2, C, 1]), dilates], axis=-1).reshape([-1, C + 1])
            self.dilates = dilates.transpose([1, 0])[:, np.newaxis, np.newaxis, :, np.newaxis]
            scale = np.linspace(0, 1, dilate_value + 2)[1:-1]
            self.scale = np.concatenate([scale, scale[::-1]]).repeat(C).reshape([1, 1, C * dilate_value * 2, 1, 1])
            print('use_dilate={}'.format(dilate_value))
        else:
            self.use_dilate = False

    def __call__(self, data, crop_range=None, rotation=None, move_range=None, scale=None):
        """
        crop_range: [[0, 64], [0, 64], [0, 64], [0, 64]] 去掉坐标在范围以外的值
        rotation: [agtd, agth, agtw, agdg, agdw, aghw] 做一个4D旋转, C42=6个自由度
        move_range: [l, h] 加一个与当前数据等大的张量,数据范围为l到h
        scale: 对各个维度进行缩放，原本低层能看到的缩放后可能得到高层
        """
        C, T, V, M = data.shape
        coords = data * self.norm_tensor
        T_tensor = np.repeat(np.linspace(0, T-1, T), V * M).reshape([1, T, V, M])
        coords = np.concatenate([T_tensor, coords], axis=0)  # add T dim

        features = np.tile(self.features, [T, 1, 1, 1])
        if self.use_edge:
            new_points = np.zeros([C + 1, T, sum(self.interval), M])
            new_features = np.zeros([T, sum(self.interval), M, V * M])
            j = 0
            for i, (v1, v2) in enumerate(self.edges):
                new_points[:, :, j:j + self.interval[i], :] = np.linspace(coords[:, :, v1], coords[:, :, v2],
                                                                          self.interval[i],
                                                                          axis=2)  # C T interval[i] M
                new_features[:, j:j + self.interval[i], :, :] = np.maximum(  # TODO
                    features[:, v1:v1 + 1] * self.weights[i],
                    features[:, v2:v2 + 1] * (1 - self.weights[i]))
                j += self.interval[i]
            coords = np.concatenate([coords, new_points], axis=2)
            features = np.concatenate([features, new_features], axis=1)

        if self.use_dilate:
            # dilates代表偏移值，加到原来的坐标上
            coords_dilate = coords[:, :, :, np.newaxis, :] + self.dilates
            coords_dilate = coords_dilate.reshape([C + 1, T, -1, M])
            # 特征乘一个scale, 类似于[0.6, 0.3, 0.3, 0.6]
            features_dilate = (np.repeat(features[:, :, np.newaxis, :, :], C * self.dilate_value * 2,
                                         axis=2) * self.scale).reshape([T, -1, M, V * M])
            coords = np.concatenate([coords, coords_dilate], axis=2)
            features = np.concatenate([features, features_dilate], axis=1)

        coords = coords.transpose((1, 2, 3, 0)).reshape([-1, C + 1])  # T V M C
        features = features.reshape([-1, V * M])

        if rotation is not None:
            coords = rot_sparse(coords, rotation)

        if scale is not None:
            coords = scale_sparse(coords, scale)

        if move_range is not None:
            coords = move_sparse(coords, move_range)

        if crop_range is not None:
            coords, features = crop_sparse(coords, features, crop_range)

        coords = coords.astype(np.int)
        coords, features = ME.utils.sparse_quantize(coords, features)
        # coords, features = sparse_quantize(coords, features)

        return np.array(coords, dtype=np.int32), np.array(features, dtype=np.float32)


def move_sparse(coords, move_range):
    move_tensor = np.random.random(coords.shape) * (move_range[1] - move_range[0]) + move_range[0]
    coords += move_tensor
    return coords


def crop_sparse(coords, features, crop_range):
    ind = (np.array(crop_range)[np.newaxis, :, 0] < coords) & (coords < np.array(crop_range)[np.newaxis, :, 1])
    coords, features = coords[ind.min(-1)], features[ind.min(-1)]
    return coords, features


def sample_sparse(coords, features, l):
    coords_new = []
    features_new = []
    for i, coord in enumerate(coords):
        new_ts = np.argwhere(np.array(l) == coord[0])
        if new_ts.shape[0] != 0:
            for new_t in new_ts[0]:
                coord[0] = new_t
                coords_new.append(coord)
                features_new.append(features[i])
    return np.array(coords_new), np.array(features_new)


def uniform_sample_sparse(coords, features, size):
    N, C = coords.shape
    T = coords[:, 0].max() + 1
    interval = T / size
    uniform_list = [int(i * interval) for i in range(size)]

    return sample_sparse(coords, features, uniform_list)


def random_choose_simple_sparse(coords, features, size, center=False):
    N, C = coords.shape
    T = coords[:, 0].max() + 1
    if size < 0:
        assert 'resize shape is not right'
    if T == size:
        return coords, features,
    elif T < size:
        return coords, features
    else:
        if center:
            begin = (T - size) // 2
        else:
            begin = random.randint(0, T - size)
        l = list(range(size))[begin:begin + size]
        return sample_sparse(coords, features, l)


def random_move_whole_sparse(coords, features, agx=0, agy=0, s=1):
    N, C = coords.shape
    agx = math.radians(agx)
    agy = math.radians(agy)
    Rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
    Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
    Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])

    coords[:, 1:] = np.dot(np.reshape(coords[:, 1:], (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
    return coords.astype(np.int), features


def rot_sparse(coords, angles):
    agtd, agth, agtw, agdh, agdw, aghw = [math.radians(rot) for rot in angles]  # tdhw->xyzu
    rtd = np.asarray([[cos(agtd), -sin(agtd), 0, 0], [sin(agtd), cos(agtd), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    rth = np.asarray([[cos(agth), 0, -sin(agth), 0], [0, 1, 0, 0], [sin(agth), 0, cos(agth), 0], [0, 0, 0, 1]])
    rtw = np.asarray([[cos(agtw), 0, 0, -sin(agtw)], [0, 1, 0, 0], [0, 0, 1, 0], [sin(agtw), 0, 0, cos(agtw)]])
    rdh = np.asarray([[1, 0, 0, 0], [0, cos(agdh), -sin(agdh), 0], [0, sin(agdh), cos(agdh), 0], [0, 0, 0, 1]])
    rdw = np.asarray([[1, 0, 0, 0], [0, cos(agdw), 0, -sin(agdw)], [0, 0, 1, 0], [0, sin(agdw), 0, cos(agdw)]])
    rhw = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, cos(aghw), -sin(aghw)], [0, 0, sin(aghw), cos(aghw)]])
    return np.dot(rtd, np.dot(rth, np.dot(rtw, np.dot(rdh, np.dot(rdw, np.dot(rhw, coords.transpose())))))).transpose()


def scale_sparse(coords, scale):
    rs = np.asarray([[scale[0], 0, 0, 0], [0, scale[1], 0, 0], [0, 0, scale[2], 0], [0, 0, 0, scale[3]]])
    return np.dot(coords, rs)

