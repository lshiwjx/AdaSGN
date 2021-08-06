import pickle
import numpy as np
from dataset.skeleton import Skeleton

edge = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
        (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
        (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
        (19, 18), (21, 22), (22, 7), (23, 24), (24, 11))
interval = [10, 7, 2, 4, 5, 7, 6, 1, 5, 7, 6, 1, 2, 11, 12, 3, 2, 11, 12, 3, 1, 1, 1, 1]
edge1 = ()
edge9 = ((0, 1), (1, 2), (1, 3), (3, 4), (1, 5), (5, 6), (0, 7), (0, 8))


class NTU_SKE(Skeleton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, edge=edge, **kwargs)

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        self.data = np.load(self.data_path, mmap_mode='r')[:, :3]  # NCTVM

        return self.data[0].shape


if __name__ == '__main__':
    from dataset.vis import plot_skeleton, test_one, test_multi, plot_points, get_pts_from_sparse_array, get_pts_from_volume
    from utility.vis_env import *

    vid = 'S004C001P003R001A058'  # ntu60
    data_path = "../../data/ntu120/CS/test_data.npy"  # 63026+50919
    label_path = "../../data/ntu120/CS/test_label.pkl"

    kwards = {
        "window_size": 20,
        "final_size": 20,
        "random_choose": False,
        "center_choose": False,
        # "turn2to1": True
        # 'rotation': [[-30, 30],[-30, 30],[-30, 30]]
        # "to_sparse": [128, 64, 64, 64],
        # "to_volume": [16, 16, 16],
        # 'fea_augment': True,
        # 'eval': True
        # 'rot_norm': True
    }

    dataset = NTU_SKE(data_path, label_path, **kwards)
    # a = dataset.__getitem__(0)

    # dataset.statistic_distance_variance(1000)

    labels = open('../prepare/ntu/statistics/class_name.txt', 'r').readlines()

    test_one(dataset, plot_skeleton, lambda x: x.transpose(1, 0, 2, 3), vid=vid, edges=edge, is_3d=True, pause=5,
             labels=labels, view=1)
    # test_multi(dataset, plot_skeleton, lambda x: x[0].numpy().transpose(1, 0, 2, 3), labels=labels, skip=1000, edges=edge,
    #            is_3d=True, pause=0.01, view=1)

    # test_one(dataset, plot_points, lambda x: get_pts_from_sparse_array(*x), vid=vid, edges=edge, is_3d=True, pause=0.1,
    #          labels=labels, view=[0, 64])
    # test_multi(dataset, plot_points, lambda x: get_pts_from_sparse_array(*x), skip=10, edges=edge, is_3d=True, pause=0.1,
    #          labels=labels, view=[0, 16])

    # test_one(dataset, plot_points, lambda x: get_pts_from_volume(x, 25), vid=vid, edges=edge, is_3d=True, pause=0.1,
    #          labels=labels, view=[0, 16])
    # test_multi(dataset, plot_points, lambda x: get_pts_from_volume(x[0].numpy(), 25), skip=10, edges=edge, is_3d=True,
    #            pause=0.1, labels=labels, view=[0, 16])

    # dhg 16:95.8%  32:99.7%  48:99.9%
    # ntu 16:93.1%  3216:96.2% 1632:98.8  32:98.6% 48:99.4% 3248:99.4 6432:99.1% 3264:99.6% 64: 99.7%