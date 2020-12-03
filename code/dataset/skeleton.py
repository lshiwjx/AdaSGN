from torch.utils.data import DataLoader, Dataset
from dataset.skeleton_data import *
from dataset.sparse_data import *


class Skeleton(Dataset):
    def __init__(self, data_path, label_path, window_size, final_size,
                 decouple_spatial=False, num_skip_frame=None,
                 random_choose=False, center_choose=False,
                 to_volume=None, to_sparse=None, edge=None, dilate_value=0, interval=None,
                 random_crop=False, crop_range=None, rotation=None, move_range=None, scale=None, fea_augment=False,
                 eval=False, turn2to1=False):
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.center_choose = center_choose
        self.window_size = window_size
        self.final_size = final_size
        self.num_skip_frame = num_skip_frame
        self.decouple_spatial = decouple_spatial
        self.fea_augment = fea_augment
        self.edge = edge
        self.eval=eval
        self.rotation = rotation
        self.move_range = move_range
        self.scale = scale
        self.turn2to1 = turn2to1
        C, T, V, M = self.load_data()
        self.to_sparse = to_sparse
        self.to_volume = to_volume
        if to_sparse is not None:
            self.SparseConverter = ConvertToSparse([C, to_sparse[0], V, M], to_sparse, dilate_value, self.edge, interval)  # T has no use
            self.random_crop = random_crop
            self.crop_range = crop_range
            self.dilate_value = dilate_value
            self.interval = interval

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        return self.data[0].shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = int(self.label[index])
        sample_name = self.sample_name[index]
        data_numpy = np.array(data_numpy)  # nctv

        data_numpy = data_numpy[:, data_numpy.sum(0).sum(-1).sum(-1) != 0]  # CTVM

        if self.turn2to1:
            data_numpy = turn_two_to_one(data_numpy)

        # data transform
        if self.decouple_spatial:
            data_numpy = decouple_spatial(data_numpy, edges=self.edge)
        if self.num_skip_frame is not None:
            velocity = decouple_temporal(data_numpy, self.num_skip_frame)
            C, T, V, M = velocity.shape
            data_numpy = np.concatenate((velocity, np.zeros((C, 1, V, M))), 1)

        if self.eval:
            data_numpy = random_sample_group(data_numpy, self.window_size, self.eval)
        else:
            if self.random_choose:
                data_numpy = random_sample_np(data_numpy, self.window_size)
            else:
                data_numpy = uniform_sample_np(data_numpy, self.window_size)
            if self.center_choose:
                data_numpy = random_choose_simple(data_numpy, self.final_size, center=True)
            else:
                data_numpy = random_choose_simple(data_numpy, self.final_size)

            if self.rotation is not None:
                rotation = [random.randint(x, y) for x, y in self.rotation]
            else:
                rotation = None
            if self.scale is not None:
                scale = [random.random() * (y - x) + x for x, y in self.scale]
            else:
                scale = None

            if self.to_volume is not None:
                data_numpy = normalize(data_numpy, [0, 1])
                data_numpy = coor_to_volume(data_numpy, self.to_volume).astype(np.float32)
            if self.to_sparse is not None:
                data_numpy = normalize(data_numpy, [0, 1])
                if self.random_crop:
                    crop_range = [[random.randint(0, x), random.randint(y, self.to_sparse[i])] for i, (x, y) in
                                  enumerate(self.crop_range)]
                else:
                    crop_range = self.crop_range

                data_numpy = self.SparseConverter(data_numpy, move_range=self.move_range, crop_range=crop_range,
                                                  rotation=rotation, scale=scale)  # [coords, features]
            else:
                data_numpy = rotate_skeleton(data_numpy, rotation)
                data_numpy = scale_skeleton(data_numpy, scale)
                data_numpy = data_numpy.astype(np.float32)
                if self.fea_augment:
                    C, T, V, M = data_numpy.shape
                    semantics = np.tile(np.eye(V * M, dtype=np.float32).reshape((M, V, V * M))[np.newaxis], [T, 1, 1, 1]).transpose((3, 0, 2, 1))  # TMVC -> CTVM
                    data_numpy = np.concatenate([semantics, data_numpy], axis=0)

        return data_numpy, label, sample_name

    def statistic_point_proportion(self, num_joint, total=100):
        """
        num_joint: num of joint
        total: number of samples for statistic
        """
        from tqdm import tqdm
        loader = DataLoader(dataset=self, batch_size=1, shuffle=False, num_workers=0)
        total_num = self.to_sparse[0] * num_joint * total
        num = 0
        for i, ([coord, data], label, index) in enumerate(tqdm(loader)):
            # num += coord.shape[1]
            if i >= total:
                break
            num += (data[0, :, :num_joint].sum(-1) > 0).sum().item()  # only statistic the first obj
        print(num / total_num)

    def skeleton_to_sparse(self, coord_path, fea_path, num_joints, num_person):
        dilate_num = self.dilate_value * 6 if self.dilate_value != 0 else 1
        num_edge_points = sum(self.interval) if self.interval is not None else 0
        l = len(self)
        coord_new = open_memmap(
            coord_path,
            dtype='uint8',
            mode='w+',
            shape=(l, (num_joints + num_edge_points) * dilate_num * self.to_sparse[0] * num_person * 3, 4))
        fea_new = open_memmap(
            fea_path,
            dtype='float16',
            mode='w+',
            shape=(l, (num_joints + num_edge_points) * dilate_num * self.to_sparse[0] * num_person * 3,
                   num_person * num_joints))
        for i in tqdm(range(l)):
            coord, fea = self.__getitem__(i)[0]
            coord_new[i, :coord.shape[0]] = coord.astype(np.uint8)
            fea_new[i, :fea.shape[0]] = fea.astype(np.float16)