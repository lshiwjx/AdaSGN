import os
from torch.utils.data import Dataset
from dataset.video_data import *


class RGB(Dataset):
    def __init__(self, arg, mode):
        self.arg = arg
        self.mode = mode
        self.load_data()

    def load_data(self):
        with open(self.arg.label_path, 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_name, num_frame, label = self.data[index].strip().split(',')
        label = int(label)
        data_dir = os.path.join(self.arg.data_path, sample_name)
        imgs = ['RGB_{:05d}.jpg'.format(i) for i in range(1, 1 + int(num_frame))]
        frames = list(map(os.path.join, [data_dir for _ in range(len(imgs))], imgs))

        if self.mode == 'train':
            data_numpy = train_video_simple(frames, self.arg.resize_shape, self.arg.final_shape, self.arg.mean, use_flip=self.arg.use_flip)
            return data_numpy.astype(np.float32), label
        else:
            data_numpy = val_video_simple(frames, self.arg.resize_shape, self.arg.final_shape, self.arg.mean)
            return data_numpy.astype(np.float32), label, sample_name


if __name__ == '__main__':
    from easydict import EasyDict as edict
    from dataset.vis import plot_img, test_multi

    arg = edict({
        'data_path': '../../data/actnet/v1-3_trainval_jpgs',
        'label_path': '../../data/actnet/val.txt',
        'resize_shape': [20, 256, 256],
        'final_shape': [16, 224, 224],
        'mean': [0.5, 0.5, 0.5],
        'use_flip': [0, 0, 1]
    })
    dataset = RGB(arg, 'train')
    # vid = 'S004C001P003R001A032'
    labels = open('../../data/actnet/classInd.txt', 'r').readlines()

    # test_one(dataset, plot_img, lambda x: ((x + 0.5) * 255).astype(np.uint8).transpose(1, 0, 2, 3), vid, labels,
    #          pause=1, view=False, is_3d=False)
    test_multi(dataset, plot_img, lambda x: ((x[0] + 0.5) * 255).numpy().astype(np.uint8).transpose(1, 0, 2, 3), 2, labels,
               pause=0.1, view=False, is_3d=False)