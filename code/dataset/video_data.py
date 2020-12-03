import cv2
from numpy import random as nprand
import random
import imutils
import numpy as np


def video_aug(video,
              brightness_delta=32,
              contrast_range=(0.5, 1.5),
              saturation_range=(0.5, 1.5),
              angle_range=(-30, 30),
              hue_delta=18):
    '''

    :param video: list of images
    :param brightness_delta:
    :param contrast_range:
    :param saturation_range:
    :param angle_range:
    :param hue_delta:
    :return:
    '''
    brightness_delta = brightness_delta
    contrast_lower, contrast_upper = contrast_range
    saturation_lower, saturation_upper = saturation_range
    angle_lower, angle_upper = angle_range
    hue_delta = hue_delta
    for index, img in enumerate(video):
        video[index] = img.astype(np.float32)

    # random brightness
    if nprand.randint(2):
        delta = nprand.uniform(-brightness_delta,
                               brightness_delta)
        for index, img in enumerate(video):
            video[index] += delta

    # random rotate
    if nprand.randint(2):
        angle = nprand.uniform(angle_lower,
                               angle_upper)
        for index, img in enumerate(video):
            video[index] = imutils.rotate(img, angle)

    # if nprand.randint(2):
    #     alpha = nprand.uniform(contrast_lower,
    #                            contrast_upper)
    #     for index, img in enumerate(video):
    #         video[index] *= alpha

    # mode == 0 --> do random contrast first
    # mode == 1 --> do random contrast last
    mode = nprand.randint(2)
    if mode == 1:
        if nprand.randint(2):
            alpha = nprand.uniform(contrast_lower,
                                   contrast_upper)
            for index, img in enumerate(video):
                video[index] *= alpha

    # convert color from BGR to HSV
    for index, img in enumerate(video):
        video[index] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # random saturation
    if nprand.randint(2):
        for index, img in enumerate(video):
            video[index][..., 1] *= nprand.uniform(saturation_lower,
                                                   saturation_upper)

    # random hue
    if nprand.randint(2):
        for index, img in enumerate(video):
            video[index][..., 0] += nprand.uniform(-hue_delta, hue_delta)
            video[index][..., 0][video[index][..., 0] > 360] -= 360
            video[index][..., 0][video[index][..., 0] < 0] += 360

    # convert color from HSV to BGR
    for index, img in enumerate(video):
        video[index] = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # random contrast
    if mode == 0:
        if nprand.randint(2):
            alpha = nprand.uniform(contrast_lower,
                                   contrast_upper)
            for index, img in enumerate(video):
                video[index] *= alpha

    # randomly swap channels
    # if nprand.randint(2):
    #     for index, img in enumerate(video):
    #         video[index] = img[..., nprand.permutation(3)]

    return video


def expand_list(l, length):
    if len(l) < length:
        while len(l) < length:
            tmp = []
            [tmp.extend([x, x]) for x in l]
            l = tmp
        return sample_uniform_list(l, length)
    else:
        return l


def sample_uniform_list(l, length):
    if len(l)==length:
        return l
    interval = len(l) / length
    uniform_list = [int(i * interval) for i in range(length)]
    tmp = [l[x] for x in uniform_list]
    return tmp


def judge_type(paths, final_shape):
    if type(paths[0]) is str:
        try:
            img = cv2.imread(paths[0])
            pre_shape = [len(paths), *img.shape]
        except:
            print(paths[0], ' is wrong')
            pre_shape = [len(paths), *final_shape[1:]]
    else:
        pre_shape = [len(paths), *paths[0].shape]

    return pre_shape


def crop_resize(imgs, starts, cshape, final_shape, mean, use_flip, other_aug):
    imgs_crop = imgs[starts[0]:starts[0] + cshape[0]]  # TODO: paths < cshape[0]
    imgs_final = sample_uniform_list(imgs_crop, final_shape[0])

    if other_aug:
        imgs_final = video_aug(imgs_final)
    clip = []
    for index, img in enumerate(imgs_final):
        clip.append(cv2.resize(img[starts[1]:starts[1] + cshape[1], starts[2]:starts[2] + cshape[2]],
                               (final_shape[2], final_shape[1])).astype(np.float32) / 255 - mean)
    clip = np.transpose(np.array(clip, dtype=np.float32), (3, 0, 1, 2))
    for i, f in enumerate(use_flip):
        if f:
            clip = np.flip(clip, i + 1).copy()  # avoid negative strides
    return clip


def resize_crop(imgs, resize_shape, final_shape, starts, mean, use_flip, other_aug):
    imgs_resize = np.array(sample_uniform_list(imgs, resize_shape[0]))
    imgs_crop = imgs_resize[starts[0]:starts[0] + final_shape[0]]
    if other_aug:
        imgs_crop = video_aug(imgs_crop)

    clip = []
    for index, img in enumerate(imgs_crop):
        clip.append(cv2.resize(img, (resize_shape[2], resize_shape[1]))[starts[1]:starts[1] + final_shape[1],
                    starts[2]:starts[2] + final_shape[2]].astype(
            np.float32) / 255 - mean)

    clip = np.transpose(np.array(clip, dtype=np.float32), (3, 0, 1, 2))
    for i, f in enumerate(use_flip):
        if f:
            clip = np.flip(clip, i + 1).copy()  # avoid negative strides
    return clip


def pose_flip(pose, use_flip):
    '''

    :param pose: T V C[x,y] M
    :param use_flip: 
    :return: 
    '''
    pose_new = pose
    if use_flip[0]:
        pose_new = pose[::-1]
    if use_flip[1] and use_flip[2]:
        pose_new = 1 - pose
        pose_new[pose_new == 1] = 0
    elif use_flip[1]:
        pose_new = 1 - pose
        pose_new[pose_new == 1] = 0
        pose_new[:, :, 0, :] = pose[:, :, 0, :]
    elif use_flip[2]:
        pose_new = 1 - pose
        pose_new[pose_new == 1] = 0
        pose_new[:, :, 1, :] = pose[:, :, 1, :]

    return pose_new


def pose_crop(pose_old, start, cshape, width, height):
    '''

    :param pose_old: T V C M
    :param start: T H,W
    :param cshape:T H,W
    :param width: 
    :param height: 
    :return: 
    '''
    # temporal crop
    pose_new = pose_old[start[0]:start[0] + cshape[0]]
    T, V, C, M = pose_new.shape
    # 复原到图像大小
    pose_new = pose_new * (np.array([width, height]).reshape([1, 1, C, 1]))  # T V C M
    # 减去边框
    pose_new -= np.array([start[2], start[1]]).reshape([1, 1, C, 1])
    # 小于0的置0
    pose_new[(np.min(pose_new, -2) < 0).reshape(T, V, 1, M).repeat(C, -2)] = 0
    # 新位置除以crop后的大小
    pose_new /= np.array([cshape[2], cshape[1]]).reshape([1, 1, C, 1])
    # 大于1的值1
    pose_new[(np.max(pose_new, -2) > 1).reshape(T, V, 1, M).repeat(C, -2)] = 0
    return pose_new


def gen_clip_simple(paths, starts, resize_shape, final_shape, mean, use_flip, other_aug=False):
    try:
        if type(paths[0]) is str:
            imgs = []
            paths = sample_uniform_list(paths, resize_shape[0])
            for path in paths:
                try:
                    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                except:
                    print(path, ' is wrong')
                    img = np.zeros([*final_shape[1:], 3], dtype=np.uint8)
                imgs.append(img)
            clip = resize_crop(imgs, resize_shape, final_shape, starts, mean, use_flip, other_aug)
            return clip
        elif type(paths[0]) is tuple:
            imgs, poses = np.array([i[0] for i in paths]), np.array([i[1] for i in paths])
            if len(imgs) != len(poses):
                imgs = np.array(sample_uniform_list(imgs, len(poses)))
            if poses.shape[2] >= 3:  # T,V,C,M
                poses = poses[:, :, :2]
            clip = resize_crop(imgs, resize_shape, final_shape, starts, mean, use_flip, other_aug).transpose(
                (1, 0, 2, 3))
            poses = np.array(sample_uniform_list(poses, resize_shape[0]))
            poses = pose_crop(poses, starts, final_shape, resize_shape[2], resize_shape[1])
            poses = pose_flip(poses, use_flip)
            return clip, poses
        else:
            imgs = paths
            clip = resize_crop(imgs, resize_shape, final_shape, starts, mean, use_flip, other_aug)
            return clip
    except:
        print(paths)


def gen_clip(paths, starts, cshape, final_shape, mean, use_flip=(0, 0, 0), other_aug=False):
    try:
        if type(paths[0]) is str:
            imgs = []
            for path in paths:
                try:
                    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                except:
                    print(path, ' is wrong')
                    img = np.zeros([*final_shape[1:], 3], dtype=np.uint8)
                imgs.append(img)
            clip = crop_resize(imgs, starts, cshape, final_shape, mean, use_flip, other_aug)
            return clip
        elif type(paths[0]) is tuple:
            imgs, poses = np.array([i[0] for i in paths]), np.array([i[1] for i in paths])
            if len(imgs) != len(poses):
                imgs = np.array(sample_uniform_list(imgs, len(poses)))
            if poses.shape[2] >= 3:  # T,V,C,M
                poses = poses[:, :, :2]
            clip = crop_resize(imgs, starts, cshape, final_shape, mean, use_flip, other_aug).transpose(
                (1, 0, 2, 3)).copy()
            poses = poses[starts[0]:starts[0] + cshape[0]]
            poses = pose_crop(poses, starts, cshape, imgs[0].shape[1], imgs[0].shape[0])
            poses = pose_flip(poses, use_flip).copy()
            return clip, poses
        else:
            imgs = paths
            clip = crop_resize(imgs, starts, cshape, final_shape, mean, use_flip, other_aug)
            return clip
    except:
        print(paths)


def train_video_simple(paths, resize_shape, final_shape, mean, use_flip=(0, 0, 0), other_aug=False):
    """

    :param paths: [frame1, frame2 ....] 
    :param resize_shape:  [l, h, w] 
    :param final_shape: [l, h, w]
    :param mean: [l, h, w, 3]
    :param use_flip: [0,0,0]
    :return: 
    """
    gap = [resize_shape[i] - final_shape[i] for i in range(3)]

    starts = [int(a * random.random()) for a in gap]

    clip = gen_clip_simple(paths, starts, resize_shape, final_shape, mean, use_flip, other_aug=other_aug)

    return clip


def val_video_simple(paths, resize_shape, final_shape, mean, use_flip=(0, 0, 0), other_aug=False):
    """

    :param paths: [frame1, frame2 ....] 
    :param resize_shape:  [l, h, w] 
    :param final_shape: [l, h, w]
    :param mean: [l, h, w, 3]
    :param use_flip: [0,0,0
    :return: 
    """

    gap = [resize_shape[i] - final_shape[i] for i in range(3)]

    starts = [int(a * 0.5) for a in gap]
    clip = gen_clip_simple(paths, starts, resize_shape, final_shape, mean, use_flip, other_aug=other_aug)

    return clip


def eval_video(paths, crop_ratios, crop_positions, final_shape, mean, use_flip=(0, 0, 0)):
    """

    :param paths: [frame1, frame2 ....] 
    :param crop_ratios:  [[t0, t1 ...], [h0, h1, ...], [w0, w1, ...]]   0-1
    :param crop_positions: [[t0, t1 ...], [h0, h1, ...], [w0, w1, ...]]   0-1
    :param final_shape: [l, h, w]
    :param mean: [l, h, w, 3]
    :param use_flip: [False, False, False]
    :return: 
    """
    pre_shape = judge_type(paths, final_shape)

    clips = []
    for crop_t in crop_ratios[0]:
        for crop_h in crop_ratios[1]:
            for crop_w in crop_ratios[2]:
                cshape = [int(x) for x in [crop_t * pre_shape[0], crop_h * pre_shape[1], crop_w * pre_shape[2]]]

                gap = [pre_shape[i] - cshape[i] for i in range(3)]
                for p_t in crop_positions[0]:
                    for p_h in crop_positions[1]:
                        for p_w in crop_positions[2]:
                            starts = [int(a * b) for a in gap for b in [p_t, p_h, p_w]]
                            clip = gen_clip(paths, starts, cshape, final_shape, mean)
                            clips.append(clip)  # clhw
                            for i, f in enumerate(use_flip):
                                if f:
                                    clip_flip = np.flip(clip, i + 1).copy()
                                    clips.append(clip_flip)

    return clips


def train_video(paths, crop_ratios, crop_positions, final_shape, mean, use_flip=(0, 0, 0)):
    """

    :param paths: [frame1, frame2 ....] 
    :param crop_ratios:  [[t0, t1 ...], [h0, h1, ...], [w0, w1, ...]]   0-1
    :param crop_positions: [[t0, t1 ...], [h0, h1, ...], [w0, w1, ...]]   0-1
    :param final_shape: [l, h, w]
    :param mean: [l, h, w, 3]
    :param use_flip: True or False
    :return: 
    """
    pre_shape = judge_type(paths, final_shape)

    crop_t = random.sample(crop_ratios[0], 1)[0]
    crop_h = random.sample(crop_ratios[1], 1)[0]
    crop_w = random.sample(crop_ratios[2], 1)[0]
    cshape = [int(x) for x in [crop_t * pre_shape[0], crop_h * pre_shape[1], crop_w * pre_shape[2]]]

    gap = [pre_shape[i] - cshape[i] for i in range(3)]

    p_t = random.sample(crop_positions[0], 1)[0]
    p_h = random.sample(crop_positions[1], 1)[0]
    p_w = random.sample(crop_positions[2], 1)[0]

    starts = [int(a * b) for a, b in list(zip(gap, [p_t, p_h, p_w]))]
    clip = gen_clip(paths, starts, cshape, final_shape, mean, use_flip, other_aug=True)

    # for i, f in enumerate(use_flip):
    #     if f:
    #         clip = np.flip(clip, i + 1)

    return clip


def val_video(paths, final_shape, mean):
    """

    :param paths: [frame1, frame2 ....] 
    :param final_shape: [l, h, w]
    :param mean: [l, h, w, 3]
    :return: 
    """
    pre_shape = judge_type(paths, final_shape)

    crop_t = 1
    crop_h = 1
    crop_w = 1
    cshape = [int(x) for x in [crop_t * pre_shape[0], crop_h * pre_shape[1], crop_w * pre_shape[2]]]

    gap = [pre_shape[i] - cshape[i] for i in range(3)]

    p_t = 0.5
    p_h = 0.5
    p_w = 0.5

    starts = [int(a * b) for a, b in list(zip(gap, [p_t, p_h, p_w]))]
    clip = gen_clip(paths, starts, cshape, final_shape, mean)

    return clip
