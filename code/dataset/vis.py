import numpy as np
from utility.vis_env import *
from torch.utils.data import DataLoader


SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 120.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    13: (190., 23., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    31: (127., 178., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    120: (100., 85., 144.),
    41: (174., 199., 232.),
    42: (152., 223., 138.),
    43: (31., 119., 180.),
    44: (255., 187., 120.),
    45: (188., 189., 34.),
    46: (140., 86., 75.),
    47: (255., 152., 150.),
    48: (214., 39., 120.),
    49: (197., 176., 213.),
    50: (148., 103., 189.),
}


def get_pts_from_volume(data, num_points):
    C, T, D, H, W = data.shape
    data = data.transpose(1, 2, 3, 4, 0)
    num_obj = C // num_points
    points = np.zeros([T, num_obj, num_points, 3])
    for t in range(T):
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    if data[t, d, h, w].max() > 0:
                        v = np.where(data[t, d, h, w] > 0)
                        for vv in v:
                            m = vv // num_points
                            vv = vv % num_points
                            points[t, m, vv] = (w, h, d)
    return points  # T M, N C


def get_pts_from_sparse_array(coords, data, color=list(SCANNET_COLOR_MAP.values())):
    """

    :param coords: NxC or 1xNxC
    :param data: NxC or 1xNxC
    :param color:
    :return:
    """
    if len(coords.shape) == 3:
        coords, data = coords[0], data[0]
    color_map = np.array(color[:data.shape[1]])
    # color_map = np.stack([np.linspace(0, 255, data.shape[1]), np.linspace(255, 0, data.shape[1]), np.linspace(255, 0, data.shape[1])], axis=-1)
    num_frame = coords[:, 0].max() + 1
    points = [[] for i in range(num_frame)]
    for i, coord in enumerate(coords):
        t = coord[0]
        # feature x color since feature is in [0, 1]
        # if (data[i]!=0).sum()==1 and data[i].sum()!=1:
        #     rgba = np.concatenate([coord[1:], color_map[np.where(data[i]!=0)[0][0]]/255, [data[i].sum()]])
        # else:
        #     rgba = np.concatenate([coord[1:], (data[i][:, np.newaxis] * color_map).sum(0)/255, [0.9]])
        rgba = np.concatenate([coord[1:], color_map[np.where(data[i] != 0)[0][0]] / 255, [data[i].sum()]])
        points[t].append(rgba)
    return points  # T N C


def plot_points(ax, pts, edges=None):
    """
    pts: M, N, 3, 6 or 7 (last 3 or 4 is rgb or rgba)
    edge: [[0, 1], ... ]
    """
    pts = np.array(pts)
    ax.view_init(azim=65, elev=-10)
    if len(pts.shape) == 3:
        M, N, C = pts.shape
        for m in range(M):
            if pts.shape[2] == 6:
                ax.scatter(pts[m, :, 0], pts[m, :, 1], pts[m, :, 2], color=pts[m, :, 3:])
            else:
                ax.scatter(pts[m, :, 0], pts[m, :, 1], pts[m, :, 2])
            if edges is not None:
                for i, j in edges:
                    ax.plot([pts[m, i, 0], pts[m, j, 0]], [pts[m, i, 1], pts[m, j, 1]], zs=[pts[m, i, 2], pts[m, j, 2]])
    elif len(pts.shape) == 2:
        N, C = pts.shape
        if C == 6:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 3:])
        elif C == 7:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=pts[:, 3:])
        elif C == 4:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=pts[:, 3])
        else:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    else:
        pass


def plot_skeleton(ax, data, edges, additional_connect=None):
    C, V, M = data.shape

    is_3d = C == 3

    p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
    for m in range(M):
        if data[:, :, m].sum() == 0:
            continue
        for i, (v1, v2) in enumerate(edges):
            if is_3d:
                ax.plot(data[0, [v1, v2], m], data[1, [v1, v2], m], data[2, [v1, v2], m], p_type[m])
            else:
                ax.plot(data[0, [v1, v2], m], data[1, [v1, v2], m], p_type[m])

    if not additional_connect is None:
        for connection in additional_connect:
            t = connection[0]
            f = connection[1]
            s = connection[2]
            ax.plot([data[0][f][0], data[0][t][0]], [data[1][f][0], data[1][t][0]], [data[2][f][0], data[2][t][0]],
                    color='darkorange', alpha=s, linewidth=2)


def plot_img(ax, data):
    ax.imshow(data.transpose(1, 2, 0))


def vis(function, data, pause=10., view=32., is_3d=True, title='', save_paths=None, show_axis=True, **kwargs):
    """

    :param function:
    :param data: Tx***
    :param pause:
    :param view:
    :param is_3d:
    :param title:
    :param kwargs:
    :return:
    """
    plt.ion()
    fig = plt.figure()
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    for i, d in enumerate(data):
        ax.clear()
        if view:
            if type(view) is int or type(view) is float:
                view = [-view, view]
            ax.axis([view[0], view[1], view[0], view[1]])
            if is_3d:
                ax.set_zlim3d(view[0], view[1])
        ax.set_title(title)
        function(ax, d, **kwargs)
        if is_3d:
            ax.view_init(azim=-45, elev=30)
        if not show_axis:
            ax.set_axis_off()
        # fig.canvas.draw()
        plt.pause(pause)
        if save_paths is not None:
            if not os.path.exists(os.path.dirname(save_paths[i])):
                os.makedirs(os.path.dirname(save_paths[i]))
            plt.savefig(save_paths[i], format='png', bbox_inches='tight')
    plt.close()
    plt.ioff()


def label_2_str(labels, label):
    if not labels:
        title = str(label)
    else:
        title = labels[label].rstrip()
    return title


def test_one(dataset, function, data_warper, vid, labels=None, **kwargs):
    sample_name = dataset.sample_name
    sample_id = [str(name).split('.')[0] for name in sample_name]
    index = sample_id.index(vid)
    datas = dataset[index]
    data, label = datas[0], datas[1]
    vis(function, data_warper(data), title=label_2_str(labels, label), **kwargs)


def test_multi(dataset, function, data_warper, skip, labels=None, **kwargs):
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, d in enumerate(loader):
        data, label = d[0], d[1]
        if i % skip == 0:
            vis(function, data_warper(data), title=label_2_str(labels, label[0]), **kwargs)