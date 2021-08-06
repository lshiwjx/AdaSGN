import numpy as np
from model.flops_count import get_model_complexity_info
from model.init_transforms import Transforms
from model.policy_layers import *
from model.ada_sgn import ADASGN

if __name__ == '__main__':
    import os

    num_js = [1, 9, 25]
    num_j = num_js[-1]
    num_t = 20
    dim = 256
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'

    model = ADASGN(60, num_js, num_t,
                   policy_type='tconv', tau=1e-5, pre_trains=None, init_type='random', init_num=5,
                   adaptive_transform=[True, True, True], gcn_types=['small', 'big'])
    pretrained_dict = torch.load(
        '../../work_dir/ntu60/sgnadapre-best.state',
        map_location='cpu')['model']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    for i, t in enumerate(model.transforms):
        model.transforms[i].data = pretrained_dict['transforms.{}'.format(i)]
    model.eval()

    from dataset.vis import plot_skeleton, test_one, test_multi, plot_points
    from dataset.ntu_skeleton import NTU_SKE, edge, edge1, edge9

    vid = 'S001C001P003R001A032'
    data_path = "../../data/ntu60/CS/test_data.npy"
    label_path = "../../data/ntu60/CS/test_label.pkl"

    kwards = {
        "window_size": 20,
        "final_size": 20,
        "random_choose": False,
        "center_choose": False,
        "rot_norm": True
    }

    dataset = NTU_SKE(data_path, label_path, **kwards)
    labels = open('../prepare/ntu60/statistics/class_name.txt', 'r').readlines()

    save_paths = ['../../vis_results/adaskeleton/{}/frame{}.png'.format(vid, i) for i in range(num_t)]
    test_one(dataset, plot_skeleton, lambda x: model.test(torch.from_numpy(x).unsqueeze(0))[1], vid=vid,
             edges=[edge1, edge9, edge], is_3d=True, pause=0.9, labels=labels, view=[-1, 1], show_axis=True, save_paths=save_paths)
    # test_multi(dataset, plot_skeleton, lambda x: model.test(x)[1], skip=1000,
    #            edges=[edge1, edge9, edge], is_3d=True, pause=1, labels=labels, view=1)
