from dataset.dhg_skeleton import DHG_SKE

num_joints = 22
num_person = 1

root = "/home/lshi/Database/shrec_hand/"
train = 'train'

data_path = "{}/{}_skeleton.pkl".format(root, train)
label_path = "{}/{}_label_14.pkl".format(root, train)

coord_path = "{}/{}_coord_joint_sparse12832_edge0_dilate0.npy".format(root, train)
fea_path = "{}/{}_fea_joint_sparse12832_edge0_dilate0.npy".format(root, train)

kwards = {
    "window_size": 150,
    "final_size": 128,
    "mode": 'val',
    "random_choose": False,
    "center_choose": True,
    "to_sparse": [128, 32, 32, 32],
    "interval": None,
    "dilate_value": 0
}

data = DHG_SKE(data_path, label_path, **kwards)
data.statistic_point_proportion(num_joints)
data.skeleton_to_sparse(coord_path, fea_path, num_joints, num_person)
