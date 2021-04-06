from dataset.skeleton import Skeleton


edge = ((0, 1),
        (1, 2), (2, 3), (3, 4), (4, 5),
        (1, 6), (6, 7), (7, 8), (8, 9),
        (1, 10), (10, 11), (11, 12), (12, 13),
        (1, 14), (14, 15), (15, 16), (16, 17),
        (1, 18), (18, 19), (19, 20), (20, 21))
edge1 = ()
edge11 = ((0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6), (0, 7), (7, 8), (0, 9), (9, 10))

class SHC_SKE(Skeleton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, edge=edge, **kwargs)


if __name__ == '__main__':
    from dataset.vis import plot_skeleton, test_one, test_multi

    data_path = "../../data/shrec/train_skeleton.pkl"
    label_path = "../../data/shrec/train_label_28.pkl"

    vid = '14_2_27_5'

    dataset = SHC_SKE(data_path, label_path, window_size=20, final_size=20,
                      random_choose=True, center_choose=False, decouple_spatial=False)
    labels = open('../prepare/shrec/label_28.txt', 'r').readlines()

    test_one(dataset, plot_skeleton, lambda x: x.transpose(1, 0, 2, 3), vid=vid, edges=edge, is_3d=True, pause=0.01,
             labels=labels, view=1)
    # test_multi(dataset, plot_skeleton, lambda x: x[0].numpy().transpose(1, 0, 2, 3), labels=labels, skip=1000, edges=edge,
    #            is_3d=True, pause=0.01, view=1)