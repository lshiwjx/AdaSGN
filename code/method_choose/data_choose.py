from __future__ import print_function, division

from torch.utils.data import DataLoader

import torch
import numpy as np
import random
import shutil
import inspect
from dataset import *


def data_choose(args, block):
    if args.data == 'ntu_skeleton':
        workers = args.worker
        data_set_train = NTU_SKE(**args.data_param.train_data_param)
        data_set_val = NTU_SKE(**args.data_param.val_data_param)
        cf = cfv = None
        shutil.copy2(inspect.getfile(NTU_SKE), args.model_saved_name)
    elif args.data == 'shrec':
        workers = args.worker
        data_set_train = SHC_SKE(**args.data_param.train_data_param)
        data_set_val = SHC_SKE(**args.data_param.val_data_param)
        cf = cfv = None
        shutil.copy2(inspect.getfile(SHC_SKE), args.model_saved_name)
    else:
        raise (RuntimeError('No data loader'))

    def init_worker_seed(_):
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    data_loader_val = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=workers, drop_last=False, pin_memory=args.pin_memory,
                                 worker_init_fn=init_worker_seed, collate_fn=cfv)
    data_loader_train = DataLoader(data_set_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=workers, drop_last=True, pin_memory=args.pin_memory,
                                   worker_init_fn=init_worker_seed, collate_fn=cf)

    block.log('Data load finished: ' + args.data)

    shutil.copy2(__file__, args.model_saved_name)
    return data_loader_train, data_loader_val
