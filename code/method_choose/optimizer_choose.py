from __future__ import print_function, division

import torch
from torch.optim.sgd import SGD
import shutil


def optimizer_choose(model, args, block):
    # params = []
    opt_param = args.optimizer_param
    # for key, value in model.named_parameters():
    #     params += [{'params': [value], 'lr': opt_param.lr, 'key': key, 'weight_decay': opt_param.wd}]

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **opt_param)
        block.log('Using Adam optimizer ' + str(opt_param))
    elif args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), **opt_param)
        block.log('Using SGD with momentum ' + str(opt_param))
    else:
        raise RuntimeError('No such optimizer')

    # shutil.copy2(inspect.getfile(optimizer), args.model_saved_name)
    shutil.copy2(__file__, args.model_saved_name)
    return optimizer
