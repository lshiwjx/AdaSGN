import torch
import torch.nn.functional as func
import shutil
import inspect
import torch.nn as nn
from loss import *


def loss_choose(args, block):
    ls_param = args.loss_param
    loss = args.loss
    if loss == 'cross_entropy':
        loss_function = torch.nn.CrossEntropyLoss(size_average=True)
    elif loss == 'label_smooth_CE':
        loss_function = LabelSmoothingLoss(args.class_num, ls_param.label_smoothing_num)
        shutil.copy2(inspect.getfile(LabelSmoothingLoss), args.model_saved_name)
    elif loss == 'smooth_CE_FLOPS':
        loss_function = AdaLoss(args.class_num, **ls_param)
        shutil.copy2(inspect.getfile(AdaLoss), args.model_saved_name)
    else:
        loss_function = torch.nn.CrossEntropyLoss(size_average=True)

    block.log('Using loss: ' + loss)
    shutil.copy2(__file__, args.model_saved_name)
    if args.device_id:
        loss_function.cuda()
        loss_function = nn.DataParallel(loss_function, device_ids=args.device_id)
        block.log('copy loss_function to gpu')
    return loss_function
