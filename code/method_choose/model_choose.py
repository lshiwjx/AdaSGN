from __future__ import print_function, division
import torch
from collections import OrderedDict
import shutil
import inspect
from model import *


def model_choose(args, block):
    m = args.model
    if m == 'sgn':
        model = SGN(num_classes=args.class_num, **args.model_param)
        shutil.copy2(inspect.getfile(SGN), args.model_saved_name)
    else:
        raise (RuntimeError("No modules"))

    shutil.copy2(__file__, args.model_saved_name)
    block.log('Model load finished: ' + args.model)

    return model
