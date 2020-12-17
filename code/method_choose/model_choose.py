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
    elif m == 'single_sgn':
        model = Single_SGN(num_classes=args.class_num, **args.model_param)
        shutil.copy2(inspect.getfile(Single_SGN), args.model_saved_name)
    elif m == 'sgn_ada':
        model = ADASGN(num_classes=args.class_num, args=args, **args.model_param)
        shutil.copy2(inspect.getfile(ADASGN), args.model_saved_name)
    elif m == 'sgn_ran':
        model = RANSGN(num_classes=args.class_num, **args.model_param)
        shutil.copy2(inspect.getfile(RANSGN), args.model_saved_name)
    elif m == 'sgn_fuse':
        model = FuseSGN(num_classes=args.class_num, **args.model_param)
        shutil.copy2(inspect.getfile(FuseSGN), args.model_saved_name)
    else:
        raise (RuntimeError("No modules"))

    shutil.copy2(__file__, args.model_saved_name)
    block.log('Model load finished: ' + args.model)

    return model
