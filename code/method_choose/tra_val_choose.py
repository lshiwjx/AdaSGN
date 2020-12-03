import shutil
import inspect
from train_val_test import *


def train_val_choose(args, block):
    if args.pipeline == 'classify':
        train_net = train_classifier
        val_net = val_classifier
    else:
        raise ValueError("args of train val is not right")

    shutil.copy2(inspect.getfile(train_net), args.model_saved_name)
    shutil.copy2(__file__, args.model_saved_name)

    return train_net, val_net