import argparse
import os
import colorama
import shutil
import yaml
from easydict import EasyDict as ed


def parser_args(block):
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', default='')

    parser.add_argument('-pipeline', default='')
    parser.add_argument('-pipeline_param', default='')

    parser.add_argument('-model', default='')
    parser.add_argument('-model_param', default={}, help=None)

    parser.add_argument('-data', default='')
    parser.add_argument('-data_param', default={}, help='')

    parser.add_argument('-loss', default='')
    parser.add_argument('-loss_param', default={})

    parser.add_argument('-lr_scheduler', default='')
    parser.add_argument('-lr_scheduler_param', default={})
    parser.add_argument('-lr_multi_keys', default=[], help='[[key, lr ratio, wd ratio, epoch], ], scale before epoch')
    parser.add_argument('-freeze_keys', default=[], help='[[key, epoch], ], freeze before epoch')

    parser.add_argument('-optimizer', default='')
    parser.add_argument('-optimizer_param', default={})

    parser.add_argument('-seed', default=1)
    parser.add_argument('-eval', default=False)
    parser.add_argument('-val_first', default=False)
    parser.add_argument('-class_num', default=0)
    parser.add_argument('-batch_size', default=0)
    parser.add_argument('-worker', default=0)
    parser.add_argument('-pin_memory', default=False)
    parser.add_argument('-max_epoch', default=0)
    parser.add_argument('-num_epoch_per_save', default=0)
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-last_model',default=None)
    parser.add_argument('-ignore_weights', default=[])
    parser.add_argument('-pre_trained_model', default='')
    parser.add_argument('-device_id', default=[])
    parser.add_argument('-debug', default=False)
    parser.add_argument('-cuda_visible_device', default='0')

    p = parser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_device

    if args.debug:
        args.device_id = [0]
        args.batch_size = 1
        args.worker = 0

    block.addr = os.path.join(args.model_saved_name, 'log.txt')

    if os.path.isdir(args.model_saved_name) and not args.last_model and not args.debug:
        print('log_dir: ' + args.model_saved_name + ' already exist')
        answer = input('delete it? y/n:')
        if answer == 'y':
            shutil.rmtree(args.model_saved_name)
            print('Dir removed: ' + args.model_saved_name)
            input('refresh it')
        else:
            print('Dir not removed: ' + args.model_saved_name)

    if not os.path.exists(args.model_saved_name):
        os.makedirs(args.model_saved_name)

    # Print all arguments
    for argument, value in sorted(vars(args).items()):
        block.log('{}: {}'.format(argument, value))

    shutil.copy2(__file__, args.model_saved_name)
    shutil.copy2(args.config, args.model_saved_name)

    args = ed(vars(args))
    return args

