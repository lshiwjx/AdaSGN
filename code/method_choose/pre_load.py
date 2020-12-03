import torch
import torch.nn as nn


def load_checkpoint(args, block, model, optimizer):
    optimizer_dict = None
    if args.pre_trained_model is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.pre_trained_model, map_location='cpu')  # ['state_dict']
        if type(pretrained_dict) is dict and ('optimizer' in pretrained_dict.keys()):
            optimizer_dict = pretrained_dict['optimizer']
            pretrained_dict = pretrained_dict['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        keys = list(pretrained_dict.keys())
        for key in keys:
            for weight in args.ignore_weights:
                if weight in key:
                    if pretrained_dict.pop(key) is not None:
                        block.log('Sucessfully Remove Weights: {}.'.format(key))
                    else:
                        block.log('Can Not Remove Weights: {}.'.format(key))
        block.log('following weight not load: ' + str(set(model_dict) - set(pretrained_dict)))
        model_dict.update(pretrained_dict)
        # block.log(model_dict)
        model.load_state_dict(model_dict)
        block.log('Pretrained model load finished: ' + args.pre_trained_model)

    global_step = 0
    global_epoch = 0
    # The name for model must be **_**-$(step).state
    if args.last_model is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.last_model, map_location='cpu')  # ['state_dict']

        try:
            global_step = int(pretrained_dict['steps'])
            global_epoch = int(pretrained_dict['epoch'])
        except:
            try:
                global_step = int(args.last_model[:-6].split('-')[2])
                global_epoch = int(args.last_model[:-6].split('-')[1])
            except:
                global_epoch = global_step = 0

        if type(pretrained_dict) is dict and ('optimizer' in pretrained_dict.keys()):
            optimizer_dict = pretrained_dict['optimizer']
            pretrained_dict = pretrained_dict['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        block.log('In last model, following weight not load: ' + str(set(model_dict) - set(pretrained_dict)))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        block.log('Training continue, last model load finished, step is {}, epoch is {}'.format(str(global_step),
                                                                                                str(global_epoch)))

    model.cuda()
    model = nn.DataParallel(model, device_ids=args.device_id)
    block.log('copy model to gpu')

    if optimizer_dict is not None and args.last_model is not None:
        try:
            optimizer.load_state_dict(optimizer_dict)
            block.log('load optimizer from state dict')
        except:
            block.log('optimizer not matched')
    else:
        block.log('no pretrained optimizer is loaded')
        
    return model, global_step, global_epoch