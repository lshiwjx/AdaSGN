import torch
import torch.nn as nn
from tqdm import tqdm
from utility.log import IteratorTimer, Recorder
import numpy as np


def train_ada(data_loader, model, loss_function, optimizer, global_step, args, writer, loger, epoch=None):
    process = tqdm(IteratorTimer(data_loader), desc='Train: ')
    for index, (inputs, labels, names) in enumerate(process):
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        outputs, actions = model(inputs)
        gflops_vector = model.module.gflops_vector
        ls_flops, ls_cls, ls_uniform = loss_function(outputs, labels, gflops_vector, actions, epoch)
        loss = ls_flops + ls_cls + ls_uniform
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        value, predict_label = torch.max(outputs.data, 1)
        acc = torch.mean((predict_label == labels.data).float()).item()
        lr = optimizer.param_groups[0]['lr']
        process.set_description(
            'Train: acc: {:2f}, ls_flops: {:2f}, ls_cls: {:2f}, ls_uniform: {:2f}, lr: {:2f}'
                .format(acc, ls_flops.item(), ls_cls.item(), ls_uniform.item(), lr))

        writer.add_scalar('acc', acc, global_step)
        writer.add_scalar('ls_flops', ls_flops.item(), global_step)
        writer.add_scalar('ls_cls', ls_cls.item(), global_step)
        writer.add_scalar('ls_uniform', ls_uniform.item(), global_step)
        writer.add_scalar('batch_time', process.iterable.last_duration, global_step)

    process.close()
    return global_step


def val_ada(data_loader, model, loss_function, global_step, args, writer, loger, epoch=None):
    process = tqdm(IteratorTimer(data_loader), desc='Val: ')
    score_frag = []
    numj_per_action = [[] for i in range(args.class_num)]
    numjs = np.array(args.model_param.num_joints)
    num_model = len(args.model_param.gcn_types)
    numjs = numjs.repeat(num_model)
    all_pre_true = []
    wrong_path_pre_ture = []
    acces = Recorder()
    loss_flops = Recorder()
    loss_cls = Recorder()
    action_list = []
    label_list = []
    for index, (inputs, labels, names) in enumerate(process):

        with torch.no_grad():
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs, actions = model(inputs)
            action_list.append(actions.cpu().numpy())  # num_act N M 1 1 T
            label_list.append(labels.cpu().numpy())
            gflops_vector = model.module.gflops_vector
            ls_flops, ls_cls, ls_uniform = loss_function(outputs, labels, gflops_vector, actions, epoch)
            _, predict_label = torch.max(outputs.data, 1)
            score_frag.append(outputs.data.cpu().numpy())

        predict = list(predict_label.cpu().numpy())
        true = list(labels.data.cpu().numpy())
        for i, x in enumerate(predict):
            all_pre_true.append(str(x) + ',' + str(true[i]) + '\n')
            if x != true[i]:
                wrong_path_pre_ture.append(str(names[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

        for i in range(actions.shape[1]):
            a = actions[:, i, :, 0, 0].mean(2).mean(-1).cpu().numpy()  # num_actionxNxMx1x1xT -> num_action, only the first obj
            numj_per_action[true[i]].append(a)  # average actions for cls true[i]

        right_num = torch.sum(predict_label == labels.data).item()
        batch_num = labels.data.size(0)

        acces.update(right_num, n=batch_num)
        loss_flops.update(ls_flops.item())
        loss_cls.update(ls_cls.item())

        process.set_description(
            'Val-batch: acc: {:2f}, ls_flops: {:2f}, ls_cls: {:2f}, ls_uniform: {:2f}'
                .format(right_num / batch_num, ls_flops.item(), ls_cls.item(), ls_uniform.item()))

    score = np.concatenate(score_frag)
    print_str = model.module.get_policy_usage_str(action_list, label_list)
    score_dict = dict(zip(data_loader.dataset.sample_name, score))

    for i, l in enumerate(numj_per_action):
        numj_per_action[i] = sum(sum(l)/len(l)*numjs)  # percent of actions x number of joints for corresponding actions

    process.close()

    loger.log(print_str)

    # print('Accuracy: ', accuracy)
    if writer is not None:
        writer.add_scalar('acc', acces.avg, global_step)
        writer.add_scalar('ls_flops', loss_flops.avg, global_step)
        writer.add_scalar('ls_cls', loss_cls.avg, global_step)
        writer.add_scalar('ls_uniform', ls_uniform.item(), global_step)
        writer.add_scalar('batch time', process.iterable.last_duration, global_step)

    return loss_flops.avg + loss_cls.avg, acces.avg, score_dict, numj_per_action, all_pre_true, wrong_path_pre_ture
