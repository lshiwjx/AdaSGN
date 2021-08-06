import torch
import torch.nn as nn
from tqdm import tqdm
from utility.log import IteratorTimer
import numpy as np


def to_onehot(num_class, label, alpha):
    return torch.zeros((label.shape[0], num_class)).fill_(alpha).scatter_(1, label.unsqueeze(1), 1 - alpha)


def mixup(input, target, gamma):
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(1 - gamma, perm_target)


def train_classifier(data_loader, model, loss_function, optimizer, global_step, args, writer, loger, *keys, **kwargs):
    process = tqdm(IteratorTimer(data_loader), desc='Train: ')
    for index, (inputs, labels, names) in enumerate(process):

        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        value, predict_label = torch.max(outputs.data, 1)
        ls = loss.data.item()
        acc = torch.mean((predict_label == labels.data).float()).item()
        lr = optimizer.param_groups[0]['lr']
        process.set_description(
            'Train: acc: {:4f}, loss: {:4f}, batch time: {:4f}, lr: {:4f}'.format(acc, ls,
                                                                                  process.iterable.last_duration,
                                                                                  lr))

        writer.add_scalar('acc', acc, global_step)
        writer.add_scalar('loss', ls, global_step)
        writer.add_scalar('batch_time', process.iterable.last_duration, global_step)

    process.close()
    return global_step


def val_classifier(data_loader, model, loss_function, global_step, args, writer, loger, *keys, **kwargs):
    process = tqdm(IteratorTimer(data_loader), desc='Val: ')
    score_frag = []
    all_pre_true = []
    wrong_path_pre_ture = []
    right_num_total = 0
    total_num = 0
    loss_total = 0
    step = 0
    for index, (inputs, labels, names) in enumerate(process):

        with torch.no_grad():
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = model(inputs)
            _, predict_label = torch.max(outputs.data, 1)
            loss = loss_function(outputs, labels)
            score_frag.append(outputs.data.cpu().numpy())

        predict = list(predict_label.cpu().numpy())
        true = list(labels.data.cpu().numpy())
        for i, x in enumerate(predict):
            all_pre_true.append(str(x) + ',' + str(true[i]) + '\n')
            if x != true[i]:
                wrong_path_pre_ture.append(str(names[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

        right_num = torch.sum(predict_label == labels.data).item()
        batch_num = labels.data.size(0)
        acc = right_num / batch_num
        ls = loss.data.item()

        right_num_total += right_num
        total_num += batch_num
        loss_total += ls
        step += 1

        process.set_description(
            'Val-batch: acc: {:4f}, loss: {:4f}, time: {:4f}'.format(acc, ls, process.iterable.last_duration))

    score = np.concatenate(score_frag)
    score_dict = dict(zip(data_loader.dataset.sample_name, score))

    process.close()
    loss = loss_total / step
    accuracy = right_num_total / total_num

    # print('Accuracy: ', accuracy)
    if writer is not None:
        writer.add_scalar('loss', loss, global_step)
        writer.add_scalar('acc', accuracy, global_step)
        writer.add_scalar('batch time', process.iterable.last_duration, global_step)

    return loss, accuracy, score_dict, [0], all_pre_true, wrong_path_pre_ture
