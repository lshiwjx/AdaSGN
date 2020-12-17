import os
import time
import torch
import setproctitle
from tensorboardX import SummaryWriter
from tqdm import tqdm
from method_choose.data_choose import data_choose
from method_choose.loss_choose import loss_choose
from method_choose.lr_scheduler_choose import lr_scheduler_choose
from method_choose.model_choose import model_choose
from method_choose.optimizer_choose import optimizer_choose
from method_choose.tra_val_choose import train_val_choose
from method_choose.pre_load import load_checkpoint
import parser_args
import pickle
import numpy as np
import random
from utility.log import IteratorTimer, Logger, Recorder
import traceback


class Processor:
    def __init__(self):
        self.block = Logger("Good Luck")
        self.args = parser_args.parser_args(self.block)
        self.init_seed()
        setproctitle.setproctitle(self.args.model_saved_name)
        self.block.log('work dir: ' + self.args.model_saved_name)
        self.train_writer = SummaryWriter(os.path.join(self.args.model_saved_name, 'train'), 'train')
        self.val_writer = SummaryWriter(os.path.join(self.args.model_saved_name, 'val'), 'val')

        self.model = model_choose(self.args, self.block)
        self.optimizer = optimizer_choose(self.model, self.args, self.block)
        self.lr = self.optimizer.param_groups[0]['lr']

        self.model, self.global_step, self.start_epoch = load_checkpoint(self.args, self.block, self.model,
                                                                         self.optimizer)

        self.loss_function = loss_choose(self.args, self.block)

        self.data_loader_train, self.data_loader_val = data_choose(self.args, self.block)

        self.lr_scheduler = lr_scheduler_choose(self.optimizer, self.args, self.start_epoch - 1, self.block)

        self.train_net, self.val_net = train_val_choose(self.args, self.block)

        self.score_dict = None
        self.all_pre_true = None
        self.wrong_path_pre_true = None
        self.epoch = 0

        self.acces = Recorder(larger_is_better=True, previous_items=self.start_epoch)
        self.losses = Recorder(larger_is_better=False, previous_items=self.start_epoch)
        self.process = tqdm(IteratorTimer(range(self.start_epoch, self.args.max_epoch)),
                            'Process: ' + self.args.model_saved_name)

    def init_seed(self):
        x = self.args.seed
        torch.cuda.manual_seed_all(x)
        torch.manual_seed(x)
        np.random.seed(x)
        random.seed(x)

    def adjust_lr(self):
        if self.args.lr_scheduler == 'reduce_by_epoch':
            self.lr_scheduler.step(epoch=self.epoch)
        elif self.args.lr_scheduler == 'reduce_by_acc':
            self.lr_scheduler.step(metric=self.acces.val, epoch=self.epoch)
        elif self.args.lr_scheduler == 'reduce_by_loss':
            self.lr_scheduler.step(metric=self.losses.val, epoch=self.epoch)
        else:
            self.lr_scheduler.step(epoch=self.epoch)

        self.lr = self.optimizer.param_groups[0]['lr']
        self.weight_decay = self.optimizer.param_groups[0]['weight_decay']

        for freeze_key, freeze_epoch in self.args.freeze_keys:
            if freeze_epoch > self.epoch:
                self.block.log('{} is froze'.format(freeze_key))
                for key, value in self.model.named_parameters():
                    if freeze_key in key:
                        value.requires_grad = False
            else:
                self.block.log('{} is not froze'.format(freeze_key))
                for key, value in self.model.named_parameters():
                    if freeze_key in key:
                        value.requires_grad = True

        for lr_key, ratio_lr, ratio_wd, lr_epoch in self.args.lr_multi_keys:
            if lr_epoch > self.epoch:
                self.block.log('lr for {}: {}*{}, wd: {}*{}'.format(lr_key, self.lr, ratio_lr, self.weight_decay, ratio_wd))
                for param in self.optimizer.param_groups:
                    if lr_key in param['key']:
                        param['lr'] = ratio_lr * self.lr
                        param['weight_decay'] = ratio_wd * self.weight_decay
            else:
                for param in self.optimizer.param_groups:
                    if lr_key in param['key']:
                        param['lr'] = self.lr
                        param['weight_decay'] = self.weight_decay

    def save_model(self):
        # save latest
        m = self.model.module.state_dict()
        save = {
            'model': m,
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'steps': self.global_step
        }
        torch.save(save, self.args.model_saved_name + '-latest.state')

        # save periodically
        if (self.epoch + 1) % self.args.num_epoch_per_save == 0:
            torch.save(save,
                       self.args.model_saved_name + '-' + str(self.epoch) + '-' + str(self.global_step) + '.state')

        if self.acces.is_current_best():  # save best
            save_score = self.args.model_saved_name + '/score.pkl'
            with open(save_score, 'wb') as f:
                pickle.dump(self.score_dict, f)
            with open(self.args.model_saved_name + '/all_pre_true.txt', 'w') as f:
                f.writelines(self.all_pre_true)
            with open(self.args.model_saved_name + '/wrong_path_pre_true.txt', 'w') as f:
                f.writelines(self.wrong_path_pre_true)
            torch.save(save, self.args.model_saved_name + '-best.state')

    def start(self):
        try:
            if self.args.val_first or self.args.eval:
                self.model.eval()
                loss, acc, self.score_dict, self.all_pre_true, self.wrong_path_pre_true = self.val_net(
                    self.data_loader_val, self.model, self.loss_function, self.global_step, self.args, None, self.block,
                    self.epoch)
                self.block.log('Init ACC: {}'.format(acc))
                if self.args.eval:
                    exit()

            self.block.log('Start epoch {} -> max epoch {}'.format(self.start_epoch, self.args.max_epoch))
            for self.epoch in self.process:
                self.adjust_lr()

                self.model.train()  # Set model to training mode
                self.global_step = self.train_net(self.data_loader_train, self.model, self.loss_function,
                                                  self.optimizer, self.global_step, self.args, self.train_writer,
                                                  self.block, self.epoch)
                # self.block.log('Training finished for epoch {}'.format(self.epoch))

                self.model.eval()
                loss, acc, self.score_dict, self.all_pre_true, self.wrong_path_pre_true = self.val_net(
                    self.data_loader_val, self.model, self.loss_function, self.global_step, self.args, self.val_writer,
                    self.block, self.epoch)
                # self.block.log('Validation finished for epoch {}'.format(self.epoch))

                self.train_writer.add_scalar('epoch', self.epoch, self.global_step)
                self.train_writer.add_scalar('lr', self.lr, self.global_step)
                self.train_writer.add_scalar('epoch_time', self.process.iterable.last_duration, self.global_step)

                self.losses.update(loss)
                self.acces.update(acc)

                self.val_writer.add_scalar('best_acc', self.acces.best_val, self.global_step)

                self.block.log('EPOCH: {}, ACC: {:4f}, LOSS: {:4f}, EPOCH_TIME: {:4f}, LR: {}, BEST_ACC: {:4f}'
                               .format(self.epoch, acc, loss, self.process.iterable.last_duration, self.lr,
                                       self.acces.best_val))
                self.process.set_description('Process: ' + self.args.model_saved_name + ' lr: ' + str(self.lr))
                self.save_model()

            self.block.log('Best model: ' + self.args.model_saved_name + '-' + str(self.acces.best_at) + '-' + str(0)
                           + '.state, acc: ' + str(self.acces.best_val))
            self.val_writer.close()
            self.train_writer.close()
        except Exception:
            print(traceback.format_exc())
            self.val_writer.close()
            self.train_writer.close()


def init_env():
    import torch
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # import cv2
    # cv2.setNumThreads(0)  # for shared memeory
    # from multiprocessing import set_start_method
    # set_start_method('spawn')
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    init_env()
    main = Processor()
    main.start()
