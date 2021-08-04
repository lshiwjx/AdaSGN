import torch
from torch.optim.optimizer import Optimizer
import numpy as np
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.getcwd()))
from util import required


class NewOptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum_I=0.9, momentum_D=0.9,
                 weight_decay=0., P=0., D=0., AP=False, writer=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum_I < 0.0 or momentum_I > 1:
            raise ValueError("Invalid momentum_I value: {}".format(momentum_I))
        if momentum_D < 0.0 or momentum_D > 1:
            raise ValueError("Invalid momentum_D value: {}".format(momentum_D))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum_I=momentum_I, momentum_D=momentum_D,
                        weight_decay=weight_decay, P=P, D=D)
        self.num = 0
        self.AP = AP
        if not writer is None:
            self.writer = writer
        super(NewOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        self.num += 1
        for i, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum_I = group['momentum_I']
            momentum_D = group['momentum_D']
            I = group['lr']
            D = group['D'] * group['lr']
            P = group['P'] * group['lr']
            # if self.AP:
            #     # P = group['lr'] / (1 + np.exp(-0.5 * self.num + 5))
            #     # P = min(group['lr'], self.num / 1000 * group['lr'])
            #     if self.num<500:
            #         P = 0
            #     else:
            #         P = group['lr']
            # else:
            #     P = group['lr']
            # if i == 0 and self.writer is not None:
            #     self.writer.add_scalar('P', P, self.num)
            #     self.writer.add_scalar('I', I, self.num)
            #     self.writer.add_scalar('D', D, self.num)
            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p_buf = torch.zeros_like(p.grad.data)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)  # a += b*c
                if I != 0:
                    param_state = self.state[p]
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf.mul_(momentum_I).add_(d_p)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum_I).add_(d_p)
                    d_p_buf.add_(I, I_buf)
                    # if i == 0 and j == 0 and self.writer is not None:
                    #     self.writer.add_histogram('Ibuf', I_buf, self.num)
                if D != 0:
                    param_state = self.state[p]
                    if 'D_buffer' not in param_state:
                        D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                        param_state['G_buffer'] = torch.zeros_like(p.data)
                    else:
                        D_buf = param_state['D_buffer']
                        G_buf = param_state['G_buffer']
                        D_buf.mul_(momentum_D).add_(d_p - G_buf)
                        param_state['G_buffer'] = d_p
                    d_p_buf.add_(-D, D_buf)
                    # if i == 0 and j == 0 and self.writer is not None:
                    #     self.writer.add_histogram('Gbuf', d_p, self.num)
                    #     self.writer.add_histogram('Dbuf', D_buf, self.num)
                # if i == 0 and j == 0 and self.writer is not None:
                #     self.writer.add_histogram('grad', d_p, self.num)
                #     self.writer.add_histogram('data', p.data, self.num)
                d_p_buf.add_(P, d_p)
                p.data.add_(-1, d_p_buf)
        return loss


class PIDOptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, I=I, D=D)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(PIDOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PIDOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the models
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            I = group['I']
            D = group['D']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)  # a += b*c
                if momentum != 0:
                    param_state = self.state[p]
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf.mul_(momentum).add_(d_p)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - dampening, d_p)
                    if 'grad_buffer' not in param_state:
                        g_buf = param_state['grad_buffer'] = torch.zeros_like(p.data)
                        g_buf = d_p

                        D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                        D_buf.mul_(momentum).add_(d_p - g_buf)
                    else:
                        D_buf = param_state['D_buffer']
                        g_buf = param_state['grad_buffer']

                        D_buf.mul_(momentum).add_(1 - momentum, d_p - g_buf)
                        g_buf = d_p.clone()  # last gradient

                    d_p = d_p.add_(I, I_buf).add_(D, D_buf)
                p.data.add_(-group['lr'], d_p)

        return loss
