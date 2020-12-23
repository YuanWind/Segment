# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : Optimizer.py
import torch
class Optimizer():
    def __init__(self,parameter,config):

        if config.optim=='SGD':
            self.optim=torch.optim.SGD(parameter,lr=config.learning_rate)
        else:
            self.optim=torch.optim.Adam(parameter,lr=config.learning_rate,betas=(config.beta_1,config.beta_2),eps=config.epsilon) #
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()