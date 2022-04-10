# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:17:11 2022

@author: Lenovo
"""

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import *


LR=0.01


class Architect(object):
    def __init__(self,model):
        self.model=model
        self.optimizer=torch.optim.Adam(self.model.alphas_parameters(),lr=LR,weight_decay=1e-3)
        
    def _backward_step(self,inputs,target):
        loss=self.model._loss(inputs,target)
        print('loss2=',loss)
        loss.backward()
        
        
    def step(self,inputs,target):
        self.optimizer.zero_grad()
        self._backward_step(inputs, target)
        self.optimizer.step()