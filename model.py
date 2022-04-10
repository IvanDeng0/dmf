# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:30:31 2022

@author: Lenovo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from operation import *
from genotype import PRIMITIVES
from torch.autograd import Variable




class MixedOP(nn.Module):
    def __init__(self,in_feature):
        super().__init__()
        self._ops=nn.ModuleList()
        for primitive in PRIMITIVES:
            op=OPS[primitive](in_feature)
            self._ops.append(op)
    
    def forward(self,x,weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
    
    
class Network(nn.Module):
    def __init__(self,steps,criterion,in_feature,out_feature):
        super().__init__()
        self._steps=steps
        self._ops=nn.ModuleList()
        self._initialize_alpha()
        self._criterion=criterion
        self._output=nn.Linear(in_feature, out_feature)
        for i in range(self._steps):
            for j in range(i+1):
                op=MixedOP(in_feature)
                self._ops.append(op)
    
    def forward(self,s0):
        weights=F.softmax(self.alphas,dim=-1)
        states=[s0]
        offset=0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset+=len(states)
            states.append(s)
        out=states[-1]+states[-2]
        logits=self._output(out)
        return logits
    
    def _initialize_alpha(self):
        k= sum(1 for i in range(self._steps) for n in range(1+i))
        num_ops=len(PRIMITIVES)
        
        self.alphas = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
        self._alphas_parameters=[self.alphas]
        
    def _loss(self,inputs,target):
        logits=self(inputs)
        return self._criterion(logits,target)
        
    def alphas_parameters(self):
        return self._alphas_parameters
            