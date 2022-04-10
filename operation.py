# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:42:22 2022

@author: Lenovo
"""


from torch import nn

OPS={'none':lambda in_feature: Zero(),
     'deep':lambda in_feature: model_deep(in_feature),
     'shaw':lambda in_feature: model_shallow(in_feature),
     'high':lambda in_feature: model_high(in_feature),
     'lina':lambda in_feature: model_linear(in_feature)
     }

class model_deep(nn.Module):
    def __init__(self,in_feature):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_feature, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, in_feature),
            )
    def forward(self,x):
        x=self.layer(x)
        return x
    
    
class model_shallow(nn.Module):
    def __init__(self,in_feature):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_feature, 20),
            nn.ReLU(),
            nn.Linear(20, in_feature),
            )
    def forward(self,x):
        x=self.layer(x)
        return x
    
class model_high(nn.Module):
    def __init__(self,in_feature):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_feature, 40),
            nn.Linear(40, in_feature),
            )
    def forward(self,x):
        x=self.layer(x)
        return x   
    

class model_linear(nn.Module):
    def __init__(self,in_feature):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_feature, 20),
            nn.Linear(20, in_feature),
            )
    def forward(self,x):
        x=self.layer(x)
        return x
    
    
class Zero(nn.Module):
    def ___init__(self):
        super().__init__()
        
    def forward(self,x):
        x=x.mul(0.)
        return x
        

