# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:46:37 2022

@author: Lenovo
"""

import numpy as np
import scipy.io as scio
import torch
import sys
sys.path.append('D:/master/multi/mycode/dmf/gp_test_frame/')
import logging
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler 
import h5py
import random
#from Exp.exp_5functions_benchmark.benchmark_setting import generate_data, BOREHOLE, BRANIN, CURRIN, HART, PARK

from torch.autograd import *
from model import Network
from architec import Architect



LR=0.01
weightdecay=1e-3
n_iters=10
sub_epoch_w=50
sub_epoch_a=50
T=10
low_train_size_1=40
low_train_size_2=20
high_train_size=10


data = scio.loadmat('data/burger_v4_02.mat')
#currin function
xl0=data['xtr']
yl1=data['Ytr_interp'][0][0]
yl2=data['Ytr_interp'][0][1]
yh0=data['Ytr_interp'][0][2]


xhte=data['xte']
yhte=[data['Yte_interp'][0][2][i].flatten() for i in range(len(xhte))]

seed=0

random.seed(seed)

low_train_set_1=random.sample(range(len(yl1)), low_train_size_1)
low_train_set_2=random.sample(range(len(yl1)), low_train_size_2)
high_train_set=random.sample(range(len(yl1)), high_train_size)

xltr,yltr_1_net1,yltr_1_net2,yltr_2_net2,yltr_2_net3,yhtr_net3=[],[],[],[],[],[]

for i in low_train_set_1:
    xltr.append(xl0[i])
    yltr_1_net1.append(yl1[i].flatten())
    
for i in low_train_set_2:
    yltr_1_net2.append(yl1[i].flatten())
    yltr_2_net2.append(yl2[i].flatten())

#print(len(yltr_1_net2[0]))
#print(len(yltr_2_net2[0]))

for i in high_train_set:
    yltr_2_net3.append(yl2[i].flatten())
    yhtr_net3.append(yh0[i].flatten())

xltr=torch.tensor(xltr,dtype=torch.float32)
yltr_1_net1=torch.tensor(yltr_1_net1,dtype=torch.float32)
yltr_1_net2=torch.tensor(yltr_1_net2,dtype=torch.float32)
yltr_2_net2=torch.tensor(yltr_2_net2,dtype=torch.float32)
yltr_2_net3=torch.tensor(yltr_2_net3,dtype=torch.float32)
yhtr_net3=torch.tensor(yhtr_net3,dtype=torch.float32)

xhte=torch.tensor(xhte,dtype=torch.float32)
yhte=torch.tensor(yhte,dtype=torch.float32)

in_feature=len(xl0[0])
out_feature=10000




criterion=nn.MSELoss()
model1=Network(2, criterion,in_feature,out_feature)
model2=Network(2, criterion,out_feature,out_feature)
model3=Network(2, criterion, out_feature, out_feature)

optimizer1 = torch.optim.Adam(model1.parameters(), LR, weight_decay=weightdecay)
optimizer2 = torch.optim.Adam(model2.parameters(), LR, weight_decay=weightdecay)
optimizer3 = torch.optim.Adam(model3.parameters(), LR, weight_decay=weightdecay)

dataset1=TensorDataset(xltr,yltr_1_net1)
train_loader1=DataLoader(dataset1,batch_size=5,shuffle=False)

dataset2 = TensorDataset(yltr_1_net2,yltr_2_net2)
train_loader2=DataLoader(dataset2,batch_size=10,shuffle=False)

dataset3 = TensorDataset(yltr_2_net3,yhtr_net3)
train_loader3=DataLoader(dataset3,batch_size=5,shuffle=False)

testset=TensorDataset(xhte,yhte)
test_loader=DataLoader(testset,batch_size=128,shuffle=False)

#scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)


#first network
archtect1=Architect(model1)
archtect2=Architect(model2)
archtect3=Architect(model3)


for i in range(n_iters):
    for j in range(sub_epoch_w):
        #scheduler.step()
        for step,(inputs,target) in enumerate(train_loader1):
            optimizer1.zero_grad()
            
            logits=model1(inputs)
            loss=criterion(logits,target)
            loss.backward()
            optimizer1.step()

    if i%5==0:
        for k in range(sub_epoch_a):
            for step,(inputs_search,target_search) in enumerate(train_loader1):       
                archtect1.step(inputs_search, target_search)
   

print('------first network training finished------')


for i in range(n_iters):
    for j in range(sub_epoch_w):
        #scheduler.step()
        for step,(inputs,target) in enumerate(train_loader2):
            optimizer2.zero_grad()  
            logits=model2(inputs)
            loss=criterion(logits,target)
            loss.backward()
            optimizer2.step()
        if j%10==0:
            print('loss1=',loss)

    if i%5==0:
        for k in range(sub_epoch_a):
            for step,(inputs_search,target_search) in enumerate(train_loader2):       
                archtect2.step(inputs_search, target_search)
print('------second network training finished------')

for i in range(n_iters):
    for j in range(sub_epoch_w):
        #scheduler.step()
        for step,(inputs,target) in enumerate(train_loader3):
            optimizer3.zero_grad()  
            logits=model3(inputs)
            loss=criterion(logits,target)
            loss.backward()
            optimizer3.step()

    if i%5==0:
        for k in range(sub_epoch_a):
            for step,(inputs_search,target_search) in enumerate(train_loader2):       
                archtect3.step(inputs_search, target_search)

model1.eval()
model2.eval()
model3.eval()
for step,(batch_x,batch_y) in enumerate(test_loader):
    temp0=model1(batch_x)
    temp1=model2(temp0)
    preds=model3(temp1)
    error=torch.sqrt(torch.mean(torch.pow(preds-batch_y,2)))
    
print(error)
