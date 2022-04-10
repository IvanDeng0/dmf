# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 20:38:25 2022

@author: Lenovo
"""

import numpy as np
import scipy.io as scio
import torch
data = scio.loadmat('data/heat_v4_02.mat')

#print(data['xtr'])

#第一个0是从列表中取出来
#第二个0,1,2分别代表三种精度的数据,每个数据包含512张图片
#第三个0是取出第一张图片100*100
#第四个0是第一行
#第五个0是第一个元素

#l0=data['xtr']

l=len(data['Ytr'][0][1][0][0])

l0=data['Ytr_interp'][0][0][0]
#l0=torch.tensor(l0)
l0=l0.flatten()

print(l0)