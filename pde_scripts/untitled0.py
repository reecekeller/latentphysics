#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:47:12 2024

@author: reecekeller
"""

from models.iPINN import *
from plots.plots import *


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split

import numpy as np
import time
import scipy.io

data = scipy.io.loadmat('burgers_shock.mat')
x = data['x']                                   # 256 points between -1 and 1 [256x1]
t = data['t']                                   # 100 time points between 0 and 1 [100x1]
usol = data['usol']                             # solution of 256x100 grid points

X, T = np.meshgrid(x,t)                         # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
plot3D(torch.from_numpy(x),torch.from_numpy(t),torch.from_numpy(usol))

X_true = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Domain bounds
lb = X_true[0]  # [-1. 0.]
ub = X_true[-1] # [1.  0.99]
U_true = usol.flatten('F')[:, None]

idx_sampler = np.random.choice(len(x)*len(t), 10000, replace=False)# Randomly chosen points for Interior
X_train_Nu = X_true[idx_sampler]
U_train_Nu = U_true[idx_sampler]

X_train_Nu = torch.from_numpy(X_train_Nu).float().to(device)
U_train_Nu = torch.from_numpy(U_train_Nu).float().to(device)
X_true = torch.from_numpy(X_true).float().to(device)
U_true = torch.from_numpy(U_true).float().to(device)
f_hat = torch.zeros(X_train_Nu.shape[0],1).to(device)

layers = np.array([2,32, 32, 32,1]) #8 hidden layers
loss_arr = []

lambda1=2.0
lambda2= 0.2
lambda3 = 1.0
lambda4 = 1.0
lambda5 = 1.0

lambdas = [lambda1, lambda2, lambda3, lambda4, lambda5]

steps=20000
lr=1e-1
nu = 0.01/np.pi #diffusion coefficient

model = iPINN(layers, lambdas, 'helmholtz', ub, lb)
params = list(model.dnn.parameters())
optimizer = optim.Adam(params, lr=0.001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

max_iter = 10000

start_time = time.time()

for i in range(max_iter):

    loss = model.loss(X_train_Nu, U_train_Nu)
           
    optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
    
    loss.backward() #backprop

    optimizer.step()
    
    if i % (max_iter/10) == 0:

        error_vec, _ = model.test()

        print(loss,error_vec)