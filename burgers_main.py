#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:47:12 2024

@author: reecekeller
"""

from iPINN import *

pde_sol = scipy.io.loadmat('data.mat')
x = pde_sol['x']; t = pde_sol['t']; u = pde_sol['wave']                        
X, T = np.meshgrid(x,t)  
N = len(x)*len(t)                       

X_test = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

s_min = X_test[0]; s_max = X_test[-1] 
U_test = u.flatten('F')[:, None]

alpha = 1

X_data = torch.from_numpy(X_test[:int(alpha*N)]).float().to(device)
X_pde = torch.from_numpy(X_test).float().to(device)
U_data = torch.from_numpy(U_test[:int(alpha*N)]).float().to(device)
train_points = np.random.choice(N, 10000, replace=False)
X_train = torch.from_numpy(X_true[train_points]).float().to(device)
U_train = torch.from_numpy(U_true[train_points]).float().to(device)
U_test = torch.from_numpy(U_true).float().to(device)
dummy = torch.zeros(N, 1).to(device)

layers = np.array([2, 32, 32, 32, 32, 32, 32, 1])
loss_arr = []
lambda_preds = []
error = []

lambda1=1.0
lambda2= 1.0
lambda3 = 1.0
lambda4 = 1.0
lambda5 = 1.0

lambdas = [lambda1, lambda2, lambda3, lambda4, lambda5]
steps=100000
lr=1e-3
nu = 0.1/np.pi 

lambda_true = [1.0, 1.0, -nu, 0, 0]

model = iPINN(layers, lambdas, 'burgers', ub, lb)
params = list(model.dnn.parameters())
optimizer = torch.optim.LBFGS(params, lr, max_iter = steps, max_eval = None, tolerance_grad = 1e-9, tolerance_change = 1e-9, history_size = 80, line_search_fn = 'strong_wolfe')

def closure():
    optimizer.zero_grad()
    loss, curr_lambda, loss_data, loss_pde = model.loss(X_train_Nu, U_train_Nu, X_train_Nu)
    lambda_preds.append(curr_lambda)
    loss_arr.append(loss.item())
    loss.backward()
    model.i+=1
    return loss
optimizer.step(closure)
error_vec, u_pred, lambdas_pred = model.test(x, t, X_true, U_true)
print('Test Error: %.5f'  % (error_vec))

