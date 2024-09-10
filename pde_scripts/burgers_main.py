#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:47:12 2024

@author: reecekeller
"""

from models.iPINN import *
from plots.plots import *

data = scipy.io.loadmat('burgers_shock.mat')
x = data['x']                                   # 256 points between -1 and 1 [256x1]
t = data['t']                                   # 100 time points between 0 and 1 [100x1]
usol = data['usol']                             # solution of 256x100 grid points

X, T = np.meshgrid(x,t)  
N = len(x)*len(t)                       # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
#plot3D(torch.from_numpy(x),torch.from_numpy(t),torch.from_numpy(usol))

X_true = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Domain bounds
lb = X_true[0]  # [-1. 0.]
ub = X_true[-1] # [1.  0.99]
U_true = usol.flatten('F')[:, None]

alpha = 1.0

X_data = torch.from_numpy(X_true[:int(alpha*N)]).float().to(device)
X_pde = torch.from_numpy(X_true).float().to(device)
U_data = torch.from_numpy(U_true[:int(alpha*N)]).float().to(device)
#idx_sampler = np.random.choice(N, 10000, replace=False)# Randomly chosen points for Interior
#X_train_Nu = X_true[idx_sampler]
#U_train_Nu = U_true[idx_sampler]

#X_train_Nu = torch.from_numpy(X_train_Nu).float().to(device)
#U_train_Nu = torch.from_numpy(U_train_Nu).float().to(device)
X_true = torch.from_numpy(X_true).float().to(device)
U_true = torch.from_numpy(U_true).float().to(device)
f_hat = torch.zeros(N,1).to(device)

layers = np.array([2, 32, 32, 32, 32, 32, 1]) #8 hidden layers
loss_arr = []
lambda_preds = []
error = []

lambda1=1.0
lambda2=1.0
lambda3 = 1.0
lambda4 = 1.0
lambda5 = 1.0
lambdas = [lambda1, lambda2, lambda3, lambda4, lambda5]
steps=20000
lr=1e-2
nu = 0.01/np.pi #diffusion coefficient

lambda_true = [1.0, 1.0, -nu, 0, 0]

model = iPINN(layers, lambdas, 'burgers', ub, lb)
params = list(model.dnn.parameters())
optimizer = torch.optim.LBFGS(params, lr,
                              max_iter = steps,
                              max_eval = None,
                              tolerance_grad = 1e-11,
                              tolerance_change = 1e-11,
                              history_size = 100,
                              line_search_fn = 'strong_wolfe')

def closure():
    optimizer.zero_grad()
    loss, curr_lambda, loss_data, loss_pde = model.loss(X_data, U_data, X_pde)
    lambda_preds.append(curr_lambda)
    loss_arr.append(loss.item())
    loss.backward()
    model.iter+=1
    if model.iter % 100 == 0:
        error_vec, u_pred, _ = model.test(x, t, X_true, U_true)
        error.append(error_vec)
        print(
            'Relative Error(Test): %.5f , 𝜆_real = [1.0,  1.0, %.5f, 0, 0], 𝜆_PINN = [%.5f, %.5f,  %.5f, %.5f, %.5f], total loss: %.5f, data loss: %.5f, pde loss: %.5f' %
            (
                error_vec.cpu().detach().numpy(),
                -nu,
                curr_lambda[0].item(),
                curr_lambda[1].item(),
                curr_lambda[2].item(),
                curr_lambda[3].item(),
                curr_lambda[4].item(),
                loss, 
                loss_data, 
                loss_pde
            )
        )

    return loss
start_time = time.time()
optimizer.step(closure)

# model = iPINN(layers, lambdas, 'burgers', ub, lb)
# params = list(model.dnn.parameters())
# optimizer = optim.SGD(params, lr, momentum = 0.9)

# max_iter = 10000

# start_time = time.time()

# loss_arr = []
# lambda_preds = []
# for i in range(max_iter):

#     optimizer.zero_grad()
#     loss, curr_lambda, loss_data, loss_pde = model.loss(X_data, U_data, X_pde)
#     lambda_preds.append(curr_lambda)
#     loss_arr.append(loss.item())
#     loss.backward()
#     loss_arr.append(loss.item())
#     optimizer.step()
#     if i % 100 == 0:

#         error_vec, _, _ = model.test(x, t, X_true, U_true)
#         print(
#             'Relative Error(Test): %.5f , 𝜆_real = [1.0,  1.0, %.5f, 0, 0], 𝜆_PINN = [%.5f, %.5f,  %.5f, %.5f, %.5f], total loss: %.5f, data loss: %.5f, pde loss: %.5f' %
#             (
#                 error_vec.cpu().detach().numpy(),
#                 -nu,
#                 curr_lambda[0].item(),
#                 curr_lambda[1].item(),
#                 curr_lambda[2].item(),
#                 curr_lambda[3].item(),
#                 curr_lambda[4].item(),
#                 loss, 
#                 loss_data, 
#                 loss_pde
#             )
#         )
        


elapsed = time.time() - start_time
print('Training time: %.2f' % (elapsed))
error_vec, u_pred, lambdas_pred = model.test(x, t, X_true, U_true)
print('Test Error: %.5f'  % (error_vec))
solutionplot(x, t, X, T, usol, u_pred, X_true,U_true)
#plt.show()
#solutionplotV2(x, t, usol, u_pred, X_true, U_true)
