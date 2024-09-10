#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:39:55 2024

@author: reecekeller
"""
from iPINN import *
from plots import *

x = np.linspace(-1,1,256)  # 256 points between -1 and 1 [256x1]
t = np.linspace(1,-1,256)  # 256 points between 1 and -1 [256x1]

X, Y = np.meshgrid(x,t) 

X_true = np.hstack((X.flatten(order='F')[:,None], Y.flatten(order='F')[:,None]))

# Domain bounds
lb = np.array([-1, -1]) #lower bound
ub = np.array([1, 1])  #upper bound

usol = np.sin(np.pi * X) * np.sin(np.pi * Y) #solution chosen for convinience  
#usol = (np.cos(np.pi*X) + np.sin(np.pi*X))*(np.cos(np.pi*Y) + np.sin(np.pi*Y))
U_true = usol.flatten('F')[:,None] 


N_u = 400 #Total number of data points for 'u'
N_f = 10000 #Total number of collocation points 

N=len(x)*len(t)

# Obtain random points for interior
id_f = np.random.choice(N, N_f, replace=False)# Randomly chosen points for Interior
X_train_Nu = X_true[id_f]
U_train_Nu= U_true[id_f]

alpha = 1.0

X_data = torch.from_numpy(X_true[:int(alpha*N)]).float().to(device)
X_pde = torch.from_numpy(X_true).float().to(device)
U_data = torch.from_numpy(U_true[:int(alpha*N)]).float().to(device)

'Convert to tensor and send to GPU'
X_train_Nu = torch.from_numpy(X_train_Nu).float().to(device)
U_train_Nu = torch.from_numpy(U_train_Nu).float().to(device)
X_true = torch.from_numpy(X_true).float().to(device)
U_true = torch.from_numpy(U_true).float().to(device)
f_hat = torch.zeros(X_train_Nu.shape[0],1).to(device)


plot3D(torch.tensor(x).unsqueeze(1), torch.tensor(t).unsqueeze(1), torch.tensor(usol))


steps=20000
lr=1e-1
layers = np.array([2, 32, 32, 32, 32, 32, 1]) #8 hidden layers
lambdas = [0.8, 1.2, 0.2]
loss_arr = []
error = []
# model = iPINN(layers, lambdas, 'helmholtz', ub, lb)
# params = list(model.dnn.parameters())
# optimizer = optim.Adam(params, lr=0.001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# max_iter = 10000

# start_time = time.time()

# loss_arr = []
# for i in range(max_iter):

#     loss = model.loss(X_train_Nu, U_train_Nu)
           
#     optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
    
#     loss.backward() #backprop

#     optimizer.step()
#     loss_arr.append(loss.item())
#     if i % 100 == 0:

#         error_vec, _, lambda_pred= model.test(x, t, X_true, U_true)
#         print(
#             'Relative Error(Test): %.5f , 𝜆_real = [1.0,  1.0], 𝜆_PINN = [%.5f, %.5f,, total loss: %.5f' %
#             (
#                 error_vec.cpu().detach().numpy(),
#                 lambda_pred[0].item(),
#                 lambda_pred[1].item(),
#                 loss
#             )
#         )


model = iPINN(layers, lambdas, 'helmholtz', ub, lb)
params = list(model.dnn.parameters())
optimizer = torch.optim.LBFGS(params, lr,
                              max_iter = steps,
                              max_eval = None,
                              tolerance_grad = 1e-11,
                              tolerance_change = 1e-11,
                              history_size = 100,
                              line_search_fn = 'strong_wolfe')
loss_arr = []
lambda_preds = []
def closure():
    optimizer.zero_grad()
    error_vec, u_pred, _ = model.test(x, t, X_true, U_true)
    error.append(error_vec)

    loss, curr_lambda, loss_data, loss_pde = model.loss(X_data, U_data, X_pde)
    lambda_preds.append(curr_lambda)
    loss_arr.append(loss.item())
    loss.backward()
    model.iter+=1
    if model.iter % 100 == 0:
        print(
            'Relative Error(Test): %.5f , 𝜆_real = [1.0,  1.0, 0.0], 𝜆_PINN = [%.5f, %.5f, %.5f], total loss: %.5f, data loss: %.5f, pde loss: %.5f' %
            (
                error_vec.cpu().detach().numpy(),
                curr_lambda[0].item(),
                curr_lambda[1].item(),
                curr_lambda[2].item(), 
                loss,
                loss_data, 
                loss_pde
            )
        )

    return loss
start_time = time.time()
optimizer.step(closure)

elapsed = time.time() - start_time
print('Training time: %.2f' % (elapsed))
error_vec, u_pred, _ = model.test(x, t, X_true, U_true)
print('Test Error: %.5f'  % (error_vec))
solutionplot(x, t, X, Y, usol, u_pred,X_train_Nu,U_train_Nu)
plt.show()
solutionplotV2(x, t, usol, u_pred,X_train_Nu,U_train_Nu)
















