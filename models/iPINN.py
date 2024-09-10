import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split
from pyDOE import lhs         #Latin Hypercube Sampling

import numpy as np
import time
import scipy.io

torch.set_default_dtype(torch.float)
# torch.manual_seed(1234)
# np.random.seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mlp(nn.Module):
    def __init__(self,layers, ub, lb):
        super().__init__() #call __init__ from parent class
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)
            
        self.ub = ub
        self.lb = lb
        self.layers = layers
        
    def forward(self,x):

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        u_b = torch.from_numpy(self.ub).float().to(device)
        l_b = torch.from_numpy(self.lb).float().to(device)

        #preprocessing input
        x = (x - u_b)/(u_b - l_b) #feature scaling
        a = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
class iPINN():
    def __init__(self, layers, lambdas, pde: str, ub, lb):
        self.loss_function = nn.MSELoss(reduction ='mean')
        'Initialize iterator'
        self.iter = 0
        'Initialize our new parameters i.e. 𝜆 (Inverse problem)'
        self.lambdas = torch.tensor(lambdas, requires_grad=True).float().to(device)
        'Register lambda to optimize'
        self.lambdas = nn.Parameter(self.lambdas)
        'Call our DNN'
        self.dnn = mlp(layers, ub, lb).to(device)
        self.dnn.register_parameter('lambdas', self.lambdas)
        self.pde = pde

    def loss_data(self,x,y):

        loss_u = self.loss_function(self.dnn(x), y)

        return loss_u

    def loss_burgers(self, X_pde):

        lambda1=self.lambdas[0]
        lambda2=self.lambdas[1]
        lambda3 = self.lambdas[2]
        lambda4 = self.lambdas[3]
        lambda5 = self.lambdas[4]


        g = X_pde.clone()
        g.requires_grad = True
        u = self.dnn(g)

        u_x_t = autograd.grad(u,g,torch.ones([X_pde.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
        u_xx_tt = autograd.grad(u_x_t,g,torch.ones(X_pde.shape).to(device), create_graph=True)[0]
        u_x = u_x_t[:,[0]]
        u_t = u_x_t[:,[1]]
        u_xx = u_xx_tt[:,[0]]
        f = lambda1*u_t + lambda2*u*u_x + lambda3*u_xx + lambda4*u_xx_tt + lambda5*u_x_t
        f_hat = torch.zeros_like(X_pde[:, 1].unsqueeze(1))
        return self.loss_function(f,f_hat) 

    def loss_diffusion(self, X_pde):
        lambda1=self.lambdas[0]
        lambda2=self.lambdas[1]

        g=X_pde.clone()
        g.requires_grad=True #Enable differentiation
        u=self.dnn(g)

        u_x_t = autograd.grad(u,g,torch.ones([g.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0] #first derivative
        u_xx_tt = autograd.grad(u_x_t,g,torch.ones(g.shape).to(device), create_graph=True)[0]#second derivative
        u_t=u_x_t[:,[1]]# we select the 2nd element for t (the first one is x) (Remember the input X=[x,t])
        u_xx=u_xx_tt[:,[0]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t])
        f= u_t - u_xx+ torch.exp(-g[:, 1:])* (torch.sin(np.pi * g[:, 0:1]) - np.pi ** 2 * torch.sin(np.pi * g[:, 0:1]))
        f_hat = torch.zeros_like(X_pde[:, 1].unsqueeze(1))

        return self.loss_function(f,f_hat) 

    def loss_helmholtz(self, X_pde):
        lambda1 = self.lambdas[0]
        lambda2 = self.lambdas[1]
        lambda3 = self.lambdas[2]

        x_1_f = X_pde[:,[0]]
        x_2_f = X_pde[:,[1]]                        
        g = X_pde.clone()                        
        g.requires_grad = True        
        u = self.dnn(g)
                
        u_x = autograd.grad(u,g,torch.ones([X_pde.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]                              
        u_xx = autograd.grad(u_x,g,torch.ones(X_pde.shape).to(device), create_graph=True)[0]                                                            
        u_xx_1 = u_xx[:,[0]]       
        u_xx_2 = u_xx[:,[1]]
        k=1
        q = ( -np.pi**2 - np.pi**2 + k**2) * torch.sin(np.pi*x_1_f) * torch.sin(np.pi*x_2_f)
        f = lambda1*u_xx_1 + lambda2*u_xx_2 + k**2* u - q + lambda3*u_x
        f_hat = torch.zeros_like(X_pde[:, 1].unsqueeze(1))

        loss_f = self.loss_function(f,f_hat)
                
        return loss_f
    def loss(self, x_data, u, x_pde):
        gamma = torch.tensor(1.4)
        
        if self.pde == 'burgers':
          loss_pde = self.loss_burgers(x_pde) - torch.min(gamma, torch.linalg.norm(self.lambdas, ord=2))
        elif self.pde == 'diffusion':
          loss_pde = self.loss_diffusion(x_pde) - torch.min(gamma, torch.linalg.norm(self.lambdas, ord=2))
        elif self.pde == 'helmholtz':
          loss_pde = self.loss_helmholtz(x_pde) - torch.min(gamma, torch.linalg.norm(self.lambdas, ord=2))

        loss_data = self.loss_data(x_data, u)
        loss_val = loss_data + loss_pde #+ torch.linalg.norm(self.lambdas, ord=1) + 10* torch.linalg.norm(self.lambdas, ord=2)

        return loss_val, self.lambdas, loss_data, loss_pde

    def test(self, x, t, X_true, U_true):
        u_pred = self.dnn(X_true)
        error_vec = torch.linalg.norm((U_true-u_pred), 2)/torch.linalg.norm(U_true,2)        # Relative L2 Norm of the error (Vector)
        u_pred = u_pred.cpu().detach().numpy()
        u_pred = np.reshape(u_pred,(x.shape[0],t.shape[0]),order='F')
        return error_vec, u_pred, self.lambdas
    
    
    
    
    
    
    
    