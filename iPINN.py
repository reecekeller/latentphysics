import torch
import torch.autograd as autograd        
from torch import Tensor                  
import torch.nn as nn                    
import torch.optim as optim               

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split

import numpy as np
import time
import scipy.io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mlp(nn.Module):
    def __init__(self,layers, ub, lb):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)
        self.ub = ub
        self.lb = lb
        self.layers = layers
        
    def forward(self,x):
        u_b = torch.from_numpy(self.ub).float().to(device)
        l_b = torch.from_numpy(self.lb).float().to(device)
        x = (x-u_b) / (u_b-l_b)
        x = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](x)
            x = self.activation(z)
        x = self.linears[-1](x)
        return x
    
class iPINN():
    def __init__(self, layers, lambdas, pde: str, ub, lb):
        self.loss_function = nn.MSELoss(reduction ='mean')
        self.i = 0
        self.lambdas = torch.tensor(lambdas, requires_grad=True).float().to(device)
        self.lambdas = nn.Parameter(self.lambdas)
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
        u_x = u_x_t[:, 0]
        u_t = u_x_t[:, 1]
        u_xx = u_xx_tt[:, 0]
        f = lambda1*u_t + lambda2*u*u_x + lambda3*u_xx + lambda4*u_xx_tt + lambda5*u_x_t
        dummy = torch.zeros_like(X_pde)
        return self.loss_function(f,dummy) 
    def loss_diffusion(self, X_pde):
        lambda1=self.lambdas[0]
        lambda2=self.lambdas[1]
        lambda3=self.lambdas[2]
        lambda4=self.lambdas[2]
        lambda5=self.lambdas[3]
        g=X_pde.clone()
        g.requires_grad=True 
        u=self.dnn(g)
        u_x_t = autograd.grad(u,g,torch.ones([g.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0] 
        u_x = u_x_t[:, 0]
        u_xx_tt = autograd.grad(u_x_t,g,torch.ones(g.shape).to(device), create_graph=True)[0]
        u_t=u_x_t[:, 1]
        u_xx=u_xx_tt[:,0]
        u_tt=  u_xx_tt[:, 1]
        f= lambda1*u_t + lambda2*u_xx + lambda3*torch.exp(-g[:, 1:])* (torch.sin(np.pi*g[:, 0]) - np.pi**2 * torch.sin(np.pi*g[:, 1])) \
            + lambda4*u_tt+lambda5*u_x
        dummy = torch.zeros_like(X_pde[:, 1].unsqueeze(1))
        return self.loss_function(f,dummy) 

    def loss_helmholtz(self, X_pde):
        lambda1 = self.lambdas[0]
        lambda2 = self.lambdas[1]
        lambda3 = self.lambdas[2]
        lambda4 = self.lambdas[3]
        lambda5 = self.lambdas[4]
        x = X_pde[:,0]
        y = X_pde[:,1]                        
        g = X_pde.clone()                        
        g.requires_grad = True        
        u = self.dnn(g)
                
        u_s = autograd.grad(u,g,torch.ones([X_pde.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]                              
        u_x = u_s[:, 0]
        u_y = u_s[:, 1]
        
        u_ss = autograd.grad(u_s,g,torch.ones(X_pde.shape).to(device), create_graph=True)[0]                                                            
        u_xx = u_ss[:, 0]       
        u_yy = u_ss[:, 1]
        residual = (-np.pi**2 - np.pi**2 + 1.0)*torch.sin(np.pi*x)*torch.sin(np.pi*y)
        f = lambda1*u_xx + lambda2*u_yy + lambda3*(u - residual) \
            + lambda4*u_x+ lambda5*u_y
        dummy = torch.zeros_like(X_pde[:, 1].unsqueeze(1))
        loss_f = self.loss_function(f, dummy)
                
        return loss_f
    
    def loss(self, x_data, u, x_pde):
        gamma_burgers = torch.tensor(1.4)
        gamma_diffusion = torch.tensor(1.7)
        gamma_helmholtz = torch.tensor(1.7)
        w = 1.0
        
        if self.pde == 'burgers':
          loss_pde = self.loss_burgers(x_pde) - torch.min(gamma_burgers, torch.linalg.norm(self.lambdas, ord=2))
        elif self.pde == 'diffusion':
          loss_pde = w*self.loss_diffusion(x_pde) - w*torch.min(gamma_diffusion, torch.linalg.norm(self.lambdas, ord=2)) #+ torch.linalg.norm(self.lambdas, ord=2)
        elif self.pde == 'helmholtz':
          loss_pde = self.loss_helmholtz(x_pde) - torch.min(gamma_helmholtz, torch.linalg.norm(self.lambdas, ord=2))

        loss_data = self.loss_data(x_data, u)
        loss_val = loss_data + loss_pde #+ torch.linalg.norm(self.lambdas, ord=1) + 10* torch.linalg.norm(self.lambdas, ord=2)

        return loss_val, self.lambdas, loss_data, loss_pde

    def test(self, x, t, X_test, U_test):
        u_hat = self.dnn(X_test)
        error_vec = torch.linalg.norm((U_test-u_hat), 2)
        error_vec /= torch.linalg.norm(U_test,2)  
        u_hat = np.reshape(u_hat.cpu().detach().numpy(),(x.shape[0],t.shape[0]),order='F')
        return error_vec, u_hat, self.lambdas
    
    
    
    
    
    
    
    