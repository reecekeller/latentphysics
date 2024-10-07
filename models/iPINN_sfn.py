import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
import numpy as np

torch.set_default_dtype(torch.float)
# torch.manual_seed(1234)
# np.random.seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SineActivation(nn.Module):
    def forward(self, input):
        return torch.pow(input, 3)
    
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
            x = torch.tensor(x)
        u_b = self.ub.float().to(device)
        l_b = self.lb.float().to(device)
        #preprocessing input
        x = (x - u_b)/(u_b - l_b) #feature scaling
        a = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
class iPINN():
    def __init__(self, layers, lambdas, ub, lb):
        self.loss_function = nn.MSELoss(reduction ='mean')
        'Initialize iterator'
        self.iter = 0
        self.dnn = mlp(layers, ub, lb).to(device)
        self.t0 = lb
        self.tf = ub

        self.lambdas = torch.tensor(lambdas, requires_grad=True).float().to(device)
        self.lambdas = nn.Parameter(self.lambdas)
        self.dnn.register_parameter('lambdas', self.lambdas)

    def loss_data(self, x_0, x_f, t, u):
        t_val = t
        u_val = u
        u_0 = self.dnn(self.t0)
        u_f = self.dnn(self.tf)
        u_hat = self.dnn(t_val)
        loss_boundary = self.loss_function(u_0, x_0) + self.loss_function(u_f, x_f)
        loss_u = self.loss_function(u_hat, u_val)
        return loss_u #+ loss_boundary

    def loss_ode(self, t):
        gamma = torch.tensor(1.)
        
        #my_vars = {f'lambda{i}': value for i, value in enumerate(my_list)}
        lambda1 = self.lambdas[0]
        lambda2 = self.lambdas[1]
        lambda3 = self.lambdas[2]
        t.requires_grad = True  # Enable gradient computation with respect to t
        u = self.dnn(t)            # u(t) = [u1, u2, u3]
        # Initialize list to store the derivatives
        derivatives = []
        
        # For each dimension u_i(t), compute du_i/dt
        for i in range(u.size(1)):  # Loop over each dimension of u(t)
            u_i = u[:, i]
            grad_u_i = torch.autograd.grad(u_i, t, grad_outputs=torch.ones_like(u_i).to(device), create_graph=True)[0]
            derivatives.append(grad_u_i)
        du_dt = torch.stack(derivatives, dim=1)
        du_dt = du_dt.squeeze(2)
       
        kinetic_energy = torch.sum(du_dt**2, dim=1)
        d_ke_dt = torch.autograd.grad(kinetic_energy, t, grad_outputs=torch.ones_like(kinetic_energy).to(device), create_graph=True)[0]
        
        U = torch.sum(u**2, dim=1)
        grad_u = torch.autograd.grad(U, u, grad_outputs=torch.ones_like(U).to(device), create_graph=True)[0]
        d_pe_dt = torch.sum(grad_u * du_dt, dim=1)
        #print(lambda3, u.shape, d_pe_dt.shape)
        constraint = lambda1*d_ke_dt.squeeze(1) + lambda2*d_pe_dt + lambda3*torch.sin(u[:, 0])
        rhs = torch.zeros_like(constraint)
        regularizer = torch.min(gamma, torch.linalg.norm(self.lambdas, ord=2)) + torch.linalg.norm(self.lambdas, ord=1)
        physics_loss = (self.loss_function(constraint, rhs), regularizer)
        return physics_loss, constraint
    
    def loss(self, x_0, x_f, t, u):
    
        loss_data = self.loss_data(x_0, x_f, t, u)
        loss_ode, energy = self.loss_ode(t)
        loss_val = loss_ode[0] - loss_ode[1] + loss_data #+ torch.linalg.norm(self.lambdas, ord=1) + 10* torch.linalg.norm(self.lambdas, ord=2)
        
        #E = torch.sum(torch.tensor([l*energy[i] for i, l in enumerate(self.lambdas)]), axis=1)
        #E = self.lambdas[0]*energy[0] + self.lambdas
        return loss_val, loss_data, loss_ode[0], energy, self.lambdas
    

    
    
    
    