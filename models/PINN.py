import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

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
        self.activation = SineActivation()
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
    
class PINN():
    def __init__(self, layers, ub, lb):
        self.loss_function = nn.MSELoss(reduction ='mean')
        'Initialize iterator'
        self.iter = 0
        self.dnn = mlp(layers, ub, lb).to(device)

    def loss_data(self,t_0, t_f, x_0, x_f):

        # y is initial condition
        u_0 = self.dnn(t_0)
        u_f = self.dnn(t_f)
        loss_u = self.loss_function(u_0, x_0) + self.loss_function(u_f, x_f)
        return loss_u

    def loss_ode(self, t):

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
        #rhs = torch.tensor([1, -5, 10]).to(torch.float32)
        rhs = torch.zeros_like(du_dt[:, 0])
        constraint = torch.pow(du_dt[:, 0], 1) + torch.pow(du_dt[:, 1], 1) + torch.pow(du_dt[:, 2], 1)
        #return self.loss_function(du_dt[0], rhs[0]) + self.loss_function(du_dt[1], rhs[1]) + self.loss_function(du_dt[2], rhs[2]), du_dt
        return self.loss_function(constraint, rhs), du_dt
    
    def loss(self, t_0, x_0, t_f, x_f, t):
    
        loss_data = self.loss_data(t_0, x_0, t_f, x_f)
        loss_ode, u_t = self.loss_ode(t)
        loss_val = loss_data + 5*loss_ode #+ torch.linalg.norm(self.lambdas, ord=1) + 10* torch.linalg.norm(self.lambdas, ord=2)
        return loss_val, loss_data, loss_ode, u_t
    
class Autodecoder(nn.Module):
    def __init__(self):
        super(Autodecoder, self).__init__()
        # Encoder: linear layers to compress input
        self.encoder = nn.Sequential(
            nn.Linear(3, 128),  # Input size is 28x28 (for MNIST images)
            nn.ReLU(),
            nn.Linear(128, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU()
        )
        # Decoder: linear layers to reconstruct input
        self.decoder = nn.Sequential(
            nn.Linear(400, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Flatten the input image into a 1D vector
        # Encode and then decode
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    
    
    
    