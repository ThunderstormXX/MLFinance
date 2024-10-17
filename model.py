from torch import nn 
import torch
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.ln1 = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.ln1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def derivative(self,x):    
        x.requires_grad = True
        y = self.forward(x)
        y.backward(torch.ones_like(y) , create_graph=True)
        first_derivative = x.grad
        return first_derivative

    def delta(self,x):
        grad = self.derivative(x)
        return grad.select(-1, 0)
    def vega(self,x):
        grad = self.derivative(x)
        return grad.select(-1, 5)
    def theta(self,x):
        grad = self.derivative(x)
        return grad.select(-1, 3)
    def rho(self, x):
        grad = self.derivative(x)
        return grad.select(-1, 4)

    def gamma(self,x):
        return torch.autograd.functional.hessian(self, x)