import torch
import torch.nn as nn
import torch.optim as optim
import os

class LinearRNN(nn.Module):
    def __init__(self, n, m, p):
        super().__init__()
        self.n = n
        self.m = m
        self.p = p
                
        # State-space matrices
        self.A = nn.Parameter(torch.randn(n, n) * 0.1)
        self.B = nn.Parameter(torch.randn(n, m) * 0.1)
        self.C = nn.Parameter(torch.randn(p, n) * 0.1)
        self.D = nn.Parameter(torch.randn(p, m) * 0.1)

    def forward(self, u, x0):
        batch_size, T, _ = u.shape
        x = x0
        y_pred = []

        for k in range(T):
            us = u[:, k, :]
            y = x @ self.C.T + us @ self.D.T
            y_pred.append(y.unsqueeze(1))
            x = x @ self.A.T + us @ self.B.T

        return  torch.cat(y_pred, dim=1)
    
class RNN_Wrapper(LinearRNN):
    
    def __init__(self, n, m, p):
        super().__init__(n=n, m=m, p=p)
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.times = []
        
    def save(self, directory, save_name):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        os.makedirs(directory, exist_ok=True)
        savename = os.path.join(directory, f'{save_name}.pt')
        torch.save(
            {
                "train_losses": self.train_losses,
                "val_losses":   self.val_losses,
                "test_losses":  self.test_losses,
                "times":      self.times,
            },
            savename,
        )
        
    def load(self, directory, save_name):
        # Load the checkpoint
        checkpoint = torch.load(os.path.join(directory, f'{save_name}.pt'), map_location="cpu", weights_only=False)
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.test_losses = checkpoint['test_losses']
        self.times = checkpoint['times']
        
def LinearRNN(n, m, p):
    return RNN_Wrapper(n=n, m=m, p=p)