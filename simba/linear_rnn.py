import torch
import torch.nn as nn
import torch.optim as optim

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