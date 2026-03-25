import torch
import torch.nn as nn

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
        """
        u: (T, m)
        x0: (n,)
        """
        T = u.shape[0]
        x = x0
        ys = []

        for k in range(T):
            y = self.C @ x + self.D @ u[k]
            ys.append(y)
            x = self.A @ x + self.B @ u[k]

        return torch.stack(ys)