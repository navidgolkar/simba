import os
import numpy as np
import torch.nn as nn
import torch
from simba.model import Simba
from simba.util import fix_seed
from simba.parameters import base_parameters, baselines_to_use
parameters = base_parameters

from simba.linear_rnn import LinearRNN
from simba.util import format_elapsed_time
import time
import copy

#import warnings
#warnings.filterwarnings('ignore')

# Load and process data as in 
# https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html

#print("Enter the dataset to use:")
#dataset = input()
#print("{dataset}")
#data = 'data/'+dataset+'.dat'
#data = np.genfromtxt(data)

data = np.genfromtxt('data/powerplant.dat')
U = data[:,1:6]    #inputs
Y = data[:,6:9]    #outputs
Yr = data[:,9:12]

nu = U.shape[1]
ny = Y.shape[1]
H = Y.shape[0]

U = U.reshape(-1, H, nu)
Y = Y.reshape(-1, H, ny)

# Normalize
um = np.mean(U, axis=1, keepdims=True)
us = np.std(U, axis=1, keepdims=True)
U = (U - um) / us    #Z-score of training data inputs

ym = np.mean(Y, axis=1, keepdims=True)
ys = np.std(Y, axis=1, keepdims=True)
Y = (Y - ym) / ys    #Z-score of training data outputs

# Define everything
X = X_val = X_test = None
U_val = U[:,:150,:].copy()
Y_val = Y[:,:150,:].copy()
U_test = U[:,150:,:].copy()
Y_test = Y[:,150:,:].copy()
U = U[:,:100,:]
Y = Y[:,:100,:]

print(U.shape, Y.shape, U_val.shape, Y_val.shape, U_test.shape)

_, _, m = U.shape
_, _, p = Y.shape
n = 2
model = LinearRNN(n=n, m=m, p=p)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def loss_fn(a, b):
    return nn.functional.mse_loss(a, b)

U_torch = torch.tensor(U, dtype=torch.float32)
Y_torch = torch.tensor(Y, dtype=torch.float32)
U_val_torch = torch.tensor(U_val, dtype=torch.float32)
Y_val_torch = torch.tensor(Y_val, dtype=torch.float32)
U_test_torch = torch.tensor(U_test, dtype=torch.float32)
Y_test_torch = torch.tensor(Y_test, dtype=torch.float32)

x0 = torch.zeros((U.shape[0], n))
times = []

epochs = 10000
print_each = 1000
best_epoch = 0
best_train_loss = float("inf")
best_val_loss = float("inf")
best_test_loss = float("inf")
start_time = time.time()

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(U_torch, x0)
    train_loss = loss_fn(y_pred, Y_torch)
    train_loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        Y_test_pred = model(U_test_torch, x0)
        test_loss = loss_fn(Y_test_pred, Y_test_torch)
        val_pred = model(U_val_torch, x0)
        val_loss = loss_fn(val_pred, Y_val_torch)
    
    # Save best model
    if test_loss.item() < best_test_loss:
        best_epoch = epoch
        best_train_loss = train_loss.item()        
        best_val_loss = val_loss.item()
        best_test_loss = test_loss.item()
    
    times.append(time.time() - start_time)
        
    if epoch == 0:
        print(f"\nEpoch\tTrain loss\tVal loss\tTest loss")
        
    if epoch % print_each == print_each-1:
        print(f"{epoch + 1}\t{train_loss.item():.2E}\t{val_loss.item():.2E}\t{test_loss.item():.2E}")

if len(times) > 100:
    print(f"\nAverage time per 100 epochs:\t{format_elapsed_time(np.mean(np.array(times[100:]) - np.array(times[:-100])))}")
else:
    print(f"")
    
print(f"Total training time:\t\t{format_elapsed_time(times[-1])}")
print(f"\nBest model performance:")
print(f"{best_epoch}\t{best_train_loss:.2E}\t{best_val_loss:.2E}\t{best_test_loss:.2E}\n")
    
# Save model
torch.save(model.state_dict(), f"saves/linear_rnn_model_{n}.pt")