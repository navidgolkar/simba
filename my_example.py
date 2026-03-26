import os
import numpy as np
import torch.nn.functional as F
import torch

from simba.model import Simba
from simba.util import fix_seed

from simba.parameters import base_parameters, baselines_to_use
parameters = base_parameters

# Parameters
seed = 1
parameters['init_from_matlab_or_ls'] = True
parameters['max_epochs'] = 10000
parameters['init_epochs'] = 150000
parameters['print_each'] = 1000

# Simulation
dt = 1228.8
path_to_matlab = parameters['path_to_matlab']
directory = os.path.join('saves', f'Daisy_init_new_{seed}')
fix_seed(seed)

# Load and process data as in 
# https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html
#print("Enter the dataset to use:")
#dataset = input()
#print("{dataset}")
#data = 'data/'+dataset+'.dat'
#data = np.genfromtxt(data)
data = np.genfromtxt('data/powerplant.dat')
U = data[:,1:6]
Y = data[:,6:9]
Yr = data[:,9:12]

nu = U.shape[1]
ny = Y.shape[1]
H = Y.shape[0]

U = U.reshape(-1, H, nu)
Y = Y.reshape(-1, H, ny)

# Normalize
um = np.mean(U, axis=1, keepdims=True)
us = np.std(U, axis=1, keepdims=True)
U = (U - um) / us

ym = np.mean(Y, axis=1, keepdims=True)
ys = np.std(Y, axis=1, keepdims=True)
Y = (Y - ym) / ys

# Define everything
X = X_val = X_test = None
U_val = U[:,:150,:].copy()
Y_val = Y[:,:150,:].copy()
U_test = U[:,150:,:].copy()
Y_test = Y[:,150:,:].copy()
U = U[:,:100,:]
Y = Y[:,:100,:]

print(U.shape, Y.shape, U_val.shape, Y_val.shape, U_test.shape)

from simba.util import check_and_initialize_data

# SIMBa
# Standard parameters
parameters['ms_horizon'] = None # No multiple shooting
parameters['base_lambda'] = 1

# Tunable parameters
parameters['learning_rate'] = 0.001
parameters['grad_clip'] = 100
parameters['train_loss'] = F.mse_loss
parameters['val_loss'] = F.mse_loss
parameters['dropout'] = 0
parameters['device'] = 'cpu'

parameters['batch_size'] = 128
parameters['horizon'] = None        # Prediction horizon of SIMBa
parameters['stride'] = 1          # Lag between two time steps to start predicting from
parameters['horizon_val'] = None  # None means entire trajectories
parameters['stride_val'] = 1

# Identify the state only
parameters['id_D'] = True
parameters['input_output'] = True
parameters['learn_x0'] = True

# Enforce stability
parameters['stable_A'] = True
parameters['LMI_A'] = True

parameters['delta'] = None

# Evaluate classical sysID baselines
baselines_to_use['parsim_s'] = False # Fails for some reason?
baselines_to_use['parsim_p'] = False # Fails for some reason?

x0 = x0_val = x0_test = np.zeros((1,1,2))
U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test = check_and_initialize_data(U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test,
                                                                                                            verbose=parameters['verbose'], autonomous=parameters['autonomous'], 
                                                                                                            input_output=parameters['input_output'], device=parameters['device'])
# Fit a state-space model with nx = 2
nx = 2
x0 = x0_val = x0_test = torch.zeros((1,1,nx))

name = f'SIMBa_{nx}'
simba = Simba(nx=nx, nu=nu, ny=ny, parameters=parameters)
simba.fit(U, U_val=U_val, U_test=U_test, X=X, X_val=X_val, X_test=X_test, Y=Y, Y_val=Y_val, Y_test=Y_test, x0=x0, x0_val=x0_val, x0_test=x0_test, baselines_to_use=baselines_to_use)
simba.save(directory=directory, save_name=name)