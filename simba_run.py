import os
import numpy as np
import torch.nn.functional as F
import torch

from simba.model import Simba
from simba.util import fix_seed
from simba.util import check_and_initialize_data
from simba.parameters import base_parameters, baselines_to_use, check_parameters
parameters = base_parameters

def simba_load(seed, nx, nu, ny):
    import os
    from simba.parameters import base_parameters as parameters
    import simba.model as sim
    parameters['device'] = 'cpu'
    parameters['id_D'] = True
    parameters['input_output'] = True
    parameters['learn_x0'] = True
    path = os.path.join("saves", f"Daisy_init_new_{seed}")
    model = sim.Simba(nx=nx, nu=nu, ny=ny, parameters=parameters)
    model.load(path, f"SIMBa_{nx}")
    parameters = model.loaded_params
    return model.val_losses, model.test_losses, model.train_losses, model.times, parameters['learning_rate'], parameters['print_each'], parameters['max_epochs']

def simba_run(seed=1, U=None, Y=None, U_val=None, Y_val=None, U_test=None, Y_test=None, X=None, X_val=None, X_test=None, nx=2, nu=100, ny=3, lr=0.001, max_ep=10000, print_each=1000, grad_clip=100, init=True):
    # Parameters
    parameters['init_from_matlab_or_ls'] = init
    parameters['max_epochs'] = max_ep
    parameters['init_epochs'] = 150000
    if (print_each <= 0):
        parameters['verbose'] = 0
        parameters['print_each'] = max_ep
    else:
        parameters['verbose'] = 1
        parameters['print_each'] = print_each
    
    # Simulation
    directory = os.path.join('saves', f'Daisy_init_new_{seed}')
    fix_seed(seed)
    
    # SIMBa
    # Standard parameters
    parameters['ms_horizon'] = None # No multiple shooting
    parameters['base_lambda'] = 1
    
    # Tunable parameters
    parameters['learning_rate'] = lr
    parameters['grad_clip'] = grad_clip
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
    x0 = x0_val = x0_test = torch.zeros((1,1,nx))
    
    name = f'SIMBa_{nx}'
    simba = Simba(nx=nx, nu=nu, ny=ny, parameters=parameters) #nx=n, nu=m, ny=p
    simba.fit(U, U_val=U_val, U_test=U_test, X=X, X_val=X_val, X_test=X_test, Y=Y, Y_val=Y_val, Y_test=Y_test, x0=x0, x0_val=x0_val, x0_test=x0_test, baselines_to_use=baselines_to_use)
    simba.save(directory=directory, save_name=name)
    
    return simba.val_losses, simba.test_losses, simba.train_losses, simba.times