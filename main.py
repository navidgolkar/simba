import numpy as np
from simba.util import format_elapsed_time
import argparse

#imports for run
from rnn_run import rnn_run
from simba_run import simba_run
#imports for load
from rnn_run import rnn_load
from simba_run import simba_load

def run(seed, lr, max_epoch, print_each, grad_clip, nx):
    #%% Load and process data
    
    # https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html
    #print("Enter the dataset to use:")
    #dataset = input()
    #print("{dataset}")
    #data = 'data/'+dataset+'.dat'
    #data = np.genfromtxt(data)
    
    data = np.genfromtxt('data/powerplant.dat')
    U = data[:,1:6]    #inputs
    Y = data[:,6:9]    #outputs
    
    nu = U.shape[1] #nu=m
    ny = Y.shape[1] #ny=p
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
    
    # %% SIMBa run or load
    # Run SIMBa
    val_losses_simba, test_losses_simba, train_losses_simba, times_simba = simba_run(seed=seed, U=U, Y=Y, U_val=U_val,
                                                                                     Y_val=Y_val, U_test=U_test, Y_test=Y_test,
                                                                                     X=X, X_val=X_val, X_test=X_test,
                                                                                     nx=nx, nu=nu, ny=ny, lr=lr, max_ep=max_epoch,
                                                                                     print_each=print_each, grad_clip=grad_clip)
    
    # # Load SIMBa checkpoint
    # val_losses_simba, test_losses_simba, train_losses_simba, times_simba = simba_load(seed=seed, nx=nx, nu=nu, ny=ny)
    best_epoch_simba = int(np.argmin(val_losses_simba))
    if len(times_simba) > 100:
        avg_time_simba = np.mean(np.array(times_simba[100:]) - np.array(times_simba[:-100]))     
    
    # %% Linear RNN run or load
    # Run Linear RNN
    val_losses_rnn, test_losses_rnn, train_losses_rnn, times_rnn = rnn_run(seed=seed, U=U, Y=Y, U_val=U_val,
                                                                                     Y_val=Y_val, U_test=U_test, Y_test=Y_test,
                                                                                     X=X, X_val=X_val, X_test=X_test,
                                                                                     n=nx, m=nu, p=ny, lr=lr, max_ep=max_epoch,
                                                                                     print_each=print_each, grad_clip=grad_clip)
    
    # # Load Linear RNN checkpoint
    # val_losses_rnn, test_losses_rnn, train_losses_rnn, times_rnn = rnn_load(seed=seed, n=nx, m=nu, p=ny)
    best_epoch_rnn = int(np.argmin(val_losses_rnn))
    if len(times_rnn) > 100:
        avg_time_rnn = np.mean(np.array(times_rnn[100:]) - np.array(times_rnn[:-100]))   
        
    # %% Printing outputs    
    param_simba = 5*(nx**2) + nx*nu + ny*nx + ny*nu + nx
    param_rnn = nx*nx + nx*nu + ny*nx + ny*nu
    
    print("\n<----- Linear RNN vs SIMBa ----->")
    print(" {:<12}".format("n"), end="")
    print("{:<12}".format("LR"), end="")
    print("{:<12}".format("Max e"), end="")
    print("{:<12}".format("Grad clip"))
    print(f"{'= '*24}")
    print(" {:<12}".format(f"{nx}"), end="")
    print("{:<12}".format(f"{lr:.2E}"), end="")
    print("{:<12}".format(f"{max_epoch}"), end="")
    print("{:<12}".format(f"{grad_clip}"))
    
    print("\nBest model performance:")
    print(" {:<12}".format("Model"), end="")
    print("{:<12}".format("Epoch"), end="")
    print("{:<12}".format("Train loss"), end="")
    print("{:<12}".format("Val loss"), end="")
    print("{:<12}".format("Test loss"), end="")
    print("{:<12}".format("Time"))
    print(f"{'= '*36}")
    print(" {:<12}".format("SIMBa"), end="")
    print("{:<12}".format(f"{best_epoch_simba+1}"), end="")
    print("{:<12}".format(f"{train_losses_simba[best_epoch_simba]:.2E}"), end="")
    print("{:<12}".format(f"{val_losses_simba[best_epoch_simba]:.2E}"), end="")
    print("{:<12}".format(f"{test_losses_simba[best_epoch_simba]:.2E}"), end="")
    print("{:<12}".format(f"{format_elapsed_time(times_simba[best_epoch_simba])}"))
    print(" {:<12}".format("RNN"), end="")
    print("{:<12}".format(f"{best_epoch_rnn+1}"), end="")
    print("{:<12}".format(f"{train_losses_rnn[best_epoch_rnn]:.2E}"), end="")
    print("{:<12}".format(f"{val_losses_rnn[best_epoch_rnn]:.2E}"), end="")
    print("{:<12}".format(f"{test_losses_rnn[best_epoch_rnn]:.2E}"), end="")
    print("{:<12}".format(f"{format_elapsed_time(times_rnn[best_epoch_rnn])}"))
    
    print("\nConvergence speed performance:")
    epsilons = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    print(" {:<12}".format("Model"), end="")
    print("{:<12}".format("Epsilon"), end="")
    print("{:<12}".format("Epoch"), end="")
    print("{:<12}".format("Train loss"), end="")
    print("{:<12}".format("Val loss"), end="")
    print("{:<12}".format("Test loss"), end="")
    print("{:<12}".format("Time"))
    print(f"{'= '*42}")
    for i in range(len(epsilons)):
        idx = np.where(np.array(val_losses_simba) <= epsilons[i])[0]
        idx_simba = int(idx[0]) if len(idx) > 0 else None
        idx = np.where(np.array(val_losses_rnn) <= epsilons[i])[0]
        idx_rnn = int(idx[0]) if len(idx) > 0 else None
        empty = f"{'-'*8}"
        if idx_simba is None:
            print(" {:<12}{:<12}{:<12}{:<12}{:<12}{:<12}{:<12}".format("SIMBa",f"{epsilons[i]:.2E}",f"{empty}",f"{empty}",f"{empty}",f"{empty}",f"{empty}"))
        else:
            print(" {:<12}{:<12}{:<12}{:<12}{:<12}{:<12}{:<12}".format("SIMBa",
                                                                      f"{epsilons[i]:.2E}",
                                                                      f"{idx_simba+1}",
                                                                      f"{train_losses_simba[idx_simba]:.2E}",
                                                                      f"{val_losses_simba[idx_simba]:.2E}",
                                                                      f"{test_losses_simba[idx_simba]:.2E}",
                                                                      f"{format_elapsed_time(times_simba[idx_simba])}"))
        if idx_rnn is None:
            print(" {:<12}{:<12}{:<12}{:<12}{:<12}{:<12}{:<12}".format("RNN",f"{epsilons[i]:.2E}",f"{empty}",f"{empty}",f"{empty}",f"{empty}",f"{empty}"))
        else:
            print(" {:<12}{:<12}{:<12}{:<12}{:<12}{:<12}{:<12}".format("SIMBa",
                                                                      f"{epsilons[i]:.2E}",
                                                                      f"{idx_rnn+1}",
                                                                      f"{train_losses_rnn[idx_rnn]:.2E}",
                                                                      f"{val_losses_rnn[idx_rnn]:.2E}",
                                                                      f"{test_losses_rnn[idx_rnn]:.2E}",
                                                                      f"{format_elapsed_time(times_rnn[idx_rnn])}"))
        print(f"{'─ '*42}")
    
    print("\nTime and space performance for training:")
    print(" {:<12}".format("Model"), end="")
    print("{:<12}".format("max e"), end="")
    print("{:<12}".format("# params"), end="")
    print("{:<12}".format("Total time"), end="")
    print("{:<12}".format("avg time/100e"))
    print(f"{'= '*30}")
    print(" {:<12}".format("SIMBa"), end="")
    print("{:<12}".format(f"{max_epoch}"), end="")
    print("{:<12}".format(f"{param_simba}"), end="")
    print("{:<12}".format(f"{format_elapsed_time(times_simba[-1])}"), end="")
    print("{:<12}".format(f"{format_elapsed_time(avg_time_simba)}"))
    print(" {:<12}".format("RNN"), end="")
    print("{:<12}".format(f"{max_epoch}"), end="")
    print("{:<12}".format(f"{param_rnn}"), end="")
    print("{:<12}".format(f"{format_elapsed_time(times_rnn[-1])}"), end="")
    print("{:<12}".format(f"{format_elapsed_time(avg_time_rnn)}"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that runs the Linear RNN and SIMBa for powerplant data from daisy"
    )
    parser.add_argument("--seed", required=False, type=int, default=1, help="Enter seed number")
    parser.add_argument("--lr", required=False, type=int, default=3, help="Enter the power of learning rate for gradient descent as Learning_rate = 0.1**lr")
    parser.add_argument("--epoch", required=False, type=int, default=10000, help="Enter the maximum number of epoch that RNN and SIMBa should run")
    parser.add_argument("--print_each", required=False, type=int, default=1000, help="Enter after how many epochs the losses should be printed")
    parser.add_argument("--grad_clip", required=False, type=float, default=100, help="Enter the value at which the gradient should be clipped to prevent explosion")
    parser.add_argument("--nx", required=False, type=int, default=2, help="Enter the output state number of dimensions")
    args = parser.parse_args()
    seed = args.seed
    lr = args.lr
    max_epoch = args.epoch
    print_each = args.print_each
    grad_clip = args.grad_clip
    nx = args.nx
    print(run(seed=seed, lr=0.1**lr, max_epoch=max_epoch, print_each=print_each, grad_clip=grad_clip, nx=nx))