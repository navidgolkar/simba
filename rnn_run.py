import os
import numpy as np
import torch
from simba.util import fix_seed, format_elapsed_time
import simba.linear_rnn as rnn
import time

def rnn_load(seed, n, m, p):
    import os
    import simba.linear_rnn as rnn
    path = os.path.join("saves", f"Daisy_init_new_{seed}")
    model = rnn.LinearRNN(n=n, m=m, p=p)
    model.load(path, f"linear_rnn_model_{n}")
    return model.val_losses, model.test_losses, model.train_losses, model.times

def rnn_run(seed=1, U=None, Y=None, U_val=None, Y_val=None, U_test=None, Y_test=None, X=None, X_val=None, X_test=None, n=2, m=5, p=3, lr=0.001, max_ep=10000, print_each=1000, grad_clip=100):
    directory = os.path.join('saves', f'Daisy_init_new_{seed}')
    fix_seed(seed)
    print("\n<-- Training of Linear RNN starts! -->")
    if (print_each > 0):
        print(f"Training data shape:\t({U.shape[0]}, {U.shape[1]}, *)\nValidation data shape:\t({U_val.shape[0]}, {U_val.shape[1]}, *)\nTest data shape:\t({U_test.shape[0]}, {U_test.shape[1]}, *)\n")
    
    model = rnn.LinearRNN(n=n, m=m, p=p)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.functional.mse_loss
    
    U_torch = torch.tensor(U, dtype=torch.float32)
    Y_torch = torch.tensor(Y, dtype=torch.float32)
    U_val_torch = torch.tensor(U_val, dtype=torch.float32)
    Y_val_torch = torch.tensor(Y_val, dtype=torch.float32)
    U_test_torch = torch.tensor(U_test, dtype=torch.float32)
    Y_test_torch = torch.tensor(Y_test, dtype=torch.float32)
    
    # x0 is always zeros, one per split (batch_size=1 for this dataset)
    x0_train = torch.zeros((U.shape[0], n))
    x0_val = torch.zeros((U_val.shape[0], n))
    x0_test = torch.zeros((U_test.shape[0], n))
    
    epochs = max_ep
    start_time = time.time()
    
    if (print_each > 0):
        print("\nEpoch\tTrain loss\tVal loss\tTest loss")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(U_torch, x0_train)
        train_loss = loss_fn(y_pred, Y_torch)
        train_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()
        
        model.eval()
        # --- Stability projection: keep spectral radius of A <= 1 ---
        with torch.no_grad():
            eigvals = torch.linalg.eigvals(model.A)
            rho = eigvals.abs().max()
            if rho > 1.0:
                model.A.data /= rho
            
            val_pred  = model(U_val_torch,  x0_val)
            model.val_losses.append(float(loss_fn(val_pred, Y_val_torch).item()))
            test_pred = model(U_test_torch, x0_test)
            model.test_losses.append(float(loss_fn(test_pred, Y_test_torch).item()))
            model.train_losses.append(float(train_loss.item()))
            model.times.append(time.time() - start_time)
            
        if print_each > 0 and ((epoch % print_each == print_each-1) or (epoch == 0)):
            print(f"{epoch + 1}\t{model.train_losses[-1]:.2E}\t{model.val_losses[-1]:.2E}\t{model.test_losses[-1]:.2E}")
                
    # Timing summary
    if print_each > 0 and len(model.times) > 100:
        avg = np.mean(np.array(model.times[100:]) - np.array(model.times[:-100]))
        print(f"\nAverage time per 100 epochs:\t{format_elapsed_time(avg)}")
        
    if print_each > 0:
        print(f"Total training time:\t\t{format_elapsed_time(model.times[-1])}")
        best_epoch_rnn = int(np.argmin(model.val_losses))
        print("\nBest model performance (by val loss):")
        print(f"{best_epoch_rnn + 1}\t{model.train_losses[best_epoch_rnn]:.2E}\t{model.val_losses[best_epoch_rnn]:.2E}\t{model.test_losses[best_epoch_rnn]:.2E}\n")  
            
    # Save best checkpoint
    name = f"linear_rnn_model_{n}"
    model.save(directory=directory, save_name=name)
    
    return model.val_losses, model.test_losses, model.train_losses, model.times
