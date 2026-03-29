import os
import numpy as np
import torch.nn as nn
import torch
from simba.util import fix_seed, format_elapsed_time
from simba.linear_rnn import LinearRNN
import time

# Load and process data as in 
# https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html
#print("Enter the dataset to use:")
#dataset = input()
#print("{dataset}")
#data = 'data/'+dataset+'.dat'
#data = np.genfromtxt(data)

seed = 1
fix_seed(seed)

data = np.genfromtxt('data/powerplant.dat')
U = data[:,1:6]    #inputs
Y = data[:,6:9]    #outputs
Yr = data[:,9:12]

nu = U.shape[1]
ny = Y.shape[1]
H = Y.shape[0]

U = U.reshape(-1, H, nu)
Y = Y.reshape(-1, H, ny)

# Z-score normalization
um = np.mean(U, axis=1, keepdims=True)
us = np.std(U, axis=1, keepdims=True)
U = (U - um) / us

ym = np.mean(Y, axis=1, keepdims=True)
ys = np.std(Y, axis=1, keepdims=True)
Y = (Y - ym) / ys

# Define everything
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
loss_fn = nn.functional.mse_loss

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

epochs = 10000
print_each = 1000
train_losses_rnn = []
val_losses_rnn = []
test_losses_rnn = []
start_time = time.time()
times_rnn = []

print(f"\nEpoch\tTrain loss\tVal loss\tTest loss")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(U_torch, x0_train)
    train_loss = loss_fn(y_pred, Y_torch)
    train_loss.backward()
    optimizer.step()
    
    model.eval()
    # --- Stability projection: keep spectral radius of A <= 1 ---
    with torch.no_grad():
        eigvals = torch.linalg.eigvals(model.A)
        rho = eigvals.abs().max()
        if rho > 1.0:
            model.A.data /= rho
        
        val_pred  = model(U_val_torch,  x0_val)
        val_losses_rnn.append(float(loss_fn(val_pred, Y_val_torch).item()))
        test_pred = model(U_test_torch, x0_test)
        test_losses_rnn.append(float(loss_fn(test_pred, Y_test_torch).item()))
        train_losses_rnn.append(float(train_loss.item()))
        times_rnn.append(time.time() - start_time)
        
    if epoch % print_each == print_each-1 or epoch == 0:
        print(f"{epoch + 1}\t{train_loss.item():.2E}\t{val_losses_rnn[-1]:.2E}\t{test_losses_rnn[-1]:.2E}")
            
# Timing summary
if len(times_rnn) > 100:
    avg = np.mean(np.array(times_rnn[100:]) - np.array(times_rnn[:-100]))
    print(f"\nAverage time per 100 epochs:\t{format_elapsed_time(avg)}")
print(f"Total training time:\t\t{format_elapsed_time(times_rnn[-1])}")

best_epoch_rnn = int(np.argmin(val_losses_rnn))
print(f"\nBest model performance (by val loss):")
print(f"{best_epoch_rnn}\t{train_losses_rnn[best_epoch_rnn]:.2E}\t{val_losses_rnn[best_epoch_rnn]:.2E}\t{test_losses_rnn[best_epoch_rnn]:.2E}\n")  
        
# Save best checkpoint
os.makedirs("saves", exist_ok=True)
torch.save(
    {"train_losses": train_losses_rnn,
     "val_losses":   val_losses_rnn,
     "test_losses":  test_losses_rnn,
     "times":      times_rnn},
    f"saves/linear_rnn_model_{n}.pt"
)

# # Load Linear RNN checkpoint
# n = 2
# seed = 1
# rnn_path = os.path.join(f"saves/linear_rnn_model_{n}.pt")
# checkpoint_rnn = torch.load(rnn_path, map_location="cpu", weights_only=False)
# train_losses_rnn = checkpoint_rnn["train_losses"]
# val_losses_rnn = checkpoint_rnn["val_losses"]
# test_losses_rnn = checkpoint_rnn["test_losses"]
# times_rnn = checkpoint_rnn["times"]
# best_epoch_rnn = int(np.argmin(val_losses_rnn))
# if len(times_rnn) > 100:
#     avg = np.mean(np.array(times_rnn[100:]) - np.array(times_rnn[:-100]))

# --- Compare with saved SIMBA results (if available) ---
simba_path = os.path.join("saves", f"Daisy_init_new_{seed}", f"SIMBa_{n}.pt")
if os.path.exists(simba_path):
    ckpt = torch.load(simba_path, map_location="cpu", weights_only=False)
    val_losses_simba  = ckpt["val_losses"]
    test_losses_simba = ckpt["test_losses"]
    train_losses_simba = ckpt["train_losses"]
    best_e = int(np.argmin(val_losses_simba))
    common_tol = np.round(np.min([val_losses_simba[best_e], val_losses_rnn[best_epoch_rnn]]), 2)
    
    print(f"\n--- Linear RNN vs SIMBa (nx={n}) ---")
    print("\nBest model performance:")
    print(f"Model\t\tEpoch\tTrain loss\tVal loss\tTest loss")
    print(f"SIMBa\t\t{best_e}\t{train_losses_simba[best_e]:.2E}\t{val_losses_simba[best_e]:.2E}\t{test_losses_simba[best_e]:.2E}")
    print(f"Linear RNN\t{best_epoch_rnn}\t{train_losses_rnn[best_epoch_rnn]:.2E}\t{val_losses_rnn[best_epoch_rnn]:.2E}\t{test_losses_rnn[best_epoch_rnn]:.2E}")
    
    simba_path = os.path.join("saves", f"simba_time_{n}.pt")
    if os.path.exists(simba_path):
        ckpt = torch.load(simba_path, map_location="cpu", weights_only=False)
        times_sim = ckpt["times"]
        if len(times_sim) > 100:
            avg_time_sim = np.mean(np.array(times_sim[100:]) - np.array(times_sim[:-100]))
        
        idx_simba = (np.abs(val_losses_simba - common_tol)).argmin()
        idx_rnn = (np.abs(val_losses_rnn - common_tol)).argmin()
        print("\nSample complexity:")
        print(f"Model\t\tTolerance\tEpoch\tTrain loss\tVal loss\tTest loss\tTime")
        print(f"SIMBa\t\t{common_tol:.2E}\t{idx_simba}\t{train_losses_simba[idx_simba]:.2E}\t{val_losses_simba[idx_simba]:.2E}\t{test_losses_simba[idx_simba]:.2E}\t{format_elapsed_time(times_sim[idx_simba])}")
        print(f"Linear RNN\t{common_tol:.2E}\t{idx_rnn}\t{train_losses_rnn[idx_rnn]:.2E}\t{val_losses_rnn[idx_rnn]:.2E}\t{test_losses_rnn[idx_rnn]:.2E}\t{format_elapsed_time(times_rnn[idx_rnn])}")
        
        print("\nTime performance:")
        print(f"Model\t\tmax epochs\ttotal train time\tAvg t per 100 epochs")
        print(f"SIMBa\t\t{len(times_sim)}\t\t{format_elapsed_time(times_sim[-1])}\t\t\t{format_elapsed_time(avg_time_sim)}")
        print(f"Linear RNN\t{len(times_rnn)}\t\t{format_elapsed_time(times_rnn[-1])}\t\t\t{format_elapsed_time(avg)}")
    else:
        print(f"\n(SIMBa time checkpoint not found at {simba_path} — run example.py again if not already)")
    
else:
    print(f"\n(SIMBa checkpoint not found at {simba_path} — run example.py first to compare)")