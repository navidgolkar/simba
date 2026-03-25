import torch
import torch.nn as nn
import numpy as np
from simba.model_linear_rnn import LinearRNN

# Load dataset (use same dataset as SIMBa)
data = np.load("data/powerplant.dat")  # adjust path if needed

u_train = torch.tensor(data["u_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.float32)

T, m = u_train.shape
p = y_train.shape[1]
n = 5  # state dimension (same as paper Section V-A)

model = LinearRNN(n, m, p)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

x0 = torch.zeros(n)

epochs = 5000

for epoch in range(epochs):
    optimizer.zero_grad()

    y_pred = model(u_train, x0)

    loss = loss_fn(y_pred, y_train)
    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "linear_rnn_model.pth")

print("Training finished.")