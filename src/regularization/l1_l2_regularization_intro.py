import torch
import torch.nn as nn
import torch.optim as optim

def example_1(dataset):
  model = nn.Linear(10, 1)
  criterion =nn.MSELoss()

  # Implementing L2(Ridge) is simple: just use weight_decay
  optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

  # Training loop
  for data, target in dataset:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)

    # L1 must be added manually
    l1_lambda = 0.001
    l1_norm = sum(p.abs().sum() for p in model.parameters())

    total_loss = loss + (l1_lambda * l1_norm)

    total_loss.backward()
    optimizer.step()

def example_2():
  model_l1 = nn.Linear(100, 1)
  model_l2 = nn.Linear(100, 1)

  epochs = 500
  lambda_val = 0.5

  optimizer_l1 = optim.SGD(model_l1.parameters(), lr=0.01)
  optimizer_l2 = optim.SGD(model_l2.parameters(), lr=0.01)

  for i in range(epochs):
    # --- Lasso (L1) ---
    optimizer_l1.zero_grad()
    # Loss = MSE + L1_penalty (sum of abs weights)
    l1_reg = torch.sum(torch.abs(model_l1.weights))
    loss_l1 = l1_reg * lambda_val
