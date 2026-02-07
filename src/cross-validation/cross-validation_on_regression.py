import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

sample_size = 100
D_in = 1
D_out = 1
train_prop = 0.8

# hyper parameters
learning_rate = 0.02
num_epochs = 300

x = torch.randn(sample_size, D_in)
y = x + torch.randn(sample_size, D_out)

dataset = TensorDataset(x, y)

def split_data(train_batch_size=8):
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, test_size])

  train_dlr = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
  test_dlr = DataLoader(test_dataset, batch_size=test_size)

  return train_dlr, test_dlr

def create_model(learning_rate=0.01, n_hidden_units=1):
  ANNreg = nn.Sequential(
    nn.Linear(D_in, 1), # input -> hidden
    nn.ReLU(),
    nn.Linear(1, D_out)
  )

  loss_func = nn.MSELoss()

  optimizer = torch.optim.SGD(ANNreg.parameters(), lr=learning_rate)

  return ANNreg, loss_func, optimizer

def train_the_model(model, loss_func, optimizer, /, train_loader, test_loader):
  train_loss = np.zeros(num_epochs)
  test_loss = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    batch_losses = []
    for batch_x, batch_y in train_loader:
      y_hat = model(batch_x)

      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      batch_losses.append(loss.item())

    train_loss[epoch_i] = np.array(batch_losses).mean()

    # test_loss
    test_x, test_y = next(iter(test_loader))
    test_pred_y = model(test_x)
    test_loss[epoch_i] = loss_func(test_pred_y, test_y).item()

  return train_loss, test_loss

train_dlr, test_dlr = split_data()
model, loss_func, optimizer = create_model(learning_rate, n_hidden_units=12)

train_losses, test_losses = train_the_model(model, loss_func, optimizer, train_dlr, test_dlr)

print(f'train_losses: ', train_losses)
print(f'test_losses: ', test_losses)
