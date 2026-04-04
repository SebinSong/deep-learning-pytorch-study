import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

F = nn.functional

num_epochs = 200
n_per_cluster = 300
blur = 1

A = [ 1, 1 ]
B = [ 5, 1 ]
C = [ 4, 3 ]

a = torch.stack([
  A[0] + torch.randn(n_per_cluster) * blur,
  A[1] + torch.randn(n_per_cluster) * blur
]).float().T
b = torch.stack([
  B[0] + torch.randn(n_per_cluster) * blur,
  B[1] + torch.randn(n_per_cluster) * blur
]).float().T
c = torch.stack([
  C[0] + torch.randn(n_per_cluster) * blur,
  C[1] + torch.randn(n_per_cluster) * blur
]).float().T

data = torch.vstack((a, b, c))
labels = torch.hstack((
  torch.zeros(n_per_cluster),
  torch.ones(n_per_cluster),
  torch.ones(n_per_cluster) + 1
)).long()
dataset = TensorDataset(data, labels)

show_data = False
if show_data:
  fig = plt.figure(figsize=(5, 5))
  plt.plot(a[:, 0], a[:, 1], 'bs', alpha=.5, label='A')
  plt.plot(b[:, 0], b[:, 1], 'ko', alpha=.5, label='B')
  plt.plot(c[:, 0], c[:, 1], 'r^', alpha=.5, label='C')
  plt.title('Qwerties')
  plt.xlabel('Dim 1')
  plt.ylabel('Dim 2')
  plt.show()

def split_data(dset, train_prop=.8, batch_size=32):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=test_size)

  return train_loader, test_loader

class QwertyNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.Sequential(
      nn.Linear(2, 16),
      nn.ReLU(),
      nn.Linear(16, 8),
      nn.ReLU(),
      nn.Linear(8, 3)
    )

  def forward (self, x):
    return self.layers(x)

def create_model(learning_rate=0.01, momentum=0):
  model =  QwertyNet()
  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  return model, loss_func, optimizer

def train_model(train_loader, test_loader, num_epochs=100, momentum=0):
  model, loss_func, optimizer = create_model(momentum=momentum)

  all_losses = np.zeros(num_epochs)
  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_acc = []
    all_batch_losses = []
    for batch_x, batch_y in train_loader:
      y_hat = model(batch_x)

      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      all_batch_losses.append( loss.item() )

      # batch accuracy
      batch_acc = (torch.argmax(y_hat, dim=1) == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )
    
    all_train_acc[epoch_i] = np.mean( all_batch_acc )
    all_losses[epoch_i] = np.mean( all_batch_losses )

    # compute test accuracy
    test_x, test_y = next(iter(test_loader))
    model.train()
    with torch.no_grad():
      test_y_hat = model(test_x)
    
    test_acc = (torch.argmax(test_y_hat, dim=1) == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc.item()
  
  return all_train_acc, all_test_acc, all_losses, model

momenta = [0, .5, .9, .95, .999]
train_loader, test_loader = split_data(dataset)

results = np.zeros((num_epochs, len(momenta), 3))

for idx, mom in enumerate(momenta):
  train_acc, test_acc, losses, net = train_model(train_loader, test_loader, num_epochs=num_epochs, momentum=mom)
  results[:, idx, 0] = train_acc
  results[:, idx, 1] = test_acc
  results[:, idx, 2] = losses

fig, ax = plt.subplots(1, 3, figsize=(16, 5))

for i in range(3):
  ax[i].plot(results[:, :, i])
  ax[i].legend(momenta)
  ax[i].set_xlabel('Epoch')
  ax[i].set_ylabel('Loss' if i == 2 else 'Accuracy')

  if i != 2:
    ax[i].set_ylim([20, 100])

ax[0].set_title('Train accuracy')
ax[1].set_title('Test accuracy')
ax[2].set_title('Train losses')

plt.show()
