import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split

F = nn.functional

plot_data = False

n_per_clust = 200
th = np.linspace(0, np.pi * 4, n_per_clust)

r1 = 10
r2 = 15

# generate data
a = np.array([
  r1 * np.cos(th) + np.random.randn(n_per_clust) * 3,
  r1 * np.sin(th) + np.random.randn(n_per_clust)
]).T
b = np.array([
  r2 * np.cos(th) + np.random.randn(n_per_clust),
  r2 * np.sin(th) + np.random.randn(n_per_clust) * 3
]).T

data_np = np.vstack([ a, b ])
labels_np = np.vstack([
  np.zeros((n_per_clust, 1)), # labels for a = 0
  np.ones((n_per_clust, 1)) # labels for b = 1
])

data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()
dataset = TensorDataset(data, labels)

D_in = 2
D_out = 1

# plot the data
if plot_data:
  plt.figure(figsize=(7, 7))
  plt.plot(
    data[torch.where(labels == 0)[0], 0],
    data[torch.where(labels == 0)[0], 1],
    'bs',
    label='cluster 1'
  )
  plt.plot(
    data[torch.where(labels == 1)[0], 0],
    data[torch.where(labels == 1)[0], 1],
    'ro',
    label='cluster 2'
  )
  plt.title('The qwerties doughnuts')
  plt.xlabel('dim 1')
  plt.ylabel('dim 2')
  plt.legend()
  plt.show()

def split_data (dset, train_prop=.9):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=16, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=(test_size - 2))

  return train_loader, test_loader

# model
class Classifier(nn.Module):
  def __init__(self):
    super().__init__()

    # layers
    self.layers = nn.ModuleDict({
      'input': nn.Linear(D_in, 128),
      'hidden': nn.Linear(128, 128),
      'output': nn.Linear(128, D_out)
    })

  def forward(self, x):
    x = F.relu( self.layers['input'](x) )
    x = F.relu( self.layers['hidden'](x) )

    return self.layers['output'](x)

def create_model(learning_rate=.01):
  model = Classifier()
  lf = nn.BCEWithLogitsLoss()
  optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

  return model, lf, optim

# global parameters
num_epochs = 500

def train_model(model, loss_func, optimizer, train_ldr, test_ldr):
  train_acc = np.zeros(num_epochs)
  test_acc = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    batch_acc = []

    model.train()
    for batch_x, batch_y in train_ldr:
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      curr_acc = 100 * torch.mean( ((y_hat > 0).float() == batch_y).float() )
      batch_acc.append( curr_acc.item() )
    
    train_acc[epoch_i] = np.mean(batch_acc)

    # compute test-accuracy
    model.eval()
    with torch.no_grad():
      test_batch_acc = []
      for tb_x, tb_y in test_ldr:
        pred_labels = (model(tb_x) > 0).float()
        curr_t_acc = 100 * (pred_labels == tb_y).float().mean()
        test_batch_acc.append( curr_t_acc.item() )

      test_acc[epoch_i] = np.mean(test_batch_acc)

  return train_acc, test_acc

train_ldr, test_ldr = split_data(dataset)
model, loss_func, optimizer = create_model()
train_acc, test_acc = train_model(model, loss_func, optimizer, train_ldr, test_ldr)

plt.figure(figsize=(10, 5))
plt.plot(train_acc, 'bo', label='Train')
plt.plot(test_acc, 'ro', label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()
