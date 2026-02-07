import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt

# switches
plot_data = False
one_time_run = False

# create data
n_per_cluster = 200
D_in = 2
D_out = 1

th = np.linspace(0, 4 * np.pi, n_per_cluster) # thetas
R1 = 10 # radius for cluster 1
R2 = 15 # radius for cluster 2

# generate data

# cluster 1 - label '0'
a = np.array([
  R1 * np.cos(th) + np.random.randn(n_per_cluster) * 3,
  R1 * np.sin(th) + np.random.randn(n_per_cluster)
])

# cluster 2 - label '1'
b = np.array([
  R2 * np.cos(th) + np.random.randn(n_per_cluster),
  R2 * np.sin(th) + np.random.randn(n_per_cluster) * 3
])

labels_np = np.vstack([
  np.zeros((n_per_cluster, 1)),
  np.ones((n_per_cluster, 1))
])

data_np = np.hstack([ a, b ]).T

# convert numpy arrays to pytorch tensors
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

if plot_data:
  # plot the data
  plt.plot(
    data[torch.where(labels == 0)[0], 0],
    data[torch.where(labels == 0)[0], 1],
    'bs'
  )

  plt.plot(
    data[torch.where(labels == 1)[0], 0],
    data[torch.where(labels == 1)[0], 1],
    'yo'
  )
  plt.title('The qwertie doughnuts!')
  plt.xlabel('qwerty dim 1')
  plt.xlabel('qwerty dim 2')
  plt.show()

# separate the data
def split_data(train_prop=.8, train_batch_size=16):
  sample_size = n_per_cluster * len(torch.unique(labels))
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  dataset = TensorDataset(data, labels)
  train_ds, test_ds = random_split(dataset, [train_size, test_size])

  train_dlr = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
  test_dlr = DataLoader(test_ds, batch_size=test_size)

  return train_dlr, test_dlr

# define the model
class QuertyClassifier(nn.Module):
  def __init__(self, dropout_rate=.5):
    super().__init__()

    # layers
    self.layers = nn.ModuleDict({
      'input': nn.Linear(D_in, 128), # input -> hidden0
      'hidden': nn.Linear(128, 128), # hidden0 -> hidden1
      'output': nn.Linear(128, D_out) # hidden1 -> output
    })

    # parameters
    self.dr = dropout_rate

  # forward pass
  def forward(self, x):
    x = F.relu( self.layers['input'](x) )
    x = F.dropout(x, p=self.dr, training=self.training)

    x = F.relu( self.layers['hidden'](x) )
    x = F.dropout(x, p=self.dr, training=self.training)

    x = self.layers['output'](x)

    return x

# function to create a model
def create_model (dropout_rate=.5, learning_rate=.002):
  model = QuertyClassifier(dropout_rate)

  loss_func = nn.BCEWithLogitsLoss()

  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  return model, loss_func, optimizer

# train the model
num_epochs = 800

def train_model(model, loss_func, optimizer, train_loader, test_loader):
  train_acc = np.zeros(num_epochs)
  test_acc = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    model.train()

    batch_acc = []
    for batch_x, batch_y in train_loader:
      # forward pass and compute the loss
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)

      # back prop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
      # compute the accuracy
      pred_labels = y_hat > 0
      acc = 100 * torch.mean( (pred_labels == batch_y).float() ).item()
      batch_acc.append(acc)

    train_acc[epoch_i] = np.mean(batch_acc)

    model.eval()
    test_x, test_y = next(iter(test_loader))
    test_y_hat = model(test_x)
    pred_labels = test_y_hat > 0
    test_acc[epoch_i] = 100 * torch.mean( (pred_labels == test_y).float() ).item()

  return train_acc, test_acc

if one_time_run:
  d_rate = 0.0
  train_loader, test_loader = split_data()
  model, loss_func, optimizer = create_model(dropout_rate=d_rate, learning_rate=0.004)
  train_acc, test_acc = train_model(model, loss_func, optimizer, train_loader, test_loader)

  def smooth_filter(x, k=5):
    return np.convolve(x, np.ones(k)/k, mode='same')

  fig = plt.figure(figsize=(10, 5))

  plt.plot(smooth_filter(train_acc), 'bs-', label='Train')
  plt.plot(smooth_filter(test_acc), 'ro-', label='Test')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy (%)')
  plt.legend()
  plt.title(f'Dropout rate = {d_rate:.3f}')
  plt.show()

# experiments
dropout_rates = np.arange(10) / 10
results = np.zeros( (len(dropout_rates), 2) )

# prepare dataloaders
train_loader, test_loader = split_data()

for di, d_rate in enumerate(dropout_rates):
  model, loss_func, optimizer = create_model(dropout_rate=d_rate, learning_rate=0.002)
  train_acc, test_acc = train_model(model, loss_func, optimizer, train_loader, test_loader)

  results[di, 0] = np.mean(train_acc[-100:])
  results[di, 1] = np.mean(test_acc[-100:])

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(dropout_rates, results, 'o-')
ax[0].set_xlabel('Dropout rate')
ax[0].set_ylabel('Average accuracy')
ax[0].legend(['Train', 'Test'])

ax[1].plot(dropout_rates, -np.diff(results, axis=1), 'o-')
ax[1].plot([0, .9], [0, 0], 'k--')
ax[1].set_xlabel('Dropout rate')
ax[1].set_ylabel('Train-test difference (acc%)')

plt.show()
