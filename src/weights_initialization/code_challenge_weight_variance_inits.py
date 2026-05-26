import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import matplotlib.pyplot as plt
from pathlib import Path

F = nn.functional

curr_dir = Path(__file__).parent
data_fpath = (curr_dir / '../data/mnist_train_small.csv').resolve()

df = pd.read_csv(data_fpath, sep=',', header=None)
data_t = torch.tensor(df.values[:, 1:], dtype=torch.float32)
labels_t = torch.tensor(df.values[:, 0], dtype=torch.long)

# minmax normalize the data
data_t /= torch.max(data_t)

dataset = TensorDataset(data_t, labels_t)

# some global variables 
D_in = data_t.shape[1]
D_out = torch.unique(labels_t).numel()
num_epochs = 10

def get_data_labels_from_ldr(ldr):
  tensors = ldr.dataset.dataset.tensors
  indices = ldr.dataset.indices

  data = tensors[0][indices]
  labels = tensors[1][indices]
  return data, labels

def split_data(dset, train_prop=.9, batch_size=32):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_ldr = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_ldr = DataLoader(test_dset, batch_size=test_size)

  return train_ldr, test_ldr

train_loader, test_loader = split_data(dataset)

class MNIST_ANN(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.Sequential(
      nn.Linear(D_in, 64), # weight matrix btw input - hidden1
      nn.ReLU(),
      nn.Linear(64, 32, bias=True), # weight matrix btw hidden1 - hidden2
      # nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Linear(32, 32, bias=True), # weight matrix btw hidden2 - hidden3
      # nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Linear(32, D_out) # the loss function will be nn.CrossEntropyLoss() which has log-softmax internally.
    )

  def forward(self, x):
    return self.layers(x)
  
def create_model(lr=0.001, std=1):
  m = MNIST_ANN()
  l = nn.CrossEntropyLoss()
  o = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)

  # custom-init the weights according to the given std value.
  for name, p in m.named_parameters():
    if 'weight' in name:
      p.data = torch.randn(p.data.shape) * std

  return m, l, o

def train_model(train_ldr, test_ldr, std=1):
  model, loss_func, optimizer = create_model(std=std)

  test_x, test_y = next(iter(test_ldr))

  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)
  all_losses = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_acc = []
    all_batch_losses = []
    for batch_x, batch_y in train_ldr
      pass
