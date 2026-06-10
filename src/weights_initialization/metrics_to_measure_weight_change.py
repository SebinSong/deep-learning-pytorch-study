import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split

F = nn.functional

curr_dir = Path(__file__).parent
data_fpath = (curr_dir / '../data/mnist_train_small.csv')

df = pd.read_csv(data_fpath, sep=',', header=None)

data_t = torch.tensor( df.values[:, 1:], dtype=torch.float32 )
labels_t = torch.tensor( df.values[:, 0], dtype=torch.long )

# minmax normalization
data_t /= torch.max(data_t)

dataset = TensorDataset(data_t, labels_t)

# some global variables
D_in = data_t.shape[1]
D_out = torch.unique(labels_t).numel()
learning_rate = 0.001
num_epochs = 60

def split_data(dset, train_prop=.9, batch_size=32):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_ldr = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_ldr = DataLoader(test_dset, batch_size=test_size)

  return train_ldr, test_ldr

class MNIST_ANN(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleDict({
      'fc1': nn.Linear(D_in, 128), # weight matrix btw input - hidden1
      'fc2': nn.Linear(128, 64, bias=True), # weight matrix btw hidden1 - hidden2
      # 'bnorm2': nn.BatchNorm1d(64),
      'fc3': nn.Linear(64, 32, bias=True), # weight matrix btw hidden2 - hidden3
      # 'bnorm3': nn.BatchNorm1d(32),
      'output': nn.Linear(32, D_out) # weight matrix btw hidden3 - output
    })

  def forward(self, x):
    x = F.relu( self.layers['fc1'](x) )

    # Linear -> Batch Norm -> Activation
    x = F.relu( self.layers['fc2'](x) )

    # Linear -> Batch Norm -> Activation
    x = F.relu( self.layers['fc3'](x) )

    return self.layers['output'](x)

def create_model():
  m = MNIST_ANN()
  l = nn.CrossEntropyLoss()
  o = torch.optim.SGD(m.parameters(), lr=learning_rate)

  return m, l, o

def get_all_weights_from_model(m):
  all_weights = []
  for name, p in m.named_parameters():
    if 'weight' in name:
      all_weights.append( copy.deepcopy(p.data) )

  return all_weights

def train_model(train_ldr, test_ldr):
  model, loss_func, optimizer = create_model()
  test_x, test_y = next(iter(test_ldr))

  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)
  all_losses = np.zeros(num_epochs)

  weight_changes = np.zeros((num_epochs, 4))
  weight_conds = np.zeros((num_epochs, 4))

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_acc = []
    all_batch_losses = []

    preW = get_all_weights_from_model(model)

    for batch_x, batch_y in train_ldr:
      y_hat = model(batch_x)

      loss = loss_func(y_hat, batch_y)

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      all_batch_losses.append( loss.item() )

      # batch_acc
      batch_acc = (torch.argmax(y_hat, dim=1) == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )
    
    all_train_acc[epoch_i] = np.mean(all_batch_acc)
    all_losses[epoch_i] = np.mean(all_batch_losses)

    # compute test_acc
    model.eval()
    with torch.no_grad():
      test_pred = torch.argmax(model(test_x), dim=1)

    test_acc = (test_pred == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc.item()

    afterW = get_all_weights_from_model(model)

    for idx in range(len(afterW)):
      pre_w_item = preW[idx].data.numpy()
      after_w_item = afterW[idx].data.numpy()

      weight_conds[num_epochs, idx] = np.linalg.cond(after_w_item)
      weight_changes[num_epochs, idx] = np.linalg.norm(pre_w_item - after_w_item, ord='fro')

  return all_train_acc, all_test_acc, all_losses, model, weight_conds, weight_changes

net = MNIST_ANN()
