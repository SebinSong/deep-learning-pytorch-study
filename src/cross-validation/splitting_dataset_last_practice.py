from pathlib import Path
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

# 1. helper function to resolve relative path - Memorize!
base_path = Path(__file__).parent
resolve_rel_path = lambda rel_path: (base_path / rel_path).resolve().as_posix()

# 2. random_splitting

sample_size = 100
D_in = 4

fake_data = np.array([[1, 2, 3, 4]] * sample_size) + \
  np.tile(np.arange(1, sample_size + 1) * 10, (4, 1)).T
fake_labels = np.random.randint(0, 2, (sample_size,))

# Method 1. Using sklearn package:
def split_by_sklearn():
  partitions = [.8, .1, .1]

  train_data, test_temp_data, train_labels, test_temp_labels = \
    train_test_split(fake_data, fake_labels, train_size=partitions[0])

  split = partitions[1] / np.sum(partitions[1:])
  devset_data, testset_data, devset_labels, testset_labels = \
    train_test_split(test_temp_data, test_temp_labels, train_size=split)
  
  return (
    (train_data, train_labels),
    (devset_data, devset_labels),
    (testset_data, testset_labels)
  )

train_pair, devset_pair, testset_pair = split_by_sklearn()

def split_by_torch(train_prop=.7, val_prop=.15, test_prop=.15, /, batch_size=4):
  if (sum([train_prop, val_prop, test_prop]) != 1.0):
    raise ValueError('all proportion parameters must be summed to 1.')

  dataset = TensorDataset(fake_data, fake_labels)

  n_sample = len(fake_data)
  train_size = int(n_sample * train_prop)
  val_size = int(n_sample * val_prop)
  test_size = n_sample - train_size - val_size

  train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
  test_loader =val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

  return (train_loader, val_loader, test_loader)

def split_using_numpy ():
  partitions = [.8, .1, .1]
  partitionBnd = np.cumsum(np.array(partitions) * sample_size).astype(int)
  rand_indices = np.random.permutation(sample_size)

  train_indices = rand_indices[:partitionBnd[0]]
  devset_indices = rand_indices[partitionBnd[0]: partitionBnd[1]]
  test_indices = rand_indices[partitionBnd[1]:]

  to_dl = lambda d, l, /, batchsize=6, shuffle=False: DataLoader(TensorDataset(torch.tensor(d), torch.tensor(l)), batch_size=batchsize, shuffle=shuffle)

  train_loader= to_dl(fake_data[train_indices], fake_labels[train_indices], shuffle=True)
  devset_loader= to_dl(fake_data[devset_indices], fake_labels[devset_indices])
  test_loader= to_dl(fake_data[test_indices], fake_labels[test_indices])

  return (train_loader, devset_loader, test_loader)

train_loader, devset_loader, test_loader = split_using_numpy()

for i, (batch_x, batch_y) in enumerate(train_loader):
  print(f'batch[{i}]:')
  print(f'x: {batch_x}')
  print(f'y: {batch_y}')
