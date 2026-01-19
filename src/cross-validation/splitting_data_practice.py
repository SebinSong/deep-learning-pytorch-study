import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

sample_size = 50

fake_data = np.array([[1, 2, 3, 4]] * sample_size) + \
  np.tile(np.arange(1, sample_size + 1) * 10, (4, 1)).T
fake_labels = np.random.randint(0, 2, (sample_size,))

## Method 1. Using sklearn package:

def split_by_sklearn():
  partitions = [.8, .1, .1]

  train_data, test_temp_data, train_labels, test_temp_labels = \
    train_test_split(fake_data, fake_labels, train_size=partitions[0])

  split = partitions[1] / np.sum(partitions[1:])
  devset_data, test_data, devset_labels, test_labels = \
    train_test_split(test_temp_data, test_temp_labels, train_size=split)

  return (
    (train_data, train_labels),
    (devset_data, devset_labels),
    (test_data, test_labels)
  )

# Method 2. use pytorch DataLoader and random_split:

def split_by_torch(train_prop = .7, val_prop = .15, test_prop = .15, batch_size=4):
  if (sum([train_prop, val_prop, test_prop]) != 1.0):
    raise ValueError('all proportion parameters must be summed to 1.')

  dataset = TensorDataset(
    torch.tensor(fake_data),
    torch.tensor(fake_labels)
  )

  train_size = int(sample_size * train_prop)
  val_size = int(sample_size * val_prop)
  test_size = sample_size - train_size - val_size

  train_data, val_data, test_data = random_split(
    dataset, [train_size, val_size, test_size]
  )

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader, test_loader

# Method 3. manual separation using numpy
def split_using_numpy():
  partitions = [.8, .1, .1]
  partitionBnd = np.cumsum(np.array(partitions) * sample_size).astype(int)
  rand_indices = np.random.permutation(sample_size)

  train_indices = rand_indices[:partitionBnd[0]]
  devset_indices = rand_indices[partitionBnd[0]:partitionBnd[1]]
  test_indices = rand_indices[partitionBnd[1]:]

  to_dloader = lambda d, l, batchsize=6: DataLoader(TensorDataset(torch.tensor(d), torch.tensor(l)), batch_size=batchsize, shuffle=False)

  train_loader = to_dloader(fake_data[train_indices], fake_labels[train_indices])
  devset_loader = to_dloader(fake_data[devset_indices], fake_labels[devset_indices])
  test_loader = to_dloader(fake_data[test_indices], fake_labels[test_indices])

  return train_loader, devset_loader, test_loader

train_loader, devset_loader, test_loader = split_using_numpy()

for i, (batch_x, batch_y) in enumerate(train_loader):
  print(f'batch #{i}.')
  print(f'x - {batch_x}')
  print(f'y - {batch_y}')
