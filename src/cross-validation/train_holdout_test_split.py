import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, TensorDataset, DataLoader

fakedata = torch.tensor(
  np.tile(np.array([1, 2, 3, 4]), (10, 1)) + np.tile(10 * np.arange(1, 11), (4, 1)).T
)
fakelabels = torch.randint(0, 2, (10,))

dataset = TensorDataset(fakedata, fakelabels)

total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, shuffle=True)
val_loader = DataLoader(val_data, shuffle=False)
test_loader = DataLoader(test_data, shuffle=False)

for batch_X, batch_y in train_loader:
  print(f'{batch_X=}, {batch_y=}')

# manual split using numpy
labels_len = len(fakelabels)
partitions = np.array([.8, .1, .1])
partitionBnd = np.cumsum(labels_len * partitions).astype(int)
randindices = np.random.permutation(labels_len)

train_dataN = fakedata[randindices[:partitionBnd[0]], :]
train_labelN = fakelabels[randindices[:partitionBnd[0]]]

devset_dataN = fakedata[
  randindices[partitionBnd[0]:partitionBnd[1]], :
]
devset_labelN = fakelabels[randindices[partitionBnd[0]:partitionBnd[1]]]

test_dataN = fakedata[
  randindices[partitionBnd[1]:partitionBnd[2]], :
]
test_labelN = fakelabels[randindices[partitionBnd[1]:partitionBnd[2]]]

print(train_dataN)
print(train_labelN)
print(devset_dataN)
print(devset_labelN)
print(test_dataN)
print(test_labelN)
