import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# rng = np.random.default_rng()
# n1 = rng.choice(100, (4, 3), replace=False)
# bool1 = n1 > 30
# print(bool1)
# print(~bool1)

# arr1 = np.tile(np.array([1, 2, 3, 4]), (10, 1))
# arr2 = 10 * np.tile(np.arange(1, 11), (4, 1)).T
# fakedata = arr1 + arr2
# print(arr2)

# n1 = np.tile(np.arange(1, 11) * 10, (4, 1)).T
# print(n1)

# fruits = ["apple", "banana", "cherry"]
# green, yellow, *rest = fruits
# print(f'{green=}, {yellow=}, {rest=}')

# data = (torch.randn(100, 3) * 100).int()
# labels = torch.randint(0, 2, (100,))
# dataset = TensorDataset(data, labels)

# batch_size = 5
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# num_epochs = 10

# for epoch_i in range(num_epochs):
#   for i, pair in enumerate(dataloader):
#     X, y = pair
#     if i == 0:
#       print(f'epoch[{epoch_i}] - X: {X}')

# t1 = torch.randint(1, 15, (4, 5))
# numel = t1.numel()
# unique_t1 = torch.unique(t1)
# print(unique_t1, unique_t1.numel())

n1 = np.arange(0, 20).reshape((4, 5))
summed_n1 = np.cumsum(n1)

n2 = np.arange(2, 50)
partitions = np.array([0.25, .35, .15, .25])
partitionsBnd = np.cumsum(len(n2) * partitions).astype(int)
print(n2)

rand_indices = np.random.permutation(len(n2))
