import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import numpy as np

sample_size = 100

# setup synthetic data
X = torch.randn(sample_size, 5) # 100 samples, 5 features
y = torch.randn(sample_size, 1) # 100 targets

dataset = TensorDataset(X, y)

# define split sizes
train_size = int(0.8 * sample_size)
val_size = sample_size - train_size

# perform the split
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# create dataloader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

def k_fold(folds = 5):
  indices = list(range(sample_size))
  fold_size = sample_size // folds

  # shuffle the indices
  np.random.shuffle(indices)

  for fold in range(folds):
    # define the start and end of the validation slice
    val_start = fold * fold_size
    val_end = val_start + fold_size

    val_indices = indices[val_start:val_end]
    train_indices = indices[:val_start] + indices[val_end:]

    # create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # create dataloader
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
