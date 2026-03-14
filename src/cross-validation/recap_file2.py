import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset, random_split

# helpers
F = nn.functional
unique_count = lambda t: torch.unique(t).numel()

# hyper parameters
num_epochs = 500

# load iris data
iris = sns.load_dataset('iris')
species_col = iris['species']
labels_mapping = { 'setosa': 0, 'versicolor': 1, 'virginica': 2 }

# prepare torch data
data = torch.tensor( iris[iris.columns[:-1]].values ).float()
labels = torch.tensor( species_col.map(labels_mapping).fillna(-1).values, dtype=torch.long)
dataset = TensorDataset(data, labels)

D_in = data.shape[1]
D_out = unique_count(labels)  # can be: species_col.nunique() as well

# split the dataset
def split_dataset(dset, train_prop=.8):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_ds, test_ds = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_ds, batch_size=12, shuffle=True)
  test_loader = DataLoader(test_ds, batch_size=test_size, shuffle=False)

  return train_loader, test_loader

def create_model(n_hidden_units = 12, learning_rate=0.1):
  # model architecture
  ANNiris = nn.Sequential(
    nn.Linear(D_in, n_hidden_units),
    nn.ReLU(),
    nn.Linear(n_hidden_units, n_hidden_units),
    nn.ReLU(),
    nn.Linear(n_hidden_units, D_out)
  )

  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(ANNiris.parameters(), lr=learning_rate)

  return ANNiris, loss_func, optimizer

def train_model(model, loss_func, optimizer, train_dlr, test_dlr):
  # init accuracies
  train_acc = np.zeros(num_epochs)
  test_acc = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    batch_acc = []

    for batch_i, (batch_x, batch_y) in enumerate(train_dlr):
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      curr_acc = 100 * torch.mean( (torch.argmax(y_hat, axis=1) == batch_y).float() ).item()
      batch_acc.append(curr_acc)

    train_acc[epoch_i] = np.array(batch_acc).mean()

    test_x, test_y = next(iter(test_dlr))
    pred_labels = torch.argmax(model(test_x), axis=1)
    test_acc[epoch_i] = 100 * torch.mean( (pred_labels == test_y).float() ).item()

  return train_acc, test_acc

train_dlr, test_dlr = split_dataset(dataset)
model, loss_func, optimizer = create_model()
train_acc, test_acc = train_model(model, loss_func, optimizer, train_dlr, test_dlr)

print(test_acc)