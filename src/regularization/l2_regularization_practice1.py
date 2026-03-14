import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import TensorDataset, DataLoader, random_split

iris = sns.load_dataset('iris')
species_col = iris['species']

# prepare dataset
data = torch.tensor( iris[iris.columns[:-1]].values).float()
labels_mapping = { 'setosa': 0, 'versicolor': 1, 'virginica': 2 }
labels = torch.tensor( species_col.map(labels_mapping).values )
dataset = TensorDataset(data, labels)

# train parameters
D_in = data.shape[1]
D_out = torch.unique(labels).numel()
n_hidden_units = 64

# hyperparameters
num_epochs = 1000

def split_dataset(dset, train_prop=.8):
  sample_size = len(dataset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_set, test_set = random_split(dset, [train_size, test_size])

  train_dlr = DataLoader(train_set, batch_size=64, shuffle=True)
  test_dlr = DataLoader(test_set, batch_size=test_size, shuffle=False)

  return train_dlr, test_dlr

def create_model(L2lambda=.01, learning_rate=0.005):
  ANNiris = nn.Sequential(
    nn.Linear(D_in, n_hidden_units),
    nn.ReLU(),
    nn.Linear(n_hidden_units, n_hidden_units),
    nn.ReLU(),
    nn.Linear(n_hidden_units, D_out)
  )
  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(ANNiris.parameters(), lr=learning_rate, weight_decay=L2lambda)

  return ANNiris, loss_func, optimizer

def train_the_model(model, loss_func, optimizer, train_dlr, test_dlr):
  train_acc = np.zeros(num_epochs)
  test_acc = np.zeros(num_epochs)
  losses = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    batch_acc = []
    batch_losses = []

    for batch_x, batch_y in train_dlr:
      # forward pass and loss
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      curr_acc = 100 * torch.mean( (torch.argmax(y_hat, dim=1) == batch_y).float() ).item()
      batch_acc.append(curr_acc)
      batch_losses.append(loss.item())

    train_acc[epoch_i] = np.mean(batch_acc)
    losses[epoch_i] = np.mean(batch_losses)

    # compute test_accuracy
    model.eval()
    test_x, test_y = next( iter(test_dlr) )
    test_pred_labels = torch.argmax(model(test_x), dim=1)
    test_acc[epoch_i] = 100 * torch.mean( (test_pred_labels == test_y).float() ).item()
    model.train()

  return train_acc, test_acc, losses

train_dlr, test_dlr = split_dataset(dataset)
model, loss_func, optimizer = create_model()
train_acc, test_acc, losses = train_the_model(model, loss_func, optimizer, train_dlr, test_dlr)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].plot(losses, 'k^-')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_title(f'Losses with L2 (lambda={0.01})')

ax[1].plot(train_acc, 'ro-')
ax[1].plot(test_acc, 'bs-')
ax[1].set_title(f'Accuracy with L2 (lambda={0.01})')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy (%)')
ax[1].legend(['Train', 'Test'])

plt.show()
