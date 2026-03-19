import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd

iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
species_col = iris['species']

data = torch.tensor(iris[iris.columns[:-1]].values, dtype=torch.float32)
labels_mapping = { 'setosa': 0, 'versicolor': 1, 'virginica': 2 }
labels = torch.tensor(species_col.map(labels_mapping).fillna(-1), dtype=torch.long)
dataset = TensorDataset(data, labels)

# train parameters
D_in = len(iris.columns[:-1])
D_out = torch.unique(labels).numel()
n_hidden_units = 64

# hyperparameters
num_epochs = 1000

def split_data (dset, train_prop=.8):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=64, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=test_size, shuffle=False)

  return train_loader, test_loader

def create_model(learning_rate=.005):
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

def train_model(model, loss_func, optimizer, train_dlr, test_dlr, /, L1lambda=.001):
  # initialize accuracies as empties
  train_acc = np.zeros(num_epochs)
  test_acc = np.zeros(num_epochs)
  losses = np.zeros(num_epochs)

  # count the total number of weights in the model
  n_weights = sum([ p.numel() for n, p in model.named_parameters() if 'bias' not in n ])

  for epoch_i in range(num_epochs):
    model.train()

    batch_accs = []
    batch_losses = []
    for batch_x, batch_y in train_dlr:
      batch_y_hat = model(batch_x)

      loss = loss_func(batch_y_hat, batch_y)

      # L1 regularization
      l1_sum = sum(p.abs().sum() for n, p in model.named_parameters() if 'bias' not in n)
      loss += (L1lambda / n_weights) * l1_sum

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      batch_acc = 100 * torch.mean( (torch.argmax(batch_y_hat, dim=1) == batch_y).float() )
      batch_accs.append(batch_acc.item())
      batch_losses.append(loss.item())

    train_acc[epoch_i] = np.mean(batch_accs)
    losses[epoch_i] = np.mean(batch_losses)

    # compute the test_accuracy
    model.eval()

    with torch.no_grad():
      test_x, test_y = next(iter(test_dlr))
      pred_test_labels = torch.argmax(model(test_x), dim=1)
      test_acc[epoch_i] = 100 * torch.mean((pred_test_labels == test_y).float())

  return train_acc, test_acc, losses

# create a 1D smoothing filter
def smooth (x, k):
  return np.convolve(x, np.ones(k)/k, mode='same')

# split dataset
train_dlr, test_dlr = split_data(dataset)

# Run parametric experiments
lambda_list = np.linspace(0, .005, 10)

acc_results_train = np.zeros((num_epochs, len(lambda_list)))
acc_results_test = np.zeros((num_epochs, len(lambda_list)))

for li, lambda_val in enumerate(lambda_list):
  ANNiris, loss_func, optimizer = create_model()
  train_acc, test_acc, _ = train_model(
    ANNiris, loss_func, optimizer,
    train_dlr, test_dlr, L1lambda=lambda_val
  )

  acc_results_train[:, li] = smooth(train_acc, 10)
  acc_results_test[:, li] = smooth(test_acc, 10)

fig, ax = plt.subplots(1, 2, figsize=(17, 7))

ax[0].plot(acc_results_train)
ax[0].set_title('Train accuracy')
ax[1].plot(acc_results_test)
ax[1].set_title('Test accuracy')

leg_labels = [np.round(l, 4) for l in lambda_list]

for i in range(2):
  ax[i].legend(leg_labels)
  ax[i].set_xlabel('Epoch')
  ax[i].set_ylabel('Accuracy (%)')
  ax[i].set_ylim([50, 101])
  ax[i].grid()

plt.show()
