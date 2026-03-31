import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url,sep=';')
data = data[data['total sulfur dioxide'] < 200] # drop outliers

# Normalize the input data
cols2zscore = data.keys().drop('quality')
data.loc[:, cols2zscore] = data[cols2zscore].apply(stats.zscore)

# prepare dataset tensors
data_t = torch.tensor(data[cols2zscore].values).float()
labels_t = torch.tensor((data['quality'] > 5).values.astype(np.float32))
labels_t = labels_t.view(-1, 1)
dataset = TensorDataset(data_t, labels_t)

def split_data(dset, train_prop=.8, batch_size=32):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=test_size)

  return train_loader, test_loader

class ANNwine(nn.Module):
  def __init__(self, act_func):
    super().__init__()
  
    self.layers = nn.ModuleDict({
      'input': nn.Linear(11, 16),
      'fc1': nn.Linear(16, 32),
      'fc2': nn.Linear(32, 32),
      'output': nn.Linear(32, 1)
    })

    self.act_func = act_func

  def forward(self, x):
    activation = getattr(torch, self.act_func)
    x = activation( self.layers['input'](x) )
    x = activation( self.layers['fc1'](x) )
    x = activation( self.layers['fc2'](x) )

    return self.layers['output'](x)

num_epochs = 1000
learning_rate = 0.01

def train_model(act_name = 'relu'):
  train_loader, test_loader = split_data(dataset)
  model = ANNwine(act_name)

  loss_func = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  all_losses = np.zeros(num_epochs)
  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    model.train()

    batch_accs = []
    batch_losses = []
    for batch_x, batch_y in train_loader:
      # forward pass and loss
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      batch_losses.append( loss.item() )
      batch_predictions = (y_hat > 0).float()
      batch_acc = 100 * (batch_predictions == batch_y).float().mean()
      batch_accs.append( batch_acc.item() )

    all_losses[epoch_i] = np.mean(batch_losses)
    all_train_acc[epoch_i] = np.mean(batch_accs)

    # compute the test accuracy
    model.eval()
    test_x, test_y = next(iter(test_loader))
    with torch.no_grad(): # deactivate auto-grad
      test_predictions = (model(test_x) > 0).float()

    test_acc = 100 * (test_predictions == test_y).float().mean()
    all_test_acc[epoch_i] = test_acc

  return all_train_acc, all_test_acc, all_losses

act_names_list = ['relu', 'tanh', 'sigmoid']

train_results_by_act = np.zeros((num_epochs, len(act_names_list)))
test_results_by_act = np.zeros((num_epochs, len(act_names_list)))
losses_by_act = np.zeros((num_epochs, len(act_names_list)))

for act_i, act_name in enumerate(act_names_list):
  train_acc, test_acc, losses = train_model(act_name)

  train_results_by_act[:, act_i] = train_acc
  test_results_by_act[:, act_i] = test_acc
  losses_by_act[:, act_i] = losses

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(train_results_by_act)
ax[0].set_title('Train accuracy')
ax[1].plot(test_results_by_act)
ax[1].set_title('Test accuracy')
ax[2].plot(losses_by_act)
ax[2].set_title('Losses')

for i in range(3):
  ax[i].legend(act_names_list)
  ax[i].set_xlabel('Epoch')
  ax[i].set_ylabel('Loss' if i == 2 else 'Accuracy (%)')

  if (i != 2):
    ax[i].set_ylim([50, 100])
  ax[i].grid()

plt.show()
