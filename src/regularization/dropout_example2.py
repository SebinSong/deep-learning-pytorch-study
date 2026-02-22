import torch
import torch.nn as nn
from torch.utils.data import random_split, TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

F = nn.functional

iris = sns.load_dataset('iris')
data = torch.tensor( iris[iris.columns[:-1]].values ).float()

# labels
labels_category = pd.Categorical(iris['species'], categories=['setosa', 'versicolor', 'virginica'])
labels = torch.tensor( labels_category.codes ).long()

# dataset
dataset = TensorDataset(data, labels)

# model params
batch_size = 16
sample_size = len(data)
D_in, D_out = len(iris.columns[:-1]), len(labels_category.categories)

# hyper params
learning_rate = 0.005
n_epochs = 400

class IrisClassifier(nn.Module):
  def __init__(self, dropout_rate):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(D_in, 12),
      'hidden': nn.Linear(12, 12),
      'output': nn.Linear(12, D_out)
    })

    self.dr = dropout_rate

  def forward(self, x):
    relu_and_dropout = lambda x: F.dropout(F.relu(x), p=self.dr, training=self.training)

    x = relu_and_dropout( self.layers['input'](x) )
    x = relu_and_dropout( self.layers['hidden'](x) )
    x = self.layers['output'](x)

    return x

def split_data(dset, train_prop=.8):
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])
  train_ldr = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
  test_ldr = DataLoader(test_dset, batch_size=test_size, shuffle=False)

  return train_ldr, test_ldr

def create_model(dropout_rate=0.0):
  model = IrisClassifier(dropout_rate)

  # CrossEntropyLoss is LogSoftmax + NLLLoss.
  loss_func = nn.CrossEntropyLoss()

  optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

  return model, loss_func, optimizer

def train_model(model, loss_func, optimizer, train_ldr, test_ldr):
  train_acc = np.zeros(n_epochs)
  test_acc = np.zeros(n_epochs)

  for epoch_i in range(n_epochs):
    model.train()

    batch_acc = []
    for batch_x, batch_y in train_ldr:
      y_hat = model(batch_x)

      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      acc = 100 * torch.mean( (torch.argmax(y_hat, dim=1) == batch_y).float() ).item()
      batch_acc.append(acc)

    train_acc[epoch_i] = np.mean(batch_acc)

    # evaluate the model - disable dropout
    model.eval()
    test_x, test_y = next(iter(test_ldr))
    test_y_hat = model(test_x)
    test_acc[epoch_i] = 100 * torch.mean( (torch.argmax(test_y_hat, dim=1) == test_y).float() ).item()

  return train_acc, test_acc

# run the test

dr_set = torch.arange(0, 10) / 10
results = torch.zeros((len(dr_set), 2))

for di, dr in enumerate(dr_set):
  train_loader, test_loader = split_data(dataset)
  model, loss_func, optimizer = create_model(dropout_rate=dr)
  train_acc, test_acc = train_model(model, loss_func, optimizer, train_loader, test_loader)

  results[di, 0] = np.mean(train_acc[-100:])
  results[di, 1] = np.mean(test_acc[-100:])

fig, ax = plt.subplots(1, 2, figsize=(12, 7))
ax[0].plot(dr_set,results, 'o-')
ax[0].set_xlabel('Dropout rate')
ax[0].set_ylabel('Average accuracy')
ax[0].legend(['Train acc', 'Test acc'])

ax[1].plot(dr_set, -np.diff(results, axis=1), 'o-')
ax[1].plot([0, .9], [0, 0], 'k--')
ax[1].set_xlabel('Dropout rate')
ax[1].set_ylabel('Train - Test difference(%)')

plt.show()
