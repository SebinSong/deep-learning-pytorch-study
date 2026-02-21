import torch
import torch.nn as nn
from  torch.utils.data import random_split, TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

F = nn.functional

plot_dataset = False

cluster_size = 200
r1 = 10
r2 = 15

# hyper paramters
learning_rate = 0.002
dr = 0.2
n_epochs = 500

thetas = np.linspace(0, 4 * np.pi, cluster_size)

a = [
  r1 * np.cos(thetas) + np.random.randn(cluster_size) * 3,
  r1 * np.sin(thetas) + np.random.randn(cluster_size)
]

b = [
  r2 * np.cos(thetas) + np.random.randn(cluster_size),
  r2 * np.sin(thetas) + np.random.randn(cluster_size) * 3
]

data_np = np.hstack([a , b]).T
labels_np = np.vstack([
  np.zeros((cluster_size, 1)),
  np.ones((cluster_size, 1))
])

data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()
dataset = TensorDataset(data, labels)

if plot_dataset:
  # plot the sample data
  fig = plt.figure(figsize=(5, 5))
  plt.plot(
    data[torch.where(labels == 0)[0], 0],
    data[torch.where(labels == 0)[0], 1],
    'bs'
  )
  plt.plot(
    data[torch.where(labels == 1)[0], 0],
    data[torch.where(labels == 1)[0], 1],
    'ko'
  )
  plt.title('The qwerties\' doughnuts!')
  plt.xlabel('qwerty dimension 1')
  plt.ylabel('qwerty dimension 2')
  plt.show()

def split_data(dset, train_prop=.8, batch_size = 16):
  sample_size = cluster_size * 2

  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size
  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dset, batch_size=test_size, shuffle=False)

  return (train_loader, test_loader)

train_dataloader, test_dataloader = split_data(dataset, batch_size=9)

class ClassifierA(nn.Module):
  def __init__(self, dropout_rate=.375):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(2, 128),
      'hidden': nn.Linear(128, 128),
      'output': nn.Linear(128, 1)
    })

    self.d_rate = dropout_rate

  def forward(self, x):
    call_relu_dropout = lambda x: F.dropout(F.relu(x), p=self.d_rate, training=self.training)

    # pass data through each layer / dropout
    x = call_relu_dropout( self.layers['input'](x) )
    x = call_relu_dropout( self.layers['hidden'](x) )
    x = self.layers['output'](x)

    return x
  
def create_model (dropout_rate=.375):
  net = ClassifierA(dropout_rate)

  # loss_function
  loss_f = nn.BCEWithLogitsLoss()

  # optimizer
  opt = torch.optim.SGD(net.parameters(), lr=learning_rate)

  return net, loss_f, opt

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

      pred_labels = (y_hat > 0).float()

      acc = 100 * torch.mean( (pred_labels == batch_y).float() ).item()
      batch_acc.append(acc)
    
    train_acc[epoch_i] = np.mean([batch_acc])

    model.eval()
    test_x, test_y = next(iter(test_ldr))
    test_y_hat = model(test_x)
    test_pred_labels = (test_y_hat > 0).float()
    test_acc[epoch_i] = 100 * torch.mean( (test_pred_labels == test_y).float() ).item()

  return train_acc, test_acc

dr_set = np.arange(10) / 10
results = np.zeros((10, 2))


for di, dr in enumerate(dr_set):
  model, loss_func, optimizer = create_model(dropout_rate=dr)
  train_acc, test_acc = train_model(model, loss_func, optimizer, train_dataloader, test_dataloader)

  results[di, 0] = np.mean(train_acc[-100:])
  results[di, 1] = np.mean(test_acc[-100:])

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].plot(dr_set, results, 'o-')
ax[0].set_xlabel('Dropout proportion')
ax[0].set_ylabel('Average accuracy')
ax[0].legend(['Train', 'Test'])

ax[1].plot(dr_set, -np.diff(results, axis=1), 'o-')
ax[1].plot([0, .9], [0, 0], 'k--')
ax[1].set_xlabel('Dropout proportion')
ax[1].set_ylabel('Train-test ave accuracy difference (%)')

plt.show()
