import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from pathlib import Path

F = nn.functional

curr_dir = Path(__file__).parent
resolve_path = lambda rel_path: (curr_dir / rel_path).resolve()
data_fpath = resolve_path('../data/mnist_train_small.csv')

df = pd.read_csv(data_fpath, sep=',', header=None)
data = torch.tensor( df.values[:, 1:], dtype=torch.float32 )
labels = torch.tensor( df.values[:, 0], dtype=torch.long )

dataset = TensorDataset(data, labels)

# hyper parameters
num_epochs = 50
learning_rate = 0.01

def split_data(dset, train_prop=.9, batch_size=32):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=test_size)

  return train_loader, test_loader

train_loader, test_loader = split_data(dataset)

class ANN_MNIST(nn.Module):
  def __init__(self, n_hidden_layers = 1, n_hidden_units=50):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(28 * 28, n_hidden_units)
    })

    for i in range(n_hidden_layers):
      self.layers[f'fc{i}'] = nn.Linear(n_hidden_units, n_hidden_units)

    self.layers['output'] = nn.Linear(n_hidden_units, 10)
    self.n_hidden_layers = n_hidden_layers
  
  def forward(self, x):
    x = F.relu( self.layers['input'](x) )

    for n_layer in range(self.n_hidden_layers):
      x = F.relu( self.layers[f'fc{n_layer}'](x) )

    return self.layers['output'](x)

def create_model(n_hidden_layers=1, n_hidden_units=50):
  global learning_rate

  model = ANN_MNIST(n_hidden_layers, n_hidden_units)
  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  return model, loss_func, optimizer

def train_model(train_ldr, test_ldr, n_hidden_layers=1, n_hidden_units=50):
  global num_epochs

  model, loss_func, optimizer = create_model(n_hidden_layers, n_hidden_units)

  last_10_train_acc = []
  last_10_test_acc = []

  for epoch_i in range(num_epochs):
    model.train()

    should_compute_accuracy = epoch_i >= (num_epochs - 10)
    test_x, test_y = next(iter(test_ldr))
    all_batch_acc = []

    for batch_x, batch_y in train_ldr:
      # forward-pass
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)

      # back prop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if should_compute_accuracy:
        # compute batch accuracy
        batch_acc = (torch.argmax(y_hat, dim=1) == batch_y).float().mean() * 100
        all_batch_acc.append( batch_acc.item() )

    if epoch_i % 5 == 0:
      print(f'[nl={n_hidden_layers}, nu={n_hidden_units}, epoch={epoch_i}]: loss={loss.item():.3f}')

    if should_compute_accuracy:
      last_10_train_acc.append( np.mean( all_batch_acc ) )

      # compute test accuracy
      model.eval()
      with torch.no_grad():
        test_pred_labels = torch.argmax( model(test_x), dim=1 )
      
      test_acc = (test_pred_labels == test_y).float().mean() * 100
      last_10_test_acc.append( test_acc.item() )

  mean_train_acc = np.mean( last_10_train_acc )
  mean_test_acc = np.mean( last_10_test_acc )
  return mean_train_acc, mean_test_acc

# Let's run the test
train_loader, test_loader = split_data(dataset)

num_layers = np.arange(1, 4, step=1)
num_hidden_units = np.arange(50, 251, step=50)

train_acc_results = np.zeros((len(num_hidden_units), len(num_layers)))
test_acc_results = np.zeros((len(num_hidden_units), len(num_layers)))

for i_layer, n_layer in enumerate(num_layers):
  for i_hu, n_hidden_units in enumerate(num_hidden_units):
    train_acc, test_acc = train_model(train_loader, test_loader, n_layer, n_hidden_units)

    train_acc_results[i_hu, i_layer] = train_acc
    test_acc_results[i_hu, i_layer] = test_acc

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

for i in range(2):
  results = train_acc_results if i == 0 else test_acc_results

  for li in range(len(num_layers)):
    num_layer = num_layers[li]

    ax[i].plot(
      num_hidden_units,
      results[:, li],
      label=f'No of layers - {num_layer}'
    )

  ax[i].set_xlabel('No of hidden units')
  ax[i].set_ylabel('Accuracy (%)')
  ax[i].legend()

ax[0].set_title('Train accuracies')
ax[1].set_title('Test accuracies')

plt.tight_layout()
plt.suptitle('Accuracies per varying number of hidden layers')
plt.show()
