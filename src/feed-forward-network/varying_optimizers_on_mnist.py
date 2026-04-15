import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path

F = nn.functional

def to_numpy(t):
  if t.requires_grad:
    return t.detach().cpu().numpy()
  return t.cpu().numpy()

curr_dir = Path(__file__).parent
data_fpath = (curr_dir / '../data/mnist_train_small.csv').resolve()

df = pd.read_csv(data_fpath, sep=',', header=None)

data = torch.tensor(df.values[:, 1:], dtype=torch.float32)
labels = torch.tensor(df.values[:, 0], dtype=torch.long)

# Normalize the data (minmax normalization)
data /= 255

dataset = TensorDataset(data, labels)

# hyperparameters
num_epochs = 60

def split_data (dset, train_prop=.9, batch_size=64):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_ldr = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_ldr = DataLoader(test_dset, batch_size=test_size)

  return train_ldr, test_ldr

class MNIST_ANN(nn.Module):
  def __init__(self, batch_norm=False):
    super().__init__()

    self.has_batch_norm = batch_norm
    self.layers = nn.ModuleDict({
      'input': nn.Linear(28*28, 128),
      'fc0': nn.Linear(128, 64, bias=(not batch_norm)),
      'fc1': nn.Linear(64, 32, bias=(not batch_norm)),
      'output': nn.Linear(32, 10),
      **({
        'bnorm0': nn.BatchNorm1d(64),
        'bnorm1': nn.BatchNorm1d(32)
      } if batch_norm else {})
    })

  def forward(self, x):
    x = F.relu( self.layers['input'](x) )

    # Linear -> Batch Norm -> Activation
    x = self.layers['fc0'](x)
    if self.has_batch_norm:
      x = self.layers['bnorm0'](x)
    x = F.relu(x)

    x = self.layers['fc1'](x)
    if self.has_batch_norm:
      x = self.layers['bnorm1'](x)
    x = F.relu(x)

    return self.layers['output'](x)

def create_model(learning_rate=0.01, optim_type='SGD'):
  model = MNIST_ANN()
  loss_func = nn.CrossEntropyLoss()

  optim_func = getattr(torch.optim, optim_type)
  optimizer = optim_func(model.parameters(), lr=learning_rate)

  return model, loss_func, optimizer

def train_model (train_ldr, test_ldr, learning_rate=0.01, optim_type='SGD'):
  model, loss_func, optimizer = create_model(learning_rate, optim_type)

  test_x, test_y = next(iter(test_ldr))
  test_accs = []

  for epoch_i in range(num_epochs):
    model.train()

    for batch_x, batch_y in train_ldr:
      y_hat = model(batch_x)

      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    if epoch_i % 5 == 0:
      print(f'[optim={optim_type}, lr={learning_rate}, epoch={epoch_i}] loss - {loss.item():.3f}')

    if epoch_i > (num_epochs - 10):
      model.eval()

      with torch.no_grad():
        test_pred_labels = torch.argmax(model(test_x), dim=1)
      
      test_acc = (test_pred_labels == test_y).float().mean() * 100
      test_accs.append( test_acc.item() )

  ave_test_acc = np.mean( test_accs )
  return ave_test_acc

optim_types = ['SGD', 'RMSprop', 'Adam']
learing_rates = np.logspace(-4, -1, 6, base=10)

all_test_accs = np.zeros((len(learing_rates), len(optim_types)))

train_loader, test_loader = split_data(dataset)

for opt_i, opt_type in enumerate(optim_types):
  for lr_i, lr in enumerate(learing_rates):
    test_acc = train_model(train_loader, test_loader, learning_rate=lr, optim_type=opt_type)
    all_test_accs[lr_i, opt_i] = test_acc

# plot the results
plt.figure(figsize=(7, 7))

for opt_i, opt_type in enumerate(optim_types):
  plt.plot(
    learing_rates,
    all_test_accs[:, opt_i],
    'o-',
    label=opt_type
  )

plt.xscale('log')
plt.xlabel('Learning rates')
plt.ylabel('Accuracy (%)')
plt.title('Test accuracies')
plt.legend()
plt.show()

