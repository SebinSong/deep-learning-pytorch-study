import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

F = nn.functional

curr_dir = Path(__file__).resolve().parent
data_fpath = (curr_dir / '../data/mnist_train_small.csv').resolve()

df = pd.read_csv(data_fpath, header=None, sep=',')

data = torch.tensor( df.values[:, 1:], dtype=torch.float32)
labels = torch.tensor( df.values[:, 0], dtype=torch.long )

data_norm = data / torch.max(data) # min-max normalization

# some global variables
batch_size = 20
learning_rate = 0.01
num_epochs = 60

def create_data(N = 2000, double_size=False):
  new_data = data_norm[:N]
  new_labels = labels[:N]

  if double_size:
    # make a noisy copy of all the data
    new_data_noisy = torch.clamp( new_data + torch.rand(new_data.shape) / 2, min=0, max=1 )
    new_data = torch.cat((new_data, new_data_noisy), dim=0)
    new_labels = torch.cat((new_labels, new_labels), dim=0)
  
  dataset = TensorDataset(new_data, new_labels)

  sample_size = new_data.shape[0]
  train_size = int(sample_size * 0.8)
  devset_size = sample_size - train_size

  train_dset, devset_dset = random_split(dataset, [train_size, devset_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  devset_loader = DataLoader(devset_dset, batch_size=devset_size)

  test_data = data_norm[N:]
  test_labels = labels[N:]

  return train_loader, devset_loader, (test_data, test_labels)

train_loader, devset_loader, test_dset = create_data(40, True)

class MNIST_ANN(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(28 * 28, 64),
      'fc0': nn.Linear(64, 32),
      'fc1': nn.Linear(32, 32),
      'output': nn.Linear(32, 10)
    })

  def forward(self, x):
    x = F.relu( self.layers['input'](x) )
    x = F.relu( self.layers['fc0'](x) )
    x = F.relu( self.layers['fc1'](x) )
    return self.layers['output'](x)

def create_model():
  m = MNIST_ANN()
  l = nn.CrossEntropyLoss()
  o = torch.optim.SGD(m.parameters(), lr=learning_rate)
  return m, l, o

def train_model(N, double_size):
  train_ldr, devset_ldr, test_dset = create_data(N, double_size)

  model, loss_func, optimizer = create_model()
  devset_x, devset_y = next(iter(devset_ldr))

  all_train_acc = np.zeros(num_epochs)
  all_devset_acc = np.zeros(num_epochs)
  all_losses = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_acc = []
    all_batch_losses = []
    for batch_x, batch_y in train_ldr:
      y_hat = model(batch_x)

      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      all_batch_losses.append( loss.item() )

      # batch_acc
      batch_acc = (torch.argmax(y_hat, dim=1) == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )

    train_acc = np.mean( all_batch_acc )
    ave_loss = np.mean( all_batch_losses )
    all_train_acc[epoch_i] = train_acc
    all_losses[epoch_i] = ave_loss

    if (epoch_i % 10 == 0):
      print(f'[N={N}, Epoch={epoch_i}] - {train_acc=:.3f}, {ave_loss=:.3f}')
 
    # devset accuracy
    model.eval()
    with torch.no_grad():
      devset_pred_labels = torch.argmax( model(devset_x), dim=1 )
    
    devset_acc = (devset_pred_labels == devset_y).float().mean() * 100
    all_devset_acc[epoch_i] = devset_acc.item()
  
  test_x, test_y = test_dset
  with torch.no_grad():
    test_pred_labels = torch.argmax(model(test_x), dim=1)
  test_acc = (test_pred_labels == test_y).float().mean().item() * 100
  return all_train_acc, all_devset_acc, all_losses, test_acc

sample_sizes = np.arange(500, 2501, 500)
len_sample_size = len(sample_sizes)

results_single = np.zeros((len_sample_size, 4))
results_double = np.zeros((len_sample_size, 4))

for s_idx, sample_size in enumerate(sample_sizes):
  def run_test(should_double=False):
    res_matrix = results_double if should_double else results_single

    train_acc, devset_acc, losses, test_acc = train_model(sample_size, should_double)
    res_matrix[s_idx, 0] = np.mean(train_acc[-5:])
    res_matrix[s_idx, 1] = np.mean(devset_acc[-5:])
    res_matrix[s_idx, 2] = test_acc
    res_matrix[s_idx, 3] = np.mean(losses[-5:])

  run_test()
  run_test(True)

fig, ax = plt.subplots(2, 2, figsize=(15, 5))
flattened_ax = ax.flatten()

titles = ['Train', 'Devset', 'Test', 'Losses']
y_labels = ['Accuracy', 'Accuracy', 'Accuracy', 'Losses']

for i in range(4):
  flattened_ax[i].plot(sample_sizes, results_single[:, i], 's-', label='Original')
  flattened_ax[i].plot(sample_sizes, results_double[:, i], 's-', label='Augmented')
  flattened_ax[i].set_title(titles[i])
  flattened_ax[i].set_xlabel('Sample sizes')
  flattened_ax[i].set_ylabel(y_labels[i])
  flattened_ax[i].legend()
  flattened_ax[i].grid('on')

  if i < 3:
    flattened_ax[i].set_ylim([20, 102])

plt.tight_layout()
plt.show()
