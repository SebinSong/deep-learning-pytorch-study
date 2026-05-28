import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import matplotlib.pyplot as plt
from pathlib import Path

F = nn.functional

curr_dir = Path(__file__).parent
data_fpath = (curr_dir / '../data/mnist_train_small.csv').resolve()

df = pd.read_csv(data_fpath, sep=',', header=None)
data_t = torch.tensor(df.values[:, 1:], dtype=torch.float32)
labels_t = torch.tensor(df.values[:, 0], dtype=torch.long)

# minmax normalize the data
data_t /= torch.max(data_t)

dataset = TensorDataset(data_t, labels_t)

# some global variables 
D_in = data_t.shape[1]
D_out = torch.unique(labels_t).numel()
num_epochs = 10
n_hist_bins = 80

def get_data_labels_from_ldr(ldr):
  tensors = ldr.dataset.dataset.tensors
  indices = ldr.dataset.indices

  data = tensors[0][indices]
  labels = tensors[1][indices]
  return data, labels

def split_data(dset, train_prop=.9, batch_size=32):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_ldr = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_ldr = DataLoader(test_dset, batch_size=test_size)

  return train_ldr, test_ldr

train_loader, test_loader = split_data(dataset)

class MNIST_ANN(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.Sequential(
      nn.Linear(D_in, 64), # weight matrix btw input - hidden1
      nn.ReLU(),
      nn.Linear(64, 32, bias=True), # weight matrix btw hidden1 - hidden2
      # nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Linear(32, 32, bias=True), # weight matrix btw hidden2 - hidden3
      # nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Linear(32, D_out) # the loss function will be nn.CrossEntropyLoss() which has log-softmax internally.
    )

  def forward(self, x):
    return self.layers(x)
  
def create_model(lr=0.001, std=1):
  m = MNIST_ANN()
  l = nn.CrossEntropyLoss()
  o = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)

  # custom-init the weights according to the given std value.
  for name, p in m.named_parameters():
    if 'weight' in name:
      p.data = torch.randn(p.data.shape) * std

  return m, l, o

def train_model(train_ldr, test_ldr, std=1):
  model, loss_func, optimizer = create_model(std=std)

  test_x, test_y = next(iter(test_ldr))

  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)
  all_losses = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_acc = []
    all_batch_losses = []

    for batch_x, batch_y in train_ldr:
      # forward-pass and compute the loss
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)
      all_batch_losses.append( loss.item() )

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      # compute the batch_acc
      batch_acc = (torch.argmax(y_hat, dim=1) == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )

    ave_loss = np.mean( all_batch_losses )
    ave_train_acc = np.mean( all_batch_acc )
    all_losses[epoch_i] = ave_loss
    all_train_acc[epoch_i] = ave_train_acc

    print(f'[Epoch {epoch_i}] loss={ave_loss:.3f}, train_accuracy={ave_train_acc:.3f}')

    # compute the test_accuracy
    model.eval()
    with torch.no_grad():
      test_predictions = torch.argmax(model(test_x), dim=1)
    
    test_acc = (test_predictions == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc.item()

  return all_train_acc, all_test_acc, all_losses, model

def get_param_histo_from_model(model):
  all_params = np.array([])
  for name, p in model.named_parameters():
    if 'weight' in name:
      all_params = np.concatenate((all_params, p.data.detach().numpy().flatten()), axis=0)

  counts, edges = np.histogram(all_params, bins=n_hist_bins)
  center_vals = (edges[:-1] + edges[1:]) / 2

  return center_vals, counts

std_dev_list = np.logspace(np.log10(1e-4), np.log10(10), 18, base=10)
acc_results = np.zeros(len(std_dev_list)) # capturing the test accuracy for each std
histo_data = np.zeros((len(std_dev_list), 2, n_hist_bins))

train_loader, test_loader = split_data(dataset, batch_size=64)

for std_idx, std in enumerate(std_dev_list):
  time_start = time.perf_counter()
  train_acc, test_acc, losses, model = train_model(train_loader, test_loader, std=std)

  model_acc =  np.mean(test_acc[-3:])
  acc_results[std_idx] = model_acc
  histo_x, histo_counts = get_param_histo_from_model(model)
  histo_data[std_idx, 0, :] = histo_x
  histo_data[std_idx, 1, :] = histo_counts

  elapsed_time_sec = time.perf_counter() - time_start
  print(f'[{std_idx + 1}/{len(std_dev_list)} iteration] Elapsed: {elapsed_time_sec:.2f} sec, final accuracy: {model_acc:.2f}%')

# plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(std_dev_list, acc_results, 's-')
ax1.set_xlabel('Std for weight init')
ax1.set_xscale('log')
ax1.set_ylabel('Test accuracy (final-3, %)')
ax1.set_title('Test accuracy per std')

len_std_dev = len(std_dev_list)
for std_idx, std in enumerate(std_dev_list):
  hist_x = histo_data[std_idx, 0, :]
  hist_y = histo_data[std_idx, 1, :]
  color_f = 1 - (std_idx / len_std_dev)
  ax2.plot(hist_x, hist_y, label=f'{std:.6f}', color=[color_f, 0.4, color_f])
ax2.set_xlabel('Weight value')
ax2.set_ylabel('Count')
ax2.legend(bbox_to_anchor=(1, 1), loc='upper left')

plt.show()
