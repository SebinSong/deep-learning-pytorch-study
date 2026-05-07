import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split

F = nn.functional

curr_dir = Path(__file__).parent
data_fpath = (curr_dir / '../data/mnist_train_small.csv').resolve()

# helpers
def to_ndarray(t):
  if t.requires_grad:
    return t.detach().cpu().numpy()
  return t.cpu().numpy()

def get_data_and_labels_from_dlr(dlr):
  dlr_tensors = dlr.dataset.dataset.tensors
  dlr_indices = dlr.dataset.indices
  dlr_data = dlr_tensors[0][dlr_indices]
  dlr_labels = dlr_tensors[1][dlr_indices]

  return dlr_data, dlr_labels

df = pd.read_csv(data_fpath, sep=',', header=None)
data = torch.tensor( df.values[:, 1:], dtype=torch.float32 )
labels = torch.tensor( df.values[:, 0], dtype=torch.long )

# some global variables
D_in = 28 * 28
D_out = 10
num_epochs = 10

def prepare_data_loaders(train_prop=.9, batch_size=64):
  # Step 1. - take out some 7s from the sample
  indices_7s = np.where(labels == 7)[0]

  # going to toss out 70% of the 7s
  exclude_count = int(len(indices_7s) * 0.9)
  indices_to_exclude = np.random.permutation(indices_7s)[:exclude_count]

  # create indices mask to toss out those 7s
  indices_mask = torch.ones(len(labels), dtype=bool)
  indices_mask[indices_to_exclude] = False

  data_filtered = data[indices_mask]
  labels_filtered = labels[indices_mask]

  # Step 2. normalize the data / split them / create train and test data loaders
  data_filtered /= 255
  dset = TensorDataset(data_filtered, labels_filtered)

  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size
  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_ldr = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_ldr = DataLoader(test_dset, batch_size=test_size)

  return train_ldr, test_ldr

train_loader, test_loader = prepare_data_loaders()
bx, by = next(iter(train_loader))

# define the model
class MNIST_ANN(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(D_in, 64),
      'fc0': nn.Linear(64, 32),
      'fc1': nn.Linear(32, 32),
      'output': nn.Linear(32, D_out)
    })

  def forward(self, x):
    x = F.relu( self.layers['input'](x) )
    x = F.relu( self.layers['fc0'](x) )
    x = F.relu( self.layers['fc1'](x) )
    return self.layers['output'](x)

def create_model(lr=0.01):
  m = MNIST_ANN()
  l = nn.CrossEntropyLoss()
  o = torch.optim.Adam(m.parameters(), lr=lr)
  return m, l, o

def train_model(train_ldr, test_ldr):
  model, loss_func, optimizer = create_model()

  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)
  all_losses = np.zeros(num_epochs)

  test_x, test_y = next(iter(test_ldr))

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

      # compute batch_acc
      batch_acc = (torch.argmax(y_hat, dim=1) == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )

    acc_ave = np.mean( all_batch_acc )
    loss_ave = np.mean( all_batch_losses )

    all_losses[epoch_i] = loss_ave
    all_train_acc[epoch_i] = acc_ave

    if epoch_i % 3 == 0 or epoch_i == (num_epochs - 1):
      print(f'[Epoch {epoch_i}] - train_acc={acc_ave:.3f}, loss={loss_ave:.3f}')

    # compute test accuracy
    model.eval()
    with torch.no_grad():
      test_pred_labels = torch.argmax(model(test_x), dim=1)
    
    test_acc = (test_pred_labels == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc.item()
  
  return all_train_acc, all_test_acc, all_losses, model

train_loader, test_loader = prepare_data_loaders()
full_model = train_model(train_loader, test_loader)[3]

train_data, train_labels = get_data_and_labels_from_dlr(train_loader)
test_data, test_labels = get_data_and_labels_from_dlr(test_loader)

train_predictions = torch.argmax(full_model(train_data), dim=1)
test_predictions = torch.argmax(full_model(test_data), dim=1)

# 0: accuracy, 1: precision, 2: recall
train_metrics = np.zeros((D_out, 3))
test_metrics = np.zeros((D_out, 3))

train_metrics[:, 0] = skm.accuracy_score(train_labels, train_predictions)
train_metrics[:, 1] = skm.precision_score(train_labels, train_predictions, average=None)
train_metrics[:, 2] = skm.recall_score(train_labels, train_predictions, average=None)

test_metrics[:, 0] = skm.accuracy_score(test_labels, test_predictions)
test_metrics[:, 1] = skm.precision_score(test_labels, test_predictions, average=None)
test_metrics[:, 2] = skm.recall_score(test_labels, test_predictions, average=None)

fig, ax = plt.subplots(3, 1, figsize=(10, 7))

titles = ['Accuracy', 'Precision', 'Recall']

for i in range(3):
  ax[i].bar(np.arange(10) - .15, train_metrics[:, i], .5)
  ax[i].bar(np.arange(10) + .15, test_metrics[:, i], .5)
  ax[i].set_xticks(range(10), np.arange(10))
  ax[i].set_xlabel('Number')
  ax[i].set_ylim([.5, 1])
  ax[i].legend(['Train', 'Test'])
  ax[i].set_title(titles[i])

plt.tight_layout()
plt.show()
