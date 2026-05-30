import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pathlib import Path

F = nn.functional

curr_dir = Path(__file__).parent
data_fpath = (curr_dir / '../data/winequality-red.csv').resolve()

# some helper functions
def draw_box_plot(df):
  # Draw box-plot for the entire dataframe
  plt.figure(figsize=(15, 7))
  sns.boxenplot(data=df)
  plt.xticks(rotation=20)
  plt.show()

def to_numpy(t):
  if t.requires_grad:
    return t.detach().cpu().numpy()
  return t.cpu().numpy()

def to_tensor_float32(arr):
  return torch.tensor(arr, dtype=torch.float32)

df = pd.read_csv(data_fpath, sep=';', header=0)

# drop the outliers
df = df[df['total sulfur dioxide'] < 200]

feature_cols = df.keys().drop('quality')

data = to_tensor_float32( df[feature_cols].values )
labels = to_tensor_float32( (df['quality'] > 5).values ).reshape(-1, 1)

# some global variables
D_in = len(feature_cols)
D_out = 1
num_epochs = 300

def split_data(train_prop=.8, batch_size=32):
  train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=(1 - train_prop))

  feature_scaler = StandardScaler()
  feature_scaler.fit(to_numpy(train_data))

  perform_zscore = lambda t: to_tensor_float32( feature_scaler.transform(to_numpy(t)) )

  # Normalize the train/test data first (z-scoring)
  train_data_norm = perform_zscore(train_data)
  test_data_norm = perform_zscore(test_data)

  train_dset = TensorDataset(train_data_norm, train_labels)
  test_dset = TensorDataset(test_data_norm, test_labels)

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=len(test_dset))
  
  return train_loader, test_loader

class ANN_WINE(nn.Module):
  def __init__(self, init_method = 'kaiming'):
    super().__init__()

    self.layers = nn.ModuleDict({
      'fc1': nn.Linear(D_in, 64), # mat btw input - hidden1
      'fc2': nn.Linear(64, 32, bias=False), # mat btw hidden1 - hidden2
      'bnorm2': nn.BatchNorm1d(32),
      'fc3': nn.Linear(32, D_out) # mat btw hidden2 - output
    })

    match init_method:
      case 'xavier':
        self.weight_initializer = lambda w: nn.init.xavier_normal_(w)
      case _:
        self.weight_initializer = lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu')
  
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      self.weight_initializer(module.weight)

  def forward(self, x):
    x = F.relu( self.layers['fc1'](x) )
    x = F.relu( self.layers['fc2'](x) )
    return self.layers['fc3'](x)

def draw_weight_histo(model):
  plt.figure(figsize=(10, 6))

  for name, p in model.named_parameters():
    if ('weight' in name) and ('bnorm' not in name):
      counts, edges = np.histogram(
        p.data.detach().numpy().flatten(), bins=10
      )

      plt.plot(
        (edges[:-1] + edges[1:]) / 2,
        counts / np.sum(counts),
        label=f'{name[:-7]}'
      )

  plt.title('Weights per layer')
  plt.xlabel('Weights')
  plt.ylabel('Density')
  plt.legend()
  plt.show()

def create_model(init_method = 'kaiming'):
  m = ANN_WINE(init_method)
  l = nn.BCEWithLogitsLoss()
  o = torch.optim.Adam(m.parameters(), lr=0.001)

  return m, l, o

def train_model(init_method='kaiming'):
  train_loader, test_loader = split_data()
  model, loss_func, optimizer = create_model(init_method)

  test_x, test_y = next(iter(test_loader))

  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)
  all_loesses = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_acc = []
    all_batch_losses = []

    for batch_x, batch_y in train_loader:
      # forward-pass and compute the loss
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)
      all_batch_losses.append( loss.item() )

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      batch_preds = (y_hat > 0).float()
      batch_acc = (batch_preds == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )

    ave_train_acc = np.mean( all_batch_acc )
    ave_loss = np.mean( all_batch_losses )
    all_train_acc[epoch_i] = ave_train_acc
    all_loesses[epoch_i] = ave_loss

    if epoch_i % 20 == 0:
      print(f'[Epoch {epoch_i}] train_acc={ave_train_acc:.3f}, loss={ave_loss:.3f}')

    # compute test_acc
    model.eval()
    with torch.no_grad():
      test_preds = (model(test_x) > 0).float()
    
    test_acc = (test_preds == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc.item()

  return all_train_acc, all_test_acc, all_loesses

num_runs = 20
kaiming_results = np.zeros((num_runs, 3))
xavier_results = np.zeros((num_runs, 3))

def run_and_record(run_idx = 0, method='kaiming'):
  res_arr = kaiming_results if method == 'kaiming' else xavier_results
  train_acc, test_acc, losses = train_model(init_method=method)

  res_arr[run_idx, 0] = np.mean(train_acc[-5:])
  res_arr[run_idx, 1] = np.mean(test_acc[-5:])
  res_arr[run_idx, 2] = np.mean(losses[-5:])

for run_idx in range(num_runs):
  run_and_record(run_idx, 'kaiming')
  run_and_record(run_idx, 'xavier')

plot_titles = ['Train acc', 'Test acc', 'Loss']
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

for i in range(3):
  kaiming_res_arr = kaiming_results[:, i]
  xavier_res_arr = xavier_results[:, i]
  t, p = stats.ttest_ind(kaiming_res_arr, xavier_res_arr)

  ax[i].plot(np.zeros(num_runs), kaiming_res_arr, 'bo')
  ax[i].plot(np.ones(num_runs), xavier_res_arr,  'ro')
  ax[i].set_xticks([0, 1], ['Kaiming', 'Xavier'])
  ax[i].set_xlim([-2, 2])
  ax[i].set_title(f'{plot_titles[i]} t={t:.3f}, p={p:.3f}')

plt.tight_layout()
plt.show()
