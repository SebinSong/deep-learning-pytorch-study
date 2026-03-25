import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import scipy.stats as stats
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

F = nn.functional

# helper func

# 1 - checking the number of unique values in a ndarray etc.
n_uniq = lambda l: len(np.unique(l))

# 2 - z-scoring the dataframe
z_score_df = lambda df: df.apply(stats.zscore)

# 3 - draw boxplot, violin plot
def draw_box_plot(df):
  plt.figure(figsize=(15, 8))
  sns.boxplot(data=df)
  plt.xticks(rotation=45)
  plt.show()

def draw_violin_plot(df):
  plt.figure(figsize=(15, 8))
  sns.violinplot(data=df)
  plt.xticks(rotation=45)
  plt.show()

# hyper-parameters
num_epochs = 700

# prepare torch dataset
wine_quality = fetch_ucirepo(id=186)
wine_features = wine_quality.data.features
wine_features_transformed = z_score_df(wine_features)

wine_target = wine_quality.data.targets
wine_target_transformed = wine_target > wine_target['quality'].mean()

D_in = len(wine_features_transformed.keys())
D_out = 1 # Since it is binary classification problem

dataset = TensorDataset(
  torch.tensor(wine_features_transformed.values).float(),
  torch.tensor(wine_target_transformed.values).float()
)

def split_data(dset, train_prop=.8):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])
  return train_dset, test_dset

# model architecture
class QualityClassifier(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(D_in, 16),
      'hidden0': nn.Linear(16, 32),
      'hidden1': nn.Linear(32, 32),
      'output': nn.Linear(32, D_out)
    })

  def forward(self, x):
    x = F.relu( self.layers['input'](x) )
    x = F.relu( self.layers['hidden0'](x) )
    x = F.relu( self.layers['hidden1'](x) )

    return self.layers['output'](x)

def create_model(learning_rate=.01):
  model = QualityClassifier()
  lf = nn.BCEWithLogitsLoss()
  opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

  return model, lf, opt

def train_model(train_ldr, test_ldr):
  model, loss_func, optimizer = create_model()

  train_acc = np.zeros(num_epochs)
  test_acc = np.zeros(num_epochs)
  for epoch_i in range(num_epochs):
    batch_acc = []

    model.train()
    for batch_x, batch_y in train_ldr:
      y_hat = model(batch_x)

      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # compute the current batch acc
      pred_labels = y_hat > 0
      acc = (pred_labels == batch_y).float().mean() * 100
      batch_acc.append(acc.item())

    train_acc[epoch_i] = np.mean(batch_acc)

    # compute the train-accuracy
    model.eval()
    test_x, test_y = next(iter(test_ldr))
    with torch.no_grad():
      test_pred_labels = model(test_x) > 0
      curr_test_acc = (test_pred_labels == test_y).float().mean() * 100
      test_acc[epoch_i] = curr_test_acc.item()

  return train_acc, test_acc

batch_sizes = np.array([2**n for n in range(1, 9)])
all_train_acc = np.zeros((num_epochs, len(batch_sizes)))
all_test_acc = np.zeros((num_epochs, len(batch_sizes)))
total_times = np.zeros(len(batch_sizes))

train_dset, test_dset = split_data(dataset)
test_ldr = DataLoader(test_dset, batch_size=len(test_dset), shuffle=False)

for bi, curr_batch_size in enumerate(batch_sizes):
  train_ldr = DataLoader(train_dset, batch_size=curr_batch_size, shuffle=True, drop_last=True)
  start_time = time.perf_counter()
  train_acc, test_acc = train_model(train_ldr, test_ldr, batch_size=curr_batch_size)
  end_time = time.perf_counter()

  all_train_acc[:, bi] = train_acc
  all_test_acc[:, bi] = test_acc
  total_times[bi] = end_time - start_time

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
ax[0].plot(all_train_acc)
ax[0].set_title('Train accuracy')
ax[1].plot(all_test_acc)
ax[1].set_title('Test accuracy')

for i in range(2):
  ax[i].set_xlabel('Epoch')
  ax[i].set_ylabel('Accuracy (%)')
  ax[i].legend(list(batch_sizes))
  ax[i].set_ylim([50, 100])
  ax[i].grid()

ax[2].plot(batch_sizes, total_times)
plt.show()

