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
num_epochs = 1000

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

def split_data(dset, train_prop=.8, batch_size=16):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dset, batch_size=test_size, shuffle=False)

  return train_loader, test_loader

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

def train_model(batch_size=16):
  train_ldr, test_ldr = split_data(dataset, batch_size=batch_size)
  model, loss_func, optimzer = create_model()

  for epoch_i in range(num_epochs):
    pass # TODO!