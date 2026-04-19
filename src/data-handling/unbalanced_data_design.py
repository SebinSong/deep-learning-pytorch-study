import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

F = nn.functional

# helper funcs
n_uniq = lambda l: len(np.unique(l))
n_uniq_t = lambda t: torch.unique(t).numel()
z_score_df = lambda df: df.apply(stats.zscore)

# global variables that are fixed
num_epochs = 500
learning_rate = 0.001 # initial lr
train_prop =.8
batch_size = 64

wine_quality = fetch_ucirepo(id=186)
wine_features = wine_quality.data.features

filtered_idx_masks = wine_features['total_sulfur_dioxide'] < 200

wine_features = wine_features[filtered_idx_masks] # Filter out the outliers
wine_features_norm = z_score_df(wine_features) # normalize the data
data_t = torch.tensor(wine_features_norm.values, dtype=torch.float32)

wine_targets = wine_quality.data.targets[filtered_idx_masks]
bool_targets = wine_targets['quality'] > 5.5

D_in = len(wine_features_norm.keys())
D_out = 1

def draw_box_plot(df):
  plt.figure(figsize=(15, 8))
  sns.boxplot(data=df)
  plt.xticks(rotation=45)
  plt.show()

def generate_labels(threshold = 5.5):
  labels_t = torch.tensor(
    (wine_targets['quality'] > threshold).values,
    dtype=torch.float32
  ).reshape(-1, 1)

  return labels_t

def split_data(threshold = 5.5):
  global batch_size

  labels_t = generate_labels(threshold)
  dataset = TensorDataset(data_t, labels_t)

  sample_size = len(dataset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dataset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=test_size)

  return train_loader, test_loader

class ANN_WINE(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.Sequential(
      nn.Linear(D_in, 32),
      nn.LeakyReLU(),
      nn.Linear(32, 16),
      nn.LeakyReLU(),
      nn.Linear(16, D_out)
    )

  def forward(self, x):
    return self.layers(x)

def create_model():
  global learning_rate

  model = ANN_WINE()
  lf = nn.BCEWithLogitsLoss()
  opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

  return model, lf, opt

def train_model(threshold = 5.5):
  global num_epochs

  logits_to_labels = lambda t: (t > 0).float()

  train_loader, test_loader = split_data(threshold)
  model, loss_func, optimizer = create_model()

  test_x, test_y = next(iter(test_loader))
  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)
  all_losses = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_acc = []
    all_batch_losses = []
    for batch_x, batch_y in train_loader:
      y_hat = model(batch_x)

      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      all_batch_losses.append( loss.item() )
      
      batch_acc = (logits_to_labels(y_hat) == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )

    acc_ave = np.mean(all_batch_acc)
    loss_ave = np.mean(all_batch_losses)
    all_train_acc[epoch_i] = acc_ave
    all_losses[epoch_i] = loss_ave

    if epoch_i % 5 == 0:
      print(f'[Threshold={threshold} Epoch {epoch_i}] - train_acc={acc_ave:.3f}, loss={loss_ave:.3f}')

    # compute test_acc
    model.eval()
    with torch.no_grad():
      test_pred_labels = logits_to_labels(model(test_x))
    
    test_acc = (test_pred_labels == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc.item()

    idxs_for_0 = torch.where(test_y == 0)[0]
    idxs_for_1 = torch.where(test_y == 1)[0]
    test_acc_0 = (test_pred_labels[idxs_for_0] == 0).float().mean() * 100
    test_acc_1 = (test_pred_labels[idxs_for_1] == 1).float().mean() * 100
    
    acc_category = {
      'bad': test_acc_0,
      'good': test_acc_1
    }

  return (all_train_acc, all_test_acc, all_losses, acc_category)

thresholds = [4.5, 5.5, 6.5]
results = np.zeros((
  num_epochs,
  len(thresholds),
  3 # 0 - train_acc, 1 - test_acc , 2 - loss
))

cate_accs = np.zeros((
  len(thresholds), 2
), dtype=np.int64)

for thres_i, threshold in enumerate(thresholds):
  train_acc, test_acc, losses, acc_category = train_model(threshold)
  results[:, thres_i, 0] = train_acc
  results[:, thres_i, 1] = test_acc
  results[:, thres_i, 2] = losses

  cate_accs[thres_i, 0] = acc_category['bad']
  cate_accs[thres_i, 1] = acc_category['good']

fig, ax = plt.subplots(3, 3, figsize=(11, 11))

for thres_i, threshold in enumerate(thresholds):
  train_acc = results[:, thres_i, 0]
  test_acc = results[:, thres_i, 1]
  losses = results[:, thres_i, 2]

  ax[thres_i, 0].plot(losses)
  ax[thres_i, 0].set_xlabel('Epoch')
  ax[thres_i, 0].set_ylabel('Loss')
  ax[thres_i, 0].set_title(f'Losses (thres={threshold})')

  ax[thres_i, 1].plot(train_acc, label='Train')
  ax[thres_i, 1].plot(test_acc, label='Test')
  ax[thres_i, 1].set_xlabel('Epoch')
  ax[thres_i, 1].set_ylabel('Accuracy (%)')
  ax[thres_i, 1].set_title(f'Accuracies (thres={threshold})')
  ax[thres_i, 1].legend()

  ax[thres_i, 2].bar(['Bad', 'Good'], cate_accs[thres_i])
  ax[thres_i, 2].set_xlabel('Wine quality')
  ax[thres_i, 2].set_ylabel('Test accuracy (%)')
  ax[thres_i, 2].set_title(f'Accuracy per quality (thres={threshold}')

plt.tight_layout()
plt.show()
