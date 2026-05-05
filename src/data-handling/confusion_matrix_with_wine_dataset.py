import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as skm

from pathlib import Path

curr_dir = Path(__file__).parent
resolve_path = lambda rel_path: (curr_dir / rel_path).resolve()
data_fpath = resolve_path('../data/winequality-red.csv')

F = nn.functional

# Util functions
def draw_box_plot(df):
  plt.figure(figsize=(15, 8))
  sns.boxplot(data=df)
  plt.xticks(rotation=20)
  plt.show()

def draw_violin_plot(df):
  plt.figure(figsize=(15, 8))
  sns.violinplot(data=df)
  plt.xticks(rotation=20)
  plt.show()

def get_data_and_labels_from_dlr(dlr):
  dlr_tensors = dlr.dataset.dataset.tensors
  dlr_indices = dlr.dataset.indices
  data = dlr_tensors[0][dlr_indices]
  labels = dlr_tensors[1][dlr_indices]

  return data, labels

z_score_df = lambda df: df.apply(stats.zscore)

df = pd.read_csv(data_fpath, sep=';', header=0)

# filter and normalize the data
df = df[df['total sulfur dioxide'] < 200] # 1. remove the outliers
data_df = z_score_df( df[df.columns.drop('quality')] )
labels_df = (df['quality'] > df['quality'].mean()).astype(np.int64)

data = torch.tensor(data_df.values, dtype=torch.float32)
labels = torch.tensor(labels_df, dtype=torch.float32).reshape(-1, 1)
dataset = TensorDataset(data, labels)

# some global variables
D_in = len(data_df.columns)
D_out = 1
num_epochs = 100

def split_data(dset, train_prop=.9, batch_size=16):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_ldr = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_ldr = DataLoader(test_dset, batch_size=test_size)

  return train_ldr, test_ldr

# define the model
class WINE_ANN(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(D_in, 32),
      'fc0': nn.Linear(32, 16),
      'bnorm0': nn.BatchNorm1d(16),
      'fc1': nn.Linear(16, 8),
      'bnorm1': nn.BatchNorm1d(8),
      'output': nn.Linear(8, D_out)
    })

  def forward(self, x):
    x = F.relu( self.layers['input'](x) )

    # Linear -> Batch Norm -> Activation
    x = self.layers['fc0'](x)
    x = F.relu( self.layers['bnorm0'](x) )

    x = self.layers['fc1'](x)
    x = F.relu( self.layers['bnorm1'](x) )

    return self.layers['output'](x)

def create_model(lr=.001):
  m = WINE_ANN()
  l = nn.BCEWithLogitsLoss()
  o = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)

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
      # forward pass
      y_hat = model(batch_x)

      loss = loss_func(y_hat, batch_y)

      # back-prop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      all_batch_losses.append( loss.item() )

      # compute batch accuracy
      batch_pred_labels = (y_hat > 0).float()
      batch_acc = (batch_pred_labels == batch_y).flatten().float().mean() * 100
      all_batch_acc.append( batch_acc.item() )

    ave_loss = np.mean( all_batch_losses )
    ave_train_acc = np.mean( all_batch_acc )
    all_losses[epoch_i] = ave_loss
    all_train_acc[epoch_i] = ave_train_acc

    if epoch_i % 50 == 0:
      print(f'[Epoch:{epoch_i}] loss={ave_loss:.3f}, train_accuracy={ave_train_acc:.3f}')

    # compute the test accuracy of this particular epoch
    model.train()
    with torch.no_grad():
      test_pred_labels = (model(test_x) > 0).float()
    
    test_acc = (test_pred_labels == test_y).float().flatten().mean() * 100
    all_test_acc[epoch_i] = test_acc.item()

  return all_train_acc, all_test_acc, all_losses, model

train_loader, test_loader = split_data(dataset)
train_acc, test_acc, losses, full_trained_model = train_model(train_loader, test_loader)

train_data, train_labels = get_data_and_labels_from_dlr(train_loader)
test_data, test_labels = get_data_and_labels_from_dlr(test_loader)

train_preds = (full_trained_model(train_data) > 0)
test_preds = (full_trained_model(test_data) > 0)

# initialize vectors
train_metrics = [0, 0, 0, 0]
test_metrics = [0, 0, 0, 0]

def compute_metrics(targets, predictions, metrics_list):
  metrics_list[0] = skm.accuracy_score(targets, predictions)
  metrics_list[1] = skm.precision_score(targets, predictions)
  metrics_list[2] = skm.recall_score(targets, predictions)
  metrics_list[3] = skm.f1_score(targets, predictions)

compute_metrics(train_labels, train_preds, train_metrics)
compute_metrics(test_labels, test_preds, test_metrics)

# plt.bar(np.arange(4) - .1, train_metrics, .5)
# plt.bar(np.arange(4) + .1, test_metrics, .5)
# plt.xticks([0, 1, 2, 3], ['Accuracy', 'Precision', 'Recall', 'F1-score'])
# plt.ylim([.5, 1])
# plt.legend(['Train', 'Test'])
# plt.title('Performance metrics')
# plt.show()

train_conf = skm.confusion_matrix(train_labels, train_preds)
test_conf = skm.confusion_matrix(test_labels, test_preds)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.imshow(train_conf, 'Blues', vmax=len(train_preds)/2)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Bad', 'Good'])
ax1.set_yticklabels(['Bad', 'Good'])
ax1.set_xlabel('Predicted quality')
ax1.set_ylabel('True quality')
ax1.set_title('Train confusion matrix')

ax1.text(0, 0, f'True negatives:\n{train_conf[0, 0]}', ha='center', va='center')
ax1.text(0, 1, f'False negatives:\n{train_conf[1, 0]}', ha='center', va='center')
ax1.text(1, 0, f'False positives:\n{train_conf[0, 1]}', ha='center', va='center')
ax1.text(1, 1, f'True positives:\n{train_conf[1, 1]}', ha='center', va='center')

ax2.imshow(test_conf, 'Blues', vmax=len(test_preds)/2)
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Bad', 'Good'])
ax2.set_yticklabels(['Bad', 'Good'])
ax2.set_xlabel('Predicted quality')
ax2.set_ylabel('True quality')
ax2.set_title('Test confusion matrix')

ax2.text(0, 0, f'True negatives:\n{test_conf[0, 0]}', ha='center', va='center')
ax2.text(0, 1, f'False negatives:\n{test_conf[1, 0]}', ha='center', va='center')
ax2.text(1, 0, f'False positives:\n{test_conf[0, 1]}', ha='center', va='center')
ax2.text(1, 1, f'True positives:\n{test_conf[1, 1]}', ha='center', va='center')

plt.tight_layout()
plt.show()
