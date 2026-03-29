import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# helpers
def draw_box_plot(df):
  plt.figure(figsize=(10, 5))
  sns.boxplot(data=df)
  plt.xticks(rotation=20)
  plt.show()

F = nn.functional

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

if 'a' == 'b':
  # Draw the box plot to check the feature distributions.
  draw_box_plot(data)

def refine_and_split_data(train_prop=.8, batch_size=32):
  data_df = data[data['total sulfur dioxide'] < 200].copy() # drop the outliers

  # Normalize the input data - 'quality' column will be used for labels
  cols2zscore = data_df.keys().drop('quality')
  data_df.loc[:, cols2zscore] = data_df[cols2zscore].apply(stats.zscore)

  data_t = torch.tensor(data_df[cols2zscore].values, dtype=torch.float32)
  labels_t = torch.tensor(
    (data_df['quality'] > 5).values,
    dtype=torch.float32
  )
  labels_t = labels_t.view(-1, 1) # tranform to matrix

  # spllt the data into dataloaders
  dset = TensorDataset(data_t, labels_t)
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_dlr = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_dlr = DataLoader(test_dset, batch_size=test_size)

  D_in = len(cols2zscore)
  return train_dlr, test_dlr, D_in

class ANNwine_withBatchNorm(nn.Module):
  def __init__(self, D_in=11, D_out=1):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(D_in, 16),
      'hidden1': nn.Linear(16, 32),
      'hidden2': nn.Linear(32, 20),
      'output': nn.Linear(20, D_out),
      'bnorm1': nn.BatchNorm1d(16),
      'bnorm2': nn.BatchNorm1d(32),
    })

  def forward(self, x, doBN = False):
    x = F.relu(self.layers['input'](x))
    
    # hidden layer 1
    if doBN:
      x = self.layers['bnorm1'](x)
    x = F.relu( self.layers['hidden1'](x) )

    # hidden 2
    if doBN:
      x = self.layers['bnorm2'](x)
    x = F.relu( self.layers['hidden2'](x) )

    return self.layers['output'](x)

def train_model (learning_rate=0.005, num_epochs=1000, doBN=False):
  train_dlr, test_dlr, D_inputs = refine_and_split_data()
  model = ANNwine_withBatchNorm(D_inputs, 1)
  loss_func = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  all_losses = np.zeros(num_epochs)
  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    all_batch_acc = []
    all_batch_losses = []

    model.train()
    for batch_x, batch_y in train_dlr:
      y_hat = model(batch_x, doBN)

      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      all_batch_losses.append( loss.item() )

      # compute batch_acc
      batch_pred_labels = (y_hat > 0).float()
      batch_acc = 100 * (batch_pred_labels == batch_y).float().mean()
      all_batch_acc.append( batch_acc.item() )

    all_train_acc[epoch_i] = np.mean(all_batch_acc)
    all_losses[epoch_i] = np.mean(all_batch_losses)

    # compute the test accuracy
    model.eval()
    test_x, test_y = next(iter(test_dlr))
    with torch.no_grad():
      test_pred_labels = (model(test_x, doBN) > 0).float()
    
    all_test_acc[epoch_i] = 100 * (test_pred_labels == test_y).float().mean().item()

  return all_train_acc, all_test_acc, all_losses

train_acc, test_acc, losses = train_model(num_epochs=600, doBN=True)
train_acc_no, test_acc_no, losses_no = train_model(num_epochs=600, doBN=False)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

ax1.plot(losses, 'b-', label='With BatchNorm')
ax1.plot(losses_no, 'y-', label='No BatchNorm')
ax1.set_title('Losses')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(train_acc, 'b-', label='With BatchNorm')
ax2.plot(train_acc_no, 'y-', label='No BatchNorm')
ax2.set_title('Train_accuracy')
ax2.set_ylabel('Accuracy (%)')
ax2.set_xlabel('Epoch')
ax2.legend()

ax3.plot(test_acc, 'b-', label='With BatchNorm')
ax3.plot(test_acc_no, 'y-', label='No BatchNorm')
ax3.set_title('Test_accuracy')
ax3.set_ylabel('Accuracy (%)')
ax3.set_xlabel('Epoch')
ax3.legend()

plt.show()
