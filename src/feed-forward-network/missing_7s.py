import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path

F = nn.functional

curr_dir = Path(__file__).parent
data_fpath = (curr_dir / '../data/mnist_train_small.csv').resolve()

df = pd.read_csv(data_fpath, sep=',', header=None)
data = torch.tensor( df.values[:, 1:], dtype=torch.float32 )
labels = torch.tensor( df.values[:, 0], dtype=torch.long )

# Min-max normalization
data /= 255

train_idxs = torch.where(labels != 7)[0]
train_dset = TensorDataset(data[train_idxs], labels[train_idxs])
train_loader = DataLoader(train_dset, batch_size=64, shuffle=True, drop_last=True)
test_data = data[torch.where(labels == 7)[0]]

class ANN_MNIST(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(28 * 28, 128),
      'fc0': nn.Linear(128, 64),
      'fc1': nn.Linear(64, 32),
      'output': nn.Linear(32, 10)
    })

  def forward(self, x):
    x = F.relu( self.layers['input'](x) )

    x = F.relu( self.layers['fc0'](x) )
    x = F.relu( self.layers['fc1'](x) )

    return self.layers['output'](x)

def create_model(learning_rate=0.01):
  model = ANN_MNIST()
  loss_func = nn.CrossEntropyLoss()
  optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

  return model, loss_func, optim

def train_model(train_ldr, num_epochs=60):
  model, loss_func, optimizer = create_model()

  all_train_acc = np.zeros(num_epochs)
  all_losses = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
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

    train_acc_mean = np.mean( all_batch_acc )
    loss_mean = np.mean( all_batch_losses )

    if epoch_i % 5 == 0:
      print(f'[Epoch: {epoch_i}] - {train_acc_mean=:.3f}, {loss_mean=:.3f}')

    all_train_acc[epoch_i] = train_acc_mean
    all_losses[epoch_i] = loss_mean

  return all_train_acc, all_losses, model

train_acc, losses, model = train_model(train_loader)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# ax1.plot(losses)
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Loss')
# ax1.set_title('Losses')

# ax2.plot(train_acc, 'o-')
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Accuracy(%)')
# ax2.set_title('Train accuracy')

# plt.show()

with torch.no_grad():
  model.eval()
  test_predictions = torch.argmax(model(test_data), dim=1)

# random_idxs = np.random.permutation(len(test_data))[:12]

# fig, ax = plt.subplots(4, 3, figsize=(20, 15))

# flattened_axs = ax.flatten()

# for i in range(12):
#   ax = flattened_axs[i]
#   sample_idx = random_idxs[i]
#   pred_label = test_predictions[sample_idx]
#   img = test_data[sample_idx].reshape(28, -1)

#   ax.imshow(img, cmap='gray', vmax=1)
#   ax.set_title(f'prediction: {pred_label}')

# plt.tight_layout()
# plt.show()

pred_labels_unique = np.unique(test_predictions)
pred_labels_unique.sort()
pred_labels_unique_prop = np.zeros(len(pred_labels_unique))

test_data_count = len(test_data)
for i, label in enumerate(pred_labels_unique):
  pred_labels_unique_prop[i] = (test_predictions == label).float().mean()

plt.bar(pred_labels_unique, pred_labels_unique_prop)
plt.xticks(pred_labels_unique)
plt.xlabel('Number')
plt.ylabel('Proportion')
plt.show()
