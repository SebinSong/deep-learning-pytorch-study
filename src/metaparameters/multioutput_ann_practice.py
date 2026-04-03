import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt

F = nn.functional

n_per_cluster = 300
blur = 1

A = [ 1, 1 ]
B = [ 5, 1 ]
C = [ 4, 4 ]

# generate data
a = np.array([
  A[0] + np.random.randn(n_per_cluster) * blur,
  A[1] + np.random.randn(n_per_cluster) * blur,
])
b = np.array([
  B[0] + np.random.randn(n_per_cluster) * blur,
  B[1] + np.random.randn(n_per_cluster) * blur,
])
c = np.array([
  C[0] + np.random.randn(n_per_cluster) * blur,
  C[1] + np.random.randn(n_per_cluster) * blur,
])

data_np = np.hstack((a, b, c)).T

labels_np = np.hstack([
  np.zeros(n_per_cluster), np.ones(n_per_cluster), 1 + np.ones(n_per_cluster)
])

data = torch.tensor(data_np, dtype=torch.float32)
labels = torch.tensor(labels_np, dtype=torch.long)
dataset = TensorDataset(data, labels)

plot_data = False
if plot_data:
  fig = plt.figure(figsize=(5, 5))
  plt.plot(a[0, :], a[1, :], 'bs', label='Cluster A(0)', alpha=.5)
  plt.plot(b[0, :], b[1, :], 'ko', label='Cluster B(1)', alpha=.5)
  plt.plot(c[0, :], c[1, :], 'r^', label='Cluster C(2)', alpha=.5)
  plt.title('Qwerties')
  plt.xlabel('Dimension 1')
  plt.ylabel('Dimension 2')
  plt.show()

def split_data(dset, train_prop=.8, batch_size=16):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=test_size)

  return train_loader, test_loader

train_loader, test_loader = split_data(dataset)

class QwertyNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.Sequential(
      nn.Linear(2, 16),
      nn.ReLU(),
      nn.Linear(16, 8),
      nn.ReLU(),
      nn.Linear(8, 3)
    )

  def forward (self, x):
    return self.layers(x)

def train_model(train_ldr, test_ldr, learning_rate=0.01, num_epochs=800):
  model = QwertyNet()
  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  all_losses = np.zeros(num_epochs)
  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    all_batch_acc = []
    all_batch_losses = []
    
    model.train()
    for batch_x, batch_y in train_ldr:
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      all_batch_losses.append( loss.item() )

      # batch accuracy
      pred_labels = torch.argmax(y_hat, dim=1)
      batch_acc = 100 * (pred_labels == batch_y).float().mean()
      all_batch_acc.append( batch_acc.item() )
  
    all_train_acc[epoch_i] = np.mean(all_batch_acc)
    all_losses[epoch_i] = np.mean(all_batch_losses)
  
    # compute test accuracy
    test_x, test_y = next(iter(test_ldr))
    model.eval()
    with torch.no_grad():
      test_pred_labels = torch.argmax(model(test_x), dim=1)
    
    test_acc = 100 * (test_pred_labels == test_y).float().mean()
    all_test_acc[epoch_i] = test_acc.item()

  return all_train_acc, all_test_acc, all_losses, model

train_loader, test_loader = split_data(dataset)
train_acc, test_acc, all_losses, model = train_model(train_loader, test_loader)

standard_way = False

if standard_way:
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

  ax1.plot(all_losses)
  ax1.set_ylabel('Loss')
  ax1.set_xlabel('Epoch')
  ax1.set_title('Losses')

  ax2.plot(train_acc, label='Train')
  ax2.plot(test_acc, label='Test')
  ax2.set_ylabel('Accuracy (%)')
  ax2.set_xlabel('Epoch')
  ax2.set_title('Accuracy')
  ax2.legend()

  plt.show()

final_y_hat = model(data)
final_preds = torch.argmax(final_y_hat, dim=1)

# plt.plot(final_preds, 'o', label='Predicted values')
# plt.plot(labels + .2, 's', label='True values')
# plt.xlabel('Qwerty number')
# plt.ylabel('Category')
# plt.yticks([0, 1, 2])
# plt.ylim([-1, 3])
# plt.legend()
# plt.show()

label_comparisons = final_preds == labels
total_acc = torch.mean( label_comparisons.float() * 100 ).item()
acc_by_group = np.zeros(3)
for i in range(3):
  acc_by_group[i] = (label_comparisons[labels == i].float().mean() * 100).item()

print(acc_by_group)

plt.bar(range(3), acc_by_group)
plt.ylim([50, 100])
plt.xticks([0, 1, 2])
plt.xlabel('Group')
plt.ylabel('Accuracy (%)')
plt.title(f'Total acc = {total_acc:.3f}%')
plt.show()
