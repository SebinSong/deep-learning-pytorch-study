import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split

F = nn.functional

n_per_cluster = 300
blur = 1

A = [1, 1]
B = [5, 1]
C = [4, 3]

a = torch.stack([
  A[0] + torch.randn(n_per_cluster) * blur,
  A[1] + torch.randn(n_per_cluster) * blur,
]).float().T
b = torch.stack([
  B[0] + torch.randn(n_per_cluster) * blur,
  B[1] + torch.randn(n_per_cluster) * blur,
]).float().T
c = torch.stack([
  C[0] + torch.randn(n_per_cluster) * blur,
  C[1] + torch.randn(n_per_cluster) * blur,
]).float().T

data = torch.vstack((a, b, c))
labels = torch.hstack((
  torch.zeros(n_per_cluster),
  torch.ones(n_per_cluster),
  torch.ones(n_per_cluster) + 1
)).long()

dataset = TensorDataset(data, labels)

# experiment constants
batch_size = 32
num_epochs = 100

draw_data_plot = False
if draw_data_plot:
  plt.figure(figsize=(5, 5))
  plt.plot(
    data[torch.where(labels == 0)[0], 0],
    data[torch.where(labels == 0)[0], 1],
    'bs',
    alpha=.5,
    label='Cluster A'
  )
  plt.plot(
    data[labels == 1, 0],
    data[labels == 1, 1],
    'ko',
    alpha=.5,
    label='Cluster B'
  )
  plt.plot(
    c[:, 0],
    c[:, 1],
    'r^',
    alpha=.5,
    label='Cluster C'
  )
  plt.xlabel('Dim 1')
  plt.ylabel('Dim 2')
  plt.title('Qwerties')
  plt.show()

def split_data (dset, train_prop=.8, batch_size=32):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=test_size)

  return train_loader, test_loader

class QwertyClassifier(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(2, 16),
      'hidden0': nn.Linear(16, 8),
      'bnorm0': nn.BatchNorm1d(8),
      'output': nn.Linear(8, 3)
    })

  def forward(self, x):
    x = F.relu( self.layers['input'](x) )

    # Linear -> Batch Norm -> Activation
    x = self.layers['hidden0'](x)
    x = F.relu( self.layers['bnorm0'](x) )

    return self.layers['output'](x)

def create_model(learning_rate=0.01):
  model = QwertyClassifier()

  loss_func = nn.CrossEntropyLoss()

  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  train_size = int(n_per_cluster * 3 * 0.8) 
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_size, gamma=.5)

  return model, loss_func, optimizer, lr_scheduler

def train_model(train_ldr, test_ldr, enable_lr_decay=False):
  model, loss_func, optimizer, lr_scheduler = create_model()

  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)
  lr_track = []

  for epoch_i in range(num_epochs):
    all_batch_acc = []

    model.train()
    for batch_x, batch_y in train_ldr:
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if enable_lr_decay:
        lr_scheduler.step()
      
      lr_track.append( lr_scheduler.get_last_lr()[0])

      # batch accuracy
      batch_acc = (torch.argmax(y_hat, dim=1) == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )

    all_train_acc[epoch_i] = np.mean(all_batch_acc)

    # compute the test accuracy
    test_x, test_y = next(iter(test_ldr))

    model.eval()
    with torch.no_grad():
      test_pred_labels = torch.argmax(model(test_x), dim=1)
    
    test_acc = (test_pred_labels == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc.item()
  
  return all_train_acc, all_test_acc, lr_track

train_ldr, test_ldr = split_data(dataset, batch_size=batch_size)

train_acc, test_acc, lrs_true = train_model(train_ldr, test_ldr, True)
train_acc_2, test_acc_2, lrs_false = train_model(train_ldr, test_ldr, False)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(train_acc, 'b-', label='Train accuracy')
ax[0].plot(test_acc, 'y-', label='Test accuracy')
ax[0].set_title('With lr decay')

ax[1].plot(train_acc_2, 'b-', label='Train accuracy')
ax[1].plot(test_acc_2, 'y-', label='Test accuracy')
ax[1].set_title('Without lr decay')

for i in range(2):
  ax[i].set_xlabel('Epoch')
  ax[i].set_ylabel('Accuracy (%)')
  ax[i].legend()
  ax[i].grid()

plt.show()
