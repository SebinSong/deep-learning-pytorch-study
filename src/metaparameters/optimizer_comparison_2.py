import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt

F = nn.functional

n_per_cluster = 300
blur = 1

A = [1, 1]
B = [5, 1]
C = [4, 3]

a = torch.stack([
  A[0] + torch.randn(n_per_cluster) * blur,
  A[1] + torch.randn(n_per_cluster) * blur
]).float().T
b = torch.stack([
  B[0] + torch.randn(n_per_cluster) * blur,
  B[1] + torch.randn(n_per_cluster) * blur
]).float().T
c = torch.stack([
  C[0] + torch.randn(n_per_cluster) * blur,
  C[1] + torch.randn(n_per_cluster) * blur
]).float().T
data = torch.vstack((a, b, c))

labels = torch.hstack((
  torch.zeros(n_per_cluster),
  torch.ones(n_per_cluster),
  torch.ones(n_per_cluster) + 1,
)).long()

dataset = TensorDataset(data, labels)

def split_data (dset, train_prop=.8, batch_size=32):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_ldr = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_ldr = DataLoader(test_dset, batch_size=test_size)

  return train_ldr, test_ldr

class QwertyClassifier(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.Sequential(
      nn.Linear(2, 16), # 2 -> two dimensions
      nn.ReLU(),
      nn.Linear(16, 8),
      nn.ReLU(),
      nn.Linear(8, 3) # 3 -> 3 labels
    )

  def forward(self, x):
    return self.layers(x)

def create_model(optimizer_name = 'SGD', learning_rate = 0.01):
  model = QwertyClassifier()
  loss_func = nn.CrossEntropyLoss()

  if optimizer_name != 'SGD':
    learning_rate = 0.001

  optim_func = getattr(torch.optim, optimizer_name)
  optimizer = optim_func(model.parameters(), lr=learning_rate)

  return model, loss_func, optimizer

def train_model(train_ldr, test_ldr, optim_name = 'SGD', learning_rate=0.01):
  num_epochs = 70

  model, loss_func, optimizer = create_model(optim_name, learning_rate)

  all_losses = np.zeros(num_epochs)
  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_acc = [] # train_accuracy per batch
    all_batch_loss = [] # loss in each batch

    for batch_x, batch_y in train_ldr:
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # batch_acc
      batch_acc = (torch.argmax(y_hat, dim=1) == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )

      all_batch_loss.append( loss.item() )
    
    all_train_acc[epoch_i] = np.mean( all_batch_acc )
    all_losses[epoch_i] = np.mean( all_batch_loss )

    # compute test_accuracy for this epoch
    test_x, test_y = next(iter(test_ldr))
    model.eval()
    with torch.no_grad():
      test_acc = (torch.argmax(model(test_x), dim=1) == test_y).float().mean() * 100
    
    all_test_acc[epoch_i] = test_acc.item()

  return (
    all_train_acc,
    all_test_acc,
    all_losses,
    model
  )

learing_rates = np.logspace(np.log10(1e-5), np.log10(0.1), 20)
optimizer_types = ['Adam', 'RMSprop', 'SGD']

final_performances = np.zeros((len(learing_rates), len(optimizer_types)))

train_loader, test_loader = split_data(dataset)

for idx_o, opt_type in enumerate(optimizer_types):
  for idx_l, lr in enumerate(learing_rates):
    all_train_acc, all_test_acc, all_losses, model =\
      train_model(train_loader, test_loader, opt_type, lr)

    final_performances[idx_l, idx_o] = np.mean( all_test_acc[-10:] )

# plot the results
plt.plot(
  learing_rates,
  final_performances,
  'o-',
  linewidth=2
)
plt.legend(optimizer_types)
plt.xscale('log')
plt.xlabel('Learing rates')
plt.ylabel('Test accuracy (ave. last 10 epochs)')
plt.title('Comparison of optimizers by learning rates')
plt.show()
