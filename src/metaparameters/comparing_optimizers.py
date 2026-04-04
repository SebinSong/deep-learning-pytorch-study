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

draw_data_plot = False
if draw_data_plot:
  fig = plt.figure(figsize=(5,5))
  plt.plot(
    data[labels == 0, 0],
    data[labels == 0, 1],
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
    data[labels == 2, 0],
    data[labels == 2, 1],
    'r^',
    alpha=.5,
    label='Cluster C'
  )

  plt.legend()
  plt.xlabel('Dim 1')
  plt.ylabel('Dim 2')
  plt.title('Qwerties!')
  plt.show()

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

def train_model(train_ldr, test_ldr, optim_name = 'SGD'):
  num_epochs = 80

  model, loss_func, optimizer = create_model(optim_name)

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

def plot_results(
  train_acc,
  test_acc,
  losses,
  model,
  optim_name = 'SGD'
):
  global data

  with torch.no_grad():
    y_hat = model(data)

  predictions = torch.argmax(y_hat, dim=1)
  total_accuracy = (predictions == labels).float().mean().item() * 100

  acc_by_group = np.zeros(3)

  for i in range(3):
    preds_per_group = predictions[labels == i] # pred results for class i
    acc_by_group[i] = (preds_per_group == i).float().mean() * 100

  # Create the figure
  fig, ax = plt.subplots(2, 2, figsize=(10, 6))

  # plot for losses
  ax[0, 0].plot(losses)
  ax[0, 0].set_ylabel('Loss')
  ax[0, 0].set_xlabel('Epoch')
  ax[0, 0].set_title(f'{optim_name}: Losses')

  # plot the  accuracies
  ax[0, 1].plot(train_acc, label='Train acc')
  ax[0, 1].plot(test_acc, label='Test acc')
  ax[0, 1].set_ylabel('Accuracy (%)')
  ax[0, 1].set_xlabel('Epoch')
  ax[0, 1].legend()
  ax[0, 1].set_title(f'{optim_name}: Accuracies')

  # bar plots for total accuracy by groups
  ax[1, 0].bar(range(3), acc_by_group)
  ax[1, 0].set_ylim([
    np.min(acc_by_group) - 5,
    np.max(acc_by_group) + 5
  ])
  ax[1, 0].set_xticks([0, 1, 2])
  ax[1, 0].set_xlabel('Group')
  ax[1, 0].set_ylabel('Accuracy (%)')
  ax[1, 0].set_title(f'{optim_name}: Accuracy by group (total acc: {total_accuracy:.2f}%)')

  # scatter plot of correct / incorrect labeled data
  idx_mask_incorrect = predictions != labels

  dot_styles = ['bs', 'ko', 'g^']
  class_name = ['Cluster A', 'Cluster B', 'Cluster C']
  ax4 = ax[1, 1]

  # Plot all dots first
  for i in range(3):
    ax4.plot(
      data[labels == i, 0],
      data[labels == i, 1],
      dot_styles[i],
      label=class_name[i],
      alpha=.5
    )

  # Draw marks for incorrect predictions
  ax4.plot(
    data[idx_mask_incorrect, 0],
    data[idx_mask_incorrect, 1],
    'rx',
    label='Incorrect'
  )
  ax4.legend()

  plt.tight_layout()
  plt.show(block=False)

optimizer_types = ['Adam', 'RMSprop', 'SGD']

train_loader, test_loader = split_data(dataset)
for oi, optim_type in enumerate(optimizer_types):
  all_train_acc, all_test_acc, all_losses, model =\
    train_model(train_loader, test_loader, optim_type)

  plot_results(
    all_train_acc, all_test_acc, all_losses, model,
    optim_name=optim_type
  )

plt.show()
