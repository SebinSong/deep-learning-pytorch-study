import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path

F = nn.functional

def to_numpy(t):
  if t.requires_grad:
    return t.detach().cpu().numpy()
  return t.cpu().numpy()

curr_dir = Path(__file__).parent
resolve_path = lambda rel_path: (curr_dir / rel_path).resolve()
data_fpath = resolve_path('../data/mnist_train_small.csv')

df = pd.read_csv(data_fpath, header=None)
data = torch.tensor(df.values[:, 1:], dtype=torch.float32)
labels = torch.tensor(df.values[:, 0], dtype=torch.long)

data /= torch.max(data).item()

dataset = TensorDataset(data, labels)

def split_data (dset, train_prop=.9, batch_size=64):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_ldr = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_ldr = DataLoader(test_dset, batch_size=test_size)

  return train_ldr, test_ldr

class MNIST_ANN(nn.Module):
  def __init__(self, batch_norm=False, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.has_batch_norm = batch_norm
    self.layers = nn.ModuleDict({
      'input': nn.Linear(28 * 28, 128),
      'fc0': nn.Linear(128, 64, bias=(not self.has_batch_norm)),
      'fc1': nn.Linear(64, 32, bias=(not self.has_batch_norm)),
      'output': nn.Linear(32, 10),
      **({
        'bnorm0': nn.BatchNorm1d(64),
        'bnorm1': nn.BatchNorm1d(32),
      } if batch_norm else {})
    })

  def forward(self, x):
    x = self.layers['input'](x)

    # Linear weighted sum -> Batch Norm -> Activation
    x = self.layers['fc0'](x)
    if self.has_batch_norm:
      x = self.layers['bnorm0'](x)
    x = F.relu(x)

    x = self.layers['fc1'](x)
    if self.has_batch_norm:
      x = self.layers['bnorm1'](x)
    x = F.relu(x)
    
    return self.layers['output'](x)

def create_model(learning_rate=0.01, has_batch_norm=False):
  model = MNIST_ANN(batch_norm=has_batch_norm)

  loss_func = nn.CrossEntropyLoss()

  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4) # L2 regularization for mitigating overfitting.

  return model, loss_func, optimizer

def weights_histogram(model):
  # initialize weight vector
  input_layer = model.layers['input']
  W = np.array([])

  flatten = lambda t: t.detach().flatten().cpu().numpy()
  for para in input_layer.parameters():
    W = np.concatenate((W, flatten(para)))

  histy, bin_edges = np.histogram(W, bins=np.linspace(-.8, .8, 101), density=True)
  histx = (bin_edges[1:] + bin_edges[:-1]) / 2
  return histx, histy

def train_model(train_ldr, test_ldr, num_epochs=100):
  model, loss_func, optimizer = create_model()

  all_losses = np.zeros(num_epochs)
  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)
  hist_x = np.zeros(100)
  all_histy = np.zeros((num_epochs, 100))

  test_x, test_y = next(iter(test_ldr))

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_acc = []
    all_batch_losses = []
    for batch_x, batch_y in train_ldr:
      y_hat = model(batch_x)

      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      all_batch_losses.append( loss.item() )
      
      batch_acc = (torch.argmax(y_hat, dim=1) == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )

    curr_train_acc = np.mean( all_batch_acc )
    curr_loss = np.mean( all_batch_losses )
    all_train_acc[epoch_i] = curr_train_acc
    all_losses[epoch_i] = curr_loss
    print(f'Epoch[{epoch_i}] - train_acc: {curr_train_acc:.2f}%, loss: {curr_loss:.2f}')

    curr_histx, curr_histy = weights_histogram(model)
    hist_x = curr_histx
    all_histy[epoch_i] = curr_histy

    # compute test_acc
    model.eval()
    with torch.no_grad():
      test_pred_labels = torch.argmax( model(test_x), dim=1 )
    
    test_acc = (test_pred_labels == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc

  return all_train_acc, all_test_acc, all_losses, hist_x, all_histy

train_loader, test_loader = split_data(dataset)

train_acc, test_acc, losses, hist_x, all_histy = train_model(train_loader, test_loader)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(losses)
ax1.set_title('Losses')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_ylim([0, 3])

ax2.plot(train_acc, label='Train acc')
ax2.plot(test_acc, label='Test acc')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim([10, 100])
ax2.set_title(f'Final model test accuracy: {test_acc[-1]:.2f}%')
ax2.legend()

plt.show()

fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 5))

total_epochs = all_histy.shape[0]
for i in range(total_epochs):
  ax3.plot(hist_x, all_histy[i, :], color=[1 - i/total_epochs, .3, i/total_epochs])

ax3.set_title('Histograms of weights')
ax3.set_xlabel('Weight values')
ax3.set_ylabel('Density')

ax4.imshow(all_histy, vmin=0, vmax=3, extent=[hist_x[0], hist_x[-1], 0, 99], aspect='auto', origin='upper', cmap='hot')
ax4.set_xlabel('Weight values')
ax4.set_ylabel('Training epoch')
ax4.set_title('Image of weight histograms')

plt.show()
