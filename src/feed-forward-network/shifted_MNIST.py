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

dataset = TensorDataset(data, labels)

def split_data(dset, train_prop=.9, batch_size=64):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_ldr = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_ldr = DataLoader(test_dset, batch_size=test_size)

  return train_ldr, test_ldr

def get_data_and_labels_from_dlr (dlr):
  dlr_tensors = dlr.dataset.dataset.tensors
  dlr_indices = dlr.dataset.indices
  data = dlr_tensors[0][dlr_indices]
  labels = dlr_tensors[1][dlr_indices]

  return data, labels

def show_single_img(t, label=''):
  plt.figure(figsize=(5,5))
  plt.imshow(t, cmap='gray', vmax=1)
  if isinstance(label, int):
    plt.title(f'Label: {label}')
  plt.show()

  return

def random_int(max=100):
  n = int(torch.rand(1).item() * max)
  return n

def random_shift_data (dlr):
  tensors = dlr.dataset.dataset.tensors
  idxs = dlr.dataset.indices

  for idx in idxs:
    img = tensors[0][idx, :].reshape(28, -1)
    rand_roll = np.random.randint(-10, 11)
    shifted = torch.roll(img, rand_roll, dims=1)

    tensors[0][idx] = shifted.reshape(1, -1)

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

def train_model(train_ldr, test_ldr, num_epochs=60):
  model, loss_func, optimizer = create_model()

  test_x, test_y = next(iter(test_ldr))
  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)
  all_losses = np.zeros(num_epochs)

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
      # compute batch_acc
      batch_acc = (torch.argmax(y_hat, dim=1) == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )

    train_acc_mean = np.mean( all_batch_acc )
    loss_mean = np.mean( all_batch_losses )

    if epoch_i % 5 == 0:
      print(f'[Epoch: {epoch_i}] - {train_acc_mean=:.3f}, {loss_mean=:.3f}')

    all_train_acc[epoch_i] = train_acc_mean
    all_losses[epoch_i] = loss_mean
    # compute test_acc
    model.eval()
    with torch.no_grad():
      test_pred_labels = torch.argmax( model(test_x), dim=1 )
    test_acc = (test_pred_labels == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc.item()

  return all_train_acc, all_test_acc, all_losses

train_loader, test_loader = split_data(dataset)
random_shift_data(test_loader)

train_acc, test_acc, losses = train_model(train_loader, test_loader)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Losses')

ax2.plot(train_acc, label='Train')
ax2.plot(test_acc, label='Test')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracies per epoch')
ax2.legend()

plt.show()
