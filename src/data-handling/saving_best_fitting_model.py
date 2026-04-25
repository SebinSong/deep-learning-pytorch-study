import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from pathlib import Path

F = nn.functional

curr_dir = Path(__file__).resolve().parent
data_fpath = (curr_dir / '../data/mnist_train_small.csv').resolve()

df = pd.read_csv(data_fpath, header=None, sep=',')

data = torch.tensor( df.values[:, 1:], dtype=torch.float32)
labels = torch.tensor( df.values[:, 0], dtype=torch.long )

data_norm = data / torch.max(data) # min-max normalization
dataset = TensorDataset(data_norm, labels)

# some global variables
batch_size = 64
learning_rate = 0.001
num_epochs = 80

def split_data(dset, train_prop=.9):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=test_size)

  return train_loader, test_loader

class MNIST_ANN(nn.Module):
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

def create_model():
  m = MNIST_ANN()
  l = nn.CrossEntropyLoss()
  o = torch.optim.Adam(m.parameters(), lr=learning_rate, weight_decay=1e-4)
  return m, l, o

def train_model(train_ldr, test_ldr):
  model, loss_func, optimizer = create_model()
  test_x, test_y = next(iter(test_ldr))

  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)
  all_losses = np.zeros(num_epochs)

  best_model = { 'accuracy': 0, 'net': None }

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

      # batch_acc
      batch_acc = (torch.argmax(y_hat, dim=1) == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )
    
    train_acc = np.mean( all_batch_acc )
    ave_loss = np.mean( all_batch_losses )
    all_train_acc[epoch_i] = train_acc
    all_losses[epoch_i] = ave_loss

    if (epoch_i % 10 == 0):
      print(f'[Epoch={epoch_i}] - {train_acc=:.3f}, {ave_loss=:.3f}')
 
    # test accuracy
    model.eval()
    with torch.no_grad():
      test_pred_labels = torch.argmax( model(test_x), dim=1 )
    
    test_acc = (test_pred_labels == test_y).float().mean().item() * 100
    all_test_acc[epoch_i] = test_acc

    if test_acc > best_model['accuracy']:
      best_model['accuracy'] = test_acc
      best_model['net'] = copy.deepcopy( model.state_dict() ) # can also be: { k: v.cpu().clone() for k, v in model.state_dict().item() }

  return all_train_acc, all_test_acc, all_losses, best_model

if __name__ == '__main__':
  train_loader, test_loader = split_data(dataset)
  train_acc, test_acc, losses, best_model = train_model(train_loader, test_loader)

  print(f'Best model acc: {best_model["accuracy"]:.3f}')
  base_net = MNIST_ANN()
  base_net.load_state_dict(best_model['net'])

  with torch.no_grad():
    final_pred_labels = torch.argmax( base_net(data), dim=1 )
  final_acc = (final_pred_labels == labels).float().mean() * 100
  print(f'Final acc: {final_acc:.3f}%')

  fig, ax = plt.subplots(1, 2, figsize=(16, 5))

  ax[0].plot(losses)
  ax[0].set_xlabel('Epoch')
  ax[0].set_ylabel('Loss')
  ax[0].set_title('Losses')

  ax[1].plot(train_acc, 'o-', label='Train')
  ax[1].plot(test_acc, 'o-', label='Test')
  ax[1].set_xlabel('Epoch')
  ax[1].set_ylabel('Accuracy (%)')
  ax[1].set_ylim([70, 100])
  ax[1].set_title('Accuracies')
  ax[1].legend()

  plt.tight_layout()
  plt.show()
