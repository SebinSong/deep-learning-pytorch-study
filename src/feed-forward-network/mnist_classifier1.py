import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

F = nn.functional

curr_dir = Path(__file__).parent
resolve_path = lambda rel_path: (curr_dir / rel_path).resolve()

def load_data():
  data_path = resolve_path('../data/mnist_train_small.csv')
  df = pd.read_csv(data_path, header=None)

  data = torch.tensor(df.values[:, 1:], dtype=torch.float32)
  labels = torch.tensor(df.values[:, 0], dtype=torch.long)

  return data, labels

data, labels = load_data()

# Normalize the data to a range of [0, 1]
# NOTE: It is minmax normalization here but if batch-norm is used inside the model, z-scoring is preferred and works better.
data_normalized = data / torch.max(data)
dataset = TensorDataset(data_normalized, labels)

show_data_distribution = False

if show_data_distribution:
  _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

  ax1.hist(data.flatten(), 50)
  ax1.set_xlabel('Pixel intensity values')
  ax1.set_ylabel('Count')
  ax1.set_title('Histogram of original data')
  ax1.set_yscale('log')

  ax2.hist(data_normalized.flatten(), 50)
  ax2.set_xlabel('Pixel intensity values')
  ax2.set_ylabel('Count')
  ax2.set_title('Histogram of normalized data')

  plt.show()

def split_data(dset, train_prop=.9, batch_size=32):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_ldr = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_ldr = DataLoader(test_dset, batch_size=test_size)

  return train_ldr, test_ldr

class MNIST_ANN(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.layers = nn.ModuleDict({
      'input': nn.Linear(28*28, 128),
      'fc1': nn.Linear(128, 64),
      'fc2': nn.Linear(64, 32),
      'output': nn.Linear(32, 10)
    })
  
  def forward(self, x):
    x = F.relu( self.layers['input'](x) )

    x = F.relu( self.layers['fc1'](x) )
    x = F.relu( self.layers['fc2'](x) )

    return torch.log_softmax( self.layers['output'](x), axis=1 ) # loss function will be nn.NLLLoss(). So log-softmax needs to be called here.

def create_model(learning_rate=0.01):
  model = MNIST_ANN()
  loss_func = nn.NLLLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  return model, loss_func, optimizer

def train_model(train_ldr, test_ldr, num_epochs=70):
  model, loss_func, optimizer = create_model()

  all_losses = np.zeros(num_epochs)
  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)

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
      pred_labels = torch.argmax(y_hat, dim=1)
      batch_acc = (pred_labels == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )

    all_losses[epoch_i] = np.mean( all_batch_losses )
    all_train_acc[epoch_i] = np.mean( all_batch_acc )

    test_x, test_y = next(iter(test_ldr))

    model.eval()
    with torch.no_grad():
      test_pred_labels = torch.argmax( model(test_x), dim=1 )

    test_acc = (test_pred_labels == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc
  
  return all_train_acc, all_test_acc, all_losses, model

train_loader, test_loader = split_data(dataset, batch_size=32)
train_acc, test_acc, losses, model = train_model(train_loader, test_loader)

draw_result_plot = False

if draw_result_plot:
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

  ax1.plot(losses)
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Loss')
  ax1.set_ylim([0, 3])
  ax1.set_title('Model loss')

  ax2.plot(train_acc, label='Train')
  ax2.plot(test_acc, label='Test')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Accuracy(%)')
  ax2.set_title(f'Final model test accuracy - {test_acc[-1]:.3f}%')
  ax2.legend()

  plt.show()

test_x, test_y = next(iter(test_loader))
with torch.no_grad():
  predictions = model(test_x)
