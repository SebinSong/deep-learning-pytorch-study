import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path

F = nn.functional

curr_dir = Path(__file__).parent
data_fpath = (curr_dir / '../data/mnist_train_small.csv').resolve()

df = pd.read_csv(data_fpath, sep=',', header=None)
data_t = torch.tensor(df.values[:, 1:], dtype=torch.float32)
# NOTE: For multi-classification problem, where the loss function is nn.CrossEntropyLoss(),
# there is no need to turn the labels into a 1-column matrix.
labels_t = torch.tensor(df.values[:, 0], dtype=torch.long)

# min-max normalize data_t
data_t /= torch.max(data_t)

dataset = TensorDataset(data_t, labels_t)

# some global variables
D_in = data_t.shape[1]
D_out = torch.unique(labels_t).numel()
num_epochs = 10

def get_data_and_labels_from_subset (subset):
  tensors = subset.dataset.tensors
  indices = subset.indices
  data = tensors[0][indices]
  labels = tensors[1][indices]

  return data, labels

def split_data(dset, train_prop=.9, batch_size=32):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  testset = get_data_and_labels_from_subset(test_dset)

  return train_loader, testset

# define the model
class MNIST_ANN(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleDict({
      'fc1': nn.Linear(D_in, 128), # input -> hidden1
      'fc2': nn.Linear(128, 64, bias=False), # hidden1 -> hidden2
      'bnorm2': nn.BatchNorm1d(64), # batch-norm for fc2
      'fc3': nn.Linear(64, 32, bias=False), # hidden2 -> hidden3
      'bnorm3': nn.BatchNorm1d(32), # batch-norm for fc3
      'output': nn.Linear(32, D_out) # hidden3 -> output
    })
  
  def forward (self, x):
    x = F.relu( self.layers['fc1'](x) )

    # Linear -> Batch Norm -> Activation
    x = self.layers['fc2'](x)
    x = self.layers['bnorm2'](x)
    x = F.relu(x)

    # Linear -> Batch Norm -> Activation
    x = self.layers['fc3'](x)
    x = self.layers['bnorm3'](x)
    x = F.relu(x)

    return self.layers['output'](x)

def create_model(lr=0.001):
  model = MNIST_ANN()
  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

  return model, loss_func, optimizer

def train_the_model(train_ldr, testset, model, loss_func, optimizer, plot_result=True):
  all_losses = np.zeros(num_epochs)
  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)

  test_x, test_y = testset

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_acc = []
    all_batch_losses = []

    for batch_x, batch_y in train_ldr:
      # forward-pass and compute the loss
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)
      all_batch_losses.append( loss.item() )

      # back-prop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # compute the batch acc
      batch_acc = (torch.argmax(y_hat, dim=1) == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )
    
    ave_loss = np.mean(all_batch_losses)
    ave_train_acc = np.mean(all_batch_acc)
    all_losses[epoch_i] = ave_loss
    all_train_acc[epoch_i] = ave_train_acc

    if epoch_i % 2 == 0:
      print(f'[Epoch {epoch_i}] train_acc={ave_train_acc:.3f}, loss={ave_loss:.3f}')

    # compute the test accuracy
    model.eval()
    with torch.no_grad():
      test_predictions = torch.argmax(model(test_x), dim=1)

    test_acc = (test_predictions == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc.item()

  # plot the result
  if plot_result:
    plt.plot(all_train_acc, 'o-', label='Train')
    plt.plot(all_test_acc, 's-', label='Test')
    plt.legend()
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.show()

  return all_train_acc, all_test_acc, all_losses, model

def get_histogram_data(weight):
  flattened_arr = weight.data.flatten().numpy()
  counts, edges = np.histogram(flattened_arr, bins=30)
  x_centers = (edges[1:] + edges[:-1]) / 2

  return x_centers, counts

train_loader, testset = split_data(dataset)
model_zero, loss_func, optimizer = create_model()

model_zero.layers['fc2'].weight.data = torch.zeros_like(model_zero.layers['fc2'].weight).float()

_, __, ___, trained_model = train_the_model(train_loader, testset, model_zero, loss_func, optimizer, plot_result=False)

centers, counts = get_histogram_data(trained_model.layers['fc2'].weight)

plt.plot(centers, counts, 'b')
plt.xlabel('Weight values')
plt.ylabel('Counts')
plt.show()
