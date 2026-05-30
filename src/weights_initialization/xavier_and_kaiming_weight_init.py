import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

F = nn.functional

class ANN_Model(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(100, 100),
      'fc1': nn.Linear(100, 100),
      'fc2': nn.Linear(100, 100),
      'fc3': nn.Linear(100, 100),
      'output': nn.Linear(100, 2)
    })

  
  def forward(self, x):
    x = F.relu( self.layers['input'](x) )
    x = F.relu( self.layers['fc1'](x) )
    x = F.relu( self.layers['fc2'](x) )
    x = F.relu( self.layers['fc3'](x) )
    return self.layers['output'](x)

class ANN_Model_Xavier(ANN_Model):
  def __init__(self):
    super().__init__()

    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

model = ANN_Model_Xavier()

# collect all weights and biases
version_one = False
version_two = True

if version_one:
  all_weights = np.array([])
  all_biases = np.array([])

  for name, p in model.named_parameters():
    # p.data() is how we access the actual tensor of the parameters, and .flatten() is how we convert them to 1D arrays.
    if 'weight' in name:
      all_weights = np.concatenate((all_weights, p.data.detach().numpy().flatten()), axis=0)
    elif 'bias' in name:
      all_biases = np.concatenate((all_biases, p.data.detach().numpy().flatten()), axis=0)

elif version_two:
  all_weights = np.concatenate([
    p.data.detach().numpy().flatten() for name, p in model.named_parameters() if 'weight' in name
  ])
  all_biases = np.concatenate([
    p.data.detach().numpy().flatten() for name, p in model.named_parameters() if 'bias' in name
  ])

# How many are there?
print(f'There are {len(all_weights)} weight parameters.')
print(f'There are {len(all_biases)} bias parameters.')

plot_weight_dist = True

if plot_weight_dist:
  fig, ax = plt.subplots(1, 3, figsize=(18, 4))
  ax[0].hist(all_biases, 40)
  ax[0].set_title('Histogram of initial biases')

  ax[1].hist(all_weights, 40)
  ax[1].set_title('Histogram of initial weights')

  yB, xB = np.histogram(all_biases, 30)
  yW, xW = np.histogram(all_weights, 30)

  ax[2].plot(
    (xB[:-1] + xB[1:]) / 2, yB / np.sum(yB), label='Bias'
  )
  ax[2].plot(
    (xW[:-1] + xW[1:]) / 2, yW / np.sum(yW), label='Weight'
  )
  ax[2].set_title('Density estimate for both')
  ax[2].legend()

  for i in range(3):
    ax[i].set_xlabel('Initial Value')
    ax[i].set_ylabel('Density' if i == 2 else 'Count')

  plt.show()

def draw_weight_hist_per_layers():
  fig, ax = plt.subplots(1, 2, figsize=(15, 4))

  for name, p in model.named_parameters():
    these_data = p.data.numpy().flatten()
    counts, edges = np.histogram(these_data, 10)

    if 'bias' in name:
      ax[0].plot(
        (edges[:-1] + edges[1:]) / 2,
        counts / np.sum(counts),
        label=f'{name[:-5]} bias (N={len(these_data)})'
      )
    elif 'weight' in name:
      ax[1].plot(
        (edges[:-1] + edges[1:]) / 2,
        counts / np.sum(counts),
        label=f'{name[:-7]} weights (N={len(these_data)})'
      )
  
  ax[0].set_title('Biases per layer')
  ax[0].legend()
  ax[1].set_title('Weights per layer')
  ax[1].legend()
  plt.show()
