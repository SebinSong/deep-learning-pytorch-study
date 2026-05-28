import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

F = nn.functional

class ANN_Model(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(100, 2**8),
      'fc1': nn.Linear(2**8, 2**7),
      'fc2': nn.Linear(2**7, 2**5),
      'fc3': nn.Linear(2**5, 2**4),
      'output': nn.Linear(2**4, 2)
    })
  
  def forward(self, x):
    x = F.relu( self.layers['input'](x) )
    x = F.relu( self.layers['fc1'](x) )
    x = F.relu( self.layers['fc2'](x) )
    x = F.relu( self.layers['fc3'](x) )
    return self.layers['output'](x)

model = ANN_Model()

# collect all weights and biases
version_one = False
version_two = True

if version_one:
  all_weights = np.array([])
  all_biases = np.array([])

  for name, p in model.named_parameters():
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
