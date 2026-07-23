import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from pathlib import Path

F = nn.functional

curr_dir = Path(__file__).parent
data_fpath = (curr_dir / '../data/mnist_train_small.csv').resolve()

dataset_df = pd.read_csv(data_fpath, sep=',', header=None)
data_t = torch.tensor( dataset_df.values[:, 1:], dtype=torch.float32 )
labels_t = torch.tensor( dataset_df.values[:, 0], dtype=torch.long )

# scale the tensor element range down to 0-1
data_t /= data_t.max()

# some global variables
D_in = 28 * 28

# define the AE model
class MNIST_AE(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(D_in, 128),
      'enc': nn.Linear(128, 16),
      'latent': nn.Linear(16, 128),
      'dec': nn.Linear(128, D_in)
    })
  
  def forward(self, x):
    x = F.relu( self.layers['input'](x) )
    code_x = F.relu( self.layers['enc'](x) )
    x = F.relu( self.layers['latent'](x) )
    y = torch.sigmoid( self.layers['dec'](x) )

    return y, code_x

def create_model(lr=1e-4):
  model = MNIST_AE()
  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  return model, loss_fn, optimizer

