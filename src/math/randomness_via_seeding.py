import numpy as np
import torch

np.random.seed(10)

series1 = np.random.randn(2, 5)
# [[ 1.3315865   0.71527897 -1.54540029 -0.00838385  0.62133597]
#  [-0.72008556  0.26551159  0.10854853  0.00429143 -0.17460021]]
np_seed1 = np.random.RandomState(10)

def _():
  # using np.random.RandomState in numpy
  arr1 = np_seed1.randn(2, 5)
  print('arr1: ', arr1)
  arr2 = np_seed1.randn(2, 5)
  print('arr2:', arr2)

# In pytorch
def _():
  g = torch.Generator()
  g.manual_seed(10)

  t1 = torch.randn((2, 5), generator=g)
  # tensor([[-0.6014, -1.0122, -0.3023, -1.2277,  0.9198],
  #       [-0.3485, -0.8692, -0.9582, -1.1920,  1.9050]])
  print('t1: ', t1)

_()
