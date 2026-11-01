import numpy as np
import matplotlib.pyplot as plt

# Computing the entropy of an event happening
def _():

  p = [0.375, 1 - 0.375]

  H = 0
  for pv in p:
    H -= pv * np.log(pv)

  print(f'Entropy: {H:^10.4f}')

# Cross entropy example
def _():
  p = [1,0] # prob distribution of the reality
  q = [.975, 1 - .975] # Model predicting (is a cat, isn't a cat)

  # computing the cross-entropy of the model prediction
  H = 0
  for (pv, qv) in zip(p, q):
    H -= pv * np.log(qv)

  print(f'Cross entropy value is : {H:.3f}')

# BCE in pytorch
def _():
  import torch
  import torch.nn.functional as F

  p = [1,0] # prob distribution of the reality
  q = [.975, 1 - .975] # Model predicting (is a cat, isn't a cat)

  pt = torch.tensor(p).float()
  qt = torch.tensor(q)

  bce = F.binary_cross_entropy(qt, pt)
  bce_swap = F.binary_cross_entropy(pt, qt)

  print(f'Cross entropy value is : {bce:.3f}')
  print(f'Cross entropy value(swapped) is : {bce_swap:.3f}')

_()

