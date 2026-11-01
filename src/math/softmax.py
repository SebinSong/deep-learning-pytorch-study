import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

z = [1, 3, 5, 7]
az = np.array(z, dtype=np.float32)
tz = torch.tensor(z, dtype=torch.float)

# softmax manual
exp_tz = torch.exp(tz)
sum_exp_tz = torch.sum(exp_tz)

softmaxed_tz = exp_tz / sum_exp_tz

# softmax common way
sm_tz = torch.softmax(tz, dim=0)

# Linear-relationship btw data and softmax?
cc = np.corrcoef(tz, sm_tz)

# softmax with dimension specified
t2 = torch.randint(-10, 10, (20,), dtype=torch.float32)
softmaxer = nn.Softmax(dim=0)
sm_t2 = softmaxer(t2)

# plotting
def _():
  plt.plot(t2, sm_t2, 'ko')
  plt.xlabel('Original number (z)')
  plt.ylabel('Soft-maxed')
  plt.yscale('log')
  plt.title(f'Softmax sum: {torch.sum(sm_tz).item():.2f}')
  plt.show()

# numpy's correlation coefficient calculation

def _():
  # 4 samples , each with 2 features
  data = np.array([
    [500, 100],
    [1000, 200],
    [1500, 300],
    [2000, 400],
  ])

  corr_matrix = np.corrcoef(data.T)
  print(corr_matrix)

# logarithm in numpy / pytorch
def _():
  arr = np.random.randint(1, 15, 200)
  logged = np.log(arr)
  cc = np.corrcoef(arr, logged)[0, 1]
  print(cc)

# Draw the log values
def _():
  x = np.linspace(1e-4, 1, 50)
  logx = np.log(x)
  ex = np.exp(x)

  fig = plt.figure(figsize=(10, 4))
  plt.plot(x, logx, 'ks-', markerfacecolor='w', label='ln(x)')
  plt.plot(x, ex, 'bo-', markerfacecolor='w', label='e^x')
  plt.plot(x, x, 'o-', color=[.8, .8, .8], label='Unity')
  plt.xlabel('x')
  plt.ylabel('f(x)')
  plt.title('ln(x), e^x graphs')
  plt.legend()
  plt.show()

_()
