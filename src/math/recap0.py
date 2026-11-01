import numpy as np
import torch

# vector/matrix transpose in pytorch and numpy
def _():
  # in numpy: 1-D vector
  arr1 = np.arange(1, 5, 1).astype(np.int64)
  arr2 = arr1.reshape(1, -1)
  arr3 = arr2.T

  # in pytorch: 1-D tensor
  t1 = torch.arange(1, 5, 1).to(torch.long)
  t2 = t1.reshape(1, -1)
  t3 = t2.T

  # in numpy: 2D-matrix
  mat1 = np.linspace(0, 20, 20).reshape((4, 5))
  mat2 = np.arange(1, 21, dtype=np.float64).reshape((4, -1))
  mat3 = np.transpose(mat2, axes=(1, 0))

  # in pytoch: 2D-tensor

  tmat_1 = torch.randn(3, 3)
  tmat_2 = tmat_1.T
  tmat_3 = (tmat_1 == tmat_2).to(torch.long)

  # zeros and ones
  t1 = torch.zeros(3, 4).long()

# Dot product between vectors and matrices
def _():
  v1 = np.array([1, 0, -1, 1])
  v2 = np.array([2, 4, 6, 8])

  # dot product btw two vectors - numpy
  d1 = np.dot(v1, v2)
  d1_manual = np.sum(v1 * v2)

  # in pytorch
  t1 = torch.from_numpy(v1)
  t2 = torch.tensor(v2)

  dot_t1 = torch.dot(t1, t2)
  dot_t1_manual = torch.sum(t1 * t2)

  a = v1 @ v2
  b = t1 @ t2

# matrix multiplication
def _():

  # numpy
  m1 = np.array([
    [1, 2, 3],
    [4, 5, 6]
  ], dtype=np.float32)
  m2 = np.arange(1, 13, dtype=np.float32).reshape(3, -1)

  res1 = m1 @ m2
  res2 = np.matmul(m1, m2)

  # pytorch
  tm1 = torch.tensor(m1)
  tm2 = torch.from_numpy(m2)

  res1 = tm1 @ tm2
  res2 = torch.mm(tm1, tm2)
  res3 = torch.matmul(tm1, tm2)

def _():
  import torch.nn as nn

  layer = nn.Linear(in_features=2, out_features=1)
  X = torch.randn(10, 2)
  pred = layer(X)

  print(pred)
_()
