import numpy as np
import numpy.random as npr
import torch

# numpy nd-array creation and addition
nv1 = np.array([
  [1, 2, 3, 4],
  [5, 6, 7, 8]
], dtype=np.int16)
nv2 = np.array([
  [1, 2, 3, 4],
  [5, 6, 7, 8]
], dtype=np.int16)
nv3 = nv1 + nv2
nv3_t = np.transpose(nv3)
# print(nv3_t)

# useful 1-D array creation methods
v1 = np.arange(0.1, 1.1, 0.1, dtype=float)
# print(v1, v1.shape)

v2 = np.linspace(2, 77, 9, dtype=np.int16)
# print(v2)

# useful 2-D array creation methods
m1 = np.eye(3,4).astype(np.int16)
# print(m1)

m2 = np.diag(range(1, 4))
m2_1 = np.diag(range(1, 4), 2)
m2_2 = np.diag(
  np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
  ])
)
# print(m2_1)
# print(m2_2)

m3 = np.zeros((3, 4))

a = npr.randn(3, 4)
b = a[:2, 2:].copy() + 1
print(a)
print(b)
