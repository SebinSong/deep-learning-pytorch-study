import numpy as np
import torch

## Calculating dot-product In numpy
nv1 = np.array([1, 2, 3, 4])
nv2 = np.array([5, 6, 7])
dot_nv12 = np.dot(nv1, nv2)

# print(dot_nv12)

# numpy.sum() method
sum_nv1 = np.sum(nv1)

# Element-wise multiplication
nv1_2 = nv1 * nv2
sum_nv1_2 = np.sum(nv1_2)

# print(nv1_2)
# print(sum_nv1_2)

# print(sum_nv1_2 == dot_nv12)

## In pytorch
t1 = torch.tensor([1, 2, 3, 4])
t2 = torch.tensor([5, 6, 7, 8])
dot_t12 = float(torch.dot(t1, t2))

# Element-wise multiplication
t1_2 = t1 * t2
sum_t1_2 = torch.sum(t1_2)
print(t1_2)
print(sum_t1_2)
print(dot_t12)
