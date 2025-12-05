import numpy as np
import torch

npr = np.random

A = npr.randn(3, 4)
B = npr.randn(4, 5)
C = npr.randn(3, 7)

# @ is a shorthand for matrix multiplication in numpy
AB = np.round(np.matmul(A, B), 3)
AB_0 = np.round(A@B, 2)
CT_A = C.T @ A
# print(AB)
# print(CT_A)

# pytorch way
ta = torch.randn(3, 4) * 0.5 + 10
tb = torch.randn(4, 5) * 0.5 + 10
tc = torch.tensor(C)
tatb = torch.matmul(ta, tb)
tatb_0 = ta @ tb
taB = ta @ B
print(taB)
print(tc)
