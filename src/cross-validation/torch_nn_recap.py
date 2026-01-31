import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# # batch size
# N = 30

# D_in = 1
# D_out = 1

# linear_layer = nn.Linear(D_in, D_out)

# print(f"Layer's weight: {linear_layer.weight}")
# print(f"Layer's Bias: {linear_layer.bias}")

# X = torch.randn(N, D_in)

# true_W = torch.zeros((D_in, D_out)) + 2.0
# true_b = torch.tensor(1.0)

# y_true = X @ true_W + true_b + (torch.randn(N, D_out) + 0.1) # add a little noise

# y_hat = linear_layer(X)

sample_data = torch.randn((4,3), dtype=torch.float32)
print(sample_data)
activated_data = nn.functional.relu(sample_data)
print(sample_data)
print(activated_data)