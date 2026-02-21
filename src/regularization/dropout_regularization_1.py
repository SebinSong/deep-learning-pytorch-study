import torch
import torch.nn as nn
import numpy as np

F = nn.functional

# define a dropout instance and make some data
prob = .5

dropout = nn.Dropout(p=prob)
x = torch.ones(10)
y = F.dropout(x, p=prob)
scaled_back = y * (1 - prob)

stacked = torch.vstack([x, y, scaled_back]).T
print(stacked)
