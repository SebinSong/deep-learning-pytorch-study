import torch
import torch.nn as nn

# define a dropout instance and make some data
prob = .5

dropout = nn.Dropout(p=prob)
x = torch.ones(10)
print(x)
