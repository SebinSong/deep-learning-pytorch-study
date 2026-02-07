import torch
import torch.nn as nn
import torch.nn.functional as F

# define a dropout instance and make some data
prob = 0.5 # probability of an element to be zeroed

dropout = nn.Dropout(p=prob)

x = torch.ones(10)
y1 = dropout(x)

# dropout is turned off when evaluating the model
dropout.eval()
y2 = dropout(x)

# another way of using dropout is to use F.dropout()
# to toggle on/off, use training=bool parameter in this case
y3 = F.dropout(x, p=0.35)
y4 = F.dropout(x, p=0.5, training=False)
print(y3)
print(y4)

# once dropout.eval() is called, the eval mode stays on until dropout.train() is called
dropout.train()
y5 = dropout(x)
