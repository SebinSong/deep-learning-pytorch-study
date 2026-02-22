import torch
import numpy as np
import pandas as pd

F = torch.nn.functional

arr1 = np.arange(0, 20, step=1)
np.random.shuffle(arr1)
t1 = torch.tensor( arr1.reshape((4, 5)), dtype=torch.float32 )
softened = F.softmax(t1, dim=1)
summed = torch.sum(softened, dim=1)
argmaxed = torch.argmax(t1, dim=0)

levels = ['low', 'medium', 'high']

cat = pd.Categorical(
  ['low', 'high', 'low', 'medium', 'high'],
  categories=levels
)
print(cat)
