import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

F = torch.nn.functional

# arr1 = np.arange(0, 20, step=1)
# np.random.shuffle(arr1)
# t1 = torch.tensor( arr1.reshape((4, 5)), dtype=torch.float32 )
# softened = F.softmax(t1, dim=1)
# summed = torch.sum(softened, dim=1)
# argmaxed = torch.argmax(t1, dim=0)

# levels = ['low', 'medium', 'high']

# cat = pd.Categorical(
#   ['low', 'high', 'low', 'medium', 'high'],
#   categories=levels
# )
# print(cat)

result1 = 5 + np.random.randn(20)
result2 = 10 + np.random.randn(20)
two_sets = np.zeros((20, 2))
two_sets[:, 0] = result1
two_sets[:, 1] = result2

plt.plot(list(range(1, 41, 2)), two_sets)
plt.xlabel('Indexes')
plt.ylabel('values')
plt.title('Values per index')
plt.legend(['Result 1', 'Result 2'])
plt.show()
