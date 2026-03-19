import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

F = torch.nn.functional

# mean() method on tensor var
# t1 = torch.randint(0, 5, (4, 3))
# mean_t1 = t1.to(torch.float32).mean()
# print(t1)
# print(mean_t1)

# arr1 = np.arange(0, 20, step=1)
# np.random.shuffle(arr1)
# t1 = torch.tensor( arr1.reshape((4, 5)), dtype=torch.float32 )
# softened = F.softmax(t1, dim=1)
# summed = torch.sum(softened, dim=1)
# argmaxed = torch.argmax(t1, dim=0)

# arr1 = np.arange(0, 20, step=1)
# np.random.shuffle(arr1)
# t1 = torch.tensor( arr1.reshape((4, 5)), dtype=torch.float32 )
# softened_t1 = F.softmax(t1, dim=0)
# summed = softened_t1.sum(dim=0)
# print(t1)
# print(softened_t1)
# print(summed)

# levels = ['low', 'medium', 'high']

# cat = pd.Categorical(
#   ['low', 'high', 'low', 'medium', 'high'],
#   categories=levels
# )
# print(cat)

# t1 = torch.randint(0, 10, (20,)).float()
# t2 = torch.randint(0, 10, (20,)).float()
# two_cols = torch.zeros((20, 2))
# two_cols[:, 0] = t1
# two_cols[:, 1] = t2
# print(t1)
# print(t2)
# print(two_cols)

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
