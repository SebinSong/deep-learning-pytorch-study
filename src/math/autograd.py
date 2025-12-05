import torch
import numpy as np

# x = torch.tensor(3.0, requires_grad=True)
# y = x * 2
# z = y + 5
# loss = z ** 2

# x2 = torch.linspace(-10, 10, 50, requires_grad=True, dtype=torch.float32).reshape(10, 5)
# ones = torch.ones(10, 5, dtype=torch.float32)
# loss2 = (x2 - ones).pow(2).mean()

# slopes = np.linspace(-2, 2, 21)
# print(slopes, slopes.size)

# for a, b in enumerate(slopes):
#   print(a, b)

dict1 = {
  'a': 100,
  'b': 200,
  'c': 300
}

print(dict1.items())
