import numpy as np
import torch

t1 = torch.randint(0, 100, (5, 10))
maxed = torch.max(t1, axis=1)
maxed_idxs = maxed[1]
argmaxed = torch.argmax(t1, dim=1)

whered = np.where(maxed_idxs == argmaxed)
# print(t1)
# print(maxed_idxs)
# print(argmaxed)
# print(whered)

# testing np.where() with bool mask
# random_bool_mask = torch.randn(10) > 0
# np_whered_1 = np.where(random_bool_mask)
# torch_whered = torch.where(random_bool_mask)
# print(random_bool_mask)
# print(np_whered_1)
# print(torch_whered)

t2 = torch.randint(0, 20, (10,))
rand_int = torch.randint(0, 10, (1,)).item()
print(rand_int)
