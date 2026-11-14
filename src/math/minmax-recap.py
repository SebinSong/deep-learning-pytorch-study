import numpy as np
import torch
import torch.nn as nn

v = np.array([1, 40, 2, -3]) # random integers - reps logits in DL context

minval = np.min(v)
maxval = np.max(v)

# Can specify the dimension for min/max calc
v2 = np.random.randint(-20, 30, (4, 3)).astype(dtype=np.float32)

minval = np.min(v2, axis=1)
maxval = np.max(v2, axis=1)
print(f'min: {minval}, max: {maxval}')

# arg max/min
minidx = np.argmin(v2, axis=1)
maxidx = np.argmax(v2, axis=1)

# With pytorch
t1 = torch.tensor([1, 40, 2, -3], dtype=torch.float32).reshape(1, -1)
min_t1 = torch.min(t1)
max_t1 = torch.max(t1)
print(f'min_t1: {min_t1}, type: {type(min_t1)}')

amin_t1 = torch.argmin(t1, dim=1)
amax_t1 = torch.argmax(t1, dim=1)

mat1 = torch.randint(-10, 30, (3, 4)).float()
min_mat1 = torch.min(mat1, axis=1)
print(f'mat1: {mat1}')

argmin_mat1 = torch.argmin(mat1, dim=1)
print(f'argmin v1: ', min_mat1.indices)
print(f'argmin v2: ', argmin_mat1)
