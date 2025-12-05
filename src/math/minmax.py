import numpy as np
import torch
import torch.nn as nn

## argmin / argmax for a vector
v = np.array([1, 40, 2, -3])

# Below two are minimaum and maximum 'values' in the array.
minval = np.min(v)
maxval = np.max(v)

# whereas, argmin and argmax finds the index of the min/max in the array
minidx = np.argmin(v)
maxidx = np.argmax(v)

## argmin / argmax for a matrix
m = np.array([
  [-3, 1, 10],
  [20, 8, 5],
  [30, 3, 7]
])

min_m = np.min(m)
min_m0 = np.min(m, axis=0) # minimum of each column
min_m1 = np.min(m, axis=1) # minimum of each row

maxidx_m = np.argmax(m)
maxidx_m0 = np.argmax(m, axis=0)
maxidx_m1 = np.argmax(m, axis=1)


### In pytorch
vt = torch.tensor([1, 40, 2, -3])

vt_min = torch.min(vt)
vt_max = torch.max(vt)
vt_minidx = torch.argmin(vt)
vt_maxidx = torch.argmax(vt)

vm = torch.tensor([
  [-3, 1, -3],
  [2, 8, -5],
  [30, 130, 7]
])

vm_min = torch.min(vm)
vm_max0 = torch.max(vm, axis=0)
vm_argmax = torch.argmax(vm)
vm_argmax0 = torch.argmax(vm, axis=0) # argmax of each column
vm_argmax1 = torch.argmax(vm, axis=1) # argmax of each row

print('vm_max0.indices:', vm_max0.indices, vm_argmax0, vm_max0.indices == vm_argmax0)
print('vm_max0.values:', vm_max0.values)
