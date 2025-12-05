import numpy as np
import torch

rn = np.random.randn(5, 4)

# Seeding in NumPy - the old way.
np.random.seed(17)
rn1 = np.random.randn(5)
rn2 = np.random.randn(5)

np.random.seed(17)
rn3 = np.random.randn(5)
same = rn1 == rn3  # should be all True

# Seeding in numPy - the new way.
randseed1 = np.random.RandomState(17)
randseed2 = np.random.RandomState(1004)

# print( randseed1.randn(5) ) # l1 - same sequence as rn1
# print( randseed2.randn(5) ) # l2 - different from above. but same each time.
# print( randseed1.randn(5) ) # It is not the same as the result of l1, despite calling the same randn(5).

# Pytorch way
t1 = torch.randn(5)
# print(t1)

torch.manual_seed(17)
np.random.seed(17)
t2 = torch.randn(5)
_rn = np.random.randn(5)
print(t2)
print(_rn)

# If compare t2 and _rn here, torch's seed does not spread to numpy.
