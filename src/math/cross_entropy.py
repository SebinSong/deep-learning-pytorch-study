import numpy as np
import torch 
import torch.nn.functional as F

p_list = [.25, .75]
H = -1 * np.sum([p * np.log(p) for p in p_list])


# Cross entropy
p_real = [1, 0] # is a cat = 1, not a cat = 0
p_model = [0.25, 0.75]

zipped = list(zip(p_real, p_model))
Cross_H = -1 * np.sum([a * np.log(b) for a, b in zipped])
print('Cross_H: ', Cross_H)

# In pytorch

p_real_tensor = torch.tensor(p_real, dtype=float)
p_model_tensor = torch.tensor(p_model, dtype=float)

Ht = F.binary_cross_entropy(p_model_tensor, p_real_tensor)
print('Ht:', Ht)
