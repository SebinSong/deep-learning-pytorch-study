import numpy as np
import torch
import torch.nn as nn # nn stands for newral networks
import matplotlib.pyplot as plt

z = [1, 2, 3]
nz = np.array(z)
tz = torch.tensor(z, dtype=torch.float32)

# Manually computing softmax in numpy
exp_nz = np.exp(nz)
sum_exp_nz = np.sum(exp_nz)
sigma_nz = exp_nz / sum_exp_nz
# print(sigma_nz)
# print(np.sum(sigma_nz) == 1.0)

# Repeat with some random integers
z = np.random.randint(-5, high=15, size=15)
exp_z = np.exp(z)
sum_exp_z = np.sum(exp_z)
sigma_z = exp_z / sum_exp_z

# create an instance of the softmax activation class
softfunc = nn.Softmax(dim=0)

sigmaT = softfunc(torch.tensor(z, dtype=torch.float32))
sum_sigmaT = torch.sum(sigmaT)

# compare

plt.plot(z, sigma_z, 'ko')
plt.plot(z, sigmaT, 'ko')
plt.xlabel('Original number (z)')
plt.ylabel('Softmax(z) $\sigma$')
plt.title('$\sum\sigma$ = %g' %np.sum(sigma_z))
plt.show()


# softmax in pytorch
