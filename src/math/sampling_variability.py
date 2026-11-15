import numpy as np
from matplotlib import pyplot as plt

# population
x = np.array([1, 2, 4, 6, 5, 4, 0, -4, 5, -2, 6, 10, -9, 1, 3, -6], dtype=np.int64)
n = len(x)
sample_size = 10
iteration = 10000

# pop mean
pop_mean = np.mean(x)
sample_means = np.zeros(iteration)

for i in range(iteration):
  sample_idxs = np.random.permutation(n)[:sample_size]
  sample = x[sample_idxs]
  sample_means[i] = np.mean(sample)

plt.hist(sample_means, bins=30, density=False)
plt.plot([pop_mean, pop_mean], [0, .3], 'm--')
plt.ylabel('Count')
plt.xlabel('Sample mean')
plt.show()

# print(f'{pop_mean=}, {sample_mean=}')
