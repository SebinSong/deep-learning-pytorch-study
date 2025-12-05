import numpy as np
from matplotlib import pyplot as plt

x = [1, 2, 4, 6, 5, 4, 0, -4, 5, -2, 6, 10, -9, 1, 3, -6]
n = len(x)

# compute the population mean:
popmean = np.mean(x)

# compute the sample mean:
sample = np.random.choice(x, size=5, replace=False)
sampmean = np.mean(sample)

print('population mean: ', popmean)
print('sample mean: ', sampmean)

# compute lots of sample means:
n_expers = 1000

sample_means = np.zeros(n_expers)
for i in range(n_expers):
  # step-1: draw a sample
  sample = np.random.choice(x, size=5, replace=True)

  # step-2: compute its mean
  sample_means[i] = np.mean(sample)

# show the results as a histogram:

plt.hist(sample_means, bins=40, density=True) # density=True here normalizes the histogram, kind of like a probability density function
plt.plot([popmean, popmean], [0, 0.5], 'm--')
plt.show()
