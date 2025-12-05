import numpy as np

x = [1, 2, 4, 6, 5, 10, 12]
n = len(x)

# Mean
mean1 = np.mean(x)
mean2 = sum(x) / n

# Variance
var1 = np.var(x, ddof=1)
var2 = (1 / (n - 1)) * np.sum((x - mean1) ** 2)

print('var1: ', var1)
print('var2: ', var2)