import numpy as np
import torch

x = [1, 2, 4, 6, 5, 10, 12]
n = len(x)

# Mean - manual
mean1 = sum(x) / n

# Mean - numpy
mean2 = np.mean(x)

# Mean - pytorch
tx = torch.tensor(x, dtype=torch.float32)
mean_tx = tx.mean()

# variance - numpy manual
arr1 = np.array(x, dtype=np.float64)
var1 = np.sum( (arr1 - mean2)**2 ) / n
var2 = np.var(arr1, ddof=1)

# Does ddof(Degree of freedom) matters for large dataset?
N = 20000
arr2 = np.random.randint(-10, 50, size=N)
var1 = np.var(arr2, ddof=0)
var2 = np.var(arr2, ddof=1)
print(var1, var2)