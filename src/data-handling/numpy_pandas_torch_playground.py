import numpy as np

rand_int = np.random.randint(1, 10, (3, 3))

arr1 = np.arange(30).reshape((5, 3, 2))
print(arr1)
flattened = arr1.flatten()
print(flattened)
