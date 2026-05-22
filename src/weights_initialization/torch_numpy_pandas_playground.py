import numpy as np

data = [1, 2, 1, 5, 6, 6]
counts, edges = np.histogram(data, 3)
print(counts)
print(edges)
