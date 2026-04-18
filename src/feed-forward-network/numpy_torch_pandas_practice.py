import numpy as np
import torch
import matplotlib.pyplot as plt

# t1 = torch.randint(0, 100, (5, 10))
# maxed = torch.max(t1, axis=1)
# maxed_idxs = maxed[1]
# argmaxed = torch.argmax(t1, dim=1)

# whered = np.where(maxed_idxs == argmaxed)

# print(t1)
# print(maxed_idxs)
# print(argmaxed)
# print(whered)

# testing np.where() with bool mask
# random_bool_mask = torch.randn(10) > 0
# np_whered_1 = np.where(random_bool_mask)
# torch_whered = torch.where(random_bool_mask)
# print(random_bool_mask)
# print(np_whered_1)
# print(torch_whered)

# t2 = torch.randint(0, 20, (10,))
# rand_int = torch.randint(0, 10, (1,)).item()
# print(rand_int)

# a = np.array([
#   [1, 2],
#   [3, 4]
# ])
# b = np.array([[5, 6]])
# concated1 = np.concatenate((a, b), axis=0)
# concated2 = np.concatenate((a, b.T), axis=1)
# print(a.shape, b.shape)
# print(concated1.shape)
# print(concated2)

# Example for np.histogram()
# data = [1.2, 2.3, 3.3, 3.1, 1.7, 3.4, 2.1, 1.25, 1.3]

# hist, bin_edges = np.histogram(data, bins=3, density=False)
# print('hist: ', hist)
# print('bin_edges: ', bin_edges)

# diffed = np.diff(bin_edges)
# print(diffed)

# plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black", align="edge")
# plt.show()

# np.diff() practice
# a1 = np.array([
#   [1, 2, 3],
#   [10, 20, 30],
#   [30, 60, 90]
# ])

# diffed_a1 = np.diff(a1, axis=0)
# diffed_a1_2 = np.diff(a1, axis=1)
# diffed_b = np.diff(a1[:, [0, 2]], axis=1)
# print(diffed_b)

# practice np.roll()

# a1 = np.array([1, 2, 3])
# b1 = np.array([3, 5, 7])
# c1 = np.vstack((a1, b1))
# c = a1 + b1

# d1 = np.roll(c1, 1, axis=0)
# d2 = np.roll(c1, 2, axis=1)
# print(c1)
# print(d1)
# print(d2)

# rand_num = np.random.randint(1, 10, 10)
# print(rand_num, type(rand_num))

# rand_idxs = np.random.permutation(100)
# print(rand_idxs[:10])


data = [1, 2, 1, 3, 4, 1, 2, 5]
max_n = np.max(data)
arr = np.arange(0, np.max(data) + 1)
uniq = np.unique(data)
uniq.sort()

np_data = np.array(data)
for n in np_data:
  print(n)


# hist, bin_edges = np.histogram(data, bins=arr)
# print(hist)
# print(bin_edges)
