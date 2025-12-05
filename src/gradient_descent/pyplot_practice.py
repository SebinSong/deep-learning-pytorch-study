import matplotlib.pyplot as plt
import numpy as np

# names = ['group_a', 'group_b', 'group_c']
# values = [15, 35, 50]

# plt.figure(figsize=(9, 3))
# plt.subplot(131)
# plt.bar(names, values)
# plt.subplot(132)
# plt.scatter(names, values)
# plt.subplot(133)
# plt.plot(names, values, 'r-o')
# plt.show()

# plt.figure(1)
# plt.subplot(2, 1, 1)
# plt.plot([1, 2, 3], 'r-')
# plt.subplot(2, 1, 2)
# plt.plot([4, 5, 6], 'b^:')

# plt.figure(2)
# plt.plot([4, 5, 6], 'gD-')

# plt.figure(1)
# plt.subplot(2, 1, 1)
# plt.title('Easy as 1, 2, 3')

# plt.show()

# lst = list(range(0, 10, 2))
# print(lst)
# nd_arr = np.array(lst)
# print(nd_arr)

# ndarr = np.linspace(0, 10, 100).reshape(50, 2)
# filtered = ndarr[:, 0]
# print(filtered)

# arr1 = np.random.rand(2)
# arr2 = np.random.rand(1, 2)

# b_arr1 = np.ones((3, 2)) + arr1
# print(arr1)
# print(b_arr1)

data = np.random.rand(10, 10)
plt.imshow(data, extent=[0, 100, 0, 100])
plt.show()
