import torch
import numpy as np
import pandas as pd


# rng = np.random.default_rng()
# n1 = rng.choice(100, (4, 3), replace=False)
# bool1 = n1 > 30
# print(bool1)
# print(~bool1)

# arr1 = np.tile(np.array([1, 2, 3, 4]), (10, 1))
# arr2 = 10 * np.tile(np.arange(1, 11), (4, 1)).T
# fakedata = arr1 + arr2
# print(arr2)

# n1 = np.tile(np.arange(1, 11) * 10, (4, 1)).T
# print(n1)

fruits = ["apple", "banana", "cherry"]
green, yellow, *rest = fruits
print(f'{green=}, {yellow=}, {rest=}')