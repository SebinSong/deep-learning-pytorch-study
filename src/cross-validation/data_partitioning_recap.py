import numpy as np
from sklearn.model_selection import train_test_split
import torch

sample_size = 20
fakedata = np.tile(np.arange(1, 5), (sample_size, 1)) + np.tile(10 * np.arange(1, 1 + sample_size), (4, 1)).T
fakelabels = np.random.randint(0, 2, (sample_size,))

def split_using_sklearn():
  partitions = [.8, .1, .1] # train / dev / test

  train_data, rest_data, train_labels, rest_labels = train_test_split(fakedata, fakelabels, train_size=partitions[0])

  # split the rest data
  split_ratio = partitions[1] / np.sum(partitions[1:])
  devset_data, test_data, devset_labels, test_labels = \
    train_test_split(rest_data, rest_labels, train_size=split_ratio)
  
  return (
    (train_data, train_labels),
    (devset_data, devset_labels),
    (test_data, test_labels)
  )

train_pair, dev_pair, test_pair = split_using_sklearn()

print(test_pair)
