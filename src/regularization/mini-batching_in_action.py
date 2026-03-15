import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import seaborn as sns

iris = sns.load_dataset('iris')

# iris.plot(marker='o', linestyle='none', figsize=(12, 6))
# plt.xlabel('Sample number')
# plt.ylabel('Value')
# plt.show()

data = torch.tensor( iris[iris.columns[:-1]].values, dtype=torch.float32)
labels_mapping = { 'setosa': 0, 'versicolor': 1, 'virginica': 2 }
labels = torch.tensor(iris['species'].map(labels_mapping).values, dtype=torch.long)
dataset = TensorDataset(data, labels)

def split_data(dset, train_prop=.8, batch_size=16):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=test_size, shuffle=False)

  return train_loader, test_loader

train_loader, test_loader = split_data(dataset)

for X, y in train_loader:
  print(X.shape, y.shape)
  