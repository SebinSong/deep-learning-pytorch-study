import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

iris = sns.load_dataset('iris')

data = torch.tensor( iris[iris.columns[:-1]].values ).float()
labels = torch.zeros(len(data), dtype=torch.long)

labels[iris['species'] == 'versicolor'] = 1
labels[iris['species'] == 'virginica'] = 2

fakedata = np.tile( np.array([1, 2, 3, 4]), (10, 1) ) + np.tile( 10 * np.arange(1, 11), (4, 1)).T
fakelabels = (np.arange(10) > 4).astype(int)

train_data, test_data, train_labels, test_labels = train_test_split(fakedata, fakelabels, test_size=0.2)

# convert these into pytorch datasets
train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
test_dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_labels))

train_loader = DataLoader(train_dataset, batch_size=3)
test_loader = DataLoader(test_dataset)

print('TRAIN DATA:')
for batch, label in train_loader:
  print(f'{batch=}, {label=}')
# dataset = TensorDataset(torch.tensor(fakedata), torch.tensor(fakelabels))
# fakedataLdr = DataLoader(dataset, shuffle=True)

# TODO: split the real-data using DataLoader!
