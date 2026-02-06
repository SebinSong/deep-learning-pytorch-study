import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
species_col = iris['species']
sample_size = len(iris)

# data
data = torch.tensor(iris[iris.columns[:-1]].values, dtype=torch.float32)

# label
labels = torch.zeros(sample_size, dtype=torch.long)
labels[(species_col == 'versicolor').values] = 1
labels[(species_col== 'virginica').values] = 2

# in/out feature spaces
D_in = data.shape[1]
D_out = species_col.nunique()

# dataset
dataset = TensorDataset(data, labels)

# sizes / split
train_size = int(sample_size * 0.8)
test_size = sample_size - train_size

train_ds, test_ds = random_split(dataset, [train_size, test_size])

# data loaders
train_dlr = DataLoader(train_ds, batch_size=12, shuffle=True)
test_dlr = DataLoader(test_ds, batch_size=test_size, shuffle=False)

def createNewModel(n_hidden_unit=12, learning_rate=0.1):
  # model architecture
  ANNiris = nn.Sequential(
    nn.Linear(D_in, n_hidden_unit), # input -> hidden0
    nn.ReLU(),
    nn.Linear(n_hidden_unit, n_hidden_unit), # hidden0 -> hidden1
    nn.ReLU(),
    nn.Linear(n_hidden_unit, D_out)
  )

  loss_func = nn.CrossEntropyLoss()

  optimizer = torch.optim.SGD(ANNiris.parameters(), lr=learning_rate)

  return ANNiris, loss_func, optimizer

num_epochs = 500

def train_the_model(model, loss_func, optimizer):
  # init accuracies
  train_acc = np.zeros(num_epochs)
  test_acc = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    # loop over training data batches

    batch_acc = []
    for batch_i, (X, y) in enumerate(train_dlr):
      # forward pass and loss
      y_hat = model(X)

      # compute the loss
      loss = loss_func(y_hat, y)

      # back-prop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      acc = 100 * torch.mean( (torch.argmax(y_hat, axis=1) == y).float() ).item()
      batch_acc.append(acc)

    # train accuracy
    train_acc[epoch_i] = np.array(batch_acc).mean()
    
    # test accuracy
    X, y = next(iter(test_dlr))
    pred_labels = torch.argmax( model(X), axis=1 )
    test_acc[epoch_i] = 100 * torch.mean( (pred_labels == y).float() ).item()

  return train_acc, test_acc

model, loss_func, optimizer = createNewModel(12, 0.02)
train_acc, test_acc = train_the_model(model, loss_func, optimizer)

# visualize the results
fig = plt.figure(figsize=(10, 5))

plt.plot(train_acc, 'ro-', label='train acc')
plt.plot(test_acc, 'bs-', label='test acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()
