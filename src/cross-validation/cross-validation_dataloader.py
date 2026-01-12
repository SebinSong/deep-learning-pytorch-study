import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

unique_count = lambda t: torch.unique(t).numel()

iris = sns.load_dataset('iris')

data = torch.tensor( iris[iris.columns[:-1]].values ).float()
labels = torch.zeros(len(data), dtype=torch.long)

labels[iris['species'] == 'versicolor'] = 1
labels[iris['species'] == 'virginica'] = 2

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

# Setting shuffle=True has the data reshuffled at every epoch.
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=12)
test_loader = DataLoader(test_dataset, batch_size=test_dataset.tensors[0].shape[0]) # shuffle option of DataLoader defaults to False.

def createNewModel(n_input, n_hidden_unit, n_output):
  ANNiris = nn.Sequential(
    nn.Linear(n_input, n_hidden_unit), # input -> hidden0
    nn.ReLU(),
    nn.Linear(n_hidden_unit, n_hidden_unit), # hidden0 -> hidden1
    nn.ReLU(),
    nn.Linear(n_hidden_unit, n_output) # hidden1 -> output
  )

  # loss function and an optimizer
  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(ANNiris.parameters(), lr=0.01)

  return ANNiris, loss_func, optimizer

num_epochs = 500

def trainTheModel (model, loss_func, optimizer):
  # init accuracy arrays
  train_acc = np.zeros(num_epochs)
  test_acc = np.zeros(num_epochs)

  # loop over epochs
  for epoch_i in range(num_epochs):
    batch_acc = np.zeros(len(train_loader))
    
    for i, pair in enumerate(train_loader):
      X, y = pair

      # forward prop
      y_hat = model(X)
      loss = loss_func(y_hat, y)

      # backward prop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      pred_labels = torch.argmax(y_hat, axis=1)
      batch_acc[i] = 100 * torch.mean( (pred_labels == y).float() )
    
    train_acc[epoch_i] = np.mean(batch_acc)

    # compute the test accuracy
    # NOTE: test_loader is an iterable. to use it with next() built-in python function, it must be
    #       converted to an iterator first (which is done by usinng iter() built-in function)
    X, y = next(iter(test_loader))
    pred_labels = torch.argmax(model(X), axis=1)
    test_acc[epoch_i] = 100 * torch.mean( (pred_labels == y).float() ).item()

  # function output
  return train_acc, test_acc

net, loss_func, optimizer = createNewModel(
  n_input=data.shape[1],
  n_hidden_unit=64,
  n_output=unique_count(labels)
)

train_acc, test_acc = trainTheModel(net, loss_func, optimizer)

fig = plt.figure(figsize=(10, 5))

plt.plot(train_acc, 'ro-')
plt.plot(test_acc, 'bs-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy %')
plt.legend(['Train', 'Test'])

plt.show()
