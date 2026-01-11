import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

iris_dataset = sns.load_dataset('iris')
data = torch.tensor(iris_dataset[iris_dataset.columns[:-1]].values, dtype=torch.float32)
labels = torch.zeros(len(data), dtype=torch.long)
labels[iris_dataset['species'] == 'versicolor'] = 1
labels[iris_dataset['species'] == 'virginica'] = 2

def createNewModel(n_input, n_output, n_hidden_units = 64):
  ANNiris = nn.Sequential(
    nn.Linear(n_input, n_hidden_units), # input layer -> hidden layer
    nn.ReLU(),
    nn.Linear(n_hidden_units, n_hidden_units), # hidden0 -> hidden1
    nn.ReLU(),
    nn.Linear(n_hidden_units, n_output)
  )

  # loss_function
  loss_func = nn.CrossEntropyLoss()

  # optimizer
  optimizer = torch.optim.SGD(ANNiris.parameters(), lr=0.01)

  return ANNiris, loss_func, optimizer

# hyperparameters
num_epochs = 300

def train_the_model(model, loss_func, optimizer, train_prop = .8):
  x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=train_prop)

  losses = np.zeros(num_epochs)
  train_acc = np.zeros(num_epochs)
  test_acc = np.zeros(num_epochs)

  # loop over the epochs
  for epoch_i in range(num_epochs):
    # forward prop
    y_hat = model(x_train)
    loss = loss_func(y_hat, y_train)
    losses[epoch_i] = loss.item()

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # compute the training accuracy
    train_acc[epoch_i] = 100 * torch.mean( (torch.argmax(y_hat, axis=1) == y_train).float() ).item()

    # test accuracy
    pred_labels = torch.argmax(model(x_test), axis=1)
    test_acc[epoch_i] = 100 * torch.mean( (pred_labels == y_test).float() ).item()

  return train_acc, test_acc, losses

n_input = data.shape[1]
n_output = torch.unique(labels).numel()

trainset_sizes = np.linspace(0.2, 0.95, 10)
all_train_acc = np.zeros((len(trainset_sizes), num_epochs))
all_test_acc = np.zeros((len(trainset_sizes), num_epochs))

for row_i, trainset_size in enumerate(trainset_sizes):
  net, loss_func, optimizer = createNewModel(
    n_input=n_input,
    n_output=n_output,
    n_hidden_units=64
  )

  train_accuracies, test_accuracies, _ = train_the_model(net, loss_func, optimizer, trainset_size)
  all_train_acc[row_i, :] = train_accuracies
  all_test_acc[row_i, :] = test_accuracies

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].imshow(
  all_train_acc,
  aspect='auto',
  vmin=50, vmax=90,
  extent=[0, num_epochs, trainset_sizes[-1], trainset_sizes[0]]
)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Training size proportion')
ax[0].set_title('Training accuracy')

p = ax[1].imshow(
  all_train_acc,
  aspect='auto',
  vmin=50, vmax=90,
  extent=[0, num_epochs, trainset_sizes[-1], trainset_sizes[0]]
)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Training size proportion')
ax[1].set_title('Test accuracy')
fig.colorbar(p, ax=ax[1])

plt.show()
