import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plot_data = False
nPerClust = 100
blur = 1
num_epochs = 3000

A = [1, 3] # center (x, y) coordinates of cluster A
B = [1, -2] # center (x, y) coordinates of cluster B

label_names = {
  'A': 0,
  'B': 1
}

# generate data
a = [
  A[0] + np.random.randn(nPerClust) * blur, A[1] + np.random.randn(nPerClust) * blur
]
b = [
  B[0] + np.random.randn(nPerClust) * blur, B[1] + np.random.randn(nPerClust) * blur
]

# true labels - cluster a has label 0, and b has label 1
labels_np = np.vstack((
  np.full((nPerClust, 1), label_names['A']),
  np.full((nPerClust, 1), label_names['B'])
))

# concatanate into a matrix
data_np = np.hstack([a, b]).T

# convert to a pytorch tensor
labels = torch.tensor(labels_np).float()
data = torch.tensor(data_np).float()

# functions to buil and train model

def createANNModel(learningRate):
  # model architecture
  ANNclassify = nn.Sequential(
    nn.Linear(2, 16), # input layer
    nn.ReLU(), # activation unit
    nn.Linear(16, 1), # hidden layer
    nn.ReLU(), # activation unit
    nn.Linear(1, 1) # output unit
  )

  # loss function
  loss_func = nn.BCEWithLogitsLoss()

  # optimizer - the unit where model-learninng happens
  optimizer = optim.SGD(ANNclassify.parameters(), lr=learningRate)

  # model output
  return ANNclassify, loss_func, optimizer

def trainTheModel (ANNModel, loss_func, optimizer):
  # initialize losses
  losses = torch.zeros(num_epochs)

  # loop over epochs
  for epoch_i in range(num_epochs):
    # forward propagation
    yHat = ANNModel(data)

    # compute the loss
    loss = loss_func(yHat, labels)
    losses[epoch_i] = loss.item()

    # back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # final forward propagation
  predictions = ANNModel(data)
  compared = torch.where(predictions > 0, 1, 0).type(torch.float32) == labels

  # compute the prediction accuracy
  totalacc = compared.float().mean()

  return losses, predictions, totalacc

# create everything
ANNclassify, loss_func, optimizer = createANNModel(0.01)

# run it
losses, predictions, totalacc = trainTheModel(ANNclassify, loss_func, optimizer)

# show the data
if plot_data:
  fig, ax = plt.figure(1, 2, figsize=(10, 5))

  ax[0].plot(
    data[np.where(labels==0)[0], 0],
    data[np.where(labels==0)[0], 1],
    'bs',
    label='cluster A'
  )
  ax[0].plot(
    data[np.where(labels==1)[0], 0],
    data[np.where(labels==1)[0], 1],
    'ro',
    label='cluster B'
  )

  validated = torch.where(predictions > 0, label_names['A'], label_names['B']) == labels

  ax[0].legend()
  ax[0].set_title('The qwerties!')
  ax[0].set_xlabel('qwerty dimension 1')
  ax[0].set_ylabel('qwerty dimension 2')


# show the losses
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(losses, 'o', markerfacecolor='w', linewidth=.1)
ax.set_xlabel('Epoch'), plt.ylabel('Loss')
ax.set_title(f'Accuracy: {totalacc:.3%}')
plt.show()
