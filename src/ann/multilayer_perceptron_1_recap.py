import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plot_data = True
with_multiple_lrs = True

nPerClust = 100
blur = 1.375
num_epochs = 2000

A = [1, 3] # center coordinate (x, y) of cluster A
B = [1, -2] # center coordinate (x, y) of cluster B

label_names = {
  'A': 0,
  'B': 1
}

np.random.seed(70)

# generate dummy data
a = [
  A[0] + np.random.randn(nPerClust) * blur,
  A[1] + np.random.randn(nPerClust) * blur
]

b = [
  B[0] + np.random.randn(nPerClust) * blur,
  B[1] + np.random.randn(nPerClust) * blur
]

data_np = np.hstack([a, b]).T

# create true labels
labels_np = np.vstack([
  np.full((nPerClust, 1), label_names['A']),
  np.full((nPerClust, 1), label_names['B'])
])

labels_true = torch.from_numpy(labels_np).float()
data = torch.from_numpy(data_np).float()

# functions to build/train the model

def createANNModel(learningRate):
  # define model architecture
  ANNclassify = nn.Sequential(
    nn.Linear(2, 16), # input layer
    # nn.ReLU(), # activation unit
    nn.Linear(16, 1), # hidden layer,
    # nn.ReLU(), # activation unit
    nn.Linear(1, 1),
    nn.Sigmoid()
  )

  # loss function
  loss_func = nn.BCELoss()

  # optimizer
  optimizer = optim.SGD(ANNclassify.parameters(), lr=learningRate)

  # model output
  return ANNclassify, loss_func, optimizer

def trainTheModel (ANNModel, loss_func, optimizer):
  # initialize losses
  losses = torch.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    # forward propagation
    yHat = ANNModel(data)

    # compute the loss
    loss = loss_func(yHat, labels_true)
    losses[epoch_i] = loss

    # back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # final forward propagation
  predictions = ANNModel(data)
  compared = torch.where(predictions > 0.5, 1, 0).type(torch.float32) == labels_true

  totalacc = compared.float().mean()

  return losses, predictions, totalacc

if with_multiple_lrs:
  num_lr = 20
  learning_rates = np.linspace(0.001, 0.1, num_lr)
  model_accuracies = np.zeros((num_lr, 2), dtype=np.float32)

  for (lr_i, lr) in enumerate(learning_rates):
    # build the model
    model, loss_func, optimizer = createANNModel(lr)

    # run the model
    losses, predictions, totalacc = trainTheModel(model, loss_func, optimizer)

    model_accuracies[lr_i, :] = [lr, totalacc]
    print(f'iteration #{lr_i + 1}. learning-rate: {float(lr):>8.4f}, accuracy: {float(totalacc):>8.4f}')

  if plot_data:
    plt.plot(
      model_accuracies[:, 0],
      model_accuracies[:, 1],
      'bs--',
      markerfacecolor='w',
      markersize=10
    )
    plt.xlabel('learning rates')
    plt.ylabel('model accuracies')
    plt.title('Learning rate vs accuracy')
    plt.show()

else : # learning rate is fixed to 0.01

  # prepare the model
  ANNclassify, loss_func, optimizer = createANNModel(0.01)

  # run the model
  losses, predictions, totalacc = trainTheModel(ANNclassify, loss_func, optimizer)
  predicted_labels = torch.where(predictions > 0.5, 1, 0).type(torch.float32)
  incorrect_pred = data[torch.where(predicted_labels != labels_true)[0], :]

  if plot_data:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(
      incorrect_pred[:, 0],
      incorrect_pred[:, 1],
      'rx',
      markersize=12,
      markeredgewidth=2,
      label='Incorrect guess'
    )

    ax[0].plot(
      data[np.where(labels_true==0)[0], 0],
      data[np.where(labels_true==0)[0], 1],
      'bs',
      label='cluster A'
    )

    ax[0].plot(
      data[np.where(labels_true==1)[0], 0],
      data[np.where(labels_true==1)[0], 1],
      'yo',
      label='cluster B'
    )

    ax[0].set_title(f'Prediction ccuracy: {totalacc:.3%}')
    ax[0].set_xlabel('feature 1')
    ax[0].set_ylabel('feature 2')
    ax[0].legend()

    ax[1].plot(losses.detach(), 'bo', markerfacecolor='w')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_title(f'Final loss: {losses[losses.numel() - 1]}')

    plt.show()