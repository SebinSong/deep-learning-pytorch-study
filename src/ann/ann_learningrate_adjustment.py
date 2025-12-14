import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

clusterSize = 100
blur = 1
num_epochs = 1000

def prepare_data():
  center = {
    'A': (1, 1),
    'B': (5, 1)
  }

  a = np.array([
    center['A'][0] + np.random.randn(clusterSize) * blur,
    center['A'][1] + np.random.randn(clusterSize) * blur
  ], dtype=np.float32).T
  b = np.array([
    center['B'][0] + np.random.randn(clusterSize) * blur,
    center['B'][1] + np.random.randn(clusterSize) * blur
  ], dtype=np.float32).T

  at = torch.from_numpy(a)
  bt = torch.from_numpy(b)
  data = torch.vstack((at, bt))
  labels = torch.vstack((
    torch.zeros(at.shape[0], 1),
    torch.ones(bt.shape[0], 1)
  ))

  return data, labels

def createANNmodel(learningRate):
  # define the model architecture
  ANNclassify = nn.Sequential(
    nn.Linear(2, 1),
    nn.ReLU(),
    nn.Linear(1, 1)
  )

  # loss function
  loss_func = nn.BCEWithLogitsLoss()

  # optimizer
  optimizer = optim.SGD(ANNclassify.parameters(), lr=learningRate)

  # model output
  return ANNclassify, loss_func, optimizer

data, labels_true = prepare_data()

def trainTheModel(ANNModel, loss_func, optimizer):
  # initialize losses
  losses = torch.zeros(num_epochs)

  # training loop - loop over epochs
  for epoch in range(num_epochs):
    # forward prop
    yHat = ANNModel(data)

    # compute loss
    loss = loss_func(yHat, labels_true)
    losses[epoch] = loss.item()

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # final forward-prop
  y_prediction = ANNModel(data)

  # compute the predictions and report accuracy
  totalacc = 100 * torch.mean(
    ((y_prediction > 0) == labels_true).float()
  )

  return losses, y_prediction, totalacc

learning_rates = np.linspace(0.001, 0.1, 30)
acc_by_lr = np.zeros((len(learning_rates), 2))
all_losses = np.zeros((len(learning_rates), num_epochs))

for lr_idx, lr in enumerate(learning_rates):
  ANNclassify, loss_func, optimizer = createANNmodel(lr)
  losses, y_prediction, totalacc = trainTheModel(ANNclassify, loss_func, optimizer)

  all_losses[lr_idx, :] = losses
  acc_by_lr[lr_idx, 0] = lr
  acc_by_lr[lr_idx, 1] = totalacc.numpy()

print(acc_by_lr[:, 1] > 0.7)

model_total_performance = sum(acc_by_lr[:, 1] > 70) / len(learning_rates)
# plot the results

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(acc_by_lr[:, 0], acc_by_lr[:, 1], 's-')
ax[0].set_xlabel('Learning rate')
ax[0].set_ylabel('Accuracy')
ax[0].set_title(f'Percent of > 70% accuracy: {model_total_performance:.2%}')

ax[1].plot(all_losses.T)
ax[1].set_title('Losses by learning rate')
ax[1].set_xlabel('Epoch number')
ax[1].set_xlabel('Loss')
ax[1].legend()

plt.show()
