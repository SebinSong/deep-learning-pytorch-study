import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

F = nn.functional

# create data

plot_data = True
n_per_cluster = 50
blur = 1

# cluster center
A = [1, 1]
B = [4, 2]

a_np = np.hstack((
  A[0] + np.random.randn(n_per_cluster, 1) * blur,
  A[1] + np.random.randn(n_per_cluster, 1) * blur
))

b_np = np.hstack((
  B[0] + np.random.randn(n_per_cluster, 1) * blur,
  B[1] + np.random.randn(n_per_cluster, 1) * blur
))

data = torch.tensor(
  np.vstack((a_np, b_np)),
  dtype=torch.float32
)
labels = torch.tensor(
  np.vstack((
    np.zeros((a_np.shape[0], 1)),
    np.ones((a_np.shape[0], 1))
  )),
  dtype=torch.float32
)

# if plot_data:
#   fig = plt.figure(figsize=(5, 5))

#   plt.plot(
#     data[torch.where(labels == 0)[0], 0],
#     data[torch.where(labels == 0)[0], 1],
#     'bs'
#   )

#   plt.plot(
#     data[torch.where(labels == 1)[0], 0],
#     data[torch.where(labels == 1)[0], 1],
#     'ko'
#   )

#   plt.title('Qwerties')
#   plt.xlabel('Qwerty dim 1')
#   plt.ylabel('Qwerty dim 2')
#   plt.show()

class theClass4ANN(nn.Module):
  def __init__(self):
    super().__init__()

    self.input = nn.Linear(2, 16)
    self.output = nn.Linear(16, 1)

  def forward (self, x):
    # forward pass
    x = self.input(x)

    x = F.relu(x)

    x = self.output(x)
    x = torch.sigmoid(x)

    return x

lealrning_rate, num_epochs = 0.01, 2000
losses = np.zeros(num_epochs)

ANNclassify = theClass4ANN()
loss_func = nn.BCELoss()
optimizer = torch.optim.SGD(ANNclassify.parameters(), lr=lealrning_rate)

for epoch in range(num_epochs):
  y_hat = ANNclassify(data)

  loss = loss_func(y_hat, labels)
  losses[epoch] = loss.item()

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

if plot_data:
  fig = plt.figure(figsize=(10, 5))
  plt.plot(losses, 'bs-', markerfacecolor='w')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Loss change')
  plt.show()
