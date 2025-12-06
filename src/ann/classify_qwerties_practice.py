import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

clusterSize = 200
blur = 1.5

# Cluster center point
A = [2, 3]
B = [7, 7]

a = [
  A[0] + np.random.randn(clusterSize) * blur,
  A[1] + np.random.randn(clusterSize) * blur
]

b = [
  B[0] + np.random.randn(clusterSize) * blur,
  B[1] + np.random.randn(clusterSize) * blur
]

labels_np = np.vstack([
  np.zeros((clusterSize, 1)), # label for A cluster is '0'
  np.ones((clusterSize, 1)) # label for B cluster is '1'
])

data_np = np.hstack([a, b]).T

labels = torch.tensor(labels_np, dtype=torch.float32)
data = torch.tensor(data_np, dtype=torch.float32)

# build a model
ANNclassify = nn.Sequential(
  nn.Linear(2, 16),
  nn.ReLU(),
  nn.Linear(16, 8),
  nn.ReLU(),
  nn.Linear(8, 1)
)

# other parameters
learning_rate = 0.01
num_epochs = 700

# loss function
loss_fun = nn.BCEWithLogitsLoss()
losses = np.zeros((num_epochs, 1))

# optimizer
optimizer = optim.Adam(ANNclassify.parameters(), lr=learning_rate)

# train the model
for epoch in range(num_epochs):
  # forward pass
  y_hat = ANNclassify(data)

  # compute the loss
  loss = loss_fun(y_hat, labels)
  losses[epoch] = loss.item()

  # backward pass
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# final prediction
y_prediction = ANNclassify(data)
pred_labels = y_prediction > 0.5
missclassified_idxs = np.where(pred_labels != labels)[0]
model_accuracy_score = 1 - len(missclassified_idxs) / (clusterSize * 2)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# left graph
ax[0].plot(
  data[missclassified_idxs, 0],
  data[missclassified_idxs, 1],
  'rx',
  markersize=12,
  markeredgewidth=2
)

ax[0].plot(
  data[np.where(labels==0)[0], 0],
  data[np.where(labels==0)[0], 1],
  'bs'
)

ax[0].plot(
  data[np.where(labels==1)[0], 0],
  data[np.where(labels==1)[0], 1],
  'yo'
)

ax[0].set_xlabel('feature 1')
ax[0].set_ylabel('feature 2')
ax[0].legend(['Misclassified', 'cluster A', 'cluster B'])
ax[0].set_title(f'{model_accuracy_score:.2%}% accurate')


ax[1].plot(losses, 'bo', markerfacecolor='w')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('loss')

plt.show()
