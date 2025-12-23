import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

F = nn.functional

# create data

n_per_clust = 100
blur = 1

A = [1, 1]
B = [5, 1]

a = [
  A[0] + np.random.randn(n_per_clust) * blur,
  A[1] + np.random.randn(n_per_clust) * blur
]
b = [
  B[0] + np.random.randn(n_per_clust) * blur,
  B[1] + np.random.randn(n_per_clust) * blur
]

data_np = np.hstack([a, b]).T
labels_np = np.vstack([
  np.zeros((n_per_clust, 1)),
  np.ones((n_per_clust, 1))
])

data = torch.tensor(data_np, dtype=torch.float32)
labels = torch.tensor(labels_np, dtype=torch.float32)

# fig = plt.figure(figsize=(5, 5))

# plt.plot(
#   data[np.where(labels == 0)[0], 0],
#   data[np.where(labels == 0)[0], 1],
#   'bs'
# )

# plt.plot(
#   data[np.where(labels == 1)[0], 0],
#   data[np.where(labels == 1)[0], 1],
#   'ko'
# )
# plt.title('qwerties')
# plt.xlabel('qwerty dim 1')
# plt.ylabel('qwerty dim 2')
# plt.show()

print(type(data_np), data_np.shape)
print(type(data), data.shape)

class theClass4ANN(nn.Module):
  def __init__(self):
    super().__init__()

    self.input = nn.Linear(2, 1)
    self.output = nn.Linear(1, 1)

  def forward(self, x):
    x = self.input(x)
    x = F.relu(x)
    x = self.output(x)
    return torch.sigmoid(x)
  
learning_rate, num_epochs = 0.01, 1000
losses = np.zeros(num_epochs)

ANNClassify = theClass4ANN()
loss_func = nn.BCELoss()
optimizer = torch.optim.SGD(ANNClassify.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  y_hat = ANNClassify(data)

  loss = loss_func(y_hat, labels)
  losses[epoch] = loss.item()

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

plt.plot(losses, 'bo-', markerfacecolor='w')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()
