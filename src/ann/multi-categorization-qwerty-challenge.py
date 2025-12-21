import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# some constants
show_result_plots = True
clusterSize, blur = 50, 1.2

centers = {
  'A': { 'x': 2, 'y': 3 },
  'B': { 'x': 4, 'y': 6 },
  'C': { 'x': 6, 'y': 1 }
}

cluster_labels = {
  'A': 0,
  'B': 1,
  'C': 2
}

## 1. create cluster data

def createCluster(name):
  center = centers[name]
  return np.array([
    np.random.randn(clusterSize) * blur + center['x'],
    np.random.randn(clusterSize) * blur + center['y']
  ], dtype=np.float32).T

a_np = createCluster('A')
b_np = createCluster('B')
c_np = createCluster('C')
labels_np = np.hstack([
  np.full(clusterSize, cluster_labels['A']),
  np.full(clusterSize, cluster_labels['B']),
  np.full(clusterSize, cluster_labels['C'])
]).T

data = torch.tensor(np.vstack([a_np, b_np, c_np]))
labels = torch.tensor(labels_np, dtype=torch.long)

## 2. Create model architecture

# hyperparameters
hidden_layer_nodes, learning_rate, epochs = 8, 0.01, 3000

# other variables
losses = np.zeros(epochs)
accuracies = np.zeros(epochs)

# model
ANNclassify = nn.Sequential(
  nn.Linear(2, hidden_layer_nodes),
  nn.ReLU(),
  nn.Linear(hidden_layer_nodes, 3)
)

# loss function
loss_func = nn.CrossEntropyLoss() # nn.CrossEntropyLoss is nn.LogSoftmax + nn.NLLLoss

# optimizer
optimizer = optim.SGD(ANNclassify.parameters(), lr=learning_rate)

for epoch in range(epochs):
  # forward prop
  y_hat = ANNclassify(data)

  # compute the loss
  loss = loss_func(y_hat, labels)
  losses[epoch] = loss.item()

  # backpropagation
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # compute accuracy
  matches = torch.argmax(y_hat, axis=1) == labels
  accuracy = torch.mean(matches.float())
  accuracies[epoch] = accuracy

  if (epoch % 20 == 0):
    print(f'epoch[{epoch}] acc - {accuracy:>7.3f}  |  loss - {loss.item():>7.3f}')

# final prediction
predictions = ANNclassify(data)

pred_labels = torch.argmax(predictions, axis=1)
total_acc = torch.mean((pred_labels == labels).float())

print(f'final accuracy: {total_acc:.3%}')

# fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# ax[0].plot(losses)
# ax[0].set_ylabel('loss')
# ax[0].set_xlabel('epoch')
# ax[0].set_title('Losses')

# ax[1].plot(accuracies)
# ax[1].set_ylabel('accuracy')
# ax[1].set_xlabel('epoch')
# ax[1].set_title('Accuracies')

# plt.show()

predictions_sm = nn.functional.softmax(predictions, dim=1).detach().numpy()
plt.figure(figsize=(10,4))
plt.plot(
  predictions_sm[:, 0],
  'bo',
  markerfacecolor='w',
  label='cluster A'
)
plt.plot(
  predictions_sm[:, 1],
  'rs',
  markerfacecolor='w',
  label='cluster B'
)
plt.plot(
  predictions_sm[:, 2],
  'y^',
  markerfacecolor='w',
  label='cluster C'
)
plt.xlabel('Sample number')
plt.ylabel('Probability')
plt.legend()
plt.show()
