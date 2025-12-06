import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Create data
nPerClust = 100
blur = 1 # basically just a std - defines how spread the data are around the center coordinate.

A = [1, 1] # dataset 'a' is centered around (1, 1) - Cluster A
B = [5, 1] # dataset 'b' is centered around (5, 1) - Cluster B

a = [
  A[0] + np.random.randn(nPerClust) * blur,
  A[1] + np.random.randn(nPerClust) * blur
]

b = [
  B[0] + np.random.randn(nPerClust) * blur,
  B[1] + np.random.randn(nPerClust) * blur
]

# true labels
labels_np = np.vstack(
  (np.zeros((nPerClust, 1)), np.ones((nPerClust, 1)))
)

# concatanate into a matrix
data_np = np.hstack((a, b)).T

# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

print(data.shape)
print(labels.shape)

fig = plt.figure(figsize=(5, 5))
plt.plot(data[np.where(labels==0)[0], 0], data[np.where(labels==0)[0], 1], 'bs')
plt.plot(data[np.where(labels==1)[0], 0], data[np.where(labels==1)[0], 1], 'ko')
plt.title('The qwerties!')
plt.xlabel('qwerty dimension 1')
plt.xlabel('qwerty dimension 2')
plt.show()

# Build the model
ANNclassify = nn.Sequential(
  nn.Linear(2, 16),
  nn.ReLU(),
  nn.Linear(16, 1),
  nn.Sigmoid()
)

# Other model features
learning_rate = 0.02
num_epochs = 1000

# loss function 
lossFun = nn.BCELoss()

# optimizer
optimizer = torch.optim.SGD(ANNclassify.parameters(), lr=learning_rate)

losses = np.zeros(num_epochs)

for epoch_i in range(num_epochs):
  # forward pass
  y_hat = ANNclassify(data)

  # compute loss
  loss = lossFun(y_hat, labels)
  losses[epoch_i] = loss

  # backprop
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

final_predictions = ANNclassify(data)
predlabels = final_predictions > 0.5

# find errors
misclassified = np.where(predlabels != labels)[0]

# total accuracy
totalacc = 100 - 100 * len(misclassified) / (2 * nPerClust)

print(misclassified)
print(totalacc)

# print(final_predictions)
# print(predlabels)

# plt.plot(losses, 'o', markerfacecolor='w', linewidth=0.1)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()