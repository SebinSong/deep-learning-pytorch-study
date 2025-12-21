import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

iris = sns.load_dataset('iris')

# convert from pandas dataframe to tensor
data = torch.tensor(iris[iris.columns[:-1]].values).float() # df.values extracts numpy ndarray from dataframe

# transform species to number and define as labels
labels = torch.zeros(len(data), dtype=torch.long)

# 'setosa' is '0'
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2

# hyperparameters
learning_rate, epochs = 0.01, 1000

ANNiris = nn.Sequential(
  nn.Linear(4, 64), # input layer
  nn.ReLU(), # activation
  nn.Linear(64, 64), # hidden layer
  nn.ReLU(), # activation
  nn.Linear(64, 3) # output layer
)

# loss function
loss_func = nn.CrossEntropyLoss() # nn.CrossEntropyLoss is nn.LogSoftmax + nn.NLLLoss
losses = np.zeros(epochs)

# optimizer
optimizer = optim.SGD(ANNiris.parameters(), lr=learning_rate)

# train the model
ongoing_acc = []

for epoch in range(epochs):
  # forward pass
  y_hat = ANNiris(data)

  # compute the loss
  loss = loss_func(y_hat, labels)
  losses[epoch] = loss.item()

  # backprop
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # compute accuracy
  matches = torch.argmax(y_hat, axis=1) == labels
  matches_numeric = matches.float()
  accuracy_pct = 100 * torch.mean(matches_numeric)
  ongoing_acc.append(accuracy_pct)

# final prediction
predictions = ANNiris(data)

predlabels = torch.argmax(predictions, axis=1)
totalacc = 100 * torch.mean((predlabels == labels).float())

print(f'- totalacc: {totalacc}')

# plot the result
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(losses)
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_title('Losses')

ax[1].plot(ongoing_acc)
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_title('Accuracy graph')

plt.show()
