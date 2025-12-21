import seaborn as sns
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

# hyperparameters
n_hidden_max = 124
learning_rate, num_epochs = 0.01, 150

data = torch.tensor(iris[iris.columns[:-1]].values).float()
labels = torch.zeros(len(data), dtype=torch.long)
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2

def create_iris_model(n_hidden):
  ANNiris = nn.Sequential(
    nn.Linear(4, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, 3)
  )

  loss_func = nn.CrossEntropyLoss()

  optimizer = torch.optim.SGD(ANNiris.parameters(), lr=0.01)

  return ANNiris, loss_func, optimizer

def train_the_model(model, loss_func, optimizer):
  for _ in range(num_epochs):
    # forward pass
    y_hat = model(data)

    # compute the loss
    loss = loss_func(y_hat, labels)

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  final_prediction_sm = nn.functional.softmax(model(data), dim=1)
  pred_labels = torch.argmax(final_prediction_sm, dim=1)
  accuracy = 100 * torch.mean((pred_labels == labels).float())

  return accuracy


nhidden_acc_pairs = np.vstack([
  np.arange(1, n_hidden_max + 1),
  np.zeros(n_hidden_max)
]).T

for pair in nhidden_acc_pairs:
  nhidden = int(pair[0])

  model, loss_func, optimizer = create_iris_model(nhidden)
  final_acc = train_the_model(model, loss_func, optimizer)

  pair[1] = final_acc
  print(f'n_hidden[{nhidden}] - accuracy: {final_acc:.3f}%')

plt.figure(figsize=(12,6))
plt.plot(
  nhidden_acc_pairs[:, 0][[0, -1]], [33, 33], '--', color=[.8, .8, .8]
)
plt.plot(
  nhidden_acc_pairs[:, 0][[0, -1]], [66, 66], '--', color=[.8, .8, .8]
)
plt.plot(
  nhidden_acc_pairs[:, 0],
  nhidden_acc_pairs[:, 1],
  'ko-',
  markerfacecolor='w',
  markersize=9
)
plt.xlabel('hidden unit')
plt.ylabel('model accuracy')
plt.title('Model accuracy vs N of hidden unit')
plt.show()