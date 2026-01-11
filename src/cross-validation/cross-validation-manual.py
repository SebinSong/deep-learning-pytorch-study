import torch
import torch.nn as nn
import numpy as np

import seaborn as sns

iris_dataset = sns.load_dataset('iris')

# convert from pandas dataframe to tensor
data = torch.tensor( iris_dataset[iris_dataset.columns[:-1]].values ).float()
labels = torch.zeros(len(iris_dataset), dtype=torch.long)

# labels : 'setosa' = 0, 'versicolor' = 1, 'virginica' = 2
labels[iris_dataset['species'] == 'versicolor'] = 1
labels[iris_dataset['species'] == 'virginica'] = 2

# (no hold-out set here)

# Number of training set
prop_training = .8 # in proportion, not percent
n_training = int(len(labels) * prop_training)

# trainingset indices
rng = np.random.default_rng()
indices_train = rng.choice(len(labels), size=n_training, replace=False)

# iniitialize a boolean vector to select data and labels
train_test_bool = np.zeros(len(labels), dtype=bool)
train_test_bool[indices_train] = True
data_trainset = data[train_test_bool, :]
data_testset = data[~train_test_bool, :]

# create the ANN model

class ANNiris(nn.Module):
  def __init__(self):
    super().__init__()

    self.pipe = nn.Sequential(
      nn.Linear(4, 64), # input -> hidden0
      nn.ReLU(),
      nn.Linear(64, 64), # hidden0 -> hidden1
      nn.ReLU(),
      nn.Linear(64, 3) # hidden1 -> output
    )

  def forward(self, x):
    return self.pipe(x)

net = ANNiris()

# loss function
loss_func = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=.01)

# train the model

num_epochs = 1000

losses = torch.zeros(num_epochs)
ongoing_acc = []

for epoch_i in range(num_epochs):
  y_hat = net(data_trainset)

  # compute accuracy
  ongoing_acc.append(
    100 * torch.mean( (torch.argmax(y_hat, dim=1) == labels[train_test_bool]).float() )
  )

  # compute the loss
  loss = loss_func(y_hat, labels[train_test_bool])
  losses[epoch_i] = loss.item()

  # backprop
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# compute train and test accuracies
predictions = net(data_trainset)
train_accuracy = 100 * torch.mean( (torch.argmax(predictions, dim=1) == labels[train_test_bool]).float() )

# final forwardpass using test data
predictions = net(data_testset)
test_accuracy = 100 * torch.mean(
  (torch.argmax(predictions, dim=1) == labels[~train_test_bool]).float()
)

# report the accuracies
print(f'final train accuracy: {train_accuracy:.3f}%')
print(f'final test accuracy: {test_accuracy:.3f}%')
