import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import TensorDataset, DataLoader, random_split

iris = sns.load_dataset('iris')
species_col = iris['species']

# data
data = torch.tensor(
  iris[iris.columns[:-1]].values
).float()
labels_mapping = { 'setosa': 0, 'versicolor': 1, 'virginica': 2 }
labels = torch.tensor(
  species_col.map(labels_mapping).fillna(-1),
  dtype=torch.long
)
dataset = TensorDataset(data, labels)

# train parameters
D_in = data.shape[1]
D_out = torch.unique(labels).numel()
n_hidden_units = 64

# hyperparameters
num_epochs = 800

def split_dataset(dset, train_props=.8):
  sample_size = len(dset)
  train_size = int(sample_size * train_props)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_dlr = DataLoader(train_dset, batch_size=64, shuffle=True)
  test_dlr = DataLoader(test_dset, batch_size=test_size, shuffle=False)

  return train_dlr, test_dlr

def create_model(learning_rate=0.005):
  # model architecture
  model = nn.Sequential(
    nn.Linear(D_in, n_hidden_units),
    nn.ReLU(),
    nn.Linear(n_hidden_units, n_hidden_units),
    nn.ReLU(),
    nn.Linear(n_hidden_units, D_out)
  )

  # loss function
  loss_f = nn.CrossEntropyLoss()

  # optimizer
  optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

  return model, loss_f, optim

def train_model(model, loss_func, optimizer, train_dlr, test_dlr, /, L1lambda=0.01):
  train_acc = np.zeros(num_epochs)
  test_acc = np.zeros(num_epochs)
  losses = np.zeros(num_epochs)

  # count the total number of weights in the model
  nweights = sum(p.numel() for n, p in model.named_parameters() if 'bias' not in n)

  for epoch_i in range(num_epochs):
    model.train()

    batch_acc = []
    batch_losses = []

    for batch_x, batch_y in train_dlr:
      y_hat = model(batch_x)

      # standard loss
      loss = loss_func(y_hat, batch_y)

      # L1 regularization
      l1_sum = sum(p.abs().sum() for n, p in model.named_parameters() if 'bias' not in n)
      loss += (L1lambda / nweights) * l1_sum

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      curr_acc = 100 * torch.mean( (torch.argmax(y_hat, dim=1) == batch_y).float() )
      batch_acc.append( curr_acc.item() )
      batch_losses.append( loss.item() )

    train_acc[epoch_i] = np.mean(batch_acc)
    losses[epoch_i] = np.mean(batch_losses)

    # compute the test_accuracy
    model.eval()
    # Use no_grad to save memory / speed
    with torch.no_grad():
      test_x, test_y = next(iter(test_dlr))
      pred_test_labels = torch.argmax( model(test_x), dim=1 )
      test_acc[epoch_i] = 100 * torch.mean( (pred_test_labels == test_y).float() )

  return train_acc, test_acc, losses

train_dlr, test_dlr = split_dataset(dataset)
ANNiris, loss_func, optimizer = create_model()
train_acc, test_acc, losses = train_model(ANNiris, loss_func, optimizer, train_dlr, test_dlr, L1lambda=0.001)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].plot(losses, 'k^-')
ax[0].set_ylabel('Losses')
ax[0].set_xlabel('Epochs')
ax[0].set_title(f'Losses with L1_lambda {0.001}')

ax[1].plot(train_acc, 'ro-')
ax[1].plot(test_acc, 'bs-')
ax[1].set_title('Accuracies')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy (%)')
ax[1].legend(['Train', 'Test'])

plt.show()

