import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import seaborn as sns

iris = sns.load_dataset('iris')

# iris.plot(marker='o', linestyle='none', figsize=(12, 6))
# plt.xlabel('Sample number')
# plt.ylabel('Value')
# plt.show()

data = torch.tensor( iris[iris.columns[:-1]].values, dtype=torch.float32)
labels_mapping = { 'setosa': 0, 'versicolor': 1, 'virginica': 2 }
labels = torch.tensor(iris['species'].map(labels_mapping).fillna(-1).values, dtype=torch.long)
dataset = TensorDataset(data, labels)

D_in = len(iris.columns[:-1])
D_out = torch.unique(labels).numel()

def split_data(dset, train_prop=.8, batch_size=16):
  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=test_size, shuffle=False)

  return train_loader, test_loader

def create_model (learning_rate=0.001, hunits=64):
  net = nn.Sequential(
    nn.Linear(D_in, hunits),
    nn.ReLU(),
    nn.Linear(hunits, hunits),
    nn.ReLU(),
    nn.Linear(hunits, D_out)
  )

  lf = nn.CrossEntropyLoss()

  op = torch.optim.SGD(net.parameters(), lr=learning_rate)

  return net, lf, op

def train_model(model, loss_func, optimizer, train_ldr, test_ldr, / , num_epochs=500):
  train_acc = np.zeros(num_epochs)
  test_acc = np.zeros(num_epochs)
  losses = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    batch_accs = []
    batch_losses = []

    model.train()
    for batch_x, batch_y in train_ldr:
      # forward pass
      y_hat = model(batch_x)

      # compute the loss
      loss = loss_func(y_hat, batch_y)
      batch_losses.append(loss.item())

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # compute the batch accuracy
      curr_acc = 100 * (torch.argmax(y_hat, dim=1) == batch_y).float().mean()
      batch_accs.append(curr_acc.item())

    train_acc[epoch_i] = np.mean(batch_accs)
    losses[epoch_i] = np.mean(batch_losses)

    # compute test accuracy
    model.eval()
    with torch.no_grad():
      test_x, test_y = next(iter(test_ldr))
      test_pred_labels = torch.argmax(model(test_x), dim=1)
      test_acc[epoch_i] = 100 * (test_pred_labels == test_y).float().mean()

  return train_acc, test_acc, losses


# parametric experiments with batch sizes
total_epochs = 600
batch_sizes = [ 2**n for n in range(1, 7) ]
acc_results_train = np.zeros((total_epochs, len(batch_sizes)))
acc_results_test = np.zeros((total_epochs, len(batch_sizes)))

for bi, batch_size in enumerate(batch_sizes):
  train_loader, test_loader = split_data(dataset, train_prop=0.8, batch_size=batch_size)
  model, loss_func, optimizer = create_model()
  train_acc, test_acc, _ = train_model(model, loss_func, optimizer, train_loader, test_loader, num_epochs=total_epochs)

  acc_results_train[:, bi] = train_acc
  acc_results_test[:, bi] = test_acc

fig, ax = plt.subplots(1, 2, figsize=(17, 7))

ax[0].plot(acc_results_train)
ax[0].set_title('Train accuracy')
ax[1].plot(acc_results_test)
ax[1].set_title('Test accuracy')


for i in range(2):
  ax[i].legend(batch_sizes)
  ax[i].set_xlabel('Epoch')
  ax[i].set_ylabel('Accuracy (%)')

plt.show()
