import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

F = nn.functional

# global values that are fixed
train_prop = .8
batch_size = 16
learning_rate = 0.01
num_epochs = 50
total_num_units = 80
D_in = 2 # Number of features/dimensions
D_out = 3 # Number of categories

def create_and_split_data(n_per_cluster = 50):
  global train_prop, batch_size

  A = [1, 1]
  B = [5, 1]
  C = [4, 4]

  a = torch.vstack([ A[0] + torch.randn(n_per_cluster), A[1] + torch.randn(n_per_cluster) ]).T
  b = torch.vstack([ B[0] + torch.randn(n_per_cluster), B[1] + torch.randn(n_per_cluster) ]).T
  c = torch.vstack([ C[0] + torch.randn(n_per_cluster), C[1] + torch.randn(n_per_cluster) ]).T

  data = torch.vstack((a, b, c)).float()
  labels = torch.hstack((
    torch.zeros(n_per_cluster),
    torch.ones(n_per_cluster),
    torch.ones(n_per_cluster) + 1
  )).long() # converting it to long is important!

  plot_data = False
  if plot_data:
    plt.plot(
      data[torch.where(labels == 0)[0], 0],
      data[torch.where(labels == 0)[0], 1],
      'ko',
      label='A'
    )
    plt.plot(
      data[labels == 1, 0],
      data[labels == 1, 1],
      'bs',
      label='B'
    )
    plt.plot(
      data[labels == 2, 0],
      data[labels == 2, 1],
      'y^',
      label='C'
    )
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend()
    plt.show()

  dataset = TensorDataset(data, labels)

  sample_size = len(dataset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dataset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=test_size)

  return train_loader, test_loader

train_loader, test_loader = create_and_split_data(150)

class QWERTY_ANN(nn.Module):
  def __init__(self, n_units = 8, n_layers = 1):
    super().__init__()

    self.n_layers = n_layers
    self.layers = nn.ModuleDict({
      'input': nn.Linear(D_in, n_units)
    })

    for li in range(n_layers):
      self.layers[f'fc{li}'] = nn.Linear(n_units, n_units)
    
    self.layers['output'] = nn.Linear(n_units, D_out)

  def forward(self, x):
    x = F.relu( self.layers['input'](x) )

    for li in range(self.n_layers):
      x = F.relu( self.layers[f'fc{li}'](x) )
    
    return self.layers['output'](x)

def create_model (n_units = 8, n_layers = 1):
  global learning_rate

  model = QWERTY_ANN(n_units=n_units, n_layers=n_layers)

  lf = nn.CrossEntropyLoss()
  opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

  return model, lf, opt

def train_model(n_per_cluster = 50, n_layers = 1):
  global num_epochs, total_num_units

  unit_per_layer = int(total_num_units / n_layers)
  train_loader, test_loader = create_and_split_data(n_per_cluster)
  model, loss_func, optimizer = create_model(unit_per_layer, n_layers)

  all_losses = np.zeros(num_epochs)
  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)

  test_x, test_y = next(iter(test_loader))

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_accs = []
    all_batch_losses = []
    for batch_x, batch_y in train_loader:
      y_hat = model(batch_x)

      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      all_batch_losses.append( loss.item() )

      # compute batch_acc
      batch_acc = (torch.argmax(y_hat, dim=1) == batch_y).float().mean() * 100
      all_batch_accs.append( batch_acc.item() )

    train_acc_ave = np.mean(all_batch_accs)
    loss_ave = np.mean(all_batch_losses)
    all_train_acc[epoch_i] = train_acc_ave
    all_losses[epoch_i] = loss_ave

    if epoch_i % 5 == 0:
      print(f'[c_size={n_per_cluster}, n_layer={n_layers} Epoch {epoch_i}] - train_acc={train_acc_ave:.3f}, loss={loss_ave:.3f}')

    # compute test_acc
    model.eval()
    with torch.no_grad():
      test_pred_labels = torch.argmax( model(test_x), dim=1 )
    
    test_acc = (test_pred_labels == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc.item()

  return all_train_acc, all_test_acc, all_losses, model

# Experiments
n_layers_range = [1, 5, 10, 20]
n_data_points = np.arange(50, 551, 50)

legends = []

for layer_idx, n_layer in enumerate(n_layers_range):
  units_per_layer = int(total_num_units / n_layer)
  model = QWERTY_ANN(units_per_layer, n_layer)

  n_params = np.sum([ p.numel() for p in model.parameters() if p.requires_grad])

  legend = f'{n_layer} layers, {units_per_layer} units, {n_params} params.'
  legends.append(legend)
  print(legend)

results = np.zeros((
  len(n_data_points), len(n_layers_range), 2
))

for size_idx, cluster_size in enumerate(n_data_points):
  for layer_idx, n_layers in enumerate(n_layers_range):
    train_acc, test_acc, losses, _ = train_model(cluster_size, n_layers)
    results[size_idx, layer_idx, 0] = np.mean(losses[-5:])
    results[size_idx, layer_idx, 1] = np.mean(test_acc[-5:])

loss_matrix = results[:, :, 0]
acc_matrix = results[:, :, 1]

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(
  n_data_points,
  loss_matrix,
  'o-'
)
ax[0].set_xlabel('Cluster size')
ax[0].set_ylabel('Loss')
ax[0].legend(legends)
ax[0].set_title('Losses')

ax[1].plot(
  n_data_points,
  acc_matrix,
  's-'
)
ax[1].set_xlabel('Cluster size')
ax[1].set_ylabel('Accuracy (%)')
ax[1].legend(legends)
ax[1].set_title('Accuracy')

plt.show()
