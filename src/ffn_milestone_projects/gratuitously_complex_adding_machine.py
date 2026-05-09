import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split

# create dataset
def generate_data_and_labels(n = 50, randomize=False):
  data = []
  labels = []

  for i in range(-n, n+1):
    for j in range(-n, n+1):
      data.append([i, j])
      labels.append(i+j)
  
  data_t = torch.tensor(data, dtype=torch.float32)
  labels_t = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)

  if randomize:
    rand_idxs = torch.randperm(len(labels_t))

    data_t = data_t[rand_idxs]
    labels_t = labels_t[rand_idxs]

  return data_t, labels_t

def create_and_split_data(n = 50, train_prop=.8, batch_size=16):
  data_t, labels_t = generate_data_and_labels(n)

  dset = TensorDataset(data_t, labels_t)

  sample_size = len(dset)
  train_size = int(sample_size * train_prop)
  test_size = sample_size - train_size

  train_dset, test_dset = random_split(dset, [train_size, test_size])

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=test_size)

  return train_loader, test_loader

# some global variables
D_in = 2
D_out = 1
num_epochs = 20

class Addition_ANN(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.Sequential(
      nn.Linear(D_in, 16), # input layer
      nn.ReLU(),
      nn.Linear(16, 8),
      nn.ReLU(),
      nn.Linear(8, D_out)
    )

  def forward(self, x):
    return self.layers(x)


def create_model(lr=.001):
  m = Addition_ANN()
  l = nn.MSELoss()
  o = optim.Adam(m.parameters(), lr=lr)
  return m, l, o

def train_model():
  train_loader, test_loader = create_and_split_data()
  model, loss_func, optimizer = create_model()

  test_x, test_y = next(iter(test_loader))

  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_acc = []
    for batch_x, batch_y in train_loader:
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)

      # back prop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  
      # compute batch accuracy
      batch_acc = (torch.round(y_hat) == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )
    
    ave_batch_acc = np.mean( all_batch_acc )
    all_train_acc[epoch_i] = ave_batch_acc

    if epoch_i % 5 == 0:
      print(f'[Epoch {epoch_i}] - acc={ave_batch_acc:.3f}')
    
    # compute test accuracy
    model.eval()
    with torch.no_grad():
      test_predictions = torch.round(model(test_x))
    
    test_acc = (test_predictions == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc.item()
  
  return all_train_acc, all_test_acc, model

num_experi = 10
accuracy_map = np.zeros((num_experi, 2))

best_test_acc = 0
best_performing_model = None

for n_experi in range(num_experi):
  all_train_acc, all_test_acc, model = train_model()

  final_test_acc = all_test_acc[-1]
  accuracy_map[n_experi, 0] = all_train_acc[-1]
  accuracy_map[n_experi, 1] = final_test_acc

  if final_test_acc >= best_test_acc:
    best_test_acc = final_test_acc
    best_performing_model = model

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.bar(np.arange(num_experi) - .2, accuracy_map[:, 0], .5)
ax1.bar(np.arange(num_experi) + .2, accuracy_map[:, 1], .5)
ax1.set_xticks(range(num_experi), np.arange(num_experi))
ax1.set_ylabel('Final Auccracy')
ax1.legend(['Train', 'Test'])
ax1.set_title('Final accuracies per experiment')

my_own_val_set = [
  [100, 124], [-120, -8], [-55, -22], [40, 25], [-30, 27], [-250, 32], [300, 5000]
]

val_data, val_labels = generate_data_and_labels(randomize=True)
pred_labels = torch.round(best_performing_model(val_data))

true_labels_np = val_labels.flatten().detach().numpy()
pred_labels_np = pred_labels.flatten().detach().numpy()
x_vals = range(len(true_labels_np))

ax2.plot(x_vals, true_labels_np, 'bs', alpha=.5, label='True sum')
ax2.plot(x_vals, pred_labels_np, 'r^', alpha=.5, label='Predicted sum')
ax2.set_xlabel('Sample index')
ax2.set_ylabel('Sum')
ax2.set_title('Predicted vs actual sum')
ax2.legend()

plt.show()

def perf_addition(pair):
  pair = torch.tensor(pair, dtype=torch.float32).reshape(1, -1)
  res = best_performing_model(pair)
  return round(res.item())

def val_my_set():
  for pair in my_own_val_set:
    sum_val = perf_addition(pair)
    print(f'{pair[0]} + {pair[1]} = {sum_val}')

val_my_set()
