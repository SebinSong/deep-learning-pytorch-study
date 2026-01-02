import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

F = nn.functional

iris_dataset = sns.load_dataset('iris')

data = torch.tensor(iris_dataset[iris_dataset.columns[:-1]].values, dtype=torch.float32)

labels = torch.zeros(len(data), dtype=torch.long)
labels[iris_dataset['species'] == 'versicolor'] = 1
labels[iris_dataset['species'] == 'virginica'] = 2

class ANNiris(nn.Module):
  def __init__(self, n_input, n_output, n_units, n_layers):
    super().__init__()

    self.n_layers = n_layers
    self.layers = nn.ModuleDict({
      'input': nn.Linear(n_input, n_units),
      **dict([ (f'hidden{i}', nn.Linear(n_units, n_units)) for i in range(n_layers) ]),
      'output': nn.Linear(n_units, n_output)
    })

  # forward pass
  def forward(self, x):
    x = self.layers['input'](x)

    for i in range(self.n_layers):
      x = F.relu( self.layers[f'hidden{i}'](x) )

    x = self.layers['output'](x)
    return x


def train_the_model(model):
  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

  for _ in range(num_epochs):
    y_hat = model(data)
    loss = loss_func(y_hat, labels)

    # back-propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # final calculation
  prediction = model(data)
  pred_labels = torch.argmax(prediction, dim=1)
  accuracy = 100 * torch.mean( (pred_labels == labels).float() )
  n_params = sum([ p.numel() for p in model.parameters() if p.requires_grad ])

  return accuracy, n_params

n_input = iris_dataset.columns.size - 1
n_output = iris_dataset['species'].nunique()
num_epochs, learning_rate = 300, 0.01

num_layers_collection = np.arange(1, 5, dtype=np.int16)
num_units_collection = np.arange(4, 101, 4)

# output matrices
mat_accuracies = np.zeros((num_units_collection.size, num_layers_collection.size))
mat_total_params = np.zeros((num_units_collection.size, num_layers_collection.size))

for i_unit, n_unit in enumerate(num_units_collection):
  for i_layer, n_layer in enumerate(num_layers_collection):
    net = ANNiris(n_input, n_output, n_units=n_unit, n_layers=n_layer)
    final_acc, n_total_params = train_the_model(net)

    mat_accuracies[i_unit, i_layer] = final_acc
    mat_total_params[i_unit, i_layer] = n_total_params
    print(f'{n_unit=}, {n_layer=} - accuracy:{final_acc:.3f}, num of params: {n_total_params}')

plt.figure(figsize=(12, 6))
plt.plot(num_units_collection, mat_accuracies, 'o-', markerfacecolor='w', markersize=9)
plt.plot(num_units_collection[[0, -1]], [33, 33], '--', color=[.8, .8, .8])
plt.plot(num_units_collection[[0, -1]], [67, 67], '--', color=[.8, .8, .8])
plt.legend([f'{n_layer} layer(s)' for n_layer in num_layers_collection])
plt.xlabel('Number of hidden units')
plt.ylabel('Accuracy %')
plt.title('Accuracy')
plt.show()
