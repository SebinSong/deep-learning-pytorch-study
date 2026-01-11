import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

F = nn.functional

nPerClust = 100
blur = 1

A = [1, 3]
B = [1, -2]

# generate data
a = [
  A[0] + np.random.randn(nPerClust) * blur,
  A[1] + np.random.randn(nPerClust) * blur
]

b = [
  B[0] + np.random.randn(nPerClust) * blur,
  B[1] + np.random.randn(nPerClust) * blur
]

labels_np = np.vstack([
  np.zeros((nPerClust, 1)),
  np.ones((nPerClust, 1))
])

data_np = np.hstack([a, b]).T

data = torch.tensor(data_np, dtype=torch.float)
labels = torch.tensor(labels_np, dtype=torch.float)

class ANNClassifier(nn.Module):
  def __init__(self, n_input, n_output, n_hidden_unit):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(n_input, n_hidden_unit),
      'hidden': nn.Linear(n_hidden_unit, n_hidden_unit),
      'output': nn.Linear(n_hidden_unit, n_output)
    })

  def forward(self, x):
    x = F.relu( self.layers['input'](x) )
    x = F.relu( self.layers['hidden'](x) )
    x = F.sigmoid( self.layers['output'](x) )
    return x

def createANNmodel(learningRate):
  ANNclassify = ANNClassifier(
    n_input=data.shape[1],
    n_output=labels.shape[1],
    n_hidden_unit=16
  )

  loss_func = nn.BCELoss()
  optimizer = torch.optim.SGD(
    ANNclassify.parameters(),
    lr=learningRate
  )

  return ANNclassify, loss_func, optimizer

numEpochs = 1000

def trainTheModel(ANNmodel, loss_func, optimizer):
  losses = torch.zeros(numEpochs)

  for epoch_i in range(numEpochs):
    y_hat = ANNmodel(data)

    loss = loss_func(y_hat, labels)
    losses[epoch_i] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  predictions = ANNmodel(data)
  pred_labels = (predictions > 0.5).float()

  totalacc = 100 * torch.mean( (pred_labels == labels).float() )

  return losses, predictions, totalacc

ANNclassify, loss_func, optimizer = createANNmodel(0.01)

losses, final_predictions, total_acc = trainTheModel(ANNclassify, loss_func, optimizer)

plt.plot(losses, 'ko', markerfacecolor='w', linewidth=0.1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Model accuracy: {total_acc:.3f}%')
plt.show()
