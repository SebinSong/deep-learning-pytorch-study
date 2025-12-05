import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def buildAndTrainTheModel(x, y):
  # build the model
  ANNreg = nn.Sequential(
    nn.Linear(1, 1), # input layer 1 input - 1 output
    nn.ReLU(), # ReLU as the activation function
    nn.Linear(1, 1) # 1 input - 1 output
  )

  lossfunc = nn.MSELoss()
  optimizer = torch.optim.SGD(ANNreg.parameters(), lr=0.05)

  # train the model
  num_epochs = 500
  losses = torch.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    # forward propagation
    yHat = ANNreg(x)

    # compute loss - The result of MSE is a scalar, in this case a tensor shapeed (1,)
    loss = lossfunc(yHat, y)
    losses[epoch_i] = loss

    # back-propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  # ending the training loop

  # compute model predictions - feed the data through the model one last time
  predictions = ANNreg(x)

  # output:
  return predictions, losses


def createThedata (m):
  N = 50
  x = torch.randn(N, 1)
  y = m*x + torch.randn(N, 1)
  return x, y

# the slopes to simulate
slopes = np.linspace(-2, 2, 21)
num_exps = 50

# initialize output matrix
results = np.zeros(
  (len(slopes), num_exps, 2)
)

for slope_i, slope in enumerate(slopes):
  for N in range(num_exps):
    # create a dataset and run the model
    x, y = createThedata(slope)
    yHat, losses = buildAndTrainTheModel(x, y)

    # store the final loss and performance
    results[slope_i, N, 0] = losses[-1]
    results[slope_i, N, 1] = np.corrcoef(y.detach().T, yHat.detach().T)[0, 1]

# Numpy supports boolean indexing, where it takes np.array of boolean in [] and performs assignment.
results[np.isnan(results)] = 0

# plot the results
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# left graph
ax[0].plot(
  slopes,
  np.mean(results[:, :, 0], axis=1),
  'ko-',
  markerfacecolor='w',
  markersize=10
)
# create a dataset
x, y = createThedata(.8)

# run the model
yHat, losses = buildAndTrainTheModel(x, y)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(losses.detach(), 'o', markerfacecolor='w', linewidth=.1)
ax[0].set_xlabel('Epoch')
ax[0].set_title('Loss')

ax[1].plot(x, y.detach(), 'bo', label='Real data')
ax[1].plot(x, yHat.detach(), 'rs', label='Predictions')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title(f'prediction-data corr = {np.corrcoef(y.detach().T, yHat.detach().T)[0, 1]:.2f}')
ax[1].legend()

plt.show()
