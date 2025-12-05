import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# create data
N = 30
x = torch.randn(N, 1)
y = x + torch.randn(N, 1) / 2 # real value data to compare with yHat later in loss computation.

# build a model
ANNreg = nn.Sequential(
  nn.Linear(1, 1), # input layer
  nn.ReLU(), # activation function
  nn.Linear(1, 1) # output layer
)

learningRate = 0.05
lossfun = nn.MSELoss()

# optimizer (the flavor of gradient descent to implement)
optimizer = torch.optim.SGD(ANNreg.parameters(), lr=learningRate)

# train the model
num_epochs = 500
losses = torch.zeros(num_epochs)

for epoch_i in range(num_epochs):
  # forward pass
  yHat = ANNreg(x)

  # compute loss
  loss = lossfun(yHat, y)
  losses[epoch_i] = loss

  # backprop
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# show the losses

#manually compute losses
# final forward pass
predictions = ANNreg(x)

# final loss (MSE)
testloss = (predictions - y).pow(2).mean()

# plot the losses
# plt.plot(losses.detach(), 'o', markerfacecolor='w', linewidth=.1)
# plt.plot(num_epochs, testloss.detach(), 'ro')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title(f'Final loss = {testloss.item()}')
# plt.show()

# plot the predictions vs real data
plt.plot(x, y, 'bo', label='Real data')
plt.plot(x, predictions.detach(), 'ro', label='Predictions')
plt.title(f'prediction data r={np.corrcoef(y.T, predictions.detach().T)[0,1]:.3f}')
plt.legend()
plt.show()

