import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Our batch of data will have 10 data points
N = 10

# Each data point has 1 input feature and 1 output value
D_in = 1
D_out = 1

# Create out input data X
X = torch.randn(N, D_in)

# define a class for the model
class LinearRegressionModel(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    # In the constructor, we DEFINE the layers we'll use.
    self.linear_layer = nn.Linear(in_features, out_features)

  def forward(self, X):
    # In the forward pass, we CONNECT the layers
    return self.linear_layer(X)

model = LinearRegressionModel(D_in, D_out)

# Create our true target labels y by using the 'true' W and b
# The 'true' W is 2.0, the 'true' b is 1.0
true_W = torch.tensor([[2.0]]) # The dimension here is D_in * D_out
true_b = torch.tensor(1.0)
y_true = X @ true_W + true_b + torch.randn(N, D_out) * 0.1 # Add a little noise

# Set hyper-parameters
learning_rate, epochs = 0.01, 350

# Optimizer - create Adam optimizer
# We pass model.parameters() to tell it which tensors to manage
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# We'll also grab a pre-built loss function from torch.nn
loss_fn = nn.MSELoss() # Mean Squared Error Loss

# tracking losses - Will draw a graph later.
tracked_losses = np.zeros(epochs, dtype=float)

# Training loop
for epoch in range(epochs):
  # Forward pass
  y_hat = model(X)

  # calculate the loss and track it
  loss = loss_fn(y_hat, y_true)
  tracked_losses[epoch] = loss.item()

  # Backward propagation
  optimizer.zero_grad()

  # compute the parameter gradients
  loss.backward()

  # update the parameters (model learning)
  optimizer.step()

plt.plot(np.arange(epochs), tracked_losses, 'bo-', markerfacecolor='w', label='loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
