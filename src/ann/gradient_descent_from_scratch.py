import torch
import numpy as np
import matplotlib.pyplot as plt

# Our batch of data will have 10 data points
N = 5

# Each data point has 1 input feature and 1 output value
D_in = 1
D_out = 1

# Create out input data X
X = torch.randn(N, D_in)

# Create our true target labels y by using the 'true' W and b
# The 'true' W is 2.0, the 'true' b is 1.0
true_W = torch.tensor([[2.0]]) # The dimension here is D_in * D_out
true_b = torch.tensor(1.0)
y_true = X @ true_W + true_b + torch.randn(N, D_out) * 0.1 # Add a little noise

# Set hyper-parameters
learning_rate, epochs = 0.01, 350
11
# Initialize the model parameters
W = torch.randn(D_in, D_out, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# tracking losses - Will draw a graph later.
tracked_losses = np.zeros(epochs, dtype=float)

# Training loop
for epoch in range(epochs):
  # Forward pass
  y_hat = X @ W + b
  loss = torch.mean((y_hat - y_true)**2)
  tracked_losses[epoch] = loss.item()

  # Backward propagation - compute all the gradients
  loss.backward()

  # Update parameters
  with torch.no_grad():
    W -= learning_rate * W.grad
    b -= learning_rate * b.grad

  # Zero gradients
  W.grad.zero_()
  b.grad.zero_()

plt.plot(np.arange(epochs), tracked_losses, 'bo-', markerfacecolor='w', label='loss')
plt.xlabel('epoch')
plt.legend()
plt.title(f'W_true={true_W.item():.3f}, W={W.item():.3f}, b_true={true_b.item():.3f}, b={b.item():.3f}')
plt.show()
