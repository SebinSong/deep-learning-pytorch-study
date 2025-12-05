import torch
import matplotlib.pyplot as plt

## STEP 1 - Create the real data

# Our batch of data will have 10 data points
N = 5

# Each data point has 1 input feature and 1 output value
D_in = 1
D_out = 1

# Create our input data X
X = torch.randn(N, D_in)

# Create our true target labels y by using the 'true' W and b
# The true W is 2.0m the 'true' b is 1.0
true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)
y_true = X @ true_W + true_b + torch.randn(N, D_out) * 0.1

# Hyperparameters
learning_rate, epochs = 0.01, 300

# parameters
W = torch.randn(D_in, D_out, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Training loop
for epoch in range(epochs):
  # Forward pass and calculate the loss
  y_hat = X @ W + b
  loss = torch.mean((y_hat - y_true)**2)

  # back-propagation
  loss.backward()

  # Update parameters
  with torch.no_grad():
    W -= learning_rate * W.grad
    b -= learning_rate * b.grad

  # Zero gradients for the next iteration
  W.grad.zero_()
  b.grad.zero_()

  if epoch % 10 == 0:
    print(f'Epoch {epoch:02d}: Loss={loss.item()}, W={W.item():.3f}, b={b.item():.3f}')

print(f'/nFinal Parameters: W={W.item():.3f}, b={b.item():.3f}')
print(f'True Parameters: W={true_W.item():.3f}, b={true_b.item():.3f}')
