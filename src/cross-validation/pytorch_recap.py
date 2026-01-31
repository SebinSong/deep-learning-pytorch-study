import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# batch size
N = 30

D_in = 1
D_out = 1

X = torch.randn(N, D_in)

true_W = torch.zeros((D_in, D_out)) + 2.0
true_b = torch.tensor(1.0)

y_true = X @ true_W + true_b + (torch.randn(N, D_out) + 0.1) # add a little noise

# define loss func
loss_func = lambda p, t: torch.pow(p - t, 2).mean()

# hyperparameters
learning_rate, epochs = 0.01, 300

# initialize our parameters with random values
# Shape must be correct for matrix multiplication
W = torch.randn(D_in, D_out, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# define the model
model = lambda x: x @ W + b

# Training loop
losses = np.zeros(epochs)
for epoch in range(epochs):
  # Forward pass and calcuate the loss
  y_hat = model(X)
  loss = loss_func(y_hat, y_true)
  losses[epoch] = loss.item()

  # backward pass
  loss.backward()

  # update parameters
  with torch.no_grad():
    W -= learning_rate * W.grad
    b -= learning_rate * b.grad
  
  # zero_grad
  W.grad.zero_()
  b.grad.zero_()

# plot the result
plt.figure(figsize=(7,5))
plt.plot(losses, 'ro-', markerfacecolor='w')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses')

plt.show()

print(f'Final W: {W.item()}, true_W: {true_W.item()}')
print(f'Final b: {b.item()}, true_b: {true_b.item()}')
