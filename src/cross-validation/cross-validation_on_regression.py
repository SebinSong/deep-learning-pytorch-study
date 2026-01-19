import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sample_size = 150

x = torch.randn((sample_size, 1))
y_true = x + torch.randn((sample_size, 1))

ANNreg = nn.Sequential(
  nn.Linear(1, 1), # input -> hidden
  nn.ReLU(),
  nn.Linear(1, 1) # hidden -> input
)

learning_rate = 0.05
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(ANNreg.parameters(), lr=learning_rate)

train_size = int(0.8 * sample_size)
train_indices = np.random.choice(range(sample_size), train_size, replace=False)
train_bool = np.zeros(sample_size, dtype=bool)
train_bool[train_indices] = True
test_bool = ~train_bool

num_epochs = 500

for epoch_i in range(num_epochs):
  # forward pass
  y_hat = ANNreg(x[train_bool])
  loss = loss_func(y_hat, y_true[train_bool])

  # backprop
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

pred_y_test = ANNreg(x[test_bool])
loss_test = loss_func(pred_y_test, y_true[test_bool])

print(f'loss_test: {loss_test}, loss_final_train: {loss}')
pred_y_train = ANNreg(x[train_bool]).detach()

plt.plot(x, y_true, 'k^', label='All data')
plt.plot(x[train_bool], pred_y_train, 'bs', markerfacecolor='w', label='Training pred')
plt.plot(x[test_bool], pred_y_test.detach(), 'ro', markerfacecolor='w', label='Test pred')
plt.legend()
plt.show()
