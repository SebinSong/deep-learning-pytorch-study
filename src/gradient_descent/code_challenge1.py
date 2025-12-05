import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

x = sym.symbols('x')

fx = sym.cos(2 * sym.pi * x) + x**2
dfx = sym.diff(fx)
lambda_fx = sym.lambdify(x, fx)
lambda_df = sym.lambdify(x, dfx)

x_pool = np.linspace(-2, 2, 1000)
local_min = np.random.choice(x_pool, 1)

num_epochs = 100
learning_rate = 0.01

modelparams = np.zeros((num_epochs, 2))

for i in range(num_epochs):
  grad = lambda_df(local_min[0])
  local_min = local_min - grad * learning_rate
  modelparams[i,0] = local_min[0]
  modelparams[i,1] = grad

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

for i in range(2):
  ax[i].plot(modelparams[:, i], 'o-')
  ax[i].set_xlabel('Iteration')
  ax[i].set_title(f'Final estimated minimum: {local_min[0]:.5f}')

ax[0].set_ylabel('Local minimum')
ax[1].set_ylabel('Derivative')

plt.show()

# def fx(x):
#   return 3*x**2 - 3*x + 4

# def deriv_fx(x):
#   return 6*x - 3

# x = np.linspace(-2, 2, 1000)

# localmin = np.random.choice(x, 1)

# learning_rate = 0.01
# training_epochs = 150

# modelparams = np.zeros((training_epochs, 2))

# for i in range(training_epochs):
#   grad = deriv_fx(localmin)
#   localmin = localmin - grad * learning_rate
#   modelparams[i,0] = localmin[0]
#   modelparams[i,1] = grad[0]

# fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# for i in range(2):
#   ax[i].plot(modelparams[:, i], 'o-')
#   ax[i].set_xlabel('Iteration')
#   ax[i].set_title(f'Final estimated minimum: {localmin[0]:.5f}')

# ax[0].set_ylabel('Local minimum')
# ax[1].set_ylabel('Derivative')

# plt.show()
