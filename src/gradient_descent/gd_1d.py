import numpy as np
import matplotlib.pyplot as plt

def fx(x):
  return 3*x**2 - 3*x + 4

def deriv_fx(x):
  return 6*x - 3

x = np.linspace(-2, 2, 1000)

localmin = np.random.choice(x, 1)

learning_rate = 0.01
training_epochs = 150

modelparams = np.zeros((training_epochs, 2))

for i in range(training_epochs):
  grad = deriv_fx(localmin)
  localmin = localmin - grad * learning_rate
  modelparams[i,0] = localmin[0]
  modelparams[i,1] = grad[0]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

for i in range(2):
  ax[i].plot(modelparams[:, i], 'o-')
  ax[i].set_xlabel('Iteration')
  ax[i].set_title(f'Final estimated minimum: {localmin[0]:.5f}')

ax[0].set_ylabel('Local minimum')
ax[1].set_ylabel('Derivative')

plt.show()
