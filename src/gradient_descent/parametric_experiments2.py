import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

x = sym.symbols('x')
fx = sym.sin(x) * sym.exp(-x**2*0.05)
dfx = sym.diff(fx)
lambda_fx = sym.lambdify(x, fx)
lambda_dfx = sym.lambdify(x, dfx)

learning_rates = np.linspace(1e-10, 1e-1, 50)
training_epochs = np.round(np.linspace(10, 500, 40))
finalres = np.zeros((len(learning_rates), len(training_epochs)))

# loop over learning rates
for l_idx, learning_rate in enumerate(learning_rates):
  for e_idx, training_epoch in enumerate(training_epochs):
    localmin = 0
    for i in range(int(training_epoch)):
      grad = lambda_dfx(localmin)
      localmin = localmin - learning_rate * grad

    finalres[l_idx, e_idx] = localmin

fig, ax = plt.subplots(figsize=(7,5))

plt.imshow(finalres,
  extent=[learning_rates[0], learning_rates[-1], training_epochs[0], training_epochs[-1]],
  aspect='auto',
  origin='lower',
  vmin=-1.45, vmax=-1.2
)
plt.xlabel('Learning rate')
plt.ylabel('Final function estimate')
plt.colorbar()
plt.title('Each line is a training epochs N')
plt.show()
