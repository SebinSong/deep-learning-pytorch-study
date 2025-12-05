import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

x = sym.symbols('x')
fx = sym.sin(x) * sym.exp(-x**2*0.05)
dfx = sym.diff(fx)
lambda_fx = sym.lambdify(x, fx)
lambda_dfx = sym.lambdify(x, dfx)

x_list = np.linspace(-2*np.pi, 2*np.pi, 401)

# localmin = float(np.random.choice(x_list, 1))

learning_rate = 0.01
training_epochs = 1000

startlocs = np.linspace(-5, 5, 50)
finalres = np.zeros(len(startlocs))

for idx, localmin in enumerate(startlocs):
  # run through training
  for i in range(training_epochs):
    grad = lambda_dfx(localmin)
    localmin = localmin - grad * learning_rate

  # store the final guess
  finalres[idx] = localmin

plt.plot(startlocs, finalres, 's-')
plt.xlabel('Starting guess')
plt.ylabel('Final guess')
plt.show()

# plt.plot(x_list, lambda_fx(x_list), x_list, lambda_dfx(x_list), '--')
# plt.plot(localmin, lambda_dfx(localmin), 'ro')
# plt.plot(localmin, lambda_fx(localmin), 'ro')
# plt.xlim([x_list[0], x_list[-1]])
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.legend(['f(x)', 'df', 'f(x) min'])
# plt.title(f'Empirical local minimum: {localmin:.5f}')
# plt.show()
