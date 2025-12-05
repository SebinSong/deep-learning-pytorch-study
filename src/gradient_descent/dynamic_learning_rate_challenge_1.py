import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

x = sym.symbols('x')

_fx = 3*x**2 - 3*x + 4
_dfx = sym.diff(_fx)
fx = sym.lambdify(x, _fx)
dfx = sym.lambdify(x, _dfx)

x = np.linspace(-2, 2, 701)

# perform gradient descent
learning_rate = 0.01
epochs = 200

localmin = np.random.choice(x, 1)[0]
trajectory = np.zeros(epochs)
start_pnt = localmin

for i in range(epochs):
  grad = dfx(localmin)
  lr = learning_rate * np.abs(grad)
  localmin = localmin - grad * lr
  trajectory[i] = localmin

# plot it!

# draw graphes
plt.plot(x, fx(x), 'b-', x, dfx(x), 'y--')
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')

plt.title('f(x) and its derivatives')

# draw gradient descent trajectory

# starting point
plt.plot(start_pnt, fx(start_pnt), 'go', markersize=8)

# trajectory
plt.plot(trajectory, fx(trajectory), 'ro-', markerfacecolor='w', markersize=4)
plt.plot(localmin, fx(localmin), 'ro', markersize=8)
plt.plot(localmin, dfx(localmin), 'ro', markersize=8)

plt.legend(['f(x)', "f'(x)", 'start pnt', 'local min'])
plt.show()
