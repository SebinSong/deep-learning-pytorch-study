import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.0001, 2, 50)
log_x = np.log(x)
exp_x = np.exp(x)
print(x, log_x, exp_x, sep='\n')

# font-size update: below line:
plt.rcParams.update({'font.size': 15})

plt.figure(figsize=(8, 6))
plt.plot(x, x, '-', color=[.8, .8, .8])
plt.plot(x, log_x, '-b^', markersize=5, markerfacecolor='w')
plt.plot(x, exp_x, '-ko', markersize=5, markerfacecolor='w')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['unity', 'log(x)', 'exp(x)'])
plt.show()
