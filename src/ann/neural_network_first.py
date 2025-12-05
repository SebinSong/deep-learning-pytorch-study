import sympy as sym
import torch

x1, x2 = sym.symbols('x1, x2')

# Define target function
_f = 2*x1**2 + 3*x2
lambda_f = sym.lambdify((x1, x2), _f)

def f(t):
  if t.ndim != 2 or t.shape[1] != 2:
    raise ValueError('wrong tensor dimension')
  
  x1 = t[:, 0]
  x2 = t[:, 1]

  return (2 * x1**2 + 3 * x2).unsqueeze(1)

# Define dataset (size = N)
N = 10
X = torch.randn(N, 2) * 3 + 20
y_hat = f(X)
print(X)
print(y_hat)