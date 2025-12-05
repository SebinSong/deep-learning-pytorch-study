import sympy as sym
import numpy as np

x = sym.symbols('x')

fx = sym.cos(2*sym.pi*x) + x**2
dfx = sym.diff(fx)

p = sym.plot(
  (fx, (x, -2, 2), 'fx'),
  (dfx, (x, -2, 2), 'f\'x'),
  legend=True,
  show=False,
  title='f(x) and its derivative'
)

p.show()
