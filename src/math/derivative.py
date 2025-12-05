import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import sympy.plotting.plot as symplot

x = sym.symbols('x')

fx = 2*x**2
gx = 4*x**3 - 3*x**4

df = sym.diff(fx)
dg = sym.diff(gx)

manual = df*gx + fx*dg
thewrongway = df*dg

viasympy = sym.diff(fx*gx)

print('The functions:')
print(manual)
print(viasympy)
print('The wrong way:')
print(thewrongway)

# Chain rule
hx = (x**2 + 4*x**3)**5
print('derivative of hx:')
print(sym.diff(hx))
