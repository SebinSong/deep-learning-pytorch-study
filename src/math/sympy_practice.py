import sympy as sym

x, t, z, nu = sym.symbols('x t z nu')

sym.init_printing(use_unicode=True)

f1 = sym.sin(x) * sym.exp(x)
df1 = sym.diff(f1, x)
idf1 = sym.integrate(df1)

f2 = sym.sin(x) / x
limf2 = sym.limit(f2, x, 0)

f3 = x**2 - 2
solved_f3 = sym.solve(f3, x)
print(solved_f3)
