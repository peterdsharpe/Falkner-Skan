# falkner_skan.py
# By Peter Sharpe

import casadi as cas

eta_edge = 7
n_points = 100
max_iter = 100
"""
Solves the Falkner-Skan equation for a given value of m.
See Wikipedia for reference: https://en.wikipedia.org/wiki/Falkner–Skan_boundary_layer

:param m: power-law exponent of the edge velocity (i.e. u_e(x) = U_inf * x ^ m)
:return: eta, f0, f1, and f2 as a tuple of 1-dimensional ndarrays.

Governing equation:
f''' + f*f'' + beta*( 1 - (f')^2 ) = 0, where:
beta = 2 * m / (m+1)
f(0) = f'(0) = 0
f'(inf) = 1

Syntax:
f0 is f
f1 is f'
f2 is f''
f3 is f'''

"""

opti = cas.Opti()

m = opti.variable()
opti.set_initial(m, -0.1)

# Assign beta
beta = 2 * m / (m + 1)

eta = cas.linspace(0, eta_edge, n_points)

trapz = lambda x: (x[:-1] + x[1:]) / 2

# Vars
f0 = opti.variable(n_points)
f1 = opti.variable(n_points)
f2 = opti.variable(n_points)

# Guess
opti.set_initial(f0,
                 -eta ** 2 * (eta - 3 * eta_edge) / (3 * eta_edge ** 2)
                 )
opti.set_initial(f1,
                 1 - (1 - eta / eta_edge) ** 2
                 )
opti.set_initial(f2,
                 2 * (eta_edge - eta) / eta_edge ** 2
                 )

# BCs
opti.subject_to([
    f0[0] == 0,
    f1[0] == 0,
    f1[-1] == 1
])

# ODE
f3 = -f0 * f2 - beta * (1 - f1 ** 2)

# Derivative definitions (midpoint-method)
df0 = cas.diff(f0)
df1 = cas.diff(f1)
df2 = cas.diff(f2)
deta = cas.diff(eta)
opti.subject_to([
    df0 == trapz(f1) * deta,
    df1 == trapz(f2) * deta,
    df2 == trapz(f3) * deta
])

# Require barely-separating solution
opti.subject_to([
    f2[0] == 0
])

p_opts = {}
s_opts = {}
s_opts["max_iter"] = max_iter  # If you need to interrupt, just use ctrl+c
opti.solver('ipopt', p_opts, s_opts)

try:
    sol = opti.solve()
except:
    raise Exception("Solver failed for m = %f!" % m)

print("m where separation occurs = %.16f" % sol.value(m))