# falkner_skan.py
# By Peter Sharpe

import casadi as cas

m=0
eta_edge = 7
n_xi = 10
n_eta = 100
max_iter = 100

opti = cas.Opti()

xi = cas.linspace(0, 1, n_xi)
eta = cas.linspace(0, eta_edge, n_eta)

def trapz(x):
    out = (x[:-1] + x[1:]) / 2
    out[0] += x[0] / 2
    out[-1] += x[-1] / 2
    return out

# Vars
f0 = opti.variable(n_eta)
f1 = opti.variable(n_eta)
f2 = opti.variable(n_eta)
f3 = opti.variable(n_eta)
g0 = opti.variable(n_eta)

# Guess (guess a quadratic velocity profile, integrate and differentiate accordingly)
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
    f1[-1] == 1,
    g0[0] == 1,
    g0[-1] == 1
])

C = cas.exp(-(g-1))

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

# Require unseparated solutions
opti.subject_to([
    f2[0] > 0
])

p_opts = {}
s_opts = {}
s_opts["max_iter"] = max_iter  # If you need to interrupt, just use ctrl+c
opti.solver('ipopt', p_opts, s_opts)

try:
    sol = opti.solve()
except:
    raise Exception("Solver failed for m = %f!" % m)
