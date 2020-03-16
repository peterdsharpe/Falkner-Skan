from falkner_skan import falkner_skan
import numpy as np
import sympy as sp
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as px
import plotly.graph_objects as go
import dash
import seaborn as sns
sns.set(font_scale=1)


### Define constants
m = 0.0733844181517584  # Exponent of the edge velocity ("a")
Vinf = 7  # Velocity at the trailing edge
nu = 1.45e-5  # kinematic viscosity
c = 0.08  # chord, in meters
x = sp.symbols("x")  # x as a symbolic variable
ue = Vinf * (x / c) ** m

x_over_c = sp.symbols("x_over_c")

### Get the nondimensional solution
eta, f0, f1, f2 = falkner_skan(m=m)  # each returned value is a ndarray

### Get parameters of interest
Re_local = ue * x / nu
dFS = sp.sqrt(nu * x / ue)

theta_over_dFS = np.trapz(
    f1 * (1 - f1),
    dx=eta[1]
) * np.sqrt(2 / (m + 1))
dstar_over_dFS = np.trapz(
    (1 - f1),
    dx=eta[1]
) * np.sqrt(2 / (m + 1))
H = dstar_over_dFS / theta_over_dFS

Cf = 2 * np.sqrt((m+1)/2) * Re_local ** -0.5 * f2[0]

# Calculate the chord-normalized values of theta, H, and Cf
theta_x_over_c = (theta_over_dFS * dFS).subs(x, x_over_c * c).simplify()
H_x_over_c = H
Cf_x_over_c = (Cf).subs(x, x_over_c * c).simplify()

### Plot parameters of interest
plt.ion()

# Generate discrete values of parameters
x_over_c_discrete = np.linspace(1 / 100, 1, 100)
theta_x_over_c_discrete = sp.lambdify(x_over_c, theta_x_over_c, "numpy")(x_over_c_discrete)
H_x_over_c_discrete = sp.lambdify(x_over_c, H, "numpy")(x_over_c_discrete)
Cf_x_over_c_discrete = sp.lambdify(x_over_c, Cf_x_over_c, "numpy")(x_over_c_discrete)

# Plot it
fig, ax = plt.subplots()

plt.subplot(311)
plt.plot(x_over_c_discrete, theta_x_over_c_discrete)
plt.title(r"$\theta$ Distribution")
plt.xlabel(r"$x/c$")
plt.ylabel(r"$\theta$ (unitless)")
plt.grid(True)

plt.subplot(312)
plt.plot(x_over_c_discrete, np.tile(H_x_over_c_discrete, len(x_over_c_discrete)))
plt.title(r"$H$ Distribution")
plt.xlabel(r"$x/c$")
plt.ylabel(r"$H$ (unitless)")
plt.grid(True)

plt.subplot(313)
plt.plot(x_over_c_discrete, Cf_x_over_c_discrete)
plt.title(r"$C_f$ Distribution")
plt.xlabel(r"$x/c$")
plt.ylabel(r"$C_f$ (unitless)")
plt.grid(True)
