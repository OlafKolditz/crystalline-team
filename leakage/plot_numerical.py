import gmsh
import sys
import pyvista as pv
import os
import ogstools as ot
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import exp1
import scipy.special as sp

#-##########################################################
# 1 Plotting framework
# Set global default font sizes
plt.rcParams.update({
    'axes.titlesize': 14,    # Default title size for all axes
    'axes.labelsize': 14,    # Default x and y label size
    'xtick.labelsize': 14,   # X-axis tick label size
    'ytick.labelsize': 14,   # Y-axis tick label size
    'legend.fontsize': 14,    # Legend text size
})
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))


# Aquifer properties
S = 0.001
T = 9.2903e-4
L = 300
h0 = 0.0
Q = 0.016  # Pumping rate from well (m3/s)
# Discretization
xn = 100
rn = 100
r = np.arange(1, rn+1, 1)
#r = np.arange(0, rn, 1)
time_vals = [8.64, 86.4, 1728.0, 24192.0, 172800.0, 604800.0, 864000.0]
len_time_vals = len(time_vals)
r_vals = [0,1,10,20]
len_r_vals = len(r_vals)
t_val = 1728
t = np.arange(1, t_val+1, 1)

#-#############################
# 1.1 Plot values along line for selected time steps
ms = ot.MeshSeries(f"h-testing.pvd")
xaxis = np.column_stack((np.linspace(0.0, xn, xn), np.zeros((xn, 2))))
pressure = ot.variables.pressure.replace(data_unit="m",output_unit="m",output_name="hydraulic head")

# NEW: use ms.probe() instead of ot.MeshSeries.extract_probe()
ms_probe = ms.probe(xaxis)

# Plot results
ax[0].set_title("Profiles at different times")
labels = [f"$t={np.round(x, 2)}s$" for x in ms_probe[1:].timevalues]
ot.plot.line(ms_probe[1:], "x", pressure, labels=labels, ax=ax[0], fontsize=14)
#-#############################
# 1.2 Plot temporal values at selected points: Drawdown over time - Temporal evolution
## Data
x_vals = [0,1,10,20]
points_observation = np.array([[x_vals[0],0.0,0.0],[x_vals[1],0.0,0.0],[x_vals[2],0.0,0.0],[x_vals[3],0.0,0.0]])
pressure = ot.variables.pressure.replace(data_unit="m",output_unit="m",output_name="hydraulic head")
# NEW: use ms.probe() instead of ot.MeshSeries.extract_probe()
point_series = ms.probe(points_observation)
## Plots
labels = [f"$x={x}m$" for x in x_vals]

#OK ax[0].set_title("Temporal profiles at selected points", fontsize=14)
#OK ax[0].set_xscale('log')
plt.tight_layout()
#-#############################
# 1.1B Plot values along line for selected time steps
#-#############################
# 1.2B Plot temporal values at selected points
## Data
x_vals = [0.3048,1,10,20]
points_observation = np.array([[x_vals[0],0.0,0.0],[x_vals[1],0.0,0.0],[x_vals[2],0.0,0.0],[x_vals[3],0.0,0.0]])
pressure = ot.variables.pressure.replace(data_unit="m",output_unit="m",output_name="hydraulic head")
# NEW: use ms.probe() instead of ot.MeshSeries.extract_probe()
point_series = ms.probe(points_observation)
## Plots
labels = [f"$x={x}m$" for x in x_vals]
fig = ot.plot.line(point_series, "time", pressure, labels=labels, ax=ax[1], fontsize=14)
#for i, r_idx in enumerate(x_vals):
#    ax[1][1].plot(time_vals, s_all_r[r_idx, :],  # All times for this distance
#            marker='s', markersize=6, markevery=1,markerfacecolor='none')
ax[1].set_title("Temporal at different locations")
ax[1].set_xscale('log')
plt.tight_layout()

plt.show()