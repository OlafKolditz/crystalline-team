import matplotlib.pyplot as plt
import numpy as np
from scipy.special import exp1


# Theis
def calc_u(r, S, T, t):
    """Calculate and return the dimensionless time parameter, u."""
    return r**2 * S / 4 / T / t

def theis_drawdown(t, S, T, Q, r):
    """Calculate and return the drawdown s(r,t) for parameters S, T.

    This version uses the Theis equation, s(r,t) = Q * W(u) / (4.pi.T),
    where W(u) is the Well function for u = Sr^2 / (4Tt).
    S is the aquifer storage coefficient,
    T is the transmissivity (m2/day),
    r is the distance from the well (m), and
    Q is the pumping rate (m3/day).
    """
    u = calc_u(r, S, T, t)
    return Q / 4 / np.pi / T * exp1(u)

time_vals = [8.64, 86.4, 1728.0, 24192.0, 172800.0, 604800.0, 864000.0]
len_time_vals = len(time_vals)
rn = 100
r = np.arange(1, rn+1, 1)

# Aquifer properties
S = 0.001
T = 9.2903e-4
L = 300
h0 = 0.0
Q = 0.016  # Pumping rate from well (m3/s)

s_all_theis = np.zeros((rn, len(time_vals)))
for ii in range(len_time_vals):
    u = calc_u(r, S, T, time_vals[ii])
    s = theis_drawdown(time_vals[ii], S, T, Q, r)
    s_all_theis[:, ii] = s    


# Assuming: r = radial distances array (length rn), time_vals = time array
# Select a few representative time indices (e.g., early, middle, late)
time_indices = [0, len(time_vals)//4, len(time_vals)//2, -1]
colors = ['darkblue', 'steelblue', 'indianred', 'maroon']
labels = [f't = {time_vals[i]:.0f} s' for i in time_indices]

fig, ax = plt.subplots(figsize=(9, 6))

for i, idx in enumerate(time_indices):
    ax.plot(r, s_all_theis[:, idx],  # All distances for this time
            color=colors[i],
            linewidth=2,
            label=labels[i])

ax.set_xlabel('Radial Distance from Well [m]', fontsize=12)
ax.set_ylabel('Drawdown [m]', fontsize=12)
ax.set_title('Theis Drawdown: Spatial Propagation', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(title='Time')
ax.set_xlim(left=0)  # Distance starts at 0

# Optional: Set log scale if distances span orders of magnitude
# ax.set_xscale('log')
plt.tight_layout()
plt.show()