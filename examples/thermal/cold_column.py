"""
Cold column example: steady-state temperature profile in a static ice slab.

A flat ice slab with geothermal heat flux at the base and a fixed surface
temperature. No ice flow. The base stays cold (below the pressure melting
point), so the steady-state temperature is a simple linear profile:

    T(sigma) = T_s + (Q_geo * H / k_i) * (1 - sigma)

Outputs a plot comparing the numerical solution against the analytical
profile for both temperature and enthalpy.
"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from glide.grid import Grid
from glide.enthalpy import (
    EnthalpyOperators, C_I, K_I, T_REF, T_MELT, RHO_I, GRAVITY, BETA_CC,
    E_SCALE
)

# ========================================================
# Parameters
# ========================================================
ny, nx = 4, 4
dx = 1000.0            # m (irrelevant — no horizontal gradients)
H_ice = 1000.0         # m
T_surface = 243.15     # K (-30 C)
Q_geo = 0.04           # W/m^2
nz = 21
dt = 1000.0 * 365.25 * 86400.0   # 1000 yr in seconds
n_smooth = 20
snapshot_times_kyr = [0, 5, 20, 50, 100, 200]  # kyr

OUT_DIR = Path(__file__).parent / 'cold_column_output'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========================================================
# Grid and operator setup
# ========================================================
grid = Grid(ny, nx, cp.float32(dx))
grid.state.H.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
grid.state.H_prev.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
grid.geometry.bed.set(cp.zeros((ny, nx), dtype=cp.float32))
grid.forcing.smb.set(cp.zeros((ny, nx), dtype=cp.float32))
grid.sliding.beta.set(cp.ones((ny, nx), dtype=cp.float32))
grid.sliding.m.set(1.0)
grid.sliding.u_reg.set(1.0)

ops = EnthalpyOperators(grid, nz=nz)
ops.initialize_from_temperature(T_surface)
ops.set_surface_enthalpy_from_temperature(
    cp.full((ny, nx), T_surface, dtype=cp.float32))
ops.enthalpy_forcing.Q_geo[:, :] = Q_geo
ops.enthalpy_forcing.phi_strain.fill(0)
ops.enthalpy_velocity.u3d.fill(0)
ops.enthalpy_velocity.v3d.fill(0)
ops.enthalpy_velocity.omega.fill(0)
ops.enthalpy_forcing.Q_fh.fill(0)

# ========================================================
# Analytical solution
# ========================================================
sigma = cp.asnumpy(ops.sigma)
depth = (1 - sigma) * H_ice   # depth below surface (m)
T_analytical = T_surface + Q_geo * H_ice / K_I * (1 - sigma)
E_analytical = C_I * (T_analytical - T_REF)
T_pmp_bed = T_MELT - BETA_CC * RHO_I * GRAVITY * H_ice

# ========================================================
# Time stepping with snapshots
# ========================================================
i_col, j_col = ny // 2, nx // 2
dt_kyr = dt / (1000.0 * 365.25 * 86400.0)  # dt in kyr

snapshots_E = {}
snapshots_T = {}
step = 0
time_kyr = 0.0

# Save initial
E_init = cp.asnumpy(ops.enthalpy_state.E[i_col, j_col, :]) * E_SCALE
snapshots_E[0.0] = E_init.copy()
snapshots_T[0.0] = E_init / C_I + T_REF

for target_kyr in snapshot_times_kyr[1:]:
    while time_kyr < target_kyr:
        ops.set_rhs()
        ops.column_sweep(dt, n_smooth)
        step += 1
        time_kyr += dt_kyr
    E_snap = cp.asnumpy(ops.enthalpy_state.E[i_col, j_col, :]) * E_SCALE
    snapshots_E[target_kyr] = E_snap.copy()
    snapshots_T[target_kyr] = E_snap / C_I + T_REF
    print(f"  t = {target_kyr:6.0f} kyr, T_bed = {snapshots_T[target_kyr][0]:.2f} K")

# ========================================================
# Plot: 2x2 (temperature / enthalpy) x (profiles / error)
# sigma on y-axis: 0 = bed (bottom), 1 = surface (top)
# ========================================================
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
(ax_T, ax_T_err), (ax_E, ax_E_err) = axs

cmap = plt.cm.coolwarm
colors = cmap(np.linspace(0.1, 0.9, len(snapshots_T)))

# --- Temperature profiles ---
for (t_kyr, T_profile), color in zip(snapshots_T.items(), colors):
    ax_T.plot(T_profile, sigma, color=color, linewidth=1.5, label=f't = {t_kyr:.0f} kyr')
ax_T.plot(T_analytical, sigma, 'k--', linewidth=2, label='Analytical (steady)')
ax_T.axvline(T_pmp_bed, color='red', linestyle=':', alpha=0.5)
ax_T.set_xlabel('Temperature (K)')
ax_T.set_ylabel(r'$\sigma$ (0 = bed, 1 = surface)')
ax_T.set_title('Temperature')
ax_T.legend(fontsize=7)
ax_T.grid(True, alpha=0.3)

# --- Temperature error ---
T_final = snapshots_T[snapshot_times_kyr[-1]]
T_err = T_final - T_analytical
ax_T_err.plot(T_err, sigma, 'b-', linewidth=2)
ax_T_err.set_xlabel('Temperature Error (K)')
ax_T_err.set_title(f'Temperature Error at t = {snapshot_times_kyr[-1]} kyr')
ax_T_err.axvline(0, color='k', linestyle='-', alpha=0.3)
ax_T_err.grid(True, alpha=0.3)
max_T_err = np.max(np.abs(T_err[:-1]))
ax_T_err.text(0.95, 0.05, f'Max error: {max_T_err:.4f} K',
              transform=ax_T_err.transAxes, ha='right', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# --- Enthalpy profiles ---
for (t_kyr, E_profile), color in zip(snapshots_E.items(), colors):
    ax_E.plot(E_profile / 1e3, sigma, color=color, linewidth=1.5, label=f't = {t_kyr:.0f} kyr')
ax_E.plot(E_analytical / 1e3, sigma, 'k--', linewidth=2, label='Analytical (steady)')
ax_E.set_xlabel('Enthalpy (kJ/kg)')
ax_E.set_ylabel(r'$\sigma$ (0 = bed, 1 = surface)')
ax_E.set_title('Enthalpy')
ax_E.legend(fontsize=7)
ax_E.grid(True, alpha=0.3)

# --- Enthalpy error ---
E_final = snapshots_E[snapshot_times_kyr[-1]]
E_err = E_final - E_analytical
ax_E_err.plot(E_err, sigma, 'b-', linewidth=2)
ax_E_err.set_xlabel('Enthalpy Error (J/kg)')
ax_E_err.set_title(f'Enthalpy Error at t = {snapshot_times_kyr[-1]} kyr')
ax_E_err.axvline(0, color='k', linestyle='-', alpha=0.3)
ax_E_err.grid(True, alpha=0.3)
max_E_err = np.max(np.abs(E_err[:-1]))
ax_E_err.text(0.95, 0.05, f'Max error: {max_E_err:.2f} J/kg',
              transform=ax_E_err.transAxes, ha='right', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('Cold Column: Steady-State Diffusion', fontsize=14, fontweight='bold')
plt.tight_layout()
out = OUT_DIR / 'cold_column.png'
plt.savefig(out, dpi=150)
plt.show()
print(f"\nSaved: {out}")
