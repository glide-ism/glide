"""
Stefan reference example: cold-to-temperate basal transition.

A flat ice slab with high geothermal heat flux can develop a temperate
basal layer. The reference profile plotted here is a steady 1D enthalpy
solution consistent with the model's cold/temperate diffusivities in the
sharp-transition limit:

  Cold zone (sigma* < sigma < 1):
    E varies linearly with slope -Q_geo H / K_cold.
    The CTS is where E first reaches E_pmp(sigma).

  Temperate zone (sigma < sigma*):
    E continues increasing downward with slope -Q_geo H / K_temp.
    Temperature is clipped to T_pmp(sigma), and the excess enthalpy
    above E_pmp becomes water content.

This is the quantity that should be compared against the modeled steady
enthalpy. Unlike temperature, enthalpy is not capped at E_pmp in the
temperate layer.
"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import brentq
from glide.grid import Grid
from glide.enthalpy import (
    EnthalpyOperators, C_I, K_I, T_REF, T_MELT, RHO_I, GRAVITY, BETA_CC,
    L_HEAT, K_COLD, K_TEMP, K_TEMP_FACTOR, E_SCALE
)

# ========================================================
# Parameters
# ========================================================
ny, nx = 4, 4
dx = 1000.0
H_ice = 1000.0
T_surface = 223.15     # K (-50 C)
Q_geo = 0.5            # W/m^2 (high — triggers temperate base)
nz = 64
n_smooth = 20

# Convergence parameters
conv_tol = 1e-6         # relative enthalpy change for steady-state
max_steps = 50000       # upper bound on total steps
log_interval = 500      # print every N steps

SEC_PER_KYR = 1000.0 * 365.25 * 86400.0

OUT_DIR = Path(__file__).parent / 'stefan_output'
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
ops.enthalpy_forcing.drain_rate.set(0.0)
ops.enthalpy_velocity.u3d.fill(0)
ops.enthalpy_velocity.v3d.fill(0)
ops.enthalpy_velocity.omega.fill(0)
ops.enthalpy_forcing.Q_fh.fill(0)

# ========================================================
# Semi-analytical enthalpy reference profile
# ========================================================
sigma = cp.asnumpy(ops.sigma)
depth = (1.0 - sigma) * H_ice

def T_pmp(sig):
    """Pressure melting point at sigma level."""
    d = (1 - sig) * H_ice
    return T_MELT - BETA_CC * RHO_I * GRAVITY * d

def E_pmp(sig):
    """Enthalpy at pressure melting point at sigma level."""
    return C_I * (T_pmp(sig) - T_REF)

T_analytical = np.zeros_like(sigma)
E_analytical = np.zeros_like(sigma)
omega_analytical = np.zeros_like(sigma)

T_pmp_profile = np.array([T_pmp(s) for s in sigma])
E_pmp_profile = np.array([E_pmp(s) for s in sigma])

T_pmp_bed = T_pmp(0.0)
E_pmp_bed = E_pmp(0.0)
Q_crit = K_I * (T_pmp_bed - T_surface) / H_ice

if Q_geo <= Q_crit:
    regime = 'cold bed'
    E_analytical[:] = C_I * (T_surface - T_REF) + Q_geo * H_ice / K_COLD * (1.0 - sigma)
    sigma_star = None
else:
    regime = 'temperate basal layer expected'

    def cts_residual(sig_star):
        return Q_geo * H_ice * (1.0 - sig_star) / K_I - (T_pmp(sig_star) - T_surface)

    sigma_star = brentq(cts_residual, 0.0, 1.0 - 1e-10)
    E_cts = E_pmp(sigma_star)

    for k, sig in enumerate(sigma):
        if sig <= sigma_star:
            E_analytical[k] = E_cts + Q_geo * H_ice / K_TEMP * (sigma_star - sig)
        else:
            E_analytical[k] = C_I * (T_surface - T_REF) + Q_geo * H_ice / K_COLD * (1.0 - sig)

omega_analytical[:] = np.maximum(E_analytical - E_pmp_profile, 0.0) / L_HEAT
T_analytical[:] = np.minimum(E_analytical / C_I + T_REF, T_pmp_profile)

# ========================================================
# Print header with equilibration time estimates
# ========================================================
kappa_cold = K_I / (RHO_I * C_I)
kappa_temp = kappa_cold * K_TEMP_FACTOR

print(f"  Regime:           {regime}")
print(f"  T_pmp at bed:     {T_pmp_bed:.2f} K")
print(f"  T_surface:        {T_surface:.2f} K")
print(f"  Q_geo:            {Q_geo:.3f} W/m^2")
print(f"  Q_crit:           {Q_crit:.3f} W/m^2")
print(f"  K_cold (k/c):     {K_COLD:.3e} kg/(m*s)")
print(f"  K_temp:           {K_TEMP:.3e} kg/(m*s)")
print(f"  kappa_cold:       {kappa_cold:.3e} m^2/s")
print(f"  kappa_temp:       {kappa_temp:.3e} m^2/s")

tau_cold = H_ice**2 / (kappa_cold * np.pi**2)
print(f"  Cold zone tau:    {tau_cold / SEC_PER_KYR:.1f} kyr")
if sigma_star is not None:
    L_temp = sigma_star * H_ice
    tau_temp = L_temp**2 / (kappa_temp * np.pi**2)
    print(f"  Temperate tau:    {tau_temp / (1e6 * SEC_PER_KYR):.1f} Myr")
    print(f"  Reference CTS:    sigma* = {sigma_star:.4f}")
    print(f"  Reference omega_bed: {omega_analytical[0]:.3e}")

# ========================================================
# Time stepping with adaptive dt and convergence check
#
# Strategy: start with a moderate dt for the transient phase,
# then ramp dt toward pseudo-steady-state to accelerate
# convergence of the slow temperate zone.
# ========================================================
i_col, j_col = ny // 2, nx // 2

dt = 100.0 * SEC_PER_KYR        # initial time step
dt_max = 1e6 * SEC_PER_KYR      # cap for pseudo-steady-state
dt_growth = 1.5                  # dt multiplier when converging

# Snapshot times (kyr) for plotting
snapshot_times_kyr = [0, 1, 5, 20, 100]
snapshots_E = {}
snapshots_T = {}
snapshots_omega = {}
step = 0
time_kyr = 0.0
converged = False

E_col = lambda: cp.asnumpy(ops.enthalpy_state.E[i_col, j_col, :]) * E_SCALE
T_col = lambda: cp.asnumpy(ops.get_temperature()[i_col, j_col, :])
W_col = lambda: cp.asnumpy(ops.get_water_content()[i_col, j_col, :])

# Store initial snapshot
snapshots_E[0.0] = E_col().copy()
snapshots_T[0.0] = T_col().copy()
snapshots_omega[0.0] = W_col().copy()

next_snap_idx = 1  # index into snapshot_times_kyr

print(f"\n  {'step':>6s}  {'time (kyr)':>12s}  {'dt (kyr)':>10s}  "
      f"{'max|dE|':>12s}  {'rel change':>12s}  {'T_bed (K)':>10s}  {'omega_bed':>12s}")
print(f"  {'-'*82}")

for step in range(1, max_steps + 1):
    E_old = ops.enthalpy_state.E[i_col, j_col, :].copy()

    ops.set_rhs()
    ops.column_sweep(dt, n_smooth)

    E_new = ops.enthalpy_state.E[i_col, j_col, :]
    dE = E_new - E_old
    max_dE = float(cp.max(cp.abs(dE)))
    E_scale = float(cp.max(cp.abs(E_new))) + 1.0
    rel_change = max_dE / E_scale

    dt_kyr = dt / SEC_PER_KYR
    time_kyr += dt_kyr

    # Check if we should capture a snapshot
    while (next_snap_idx < len(snapshot_times_kyr)
           and time_kyr >= snapshot_times_kyr[next_snap_idx]):
        t_snap = snapshot_times_kyr[next_snap_idx]
        snapshots_E[t_snap] = E_col().copy()
        snapshots_T[t_snap] = T_col().copy()
        snapshots_omega[t_snap] = W_col().copy()
        next_snap_idx += 1

    # Log periodically
    if step % log_interval == 0 or step <= 5 or rel_change < conv_tol:
        T_bed = float(cp.asnumpy(ops.get_temperature()[i_col, j_col, 0]))
        omega_bed = float(cp.asnumpy(ops.get_water_content()[i_col, j_col, 0]))
        print(f"  {step:6d}  {time_kyr:12.1f}  {dt_kyr:10.1f}  "
              f"{max_dE:12.3e}  {rel_change:12.3e}  {T_bed:10.2f}  {omega_bed:12.3e}")

    # Convergence check
    if rel_change < conv_tol:
        print(f"\n  Converged at step {step}, t = {time_kyr:.1f} kyr (rel change = {rel_change:.2e})")
        converged = True
        break

    # Adaptive dt: grow when converging, to accelerate approach to steady state
    if rel_change < 1e-2:
        dt = min(dt * dt_growth, dt_max)

# Store final state as the last snapshot
final_kyr = time_kyr
snapshots_E[final_kyr] = E_col().copy()
snapshots_T[final_kyr] = T_col().copy()
snapshots_omega[final_kyr] = W_col().copy()

if not converged:
    print(f"\n  Did not converge in {max_steps} steps (t = {time_kyr:.1f} kyr, rel = {rel_change:.2e})")

# ========================================================
# Plot: steady-state model vs semi-analytical reference
# 2x1: temperature and enthalpy
# ========================================================
T_final = T_col()
E_final = E_col()
omega_final = W_col()

fig, (ax_T, ax_E) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# --- Temperature ---
ax_T.plot(T_final, sigma, 'b-', linewidth=2, label='Model (steady state)')
ax_T.plot(T_analytical, sigma, 'k--', linewidth=2, label='Semi-analytical')
ax_T.plot(T_pmp_profile, sigma, 'r:', linewidth=1.5, alpha=0.7,
          label=r'$T_{\mathrm{pmp}}(\sigma)$')
if sigma_star is not None:
    ax_T.axhline(sigma_star, color='green', linestyle='--', linewidth=1,
                 alpha=0.6, label=rf'CTS $\sigma^*={sigma_star:.3f}$')
ax_T.set_xlabel('Temperature (K)')
ax_T.set_ylabel(r'$\sigma$ (0 = bed, 1 = surface)')
ax_T.set_title('Temperature')
ax_T.legend(fontsize=9)
ax_T.grid(True, alpha=0.3)

# --- Enthalpy ---
ax_E.plot(E_final / 1e3, sigma, 'b-', linewidth=2, label='Model (steady state)')
ax_E.plot(E_analytical / 1e3, sigma, 'k--', linewidth=2, label='Semi-analytical')
ax_E.plot(E_pmp_profile / 1e3, sigma, 'r:', linewidth=1.5, alpha=0.7,
          label=r'$E_{\mathrm{pmp}}(\sigma)$')
if sigma_star is not None:
    ax_E.axhline(sigma_star, color='green', linestyle='--', linewidth=1, alpha=0.6)
ax_E.set_xlabel('Enthalpy (kJ/kg)')
ax_E.set_title('Enthalpy')
ax_E.legend(fontsize=9)
ax_E.grid(True, alpha=0.3)

max_T_err = np.max(np.abs(T_final[:-1] - T_analytical[:-1]))
max_E_rel = np.max(np.abs(E_final[:-1] - E_analytical[:-1])) / np.max(np.abs(E_analytical[:-1]))
fig.suptitle(f'Polythermal Column: Steady State  |  '
             f'max T err = {max_T_err:.2e} K, max E rel err = {max_E_rel:.2e}',
             fontsize=12)
plt.tight_layout()
out = OUT_DIR / 'stefan.png'
plt.savefig(out, dpi=150)
plt.show()
print(f"\nSaved: {out}")
