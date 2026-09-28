"""
Horizontal advection example: temperature bump on a static ice slab.

A flat ice slab with fixed geometry and a prescribed uniform horizontal
velocity. A warm Gaussian temperature bump at the center advects
downstream. No vertical velocity, no heat sources.

This tests that horizontal advection:
  - Transports the temperature pattern in the correct direction
  - Preserves symmetry perpendicular to the flow
  - Converges without residual growth

Outputs a plot of the depth-averaged temperature at selected times.
"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from glide.grid import Grid
from glide.enthalpy import (
    EnthalpyOperators, C_I, K_I, T_REF, T_MELT, RHO_I, E_SCALE
)

# ========================================================
# Parameters
# ========================================================
ny, nx = 32*5, 64*5
nz = 9
dx = 250.0           # m
H_ice = 1000.0        # m
T_background = 243.15 # K (-30 C)
T_bump = 260.0        # K (-13 C, well below melting)

# Velocity: uniform rightward, 500 m/yr converted to m/s
U_MYR = 1000.0
SEC_PER_YR = 365.25 * 86400.0
U_MS = U_MYR / SEC_PER_YR

# Gaussian bump parameters
bump_sigma_x = 5000.0  # m (std dev in x)
bump_sigma_y = 5000.0  # m (std dev in y)

# Time stepping
DT_YR = 1.0
DT_SEC = DT_YR * SEC_PER_YR
N_STEPS = 100
N_SMOOTH = 20

OUT_DIR = Path(__file__).parent / 'horizontal_advection_output'
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

# Initial temperature: background + Gaussian bump centered at 1/4 of domain
x_c = (cp.arange(nx, dtype=cp.float32) + 0.5) * dx
y_c = (cp.arange(ny, dtype=cp.float32) + 0.5) * dx
xx, yy = cp.meshgrid(x_c, y_c)

x_center = nx * dx * 0.25  # bump starts at 1/4 from left
y_center = ny * dx * 0.5   # centered in y
bump = (T_bump - T_background) * cp.exp(
    -0.5 * ((xx - x_center)**2 / bump_sigma_x**2
           + (yy - y_center)**2 / bump_sigma_y**2))

T_init_2d = T_background + bump  # (ny, nx)
T_init_3d = cp.broadcast_to(T_init_2d[:, :, None],
                             (ny, nx, nz)).copy()

ops.enthalpy_state.E[:] = C_I * (T_init_3d - T_REF) / E_SCALE
ops.enthalpy_state.E_prev[:] = ops.enthalpy_state.E
ops.set_surface_enthalpy_from_temperature(
    cp.full((ny, nx), T_background, dtype=cp.float32))

# No heat sources
ops.enthalpy_forcing.Q_geo.fill(0)
ops.enthalpy_forcing.Q_fh.fill(0)
ops.enthalpy_forcing.phi_strain.fill(0)
ops.enthalpy_forcing.drain_rate.set(0.0)

# Uniform rightward velocity (m/s) on all sigma layers
ops.enthalpy_velocity.u3d.fill(cp.float32(U_MS))
ops.enthalpy_velocity.v3d.fill(0)
ops.enthalpy_velocity.omega.fill(0)

# Smoother config
ops.smoother_config.report_norms = False

# ========================================================
# Time stepping
# ========================================================
x_km = cp.asnumpy(x_c) / 1000.0
y_km = cp.asnumpy(y_c) / 1000.0

snapshot_steps = [0, N_STEPS // 4, N_STEPS // 2, 3 * N_STEPS // 4, N_STEPS]
snapshots = {}

# Save initial
T_3d = cp.asnumpy(ops.get_temperature())
snapshots[0] = {'t_yr': 0.0, 'T_avg': np.mean(T_3d, axis=2)}

print(f"Horizontal advection example: {nx}x{ny} grid, u = {U_MYR} m/yr")
print(f"  dt = {DT_YR} yr, {N_STEPS} steps, {N_SMOOTH} smooths/step\n")
print(f"  {'step':>5s}  {'t (yr)':>8s}  {'|r|':>10s}  "
      f"{'T_max (K)':>10s}  {'y-symmetry':>12s}")
print(f"  {'-'*55}")

for step in range(1, N_STEPS + 1):
    ops.set_rhs()

    # Per-sweep residuals for first and last step
    if step == 1 or step == N_STEPS:
        cfg = ops.smoother_config
        r0 = float(ops.compute_residual(DT_SEC))
        print(f"\n  Step {step} per-sweep residuals:")
        print(f"    initial |r0| = {r0:.2e}")
        for sweep in range(N_SMOOTH):
            ops.column_smooth(DT_SEC)
            ops.enthalpy_state.E[:] += cfg.omega * ops.delta_E
            ops.layer_smooth(DT_SEC)
            ops.enthalpy_state.E[:] += cfg.omega * ops.delta_E
            r = float(ops.compute_residual(DT_SEC))
            rel = r / r0 if r0 > 0 else 0.0
            print(f"    sweep {sweep:2d}: |r|/|r0| = {rel:.2e}, |r| = {r:.2e}")
            if rel < cfg.relative_tolerance or r < cfg.absolute_tolerance:
                break
        r_final = r
    else:
        r0 = float(ops.compute_residual(DT_SEC))
        ops.column_sweep(DT_SEC, N_SMOOTH)
        r_final = float(ops.compute_residual(DT_SEC))

    # Diagnostics
    T_3d = cp.asnumpy(ops.get_temperature())
    T_avg = np.mean(T_3d, axis=2)
    T_max = np.max(T_avg)

    # y-symmetry: the bump is y-centered, so the field should be
    # symmetric about y = ny/2.  Measure max asymmetry.
    T_flip = T_avg[::-1, :]
    y_asym = np.max(np.abs(T_avg - T_flip))

    if step % 5 == 0 or step == 1:
        print(f"  {step:5d}  {step * DT_YR:8.1f}  {r_final:10.2e}  "
              f"{T_max:10.4f}  {y_asym:12.2e}")

    if step in snapshot_steps:
        snapshots[step] = {'t_yr': step * DT_YR, 'T_avg': T_avg.copy()}

# Expected bump displacement
displacement_km = U_MYR * N_STEPS * DT_YR / 1000.0
print(f"\n  Expected bump displacement: {displacement_km:.1f} km "
      f"({U_MYR * N_STEPS * DT_YR:.0f} m)")

# ========================================================
# Plot: depth-averaged temperature at selected times
# ========================================================
n_snaps = len(snapshots)
fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps, 4),
                          sharex=True, sharey=True)

vmin = T_background
vmax = T_bump

for ax, (step_key, snap) in zip(axes, sorted(snapshots.items())):
    im = ax.pcolormesh(x_km, y_km, snap['T_avg'],
                       cmap='coolwarm', shading='auto',
                       vmin=vmin, vmax=vmax)
    ax.set_title(f"t = {snap['t_yr']:.0f} yr")
    ax.set_aspect('equal')
    ax.set_xlabel('x (km)')

axes[0].set_ylabel('y (km)')
fig.colorbar(im, ax=axes, label='Depth-avg Temperature (K)',
             fraction=0.02, pad=0.04)
fig.suptitle(f'Horizontal Advection: u = {U_MYR} m/yr, '
             f'bump advects {displacement_km:.0f} km rightward',
             fontsize=13, fontweight='bold')
plt.tight_layout()

out = OUT_DIR / 'horizontal_advection.png'
plt.savefig(out, dpi=150)
plt.show()
print(f"  Saved: {out}")
