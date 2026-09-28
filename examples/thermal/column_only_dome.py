"""
Column-only dome thermal experiment.

Tests the enthalpy column solver in isolation on a dome geometry:
  - Horizontal advection: DISABLED
  - Strain heating:       DISABLED
  - Sigma dot:            ENABLED (vertical transport from accumulation)
  - Drainage:             DISABLED

Surface temperature follows a lapse rate: cold at the summit (~-15 C),
warm at the margins (above freezing, capped at T_melt). This creates
columns spanning the full cold-to-temperate range.

The momentum solver evolves the geometry but its velocity does not
enter the enthalpy equation (horizontal advection off). Only sigma_dot
from the mass continuity (SMB-driven vertical transport) is active.

Run:
    python examples/thermal/column_only_dome.py
"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from glide.model import IceDynamics, ThermalModel
from glide.enthalpy import T_MELT, BETA_CC, RHO_I, GRAVITY, E_SCALE

# ========================================================
# Parameters
# ========================================================
ny, nx = 512, 512
dx = 250.0             # m
n_levels = 3

DOME_RADIUS = 50000.0   # m
DOME_HEIGHT = 2000.0    # m

SMB_CENTER = 0.5        # m/yr accumulation at center
SMB_EDGE = -2.0         # m/yr ablation at margin
Q_GEO = 0.1            # W/m^2

# Surface temperature
T_SEA_LEVEL = 278.15    # K (+5 C — margins above freezing, capped at T_melt)
LAPSE_RATE = -6.5e-3    # K/m
T_INIT = 253.15         # K (-20 C uniform initial)

# Rheology (fixed, no thermal feedback for this test)
N_GLEN = 3.0
BETA_SLIDING = 10.0

# Thermal
NZ = 11
N_SMOOTH = 20

# Time stepping
DT_YR = 5.0
N_STEPS = 50
SEC_PER_YR = 365.25 * 86400.0

OUT_DIR = Path(__file__).parent / 'column_only_dome_output'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========================================================
# Build dome
# ========================================================
x = (cp.arange(nx, dtype=cp.float32) + 0.5) * dx - 0.5 * nx * dx
y = (cp.arange(ny, dtype=cp.float32) + 0.5) * dx - 0.5 * ny * dx
xx, yy = cp.meshgrid(x, y)
radius = cp.sqrt(xx**2 + yy**2)
radial_fraction = cp.clip(radius / DOME_RADIUS, 0.0, 1.0)
hemisphere = cp.sqrt(cp.clip(1.0 - radial_fraction**2, 0.0, 1.0))

thickness = DOME_HEIGHT * hemisphere
bed = cp.zeros((ny, nx), dtype=cp.float32)
smb = cp.float32(SMB_EDGE) + cp.float32(SMB_CENTER - SMB_EDGE) * hemisphere


def surface_temperature(H, bed):
    surface_elev = bed + H
    T = cp.float32(T_SEA_LEVEL) + cp.float32(LAPSE_RATE) * surface_elev
    return cp.minimum(T, cp.float32(T_MELT))


# ========================================================
# Initialize momentum solver
# ========================================================
model = IceDynamics(n_levels=n_levels, ny=ny, nx=nx, dx=cp.float32(dx))
mg = model.mg

mg.geometry.bed.set(bed)
mg.state.H.set(thickness)
mg.state.H_prev.set(thickness)
mg.forcing.smb.set(smb)
mg.rheology.B.set(cp.ones((ny, nx), dtype=cp.float32))
mg.rheology.n.set(N_GLEN)
mg.rheology.eps_reg.set(1e-6)
mg.sliding.beta.set(cp.full((ny, nx), BETA_SLIDING, dtype=cp.float32))
mg.sliding.m.set(1.0 / N_GLEN)
mg.sliding.u_reg.set(1.0)
mg.calving.timescale.set(cp.inf)  # no calving (molho API)

model.forward_solver.fas_options.set(
    coarsest_steps=200,
    pre_steps=5,
    post_steps=20,
    finest_steps=0,
    relative_tolerance=5e-5,
    absolute_tolerance=1e-3,
    report_norms=False,
)

# ========================================================
# Initialize thermal solver (column-only mode)
# ========================================================
grid = mg.levels[0]
thermal = ThermalModel(grid, nz=NZ, n_smooth=N_SMOOTH,
                       update_rheology=True, frictional_heating=False)

# phi_strain defaults to zero (no strain heating computation wired up)
thermal.ops.term_flags.drainage = False
thermal.ops.term_flags.omega = True
thermal.ops.smoother_config.report_norms = True
thermal.ops.smoother_config.n_newton = 5
thermal.ops.term_flags.horizontal_advection = False

T_surf_init = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
thermal.initialize(T_surface=T_surf_init, T_field=T_INIT, Q_geo=Q_GEO)
thermal.ops.enthalpy_forcing.h_thin.set(50.0)

# Initialize B from Paterson-Budd (fixed for this experiment)
B_init = thermal.ops.get_arrhenius_factor() / thermal.B_scale
mg.rheology.B.set(B_init)

T_summit = float(T_surf_init[ny//2, nx//2])
T_margin = float(T_SEA_LEVEL)
print(f"Surface T: summit = {T_summit:.1f} K ({T_summit-273.15:.1f} C), "
      f"margin = {T_margin:.1f} K ({T_margin-273.15:.1f} C)")
print(f"Term flags: bitmask = 0x{thermal.ops.term_flags.bitmask:x} "
      f"(h_adv={thermal.ops.term_flags.horizontal_advection}, "
      f"omega={thermal.ops.term_flags.omega}, "
      f"drain={thermal.ops.term_flags.drainage})")

# ========================================================
# Momentum spin-up
# ========================================================
dt_yr = cp.float32(DT_YR)
dt_sec = DT_YR * SEC_PER_YR
t_yr = 0.0

N_SPINUP = 5
print(f"Momentum spin-up ({N_SPINUP} steps)...")
for _ in range(N_SPINUP):
    model.forward(cp.float32(t_yr), dt_yr)
    t_yr += DT_YR

T_surf_relaxed = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
thermal.initialize(T_surface=T_surf_relaxed, T_field=T_INIT, Q_geo=Q_GEO)

# ========================================================
# Time stepping
# ========================================================
# Storage
times = [t_yr]
T_bed_center = []
T_bed_margin = []

# Pick a margin column (at ~0.8 * dome radius along x-axis)
j_margin = nx // 2 + int(0.8 * DOME_RADIUS / dx)
j_margin = min(j_margin, nx - 1)
i_center = ny // 2

# Extract initial profiles
sigma = cp.asnumpy(thermal.ops.sigma)


def get_T_profile(i, j):
    E = cp.asnumpy(thermal.ops.enthalpy_state.E[i, j, :]) * E_SCALE
    H = float(grid.state.H.data[i, j])
    T = np.zeros(NZ)
    for k in range(NZ):
        depth = (1.0 - sigma[k]) * H
        T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * depth
        T[k] = min(E[k] / 2009.0 + 223.15, T_pmp)
    return T


# Store profile snapshots
profile_times = [0, 50, 200, 500, 1000]
profile_snapshots_center = {}
profile_snapshots_margin = {}

print(f"\n{'step':>5s}  {'t (yr)':>8s}  {'T_bed center':>12s}  {'T_bed margin':>12s}  "
      f"{'H center':>9s}  {'H margin':>9s}")
print("-" * 65)

for step in range(N_STEPS):
    # Snapshot E and H before momentum step
    thermal.pre_momentum()

    # Momentum solve
    model.forward(cp.float32(t_yr), dt_yr)

    # Update surface T from current geometry
    T_surf = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
    thermal.ops.set_surface_enthalpy_from_temperature(T_surf)

    # Thermal solve (syncs velocities, computes omega, solves)
    thermal.step(dt_sec)

    t_yr += DT_YR

    # Diagnostics
    T_bc = float(get_T_profile(i_center, nx // 2)[0])
    H_c = float(grid.state.H.data[i_center, nx // 2])
    T_bm = float(get_T_profile(i_center, j_margin)[0])
    H_m = float(grid.state.H.data[i_center, j_margin])

    times.append(t_yr)
    T_bed_center.append(T_bc)
    T_bed_margin.append(T_bm)

    # Store profile snapshots
    t_total = t_yr - N_SPINUP * DT_YR
    for t_snap in profile_times:
        if abs(t_total - t_snap) < DT_YR / 2 and t_snap not in profile_snapshots_center:
            profile_snapshots_center[t_snap] = get_T_profile(i_center, nx // 2)
            profile_snapshots_margin[t_snap] = get_T_profile(i_center, j_margin)

    if (step + 1) % 20 == 0 or step == 0:
        print(f"{step+1:5d}  {t_yr:8.1f}  {T_bc:12.2f}  {T_bm:12.2f}  "
              f"{H_c:9.1f}  {H_m:9.1f}")

print(f"\nDone. {N_STEPS} steps, final t = {t_yr:.0f} yr")

# ========================================================
# Plots
# ========================================================

# 1. Time series
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(times[1:], T_bed_center, 'r-', label='Center (summit)')
ax1.plot(times[1:], T_bed_margin, 'b-', label=f'Margin (j={j_margin})')
T_pmp_center = T_MELT - BETA_CC * RHO_I * GRAVITY * DOME_HEIGHT
ax1.axhline(T_pmp_center, color='k', linestyle='--', alpha=0.4,
            label=f'$T_{{pmp}}$ at {DOME_HEIGHT:.0f} m')
ax1.axhline(T_MELT, color='gray', linestyle=':', alpha=0.4, label='$T_0$')
ax1.set_xlabel('Time (yr)')
ax1.set_ylabel('Basal temperature (K)')
ax1.set_title('Basal temperature evolution')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2D basal temperature at final time
T_3d = cp.asnumpy(thermal.temperature)
H_np = cp.asnumpy(grid.state.H.data)
T_bed_2d = T_3d[:, :, 0].copy()
T_bed_2d[H_np < 50] = np.nan

x_km = cp.asnumpy(x) / 1000.0
y_km = cp.asnumpy(y) / 1000.0
im = ax2.pcolormesh(x_km, y_km, T_bed_2d, cmap='coolwarm', shading='auto')
ax2.set_xlabel('x (km)')
ax2.set_ylabel('y (km)')
ax2.set_title(f'Basal temperature at t = {t_yr:.0f} yr')
ax2.set_aspect('equal')
plt.colorbar(im, ax=ax2, label='T (K)')

plt.tight_layout()
plt.savefig(OUT_DIR / 'column_only_time_series.png', dpi=150)
print(f"Saved {OUT_DIR / 'column_only_time_series.png'}")

# 2. Profile snapshots
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

for t_snap in sorted(profile_snapshots_center.keys()):
    T_prof = profile_snapshots_center[t_snap]
    ax3.plot(T_prof, sigma, label=f't = {t_snap} yr')

H_c = float(grid.state.H.data[i_center, nx // 2])
T_pmp_prof = np.array([T_MELT - BETA_CC * RHO_I * GRAVITY * (1 - s) * H_c
                        for s in sigma])
ax3.plot(T_pmp_prof, sigma, 'k--', alpha=0.4, label='$T_{pmp}$')
ax3.set_xlabel('Temperature (K)')
ax3.set_ylabel(r'$\sigma$ (bed=0, surface=1)')
ax3.set_title('Center column (summit)')
ax3.legend(fontsize=7)

for t_snap in sorted(profile_snapshots_margin.keys()):
    T_prof = profile_snapshots_margin[t_snap]
    ax4.plot(T_prof, sigma, label=f't = {t_snap} yr')

H_m = float(grid.state.H.data[i_center, j_margin])
if H_m > 10:
    T_pmp_prof_m = np.array([T_MELT - BETA_CC * RHO_I * GRAVITY * (1 - s) * H_m
                              for s in sigma])
    ax4.plot(T_pmp_prof_m, sigma, 'k--', alpha=0.4, label='$T_{pmp}$')
ax4.set_xlabel('Temperature (K)')
ax4.set_ylabel(r'$\sigma$ (bed=0, surface=1)')
ax4.set_title(f'Margin column (j={j_margin})')
ax4.legend(fontsize=7)

plt.tight_layout()
plt.savefig(OUT_DIR / 'column_only_profiles.png', dpi=150)
print(f"Saved {OUT_DIR / 'column_only_profiles.png'}")

# 3. Cross-section heatmap: temperature along central row
fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(12, 8))

T_row = T_3d[i_center, :, :]  # (nx, nz)
H_row = H_np[i_center, :]

# Mask where no ice
T_row_masked = T_row.copy()
for j in range(nx):
    if H_row[j] < 50:
        T_row_masked[j, :] = np.nan

im5 = ax5.pcolormesh(cp.asnumpy(x) / 1000.0, sigma, T_row_masked.T,
                      cmap='coolwarm', shading='auto')
ax5.set_xlabel('x (km)')
ax5.set_ylabel(r'$\sigma$')
ax5.set_title(f'Temperature cross-section (y=0) at t = {t_yr:.0f} yr')
plt.colorbar(im5, ax=ax5, label='T (K)')

ax6.plot(cp.asnumpy(x) / 1000.0, H_row, 'k-', linewidth=2)
ax6.fill_between(cp.asnumpy(x) / 1000.0, 0, H_row, alpha=0.2)
ax6.set_xlabel('x (km)')
ax6.set_ylabel('Ice thickness (m)')
ax6.set_title('Thickness cross-section (y=0)')

plt.tight_layout()
plt.savefig(OUT_DIR / 'column_only_cross_section.png', dpi=150)
print(f"Saved {OUT_DIR / 'column_only_cross_section.png'}")

plt.show()
