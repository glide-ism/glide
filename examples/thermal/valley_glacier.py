"""
Valley glacier thermal evolution.

Simulates the thermal development of an idealized alpine valley glacier
on a U-shaped bed carved into a mountain slope. The glacier has a
prescribed geometry and a SIA-like velocity field. Starting from a
uniform cold temperature, the enthalpy solver evolves the 3D thermal
field under:

  - Horizontal advection (down-valley ice transport)
  - Vertical diffusion (geothermal heat flux warming the base)
  - Surface temperature set by an atmospheric lapse rate
  - Drainage of meltwater in temperate ice

The simulation runs until the thermal profile approaches steady state,
producing cross-section plots of the temperature field at several times.

Run:
    python examples/thermal/valley_glacier.py
"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from glide.grid import Grid
from glide.enthalpy import (
    EnthalpyOperators, C_I, T_REF, T_MELT, RHO_I, GRAVITY, K_I,
)

# ========================================================
# Parameters
# ========================================================
ny, nx = 32, 128            # across-valley x along-valley
nz = 21
dx = 200.0                  # m

# Valley geometry
VALLEY_LENGTH = nx * dx      # m (along x)
VALLEY_WIDTH = ny * dx        # m (across y)
HEAD_ELEVATION = 3200.0       # m (bed at x=0)
TOE_ELEVATION = 1800.0        # m (bed at x=max)
VALLEY_DEPTH = 400.0          # m (U-shape depth below ridgeline)
VALLEY_FLOOR_WIDTH = 0.4      # fraction of domain width

# Ice geometry (prescribed steady-state-like profile)
MAX_THICKNESS = 350.0         # m (center of upper valley)
TERMINUS_FRACTION = 0.75      # glacier fills this fraction of valley length

# Climate
T_SEA_LEVEL = 288.15          # K (15 C)
LAPSE_RATE = -6.5e-3          # K/m
Q_GEO = 0.06                  # W/m^2 (moderate geothermal)
T_INIT = 253.15               # K (-20 C uniform initial)

# Velocity (SIA-like, prescribed)
A_GLEN = 2.4e-24              # Pa^-3 s^-1 (Glen's law rate factor)
N_GLEN = 3

# Time stepping
DT_YR = 5.0
N_STEPS = 400                 # 2000 yr total
N_SMOOTH = 20
SEC_PER_YR = 365.25 * 86400.0
DT_SEC = DT_YR * SEC_PER_YR

OUT_DIR = Path(__file__).parent / 'valley_glacier_output'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========================================================
# Build valley bed topography
# ========================================================
x = (cp.arange(nx, dtype=cp.float32) + 0.5) * dx
y = (cp.arange(ny, dtype=cp.float32) + 0.5) * dx
xx, yy = cp.meshgrid(x, y)

# Along-valley slope (headwall to toe)
x_frac = xx / VALLEY_LENGTH
bed_slope = HEAD_ELEVATION + (TOE_ELEVATION - HEAD_ELEVATION) * x_frac

# Cross-valley U-shape: parabolic, centered at y = VALLEY_WIDTH/2
y_center = VALLEY_WIDTH / 2.0
y_rel = (yy - y_center) / (VALLEY_WIDTH / 2.0)  # [-1, 1]

# Smooth U-shape: flat floor in center, steep walls at edges
floor_half = VALLEY_FLOOR_WIDTH / 2.0
y_wall = cp.clip((cp.abs(y_rel) - floor_half) / (1.0 - floor_half), 0.0, 1.0)
cross_profile = VALLEY_DEPTH * y_wall**2

bed = bed_slope + cross_profile

# ========================================================
# Build ice thickness (prescribed mature glacier profile)
# ========================================================
# Along-valley: thick near head, tapers to zero at terminus
x_glacier = cp.clip(1.0 - x_frac / TERMINUS_FRACTION, 0.0, 1.0)
along_profile = cp.sqrt(x_glacier)  # Vialov-like shape

# Cross-valley: thick in center, zero at walls
cross_thickness = cp.clip(1.0 - (cp.abs(y_rel) / (floor_half + 0.3))**2, 0.0, 1.0)

thickness = MAX_THICKNESS * along_profile * cross_thickness
thickness = cp.maximum(thickness, cp.float32(0.0))

surface = bed + thickness

# ========================================================
# Build prescribed velocity (SIA-like)
# ========================================================
def build_sia_velocity(bed, thickness):
    """
    Compute depth-averaged SIA velocity from bed slope and thickness.
    u_bar = -2A/(n+2) * (rho*g)^n * H^(n+1) * |grad(s)|^(n-1) * ds/dx
    """
    H = cp.asnumpy(thickness)
    b = cp.asnumpy(bed)
    s = b + H

    # Surface gradient (central differences)
    dsdx = np.zeros_like(s)
    dsdx[:, 1:-1] = (s[:, 2:] - s[:, :-2]) / (2 * dx)
    dsdx[:, 0] = (s[:, 1] - s[:, 0]) / dx
    dsdx[:, -1] = (s[:, -1] - s[:, -2]) / dx

    dsdy = np.zeros_like(s)
    dsdy[1:-1, :] = (s[2:, :] - s[:-2, :]) / (2 * dx)
    dsdy[0, :] = (s[1, :] - s[0, :]) / dx
    dsdy[-1, :] = (s[-1, :] - s[-2, :]) / dx

    slope_mag = np.sqrt(dsdx**2 + dsdy**2 + 1e-12)

    # Depth-averaged SIA velocity (m/s)
    coeff = 2.0 * A_GLEN / (N_GLEN + 2) * (RHO_I * GRAVITY)**N_GLEN
    u_cell = -coeff * H**(N_GLEN + 1) * slope_mag**(N_GLEN - 1) * dsdx
    v_cell = -coeff * H**(N_GLEN + 1) * slope_mag**(N_GLEN - 1) * dsdy

    # Interpolate to MAC faces
    u_face = np.zeros((ny, nx + 1), dtype=np.float32)
    u_face[:, 1:-1] = 0.5 * (u_cell[:, :-1] + u_cell[:, 1:])
    u_face[:, 0] = u_cell[:, 0]
    u_face[:, -1] = u_cell[:, -1]

    v_face = np.zeros((ny + 1, nx), dtype=np.float32)
    v_face[1:-1, :] = 0.5 * (v_cell[:-1, :] + v_cell[1:, :])
    v_face[0, :] = v_cell[0, :]
    v_face[-1, :] = v_cell[-1, :]

    # Broadcast to all sigma layers
    u3d = np.zeros((nz, ny, nx + 1), dtype=np.float32)
    v3d = np.zeros((nz, ny + 1, nx), dtype=np.float32)
    for k in range(nz):
        u3d[k] = u_face
        v3d[k] = v_face

    return cp.asarray(u3d), cp.asarray(v3d)


# ========================================================
# Surface temperature from lapse rate
# ========================================================
def surface_temperature(surface_elev):
    return cp.minimum(
        cp.float32(T_SEA_LEVEL) + cp.float32(LAPSE_RATE) * surface_elev,
        cp.float32(T_MELT))


# ========================================================
# Initialize
# ========================================================
grid = Grid(ny, nx, cp.float32(dx))
grid.state.H.set(thickness)
grid.state.H_prev.set(thickness)
grid.geometry.bed.set(bed)
grid.forcing.smb.set(cp.zeros((ny, nx), dtype=cp.float32))
grid.sliding.beta.set(cp.ones((ny, nx), dtype=cp.float32))
grid.sliding.m.set(1.0)
grid.sliding.u_reg.set(1.0)

ops = EnthalpyOperators(grid, nz=nz)
ops.smoother_config.report_norms = False
ops.smoother_config.n_newton = 5
ops.smoother_config.lf_c = cp.float32(1e-4)
ops.enthalpy_forcing.h_thin.set(30.0)

# Temperature initialization
T_surf = surface_temperature(surface)
ops.initialize_from_temperature(T_INIT)
ops.set_surface_enthalpy_from_temperature(T_surf)
ops.enthalpy_forcing.Q_geo[:] = Q_GEO
ops.enthalpy_forcing.Q_fh.fill(0)
ops.enthalpy_forcing.phi_strain.fill(0)
ops.enthalpy_forcing.drain_rate.set(0.01 / SEC_PER_YR)

# Velocity
u3d, v3d = build_sia_velocity(bed, thickness)
ops.enthalpy_velocity.u3d[:] = u3d
ops.enthalpy_velocity.v3d[:] = v3d

# Compute omega from continuity
smb_zero = cp.zeros((ny, nx), dtype=cp.float32)
ops.compute_omega(smb_zero)

# Print setup info
H_np = cp.asnumpy(thickness)
u_np = cp.asnumpy(u3d[0])
T_surf_np = cp.asnumpy(T_surf)
print(f"Valley glacier thermal evolution")
print(f"  Grid: {ny}x{nx}, nz={nz}, dx={dx}m")
print(f"  Valley: {VALLEY_LENGTH/1000:.0f} km long, "
      f"{VALLEY_WIDTH/1000:.0f} km wide")
print(f"  Bed elevation: {float(cp.min(bed)):.0f} to "
      f"{float(cp.max(bed)):.0f} m")
print(f"  Max thickness: {np.max(H_np):.0f} m")
print(f"  Max velocity: {np.max(np.abs(u_np))*SEC_PER_YR:.1f} m/yr")
print(f"  Surface T range: {np.min(T_surf_np[H_np>30]):.1f} to "
      f"{np.max(T_surf_np[H_np>30]):.1f} K")
print(f"  Q_geo = {Q_GEO} W/m^2")
print(f"  dt = {DT_YR} yr, {N_STEPS} steps = "
      f"{N_STEPS * DT_YR:.0f} yr")

# ========================================================
# Time stepping
# ========================================================
sigma = cp.asnumpy(ops.sigma)
j_center = ny // 2  # center of valley (cross-valley)

snapshot_steps = sorted({0, 10, 50, 100, N_STEPS // 4,
                         N_STEPS // 2, N_STEPS})
snapshots = {}

# Save initial
T_3d = cp.asnumpy(ops.get_temperature())
snapshots[0] = {'t_yr': 0.0, 'T_3d': T_3d.copy()}

print(f"\n{'step':>5s}  {'t (yr)':>8s}  {'T_bed max':>10s}  "
      f"{'T_bed ctr':>10s}  {'|r|':>10s}")
print("-" * 52)

for step in range(1, N_STEPS + 1):
    ops.set_rhs()
    ops.column_sweep(DT_SEC, N_SMOOTH)

    if step % 20 == 0 or step <= 5 or step in snapshot_steps:
        T_3d = cp.asnumpy(ops.get_temperature())

        # Basal temperature stats (only where ice is thick)
        T_bed = T_3d[:, :, 0].copy()
        T_bed[H_np < 30] = np.nan
        T_bed_max = np.nanmax(T_bed)

        # Center flowline, mid-glacier
        i_mid = nx // 3
        T_bed_ctr = T_3d[j_center, i_mid, 0]

        r = float(cp.max(cp.abs(ops.r_E)))

        if step % 20 == 0 or step <= 5:
            print(f"{step:5d}  {step*DT_YR:8.1f}  {T_bed_max:10.2f}  "
                  f"{T_bed_ctr:10.2f}  {r:10.2e}")

    if step in snapshot_steps:
        T_3d = cp.asnumpy(ops.get_temperature())
        snapshots[step] = {'t_yr': step * DT_YR, 'T_3d': T_3d.copy()}

t_final = N_STEPS * DT_YR
print(f"\nDone. Final time: {t_final:.0f} yr")

# ========================================================
# Plots
# ========================================================
x_km = cp.asnumpy(x) / 1000.0
y_km = cp.asnumpy(y) / 1000.0
bed_np = cp.asnumpy(bed)

# 1. Along-valley cross-sections (center of valley) at selected times
snap_keys = sorted(snapshots.keys())
n_snaps = min(len(snap_keys), 6)
snap_keys = snap_keys[:n_snaps]

fig, axes = plt.subplots(n_snaps, 1, figsize=(14, 3.0 * n_snaps),
                          sharex=True)
if n_snaps == 1:
    axes = [axes]

# Colorbar range from all snapshots
T_vals = []
for sk in snap_keys:
    T = snapshots[sk]['T_3d'][j_center, :, :]
    mask = H_np[j_center, :] > 30
    T_vals.extend(T[mask, :].flatten().tolist())
T_vmin, T_vmax = min(T_vals), max(T_vals)

for idx, sk in enumerate(snap_keys):
    ax = axes[idx]
    T_slice = snapshots[sk]['T_3d'][j_center, :, :]  # (nx, nz)
    bed_row = bed_np[j_center, :]
    H_row = H_np[j_center, :]

    # Build physical z-coordinates for plotting
    z_plot = np.zeros((nx, nz))
    T_plot = np.full((nx, nz), np.nan)
    for j in range(nx):
        if H_row[j] > 30:
            for k in range(nz):
                z_plot[j, k] = bed_row[j] + sigma[k] * H_row[j]
                T_plot[j, k] = T_slice[j, k]

    im = ax.pcolormesh(np.broadcast_to(x_km[:, None], (nx, nz)),
                        z_plot, T_plot,
                        cmap='RdYlBu_r', shading='gouraud',
                        vmin=T_vmin, vmax=T_vmax)
    ax.fill_between(x_km, bed_row, 0, color='saddlebrown', alpha=0.6)
    ax.plot(x_km, bed_row, 'k-', lw=1.0)
    ax.plot(x_km, np.where(H_row > 30, bed_row + H_row, np.nan),
            'b-', lw=1.2, label='Surface')
    ax.set_ylabel('Elevation (m)')
    ax.set_title(f't = {snapshots[sk]["t_yr"]:.0f} yr', fontsize=10)
    ax.set_ylim(float(np.min(bed_row)) - 100,
                float(np.max(bed_row + H_row)) + 200)

axes[-1].set_xlabel('Along-valley distance (km)')
fig.colorbar(im, ax=axes, label='Temperature (K)', fraction=0.015, pad=0.02)
fig.suptitle('Valley Glacier: Along-Valley Temperature (center flowline)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / 'valley_glacier_along.png', dpi=150)
print(f"Saved: {OUT_DIR / 'valley_glacier_along.png'}")

# 2. Cross-valley section at mid-glacier, final time
fig2, ax2 = plt.subplots(figsize=(8, 5))
T_final = snapshots[snap_keys[-1]]['T_3d']
i_cross = nx // 3  # upper third of glacier

T_cross = T_final[:, i_cross, :]  # (ny, nz)
bed_cross = bed_np[:, i_cross]
H_cross = H_np[:, i_cross]

z_cross = np.zeros((ny, nz))
T_cross_plot = np.full((ny, nz), np.nan)
for i in range(ny):
    if H_cross[i] > 30:
        for k in range(nz):
            z_cross[i, k] = bed_cross[i] + sigma[k] * H_cross[i]
            T_cross_plot[i, k] = T_cross[i, k]

im2 = ax2.pcolormesh(np.broadcast_to(y_km[:, None], (ny, nz)),
                      z_cross, T_cross_plot,
                      cmap='RdYlBu_r', shading='gouraud',
                      vmin=T_vmin, vmax=T_vmax)
ax2.fill_between(y_km, bed_cross, 0, color='saddlebrown', alpha=0.6)
ax2.plot(y_km, bed_cross, 'k-', lw=1.0)
ax2.plot(y_km, np.where(H_cross > 30, bed_cross + H_cross, np.nan),
         'b-', lw=1.2)
ax2.set_xlabel('Across-valley distance (km)')
ax2.set_ylabel('Elevation (m)')
ax2.set_title(f'Cross-Valley Temperature at x = {x_km[i_cross]:.1f} km, '
              f't = {t_final:.0f} yr')
plt.colorbar(im2, ax=ax2, label='Temperature (K)')
plt.tight_layout()
fig2.savefig(OUT_DIR / 'valley_glacier_cross.png', dpi=150)
print(f"Saved: {OUT_DIR / 'valley_glacier_cross.png'}")

# 3. Basal temperature map (plan view)
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

# Bed topography
bed_plot = bed_np.copy()
im3a = ax3a.pcolormesh(x_km, y_km, bed_plot, cmap='terrain',
                        shading='auto')
ax3a.contour(x_km, y_km, H_np, levels=[30], colors='blue',
             linewidths=1.5)
ax3a.set_title('Bed Elevation + Ice Extent')
ax3a.set_xlabel('Along-valley (km)')
ax3a.set_ylabel('Across-valley (km)')
ax3a.set_aspect('equal')
plt.colorbar(im3a, ax=ax3a, label='Bed elevation (m)')

# Basal temperature
T_bed_final = T_final[:, :, 0].copy()
T_bed_final[H_np < 30] = np.nan
im3b = ax3b.pcolormesh(x_km, y_km, T_bed_final, cmap='RdYlBu_r',
                        shading='auto')
ax3b.contour(x_km, y_km, H_np, levels=[30], colors='blue',
             linewidths=1.5)
ax3b.set_title(f'Basal Temperature at t = {t_final:.0f} yr')
ax3b.set_xlabel('Along-valley (km)')
ax3b.set_ylabel('Across-valley (km)')
ax3b.set_aspect('equal')
plt.colorbar(im3b, ax=ax3b, label='T (K)')

plt.tight_layout()
fig3.savefig(OUT_DIR / 'valley_glacier_basal.png', dpi=150)
print(f"Saved: {OUT_DIR / 'valley_glacier_basal.png'}")

plt.show()
