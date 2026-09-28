"""
Coupled valley glacier simulation.

A fully coupled ice-dynamics + enthalpy simulation of an idealized alpine
valley glacier on a U-shaped bed carved into a mountain slope:

  Momentum (SSA)  -->  velocity u, v  -->  Enthalpy advection + frictional heating
  Enthalpy        -->  Arrhenius B(T) -->  Momentum viscosity
  Mass balance    -->  ice front advance/retreat

The glacier starts from a prescribed initial geometry and evolves under
an elevation-dependent surface mass balance (accumulation above the ELA,
ablation below). The enthalpy solver tracks the full 3D thermal field
with geothermal heating, frictional heating, lapse-rate surface BCs,
and meltwater drainage.

Run:
    python examples/thermal/coupled_valley_glacier.py
"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from glide.model import IceDynamics, ThermalModel
from glide.enthalpy import T_MELT, BETA_CC, RHO_I, GRAVITY

# ========================================================
# Parameters
# ========================================================
# Grid
n_levels = 3
ny, nx = 64*2, 128*2
nz = 21
dx = 250.0  # m

# Valley geometry
VALLEY_LENGTH = nx * dx       # m (along x)
VALLEY_WIDTH = ny * dx        # m (across y)
HEAD_ELEVATION = 3000.0       # m (bed at x=0)
TOE_ELEVATION = 1500.0        # m (bed at x=max)
VALLEY_DEPTH = 500.0          # m (U-shape depth below ridgeline)
VALLEY_FLOOR_WIDTH = 0.5      # fraction of domain width

# Initial ice geometry (prescribed starting profile)
MAX_THICKNESS = 350.0         # m (center of upper valley)
TERMINUS_FRACTION = 0.75      # initial glacier fills this fraction of valley

# Climate / forcing
ELA = 2750.0                  # m (equilibrium line altitude)
SMB_GRAD = 0.007              # m/yr per m elevation (mass balance gradient)
SMB_MAX = 2.0               # m/yr (accumulation cap)
SMB_MIN = -4.0                # m/yr (ablation cap)
T_HEAD = 248.15               # K at highest bed elevation)
T_TOE = 288.15                # K at lowest bed elevation)
LAPSE_RATE = (T_TOE - T_HEAD) / (TOE_ELEVATION - HEAD_ELEVATION)  # K/m (derived)
Q_GEO = 0.0                  # W/m^2 (geothermal heat flux)
T_INIT = T_HEAD               # K (uniform initial, coldest surface T)

# Rheology
N_GLEN = 3.0

# Sliding
BETA_SLIDING = 2.5          # basal friction coefficient (GLIDE head units)

# Thermal
N_SMOOTH = 25                 # enthalpy smoothing sweeps

# Time stepping
DT_YR = 5.0
N_STEPS = 100              
SEC_PER_YR = 365.25 * 86400.0

OUT_DIR = Path(__file__).parent / 'coupled_valley_glacier_output'
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

floor_half = VALLEY_FLOOR_WIDTH / 2.0
y_wall = cp.clip((cp.abs(y_rel) - floor_half) / (1.0 - floor_half), 0.0, 1.0)
cross_profile = VALLEY_DEPTH * y_wall**2

bed = bed_slope + cross_profile

# ========================================================
# Initial ice thickness (prescribed starting profile)
# ========================================================
x_glacier = cp.clip(1.0 - x_frac / TERMINUS_FRACTION, 0.0, 1.0)
along_profile = cp.sqrt(x_glacier)  # Vialov-like shape

cross_thickness = cp.clip(1.0 - (cp.abs(y_rel) / (floor_half + 0.3))**2, 0.0, 1.0)

thickness = MAX_THICKNESS * along_profile * cross_thickness
thickness = cp.maximum(thickness, cp.float32(0.0))

# ========================================================
# Surface mass balance (elevation-dependent)
# ========================================================
def compute_smb(H, bed):
    """Elevation-dependent SMB: linear gradient around the ELA."""
    surface = bed + H
    smb = cp.float32(SMB_GRAD) * (surface - cp.float32(ELA))
    smb = cp.clip(smb, cp.float32(SMB_MIN), cp.float32(SMB_MAX))
    return smb


def surface_temperature(H, bed):
    """Elevation-dependent surface temperature from lapse rate."""
    surface_elev = bed + H
    return cp.minimum(
        cp.float32(T_HEAD) + cp.float32(LAPSE_RATE) * (surface_elev - cp.float32(HEAD_ELEVATION)),
        cp.float32(T_MELT))


# ========================================================
# Initialize momentum solver
# ========================================================
model = IceDynamics(n_levels=n_levels, ny=ny, nx=nx, dx=cp.float32(dx))
mg = model.mg
grid = mg.levels[0]

mg.geometry.bed.set(bed)
mg.state.H.set(thickness)
mg.state.H_prev.set(thickness)
mg.forcing.smb.set(compute_smb(thickness, bed))

# Rheology (placeholder B — will be set from thermal model)
mg.rheology.B.set(cp.ones((ny, nx), dtype=cp.float32))
mg.rheology.n.set(N_GLEN)
mg.rheology.eps_reg.set(1e-6)

# Sliding
beta_field = cp.full((ny, nx), BETA_SLIDING, dtype=cp.float32)
mg.sliding.beta.set(beta_field)
mg.sliding.m.set(1.0 / N_GLEN)
mg.sliding.u_reg.set(1.0)
mg.calving.timescale.set(cp.inf)  # no calving (molho API)

# FAS multigrid solver options
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
# Initialize thermal solver
# ========================================================
thermal = ThermalModel(grid, nz=nz,
                       n_smooth=N_SMOOTH,
                       update_rheology=True,
                       frictional_heating=False)

thermal.ops.smoother_config.report_norms = True
thermal.ops.smoother_config.omega = cp.float32(1.0)
thermal.ops.smoother_config.n_newton = 5
thermal.ops.smoother_config.relaxation = cp.float32(1.0)
thermal.ops.smoother_config.lf_c = cp.float32(1e-4)
thermal.ops.smoother_config.absolute_tolerance = cp.float32(1e-3)
thermal.ops.smoother_config.relative_tolerance = cp.float32(1e-7)

T_surf_init = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
thermal.initialize(T_surface=T_surf_init, T_field=T_INIT, Q_geo=Q_GEO)
thermal.ops.enthalpy_forcing.h_thin.set(25.0)
thermal.ops.enthalpy_forcing.drain_rate.set(0.01 / SEC_PER_YR)

thermal.ops.term_flags.horizontal_advection = True
thermal.ops.term_flags.drainage = True
thermal.ops.term_flags.omega = True

# Set initial B from Paterson-Budd at the initial temperature
B_init = thermal.ops.get_arrhenius_factor() / thermal.B_scale
mg.rheology.B.set(B_init)

# Print setup info
H_np = cp.asnumpy(thickness)
T_surf_np = cp.asnumpy(T_surf_init)
print("Coupled valley glacier simulation")
print(f"  Grid: {ny}x{nx}, nz={nz}, dx={dx}m, n_levels={n_levels}")
print(f"  Valley: {VALLEY_LENGTH/1000:.0f} km long, "
      f"{VALLEY_WIDTH/1000:.0f} km wide")
print(f"  Bed elevation: {float(cp.min(bed)):.0f} to "
      f"{float(cp.max(bed)):.0f} m")
print(f"  Initial max thickness: {np.max(H_np):.0f} m")
print(f"  ELA = {ELA:.0f} m, SMB gradient = {SMB_GRAD} m/yr/m")
print(f"  Q_geo = {Q_GEO} W/m^2, T_init = {T_INIT:.1f} K")
print(f"  Beta_sliding = {BETA_SLIDING}")
tf = thermal.ops.term_flags
print(f"  Term flags: bitmask=0x{tf.bitmask:x} "
      f"(h_adv={tf.horizontal_advection}, omega={tf.omega}, "
      f"drain={tf.drainage})")
print(f"  dt = {DT_YR} yr, {N_STEPS} steps = {N_STEPS * DT_YR:.0f} yr")

# ========================================================
# Momentum spin-up
# ========================================================
dt_yr = cp.float32(DT_YR)
dt_sec = DT_YR * SEC_PER_YR
t_yr = 0.0

N_SPINUP = 5
print(f"\n  Momentum spin-up ({N_SPINUP} steps, no thermal coupling)...")
thermal.update_rheology = False
for _ in range(N_SPINUP):
    mg.forcing.smb.set(compute_smb(grid.state.H.data, grid.geometry.bed.data))
    model.forward(cp.float32(t_yr), dt_yr)
    t_yr += DT_YR
print(f"    done (t = {t_yr:.0f} yr)")

# Re-initialize thermal state from relaxed geometry
T_surf_relaxed = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
thermal.initialize(T_surface=T_surf_relaxed, T_field=T_INIT, Q_geo=Q_GEO)
B_init = thermal.ops.get_arrhenius_factor() / thermal.B_scale
mg.rheology.B.set(B_init)
thermal.update_rheology = True

# ========================================================
# Time stepping (fully coupled)
# ========================================================
# Storage for time-series diagnostics
times = [t_yr]
vol = [float(cp.sum(grid.state.H.data) * dx**2 / 1e9)]
T_bed_center_ts = [float(cp.asnumpy(thermal.temperature[ny//2, nx//3, 0]))]
B_center_ts = [float(cp.asnumpy(grid.rheology.B.data[ny//2, nx//3]))]
max_speed_ts = [0.0]
terminus_pos_ts = [0.0]

# Snapshots for spatial plots
snapshot_steps = sorted({1, N_STEPS // 4, N_STEPS // 2, N_STEPS})
snapshots = {}

sigma = cp.asnumpy(thermal.ops.sigma)
j_center = ny // 2

print(f"\n  {'step':>5s}  {'t (yr)':>8s}  {'vol (km3)':>10s}  "
      f"{'T_bed (K)':>10s}  {'max |u|':>12s}  {'terminus':>10s}")
print(f"  {'-'*64}")

for step in range(1, N_STEPS + 1):
    # 1. Update SMB from current geometry
    mg.forcing.smb.set(compute_smb(grid.state.H.data, grid.geometry.bed.data))

    # 2. Snapshot E and H before momentum step
    thermal.pre_momentum()

    # 3. Momentum solve (dt in years)
    model.forward(cp.float32(t_yr), dt_yr)

    # 4. Update surface temperature from new geometry
    T_surf = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
    thermal.ops.set_surface_enthalpy_from_temperature(T_surf)

    # 5. Thermal solve (dt in seconds)
    thermal.step(dt_sec)

    t_yr += DT_YR

    # --- Diagnostics ---
    H_np = cp.asnumpy(grid.state.H.data)
    u_np = cp.asnumpy(grid.state.u.data)
    v_np = cp.asnumpy(grid.state.v.data)
    u_c = 0.5 * (u_np[:, 1:] + u_np[:, :-1])
    v_c = 0.5 * (v_np[1:, :] + v_np[:-1, :])
    speed = np.sqrt(u_c**2 + v_c**2)

    volume_km3 = float(np.sum(H_np) * dx**2 / 1e9)
    T_bed_val = float(cp.asnumpy(thermal.temperature[j_center, nx//3, 0]))
    B_val = float(cp.asnumpy(grid.rheology.B.data[j_center, nx//3]))
    max_spd = float(np.max(speed))

    # Terminus position: furthest x where ice is present along center flowline
    ice_center = H_np[j_center, :] > 10.0
    if np.any(ice_center):
        terminus_km = float(np.max(np.where(ice_center)[0]) + 1) * dx / 1000.0
    else:
        terminus_km = 0.0

    times.append(t_yr)
    vol.append(volume_km3)
    T_bed_center_ts.append(T_bed_val)
    B_center_ts.append(B_val)
    max_speed_ts.append(max_spd)
    terminus_pos_ts.append(terminus_km)

    if step % 20 == 0 or step <= 5:
        print(f"  {step:5d}  {t_yr:8.1f}  {volume_km3:10.3f}  "
              f"{T_bed_val:10.2f}  {max_spd:12.2f}  {terminus_km:10.2f} km")

    # --- Spatial snapshots ---
    if step in snapshot_steps:
        T_3d = cp.asnumpy(thermal.temperature)
        snapshots[step] = {
            't_yr': t_yr,
            'H': H_np.copy(),
            'speed': speed.copy(),
            'T_3d': T_3d.copy(),
            'B': cp.asnumpy(grid.rheology.B.data).copy(),
            'bed': cp.asnumpy(bed),
        }

t_final = t_yr
print(f"\n  Done. Final time: {t_final:.0f} yr")

# ========================================================
# Plots
# ========================================================
x_km = cp.asnumpy(x) / 1000.0
y_km = cp.asnumpy(y) / 1000.0
bed_np = cp.asnumpy(bed)

# ----- 1. Summary time series -----
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

axes[0, 0].plot(times, vol, 'b-', lw=2)
axes[0, 0].set_ylabel('Volume (km$^3$)')
axes[0, 0].set_title('Ice Volume')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(times, T_bed_center_ts, 'r-', lw=2)
T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * MAX_THICKNESS
axes[0, 1].axhline(T_pmp, color='k', ls='--', alpha=0.5,
                    label=f'$T_{{pmp}}$ = {T_pmp:.1f} K')
axes[0, 1].set_ylabel('Temperature (K)')
axes[0, 1].set_title('Basal Temperature (upper glacier)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(times, B_center_ts, 'g-', lw=2)
axes[0, 2].set_ylabel('B (GLIDE units)')
axes[0, 2].set_title('Arrhenius B (upper glacier)')
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].plot(times, max_speed_ts, 'm-', lw=2)
axes[1, 0].set_ylabel('Speed (m/yr)')
axes[1, 0].set_xlabel('Time (yr)')
axes[1, 0].set_title('Max Surface Speed')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(times, terminus_pos_ts, 'c-', lw=2)
axes[1, 1].set_ylabel('Distance (km)')
axes[1, 1].set_xlabel('Time (yr)')
axes[1, 1].set_title('Terminus Position (center flowline)')
axes[1, 1].grid(True, alpha=0.3)

# SMB profile along center flowline (final state)
smb_final = cp.asnumpy(compute_smb(grid.state.H.data, grid.geometry.bed.data))
axes[1, 2].plot(x_km, smb_final[j_center, :], 'k-', lw=2)
axes[1, 2].axhline(0, color='gray', ls='--', alpha=0.5)
axes[1, 2].set_ylabel('SMB (m/yr)')
axes[1, 2].set_xlabel('Along-valley (km)')
axes[1, 2].set_title('SMB (center flowline, final)')
axes[1, 2].grid(True, alpha=0.3)

fig.suptitle('Coupled Valley Glacier Evolution', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / 'coupled_valley_glacier_summary.png', dpi=150)
print(f"Saved: {OUT_DIR / 'coupled_valley_glacier_summary.png'}")

# ----- 2. Along-valley cross-sections at selected times -----
snap_keys = sorted(snapshots.keys())
n_snaps = len(snap_keys)

# Global T range across all snapshots
T_all = []
for sk in snap_keys:
    snap = snapshots[sk]
    T_row = snap['T_3d'][j_center, :, :]
    H_row = snap['H'][j_center, :]
    for j in range(nx):
        if H_row[j] >= 30.0:
            T_all.extend(T_row[j, :].tolist())
if T_all:
    T_vmin, T_vmax = min(T_all), max(T_all)
else:
    T_vmin, T_vmax = T_INIT, T_MELT

fig2, axes2 = plt.subplots(n_snaps, 1, figsize=(14, 3.5 * n_snaps),
                             sharex=True)
if n_snaps == 1:
    axes2 = [axes2]

for idx, sk in enumerate(snap_keys):
    ax = axes2[idx]
    snap = snapshots[sk]
    T_slice = snap['T_3d'][j_center, :, :]  # (nx, nz)
    H_row = snap['H'][j_center, :]
    bed_row = bed_np[j_center, :]

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
    ax.set_title(f't = {snap["t_yr"]:.0f} yr', fontsize=10)
    ax.set_ylim(float(np.min(bed_row)) - 100,
                float(np.max(bed_row + np.where(H_row > 0, H_row, 0))) + 200)

axes2[-1].set_xlabel('Along-valley distance (km)')
fig2.colorbar(im, ax=axes2, label='Temperature (K)', fraction=0.015, pad=0.02)
fig2.suptitle('Coupled Valley Glacier: Along-Valley Temperature (center flowline)',
              fontsize=13, fontweight='bold')
plt.tight_layout()
fig2.savefig(OUT_DIR / 'coupled_valley_glacier_along.png', dpi=150)
print(f"Saved: {OUT_DIR / 'coupled_valley_glacier_along.png'}")

# ----- 3. Plan-view maps at final time -----
snap_final = snapshots[snap_keys[-1]]
H_final = snap_final['H']
speed_final = snap_final['speed']
T_bed_final = snap_final['T_3d'][:, :, 0].copy()
B_final = snap_final['B']
ice_mask = H_final < 10.0

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

# Thickness
H_plot = H_final.copy()
H_plot[ice_mask] = np.nan
im0 = axes3[0, 0].pcolormesh(x_km, y_km, H_plot, cmap='cividis', shading='auto')
axes3[0, 0].set_title(f'Thickness (m), t = {t_final:.0f} yr')
axes3[0, 0].set_aspect('equal')
plt.colorbar(im0, ax=axes3[0, 0])

# Speed
spd_plot = speed_final.copy()
spd_plot[ice_mask] = np.nan
im1 = axes3[0, 1].pcolormesh(x_km, y_km, spd_plot, cmap='magma', shading='auto')
axes3[0, 1].set_title(f'Speed (m/yr), t = {t_final:.0f} yr')
axes3[0, 1].set_aspect('equal')
plt.colorbar(im1, ax=axes3[0, 1])

# Basal temperature
T_bed_plot = T_bed_final.copy()
T_bed_plot[ice_mask] = np.nan
im2 = axes3[1, 0].pcolormesh(x_km, y_km, T_bed_plot, cmap='RdYlBu_r',
                               shading='auto')
axes3[1, 0].set_title(f'Basal Temperature (K), t = {t_final:.0f} yr')
axes3[1, 0].set_aspect('equal')
plt.colorbar(im2, ax=axes3[1, 0])

# B field
B_plot = B_final.copy()
B_plot[ice_mask] = np.nan
im3 = axes3[1, 1].pcolormesh(x_km, y_km, B_plot, cmap='viridis', shading='auto')
axes3[1, 1].set_title(f'Arrhenius B, t = {t_final:.0f} yr')
axes3[1, 1].set_aspect('equal')
plt.colorbar(im3, ax=axes3[1, 1])

for ax in axes3.flat:
    ax.set_xlabel('Along-valley (km)')
    ax.set_ylabel('Across-valley (km)')

fig3.suptitle('Coupled Valley Glacier: Final State', fontsize=14, fontweight='bold')
plt.tight_layout()
fig3.savefig(OUT_DIR / 'coupled_valley_glacier_maps.png', dpi=150)
print(f"Saved: {OUT_DIR / 'coupled_valley_glacier_maps.png'}")

# ----- 4. Cross-valley section at upper glacier, final time -----
fig4, ax4 = plt.subplots(figsize=(8, 5))
i_cross = nx // 3
T_cross = snap_final['T_3d'][:, i_cross, :]  # (ny, nz)
H_cross = H_final[:, i_cross]
bed_cross = bed_np[:, i_cross]

z_cross = np.zeros((ny, nz))
T_cross_plot = np.full((ny, nz), np.nan)
for i in range(ny):
    if H_cross[i] > 30:
        for k in range(nz):
            z_cross[i, k] = bed_cross[i] + sigma[k] * H_cross[i]
            T_cross_plot[i, k] = T_cross[i, k]

im4 = ax4.pcolormesh(np.broadcast_to(y_km[:, None], (ny, nz)),
                      z_cross, T_cross_plot,
                      cmap='RdYlBu_r', shading='gouraud',
                      vmin=T_vmin, vmax=T_vmax)
ax4.fill_between(y_km, bed_cross, 0, color='saddlebrown', alpha=0.6)
ax4.plot(y_km, bed_cross, 'k-', lw=1.0)
ax4.plot(y_km, np.where(H_cross > 30, bed_cross + H_cross, np.nan),
         'b-', lw=1.2)
ax4.set_xlabel('Across-valley distance (km)')
ax4.set_ylabel('Elevation (m)')
ax4.set_title(f'Cross-Valley Temperature at x = {x_km[i_cross]:.1f} km, '
              f't = {t_final:.0f} yr')
plt.colorbar(im4, ax=ax4, label='Temperature (K)')
plt.tight_layout()
fig4.savefig(OUT_DIR / 'coupled_valley_glacier_cross.png', dpi=150)
print(f"Saved: {OUT_DIR / 'coupled_valley_glacier_cross.png'}")

plt.show()
