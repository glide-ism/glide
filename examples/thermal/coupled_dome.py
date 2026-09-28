"""
Coupled momentum-thermal dome example.

A hemispherical ice dome on a flat bed evolves under its own weight
with two-way coupling between the SSA momentum solver and the enthalpy
solver:

  Momentum  -->  velocity u, v  -->  Enthalpy advection + frictional heating
  Enthalpy  -->  Arrhenius B(T) -->  Momentum viscosity

The dome starts with a uniform cold temperature and warms from below
via geothermal heat flux. As the base warms, the ice softens (B decreases),
which accelerates flow, which in turn increases frictional heating and
advective heat transport.

Outputs:
  - VTI files for ParaView (thickness, velocity, basal temperature, B)
  - Summary plot of dome evolution
"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from glide.model import IceDynamics, ThermalModel
try:
    from glide.io import VTIWriter
    HAS_VTI = True
except ImportError:
    HAS_VTI = False
from glide.enthalpy import T_MELT, BETA_CC, RHO_I, GRAVITY

# ========================================================
# Parameters
# ========================================================
# Grid
n_levels = 4
ny, nx = 1024, 1024
dx = 250.0           # m

# Dome geometry
DOME_RADIUS = 100000.0 # m
DOME_HEIGHT = 3000.0   # m

# Forcing
SMB_CENTER = 2.       # m/yr ice equivalent (accumulation at center)
SMB_EDGE = -4.0        # m/yr ice equivalent (ablation at margin)
Q_GEO = 0.0          # W/m^2 (geothermal heat flux)
T_SUMMIT = 243.15      # K (-30 C, surface T at dome summit)
T_MARGIN = 278.15      # K (-5 C, surface T at ice margin / sea level)
LAPSE_RATE = (T_MARGIN - T_SUMMIT) / DOME_HEIGHT  # K/m (derived)
T_INIT = T_SUMMIT      # K (uniform initial ice temperature)

# Rheology
# GLIDE works in "head" units: B_head = B_SI / (rho_i * g)
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0

# Sliding (head units, comparable to Greenland example values)
BETA_SLIDING = 10.0

# Thermal
NZ = 11               # sigma levels
N_SMOOTH = 20         # enthalpy smoothing sweeps

# Time stepping
DT_YR = 5.0           # years
N_STEPS = 50
SEC_PER_YR = 365.25 * 86400.0

# Output
OUT_DIR = Path(__file__).parent / 'coupled_dome_output'

# ========================================================
# Build dome geometry
# ========================================================
x = (cp.arange(nx, dtype=cp.float32) + 0.5) * dx - 0.5 * nx * dx
y = (cp.arange(ny, dtype=cp.float32) + 0.5) * dx - 0.5 * ny * dx
xx, yy = cp.meshgrid(x, y)
radius = cp.sqrt(xx**2 + yy**2)
radial_fraction = cp.clip(radius / DOME_RADIUS, 0.0, 1.0)
hemisphere = cp.sqrt(cp.clip(1.0 - radial_fraction**2, 0.0, 1.0))

thickness = DOME_HEIGHT * hemisphere
bed = cp.zeros((ny, nx), dtype=cp.float32)
smb = (cp.float32(SMB_EDGE)
       + cp.float32(SMB_CENTER - SMB_EDGE) * hemisphere)


def surface_temperature(H, bed):
    """Elevation-dependent surface temperature from lapse rate."""
    surface_elev = bed + H
    return cp.float32(T_SUMMIT) + cp.float32(LAPSE_RATE) * (surface_elev - cp.float32(DOME_HEIGHT))

# ========================================================
# Initialize momentum solver
# ========================================================
model = IceDynamics(n_levels=n_levels, ny=ny, nx=nx, dx=cp.float32(dx))
mg = model.mg

mg.geometry.bed.set(bed)
mg.state.H.set(thickness)
mg.state.H_prev.set(thickness)
mg.forcing.smb.set(smb)

# B will be set from the thermal model after initialization below
mg.rheology.B.set(cp.ones((ny, nx), dtype=cp.float32))  # placeholder
mg.rheology.n.set(N_GLEN)
mg.rheology.eps_reg.set(1e-6)

beta_field = cp.full((ny, nx), BETA_SLIDING, dtype=cp.float32)
mg.sliding.beta.set(beta_field)
mg.sliding.m.set(1.0 / N_GLEN)
mg.sliding.u_reg.set(1.0)
mg.calving.timescale.set(cp.inf)  # no calving (molho API)

# Solver configuration
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
grid = mg.levels[0]
thermal = ThermalModel(grid, nz=NZ,
                       n_smooth=N_SMOOTH, update_rheology=False,
                       frictional_heating=False)

# All terms enabled (horizontal advection, sigma_dot, strain heating, drainage)

thermal.ops.smoother_config.report_norms = True
thermal.ops.smoother_config.omega = cp.float32(1.0)
thermal.ops.smoother_config.n_newton = 5
thermal.ops.smoother_config.relaxation = cp.float32(1.0)
thermal.ops.smoother_config.lf_c = cp.float32(1e-4)
thermal.ops.smoother_config.absolute_tolerance = cp.float32(1e-3)
thermal.ops.smoother_config.relative_tolerance = cp.float32(1e-7)
T_surf_init = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
thermal.initialize(T_surface=T_surf_init, T_field=T_INIT, Q_geo=Q_GEO)
thermal.ops.enthalpy_forcing.h_thin.set(50.0)

thermal.ops.term_flags.horizontal_advection = True 
thermal.ops.term_flags.drainage = True

# Set initial B from the Paterson-Budd law at the initial temperature,
# so there is no discontinuity when the thermal model starts updating B.
B_init = thermal.ops.get_arrhenius_factor() / thermal.B_scale
mg.rheology.B.set(B_init)
T_summit = float(T_surf_init[ny//2, nx//2])
T_margin = float(T_MARGIN)
print(f"  Surface T: summit = {T_summit:.2f} K ({T_summit-273.15:.1f} C), "
      f"margin = {T_margin:.2f} K ({T_margin-273.15:.1f} C)")
print(f"  Initial B (GLIDE units): {float(B_init[ny//2, nx//2]):.4e}")
tf = thermal.ops.term_flags
print(f"  Term flags: bitmask=0x{tf.bitmask:x} "
      f"(h_adv={tf.horizontal_advection}, omega={tf.omega}, "
      f"drain={tf.drainage})")

# ========================================================
# Output setup
# ========================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

if HAS_VTI:
    origin = (float(x[0]), float(y[0]))

    # 2D: surface / depth-averaged fields. Mirrors the momentum-solver
    # pattern used in examples/bitterroot, examples/antarctica, etc.
    vti_2d = VTIWriter(
        out_dir=str(OUT_DIR / 'vti_2d'),
        base='dome',
        dx=dx,
        origin=origin,
        static_fields={
            'bed': grid.geometry.bed,
        },
        dynamic_fields={
            'H': grid.state.H,
            'U': [grid.state.u, grid.state.v],
            'B': grid.rheology.B,
            'T_bed': lambda: thermal.temperature[:, :, 0],
        },
    )
    vti_2d.initialize(grid)

    # 3D: volumetric thermal fields. Sigma is the third axis with
    # uniform spacing dz = 1/(NZ-1). Origin z=0 is the bed.
    vti_3d = VTIWriter(
        out_dir=str(OUT_DIR / 'vti_3d'),
        base='dome_thermal',
        dx=dx,
        dz=1.0 / (NZ - 1),
        origin=(*origin, 0.0),
        dynamic_fields={
            'E': thermal.ops.enthalpy_state.E,
            'T': lambda: thermal.temperature,
            'omega_3d': thermal.ops.enthalpy_velocity.omega,
            'water_content': lambda: thermal.water_content,
        },
    )
    vti_3d.initialize(None)

# ========================================================
# Time stepping
# Note: SSA solver uses dt in years (matching smb in m/yr),
#       enthalpy solver uses dt in seconds (SI units).
# ========================================================
dt_yr = cp.float32(DT_YR)
dt_sec = DT_YR * SEC_PER_YR
t_yr = 0.0

# Spin up the momentum solver for a few steps before thermal coupling.
# The initial hemisphere dome is very steep, producing extreme first-step
# velocities that can cause slow convergence in the enthalpy smoother.
# A short momentum-only spin-up lets the geometry relax.
N_SPINUP = 5
print(f"  Momentum spin-up ({N_SPINUP} steps, no thermal coupling)...")
for _ in range(N_SPINUP):
    model.forward(cp.float32(t_yr), dt_yr)
    t_yr += DT_YR
print(f"    done (t = {t_yr:.0f} yr)")

# Re-initialize thermal state from the relaxed geometry
T_surf_relaxed = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
thermal.initialize(T_surface=T_surf_relaxed, T_field=T_INIT, Q_geo=Q_GEO)
B_init = thermal.ops.get_arrhenius_factor() / thermal.B_scale
mg.rheology.B.set(B_init)
thermal.update_rheology = False

# Storage for time-series plot
times = [t_yr]
vol = [float(cp.sum(grid.state.H.data) * dx**2 / 1e9)]
T_bed_center = [float(cp.asnumpy(thermal.temperature[ny//2, nx//2, 0]))]
B_center = [float(cp.asnumpy(grid.rheology.B.data[ny//2, nx//2]))]
max_speed = [0.0]

# Symmetry diagnostics time series
H_asym_ts = []
T_asym_ts = []
speed_asym_ts = []

# Snapshots for spatial map plots (at selected steps)
snapshot_steps = {1, N_STEPS // 4, N_STEPS // 2, N_STEPS}
snapshots_map = {}  # step -> dict of 2D fields

print(f"\n  {'step':>5s}  {'t (yr)':>8s}  {'vol (km3)':>10s}  "
      f"{'T_bed (K)':>10s}  {'B_center':>12s}  {'max |u| (m/yr)':>14s}")
print(f"  {'-'*68}")

for step in range(N_STEPS):
    # --- Snapshot E and H before momentum step ---
    thermal.pre_momentum()

    # --- Momentum solve (dt in years) ---
    model.forward(cp.float32(t_yr), dt_yr)

    # --- Update surface temperature from current geometry ---
    T_surf = surface_temperature(grid.state.H.data, grid.geometry.bed.data)
    thermal.ops.set_surface_enthalpy_from_temperature(T_surf)

    # --- Thermal solve (dt in seconds) ---
    thermal.step(dt_sec)

    t_yr += DT_YR

    # --- Diagnostics ---
    u = cp.asnumpy(grid.state.u.data)
    v = cp.asnumpy(grid.state.v.data)
    u_c = 0.5 * (u[:, 1:] + u[:, :-1])
    v_c = 0.5 * (v[1:, :] + v[:-1, :])
    speed = np.sqrt(u_c**2 + v_c**2)  # already m/yr (SSA uses year-based units)

    H_np = cp.asnumpy(grid.state.H.data)
    volume_km3 = float(np.sum(H_np) * dx**2 / 1e9)
    T_bed = float(cp.asnumpy(thermal.temperature[ny//2, nx//2, 0]))
    B_val = float(cp.asnumpy(grid.rheology.B.data[ny//2, nx//2]))
    max_spd = float(np.max(speed))

    times.append(t_yr)
    vol.append(volume_km3)
    T_bed_center.append(T_bed)
    B_center.append(B_val)
    max_speed.append(max_spd)

    # Symmetry diagnostics: compare field with 90-degree rotation.
    # For a dome centered on the grid, 90-degree rotation (transpose + flip)
    # should leave scalar fields invariant and rotate the velocity vector.
    # Compute both full-domain and interior-only (H > 200m) metrics to
    # separate margin effects from the bulk solution.
    thick = H_np > 200.0
    thick_rot90 = np.rot90(thick)
    interior = thick & thick_rot90  # cells that are thick in both orientations

    H_rot90 = np.rot90(H_np)
    H_diff = np.abs(H_np - H_rot90)
    H_asym = np.max(H_diff)
    H_asym_int = np.max(H_diff[interior]) if np.any(interior) else 0.0

    T_3d_diag = cp.asnumpy(thermal.temperature)
    T_mid = T_3d_diag[:, :, NZ // 2]
    T_rot90 = np.rot90(T_mid)
    T_diff = np.abs(T_mid - T_rot90)
    T_asym = np.max(T_diff)
    T_asym_int = np.max(T_diff[interior]) if np.any(interior) else 0.0

    speed_rot90 = np.rot90(speed)
    spd_diff = np.abs(speed - speed_rot90)
    speed_asym = np.max(spd_diff)
    speed_asym_int = np.max(spd_diff[interior]) if np.any(interior) else 0.0

    H_asym_ts.append(H_asym)
    T_asym_ts.append(T_asym)
    speed_asym_ts.append(speed_asym)

    if (step + 1) % 10 == 0 or step == 0:
        print(f"  {step+1:5d}  {t_yr:8.1f}  {volume_km3:10.1f}  "
              f"{T_bed:10.2f}  {B_val:12.4e}  {max_spd:14.2f}  "
              f"H={H_asym:.2e}/{H_asym_int:.2e}  "
              f"T={T_asym:.2e}/{T_asym_int:.2e}  "
              f"spd={speed_asym:.2e}/{speed_asym_int:.2e}")

    # --- Store spatial snapshot ---
    if (step + 1) in snapshot_steps:
        T_3d = cp.asnumpy(thermal.temperature)
        # Thin marginal ice has unreliable thermal state — clamp it
        thin = H_np < 100.0
        T_bed = np.clip(T_3d[:, :, 0], T_MARGIN - 5.0, T_MELT)
        T_avg = np.clip(np.mean(T_3d, axis=2), T_MARGIN - 5.0, T_MELT)
        T_bed[thin] = T_MARGIN
        T_avg[thin] = T_MARGIN
        T_surf_np = T_3d[:, :, -1].copy()  # modeled temperature at top sigma level
        T_surf_np[thin] = np.nan
        snapshots_map[step + 1] = {
            't_yr': t_yr,
            'T_surf': T_surf_np,
            'T_bed': T_bed,
            'T_avg': T_avg,
            'T_3d': T_3d.copy(),
            'B': cp.asnumpy(grid.rheology.B.data).copy(),
            'H': H_np.copy(),
            'speed': speed.copy(),
            'omega': cp.asnumpy(thermal.ops.enthalpy_velocity.omega).copy(),
            'u': cp.asnumpy(grid.state.u.data).copy(),
            'v': cp.asnumpy(grid.state.v.data).copy(),
        }

    # --- Write VTI snapshots ---
    if (step + 1) % 10 == 0 and HAS_VTI:
        vti_2d.append(grid, time=t_yr)
        vti_2d.write_pvd()
        vti_3d.append(None, time=t_yr)
        vti_3d.write_pvd()

print(f"\n  Done. {N_STEPS} steps, final t = {t_yr:.0f} yr")

# ========================================================
# Summary plot
# ========================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Volume
axes[0, 0].plot(times, vol, 'b-', linewidth=2)
axes[0, 0].set_ylabel('Volume (km$^3$)')
axes[0, 0].set_title('Ice Volume')
axes[0, 0].grid(True, alpha=0.3)

# Basal temperature at dome center
T_pmp_center = T_MELT - BETA_CC * RHO_I * GRAVITY * DOME_HEIGHT
axes[0, 1].plot(times, T_bed_center, 'r-', linewidth=2)
axes[0, 1].axhline(T_pmp_center, color='k', linestyle='--', alpha=0.5,
                    label=f'$T_{{pmp}}$ = {T_pmp_center:.1f} K')
axes[0, 1].set_ylabel('Temperature (K)')
axes[0, 1].set_title('Basal Temperature (dome center)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Rate factor B at dome center
axes[1, 0].plot(times, B_center, 'g-', linewidth=2)
axes[1, 0].set_ylabel('B (Pa$^{-n}$ s$^{-1}$)')
axes[1, 0].set_xlabel('Time (yr)')
axes[1, 0].set_title('Arrhenius B (dome center)')
axes[1, 0].grid(True, alpha=0.3)

# Max speed
axes[1, 1].plot(times, max_speed, 'm-', linewidth=2)
axes[1, 1].set_ylabel('Speed (m/yr)')
axes[1, 1].set_xlabel('Time (yr)')
axes[1, 1].set_title('Max Surface Speed')
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Coupled Momentum-Thermal Dome Evolution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'coupled_dome_summary.png', dpi=150)
plt.show()
print(f"  Saved: {OUT_DIR / 'coupled_dome_summary.png'}")

# ========================================================
# Radial symmetry plot
# ========================================================
sym_times = times[1:]  # skip t=0 (no diagnostics stored)

fig_sym, axes_sym = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes_sym[0].semilogy(sym_times, H_asym_ts, 'b-', lw=2)
axes_sym[0].set_ylabel('max $|H - H_{rot90}|$ (m)')
axes_sym[0].set_title('Thickness')
axes_sym[0].grid(True, alpha=0.3)

axes_sym[1].semilogy(sym_times, speed_asym_ts, 'm-', lw=2)
axes_sym[1].set_ylabel('max $|speed - speed_{rot90}|$ (m/yr)')
axes_sym[1].set_title('Speed')
axes_sym[1].grid(True, alpha=0.3)

axes_sym[2].semilogy(sym_times, T_asym_ts, 'r-', lw=2)
axes_sym[2].set_ylabel('max $|T - T_{rot90}|$ (K)')
axes_sym[2].set_title('Mid-Column Temperature')
axes_sym[2].set_xlabel('Time (yr)')
axes_sym[2].grid(True, alpha=0.3)

fig_sym.suptitle('90-Degree Rotational Symmetry (lower = better)',
                 fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'coupled_dome_symmetry.png', dpi=150)
plt.show()
print(f"  Saved: {OUT_DIR / 'coupled_dome_symmetry.png'}")

# ========================================================
# Spatial maps at selected time steps
# Rows: T_bed, T_avg, B, speed
# Columns: one per snapshot time
# ========================================================
x_km = cp.asnumpy(x) / 1000.0
y_km = cp.asnumpy(y) / 1000.0
snap_keys = sorted(snapshots_map.keys())
n_snaps = len(snap_keys)

# Compute global color ranges from ice-covered cells across all snapshots
all_H, all_T_surf, all_T_bed, all_T_avg, all_B, all_spd = [], [], [], [], [], []
for snap in snapshots_map.values():
    ice = snap['H'] >= 100.0
    all_H.append(snap['H'][ice])
    T_s = snap['T_surf']
    all_T_surf.append(T_s[ice & ~np.isnan(T_s)])
    all_T_bed.append(snap['T_bed'][ice])
    all_T_avg.append(snap['T_avg'][ice])
    all_B.append(snap['B'][ice])
    all_spd.append(snap['speed'][ice])

field_specs = [
    ('H',      'Thickness (m)',                   'cividis',
     0.0,                               np.max(np.concatenate(all_H))),
    ('T_surf', 'Surface Temperature (K)',         'coolwarm',
     np.min(np.concatenate(all_T_surf)), np.max(np.concatenate(all_T_surf))),
    ('T_bed',  'Basal Temperature (K)',          'coolwarm',
     np.min(np.concatenate(all_T_bed)), np.max(np.concatenate(all_T_bed))),
    ('T_avg',  'Column-Avg Temperature (K)',     'coolwarm',
     np.min(np.concatenate(all_T_avg)), np.max(np.concatenate(all_T_avg))),
    ('B',      'Arrhenius B (GLIDE units)',       'viridis',
     np.min(np.concatenate(all_B)),     np.max(np.concatenate(all_B))),
    ('speed',  'Speed (m/yr)',                    'magma',
     0.0,                               np.max(np.concatenate(all_spd))),
]
n_rows = len(field_specs)

fig2, axes2 = plt.subplots(n_rows, n_snaps, figsize=(4 * n_snaps, 3.5 * n_rows),
                            squeeze=False, sharex=True, sharey=True)

for col, step_key in enumerate(snap_keys):
    snap = snapshots_map[step_key]
    H_mask = snap['H'] < 1.0

    for row, (field_name, label, cmap, vmin, vmax) in enumerate(field_specs):
        ax = axes2[row, col]
        data = snap[field_name].copy()
        data[H_mask] = np.nan

        im = ax.pcolormesh(x_km, y_km, data, cmap=cmap, shading='auto',
                           vmin=vmin, vmax=vmax)
        cb = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.get_major_formatter().set_useOffset(False)
        cb.ax.yaxis.get_major_formatter().set_scientific(False)

        if row == 0:
            ax.set_title(f't = {snap["t_yr"]:.0f} yr', fontsize=11)
        if col == 0:
            ax.set_ylabel(label, fontsize=10)
        ax.set_aspect('equal')

for ax in axes2[-1, :]:
    ax.set_xlabel('x (km)')

fig2.suptitle('Spatial Fields at Selected Times', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'coupled_dome_maps.png', dpi=150)
plt.show()
print(f"  Saved: {OUT_DIR / 'coupled_dome_maps.png'}")

# ========================================================
# Cross-section plots: temperature in physical elevation
# Two figures: x-axis (y=0) and y-axis (x=0), each with
# one subplot per snapshot time showing T(x/y, z) with
# bed topography and ice surface overlaid.
# ========================================================
sigma = cp.asnumpy(thermal.ops.sigma)
i_mid = ny // 2
j_mid = nx // 2
bed_np = cp.asnumpy(bed)
H_THRESH = 50.0

# Global T range across all cross-sections (both axes)
T_all = []
for snap in snapshots_map.values():
    for row_idx, n_cells in [(i_mid, nx), (j_mid, ny)]:
        if n_cells == nx:
            T_row = snap['T_3d'][i_mid, :, :]
            H_row = snap['H'][i_mid, :]
        else:
            T_row = snap['T_3d'][:, j_mid, :]
            H_row = snap['H'][:, j_mid]
        T_all.extend(T_row[H_row >= H_THRESH, :].flatten().tolist())
T_vmin, T_vmax = min(T_all), max(T_all)

for axis_label, dist_km, n_cells, get_slice in [
    ('x (y = 0)', x_km, nx,
     lambda snap: (snap['T_3d'][i_mid, :, :],
                   snap['H'][i_mid, :],
                   bed_np[i_mid, :])),
    ('y (x = 0)', y_km, ny,
     lambda snap: (snap['T_3d'][:, j_mid, :],
                   snap['H'][:, j_mid],
                   bed_np[:, j_mid])),
]:
    fig_cs, axes_cs = plt.subplots(len(snap_keys), 1,
                                    figsize=(14, 3.0 * len(snap_keys)),
                                    sharex=True)
    if len(snap_keys) == 1:
        axes_cs = [axes_cs]

    for idx, step_key in enumerate(snap_keys):
        ax = axes_cs[idx]
        snap = snapshots_map[step_key]
        T_slice, H_row, bed_row = get_slice(snap)
        nz = len(sigma)

        # Build physical z-coordinates
        z_plot = np.zeros((n_cells, nz))
        T_plot = np.full((n_cells, nz), np.nan)
        for c in range(n_cells):
            if H_row[c] > H_THRESH:
                for k in range(nz):
                    z_plot[c, k] = bed_row[c] + sigma[k] * H_row[c]
                    T_plot[c, k] = T_slice[c, k]

        im = ax.pcolormesh(
            np.broadcast_to(dist_km[:, None], (n_cells, nz)),
            z_plot, T_plot,
            cmap='RdYlBu_r', shading='gouraud',
            vmin=T_vmin, vmax=T_vmax)
        ax.fill_between(dist_km, bed_row, bed_row.min() - 200,
                         color='saddlebrown', alpha=0.6)
        ax.plot(dist_km, bed_row, 'k-', lw=1.0)
        ax.plot(dist_km,
                np.where(H_row > H_THRESH, bed_row + H_row, np.nan),
                'b-', lw=1.2)
        ax.set_ylabel('Elevation (m)')
        ax.set_title(f't = {snap["t_yr"]:.0f} yr', fontsize=10)
        z_max = float(np.max(bed_row + np.where(H_row > 0, H_row, 0)))
        ax.set_ylim(float(bed_row.min()) - 200, z_max + 300)

    axes_cs[-1].set_xlabel(f'{axis_label} (km)')
    fig_cs.colorbar(im, ax=axes_cs, label='Temperature (K)',
                    fraction=0.015, pad=0.02)
    fig_cs.suptitle(f'Temperature Cross-Section along {axis_label}',
                    fontsize=13, fontweight='bold')
    plt.tight_layout()

    tag = axis_label[0]  # 'x' or 'y'
    fname = OUT_DIR / f'coupled_dome_cross_{tag}.png'
    fig_cs.savefig(fname, dpi=150)
    plt.show()
    print(f"  Saved: {fname}")

# ========================================================
# Omega cross-sections along x and y axes
# Shows omega in physical elevation coordinates, like the
# temperature cross-sections. Units converted to m/yr for
# readability.
# ========================================================
# Global omega range (in m/yr) across all snapshots for consistent colorbar
omega_all = []
for snap in snapshots_map.values():
    omega_yr = snap['omega'] * SEC_PER_YR  # m/s -> m/yr
    for axis_idx, n_cells in [(i_mid, nx), (j_mid, ny)]:
        if n_cells == nx:
            omega_row = omega_yr[i_mid, :, :]
            H_row = snap['H'][i_mid, :]
        else:
            omega_row = omega_yr[:, j_mid, :]
            H_row = snap['H'][:, j_mid]
        omega_all.extend(omega_row[H_row >= H_THRESH, :].flatten().tolist())
omega_abs_max = max(abs(min(omega_all)), abs(max(omega_all)))

for axis_label, dist_km, n_cells, get_slice in [
    ('x (y = 0)', x_km, nx,
     lambda snap: (snap['omega'][i_mid, :, :] * SEC_PER_YR,
                   snap['H'][i_mid, :],
                   bed_np[i_mid, :])),
    ('y (x = 0)', y_km, ny,
     lambda snap: (snap['omega'][:, j_mid, :] * SEC_PER_YR,
                   snap['H'][:, j_mid],
                   bed_np[:, j_mid])),
]:
    fig_om, axes_om = plt.subplots(len(snap_keys), 1,
                                    figsize=(14, 3.0 * len(snap_keys)),
                                    sharex=True)
    if len(snap_keys) == 1:
        axes_om = [axes_om]

    for idx, step_key in enumerate(snap_keys):
        ax = axes_om[idx]
        snap = snapshots_map[step_key]
        omega_slice, H_row, bed_row = get_slice(snap)
        nz = len(sigma)

        z_plot = np.zeros((n_cells, nz))
        om_plot = np.full((n_cells, nz), np.nan)
        for c in range(n_cells):
            if H_row[c] > H_THRESH:
                for k in range(nz):
                    z_plot[c, k] = bed_row[c] + sigma[k] * H_row[c]
                    om_plot[c, k] = omega_slice[c, k]

        im = ax.pcolormesh(
            np.broadcast_to(dist_km[:, None], (n_cells, nz)),
            z_plot, om_plot,
            cmap='RdBu_r', shading='gouraud',
            vmin=-omega_abs_max, vmax=omega_abs_max)
        ax.fill_between(dist_km, bed_row, bed_row.min() - 200,
                         color='saddlebrown', alpha=0.6)
        ax.plot(dist_km, bed_row, 'k-', lw=1.0)
        ax.plot(dist_km,
                np.where(H_row > H_THRESH, bed_row + H_row, np.nan),
                'b-', lw=1.2)
        ax.set_ylabel('Elevation (m)')
        ax.set_title(f't = {snap["t_yr"]:.0f} yr', fontsize=10)
        z_max = float(np.max(bed_row + np.where(H_row > 0, H_row, 0)))
        ax.set_ylim(float(bed_row.min()) - 200, z_max + 300)

    axes_om[-1].set_xlabel(f'{axis_label} (km)')
    fig_om.colorbar(im, ax=axes_om, label='$\\omega$ (m/yr)',
                    fraction=0.015, pad=0.02)
    fig_om.suptitle(f'Omega Cross-Section along {axis_label}',
                    fontsize=13, fontweight='bold')
    plt.tight_layout()

    tag = axis_label[0]
    fname = OUT_DIR / f'coupled_dome_omega_{tag}.png'
    fig_om.savefig(fname, dpi=150)
    plt.show()
    print(f"  Saved: {fname}")

# ========================================================
# Radial profiles of div(Hu), omega, and speed at multiple
# angles, for each snapshot. Shows whether the flux divergence
# (and hence omega) is truly radially symmetric.
# ========================================================
def compute_div_Hu_centered(u_face, v_face, H_field, dx_val):
    """Compute div(Hu) at cell centers using centered mass flux."""
    ny_h, nx_h = H_field.shape
    div_out = np.zeros_like(H_field)
    for i in range(ny_h):
        for j in range(nx_h):
            h = H_field[i, j]
            if h < H_THRESH:
                continue
            H_xp = H_field[i, min(j+1, nx_h-1)]
            H_xm = H_field[i, max(j-1, 0)]
            H_yp = H_field[min(i+1, ny_h-1), j]
            H_ym = H_field[max(i-1, 0), j]
            u_r = u_face[i, j+1]
            u_l = u_face[i, j]
            v_b = v_face[i+1, j]
            v_t = v_face[i, j]
            flux_r = 0.5*(h + H_xp)*u_r
            flux_l = 0.5*(H_xm + h)*u_l
            flux_b = 0.5*(h + H_yp)*v_b
            flux_t = 0.5*(H_ym + h)*v_t
            div_out[i, j] = (flux_r - flux_l + flux_b - flux_t) / dx_val
    return div_out

# Precompute cell-centered coordinates and radius
x_np = cp.asnumpy(x)
y_np = cp.asnumpy(y)
xx_np, yy_np = np.meshgrid(x_np, y_np)
rr_np = np.sqrt(xx_np**2 + yy_np**2)

# Angles to sample (degrees from x-axis)
angles_deg = [0, 30, 45, 60, 90]
n_angles = len(angles_deg)

# For each angle, extract a radial profile by sampling cells along a ray
def extract_radial_profile(field_2d, H_field, angle_rad, r_max, dr):
    """Sample a 2D field along a radial ray from the dome center."""
    r_vals = np.arange(0, r_max, dr)
    vals = np.full_like(r_vals, np.nan)
    cx = 0.0  # dome center
    cy = 0.0
    for ir, r in enumerate(r_vals):
        px = cx + r * np.cos(angle_rad)
        py = cy + r * np.sin(angle_rad)
        # Find nearest cell
        j = int(round((px - float(x_np[0])) / dx))
        i = int(round((py - float(y_np[0])) / dx))
        if 0 <= i < ny and 0 <= j < nx and H_field[i, j] > H_THRESH:
            vals[ir] = field_2d[i, j]
    return r_vals / 1000.0, vals  # r in km

r_max = float(DOME_RADIUS * 1.05)
dr = dx  # sample at grid spacing

n_snaps_r = len(snap_keys)
fig_rad, axes_rad = plt.subplots(3, n_snaps_r, figsize=(5*n_snaps_r, 12),
                                  squeeze=False, sharey='row')

for col, step_key in enumerate(snap_keys):
    snap = snapshots_map[step_key]
    H_snap = snap['H']
    u_snap = snap['u']
    v_snap = snap['v']
    omega_snap = snap['omega'] * SEC_PER_YR  # m/yr

    # Compute div(Hu) for this snapshot
    div_Hu = compute_div_Hu_centered(u_snap, v_snap, H_snap, dx)

    # Compute cell-centered speed
    u_c = 0.5 * (u_snap[:, 1:] + u_snap[:, :-1])
    v_c = 0.5 * (v_snap[1:, :] + v_snap[:-1, :])
    spd = np.sqrt(u_c**2 + v_c**2)

    # Omega at mid-sigma
    omega_mid = omega_snap[:, :, NZ // 2]

    for ia, angle in enumerate(angles_deg):
        angle_rad = np.radians(angle)
        color = plt.cm.viridis(ia / (n_angles - 1))
        label = f'{angle}$^\\circ$'

        r_km, div_prof = extract_radial_profile(div_Hu, H_snap, angle_rad, r_max, dr)
        axes_rad[0, col].plot(r_km, div_prof, color=color, lw=1.5,
                               label=label if col == 0 else None)

        r_km, om_prof = extract_radial_profile(omega_mid, H_snap, angle_rad, r_max, dr)
        axes_rad[1, col].plot(r_km, om_prof, color=color, lw=1.5)

        r_km, spd_prof = extract_radial_profile(spd, H_snap, angle_rad, r_max, dr)
        axes_rad[2, col].plot(r_km, spd_prof, color=color, lw=1.5)

    axes_rad[0, col].set_title(f't = {snap["t_yr"]:.0f} yr', fontsize=11)
    axes_rad[0, col].axhline(0, color='gray', ls='--', alpha=0.4)
    axes_rad[1, col].axhline(0, color='gray', ls='--', alpha=0.4)
    axes_rad[2, col].set_xlabel('Radius (km)')

axes_rad[0, 0].set_ylabel('div(Hu) (m/yr)')
axes_rad[1, 0].set_ylabel('$\\omega_{mid}$ (m/yr)')
axes_rad[2, 0].set_ylabel('Speed (m/yr)')
axes_rad[0, 0].legend(fontsize=9)

fig_rad.suptitle('Radial Profiles at Multiple Angles', fontsize=14, fontweight='bold')
plt.tight_layout()
fname = OUT_DIR / 'coupled_dome_radial_profiles.png'
fig_rad.savefig(fname, dpi=150)
plt.show()
print(f"  Saved: {fname}")
