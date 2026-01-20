"""
Greenland inverse modeling example.

Infers basal friction (beta) from observed surface velocities using
adjoint-based optimization. Run interactively or as a script.
"""

import xarray as xr
import pickle
import cupy as cp
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def allocate_pinned(shape, dtype=np.float64):
    """Allocate a pinned (page-locked) numpy array for fast GPU transfers."""
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    return np.frombuffer(mem, dtype=dtype, count=int(np.prod(shape))).reshape(shape)

from glide import IcePhysics
from glide.io import VTIWriter, write_vti
from glide.physics import abs_loss, abs_grad, tikhonov_regularization
from glide.data import (
    load_bedmachine,
    load_velocity_mosaic,
    load_smb_mar,
    prepare_grid,
    interpolate_to_grid,
    load_greenland_preprocessed
)
from glide.kernels import restrict_vfacet, restrict_hfacet, get_kernels
from glide.solver import fascd_vcycle_frozen, adjoint_vcycle, restrict_parameters_to_hierarchy, restrict_frozen_fields_to_hierarchy
from glide.kernels import prolongate_cell_centered

# =============================================================================
# Configuration - modify these paths and parameters
# =============================================================================

GEOMETRY_PATH = "./data/BedMachineGreenland-v5.nc"
U_OBS_PATH = "./data/greenland_vel_mosaic250_vx_v1.tif"
V_OBS_PATH = "./data/greenland_vel_mosaic250_vy_v1.tif"
SMB_PATH = "./data/MARv3.9-yearly-MIROC5-rcp85-ltm1995-2014.nc"
OUTPUT_DIR = "./inverse_output"

SKIP = 6              # Geometry downsampling factor
DT = 10.0             # Time step (years)
N_LEVELS = 5          # Multigrid levels
REG_WEIGHT = 1e-4     # Tikhonov regularization weight

# Physical constants
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0

kernels = get_kernels()
# =============================================================================
# Load data - from source files
# =============================================================================
"""

print("Loading geometry...")
geometry = load_bedmachine(GEOMETRY_PATH, skip=SKIP, thklim=0.1)
geometry = prepare_grid(geometry, n_levels=N_LEVELS)

ny, nx = geometry['ny'], geometry['nx']
dx = geometry['dx']
x, y = geometry['x'], geometry['y']

print(f"Grid: {ny} x {nx}, dx = {dx:.1f} m")

print("Loading observed velocities...")
vel_data = load_velocity_mosaic(U_OBS_PATH, V_OBS_PATH)

u_obs_cell = interpolate_to_grid(vel_data['u'], vel_data['x'], vel_data['y'], x, y)
v_obs_cell = interpolate_to_grid(vel_data['v'], vel_data['x'], vel_data['y'], x, y)

# Interpolate to faces
u_obs = cp.zeros((ny, nx + 1), dtype=cp.float32)
u_obs[:, 1:-1] = (u_obs_cell[:, 1:] + u_obs_cell[:, :-1]) / 2.0
v_obs = cp.zeros((ny + 1, nx), dtype=cp.float32)
v_obs[1:-1] = (v_obs_cell[1:] + v_obs_cell[:-1]) / 2.0

print("Loading SMB...")
smb_data = load_smb_mar(SMB_PATH)
smb = interpolate_to_grid(smb_data['smb'], smb_data['x'], smb_data['y'], x, y)
"""
# =============================================================================
# Load data - From prepackaged
# =============================================================================

dataset = load_greenland_preprocessed()
ny,nx = dataset.ny,dataset.nx
dx = dataset.dx
bed = dataset.bed.values
surface = dataset.surface.values
thickness = dataset.thickness.values
smb = dataset.smb.values
u_obs_cell = dataset.vx.values
v_obs_cell = dataset.vy.values

# Interpolate to faces
u_obs = cp.zeros((ny, nx + 1), dtype=cp.float32)
u_obs[:, 1:-1] = cp.array((u_obs_cell[:, 1:] + u_obs_cell[:, :-1]) / 2.0)
v_obs = cp.zeros((ny + 1, nx), dtype=cp.float32)
v_obs[1:-1] = cp.array((v_obs_cell[1:] + v_obs_cell[:-1]) / 2.0)


# =============================================================================
# Initialize physics
# =============================================================================

# Compute B (rate factor)
B_scalar = cp.float32(1e-17 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)


print("Initializing physics...")
physics = IcePhysics(ny, nx, dx, n_levels=N_LEVELS, thklim=0.1, water_drag=1e-6,calving_rate=2.0)
physics.set_geometry(bed, thickness)
physics.set_parameters(B=B, beta=0.01, smb=smb)

grid = physics.grid

# =============================================================================
# Build observation hierarchy for multi-resolution optimization
# =============================================================================

obs_hierarchy = [(u_obs, v_obs)]
current_u, current_v = u_obs, v_obs
g = grid
while g.child is not None:
    current_u = restrict_vfacet(current_u, kernels)
    current_v = restrict_hfacet(current_v, kernels)
    obs_hierarchy.append((current_u, current_v))
    g = g.child

# Build list of grids from finest to coarsest
level_grids = [grid]
g = grid
while g.child is not None:
    level_grids.append(g.child)
    g = g.child

# =============================================================================
# Multi-resolution optimization
# =============================================================================

for level_idx in range(len(level_grids) - 1, -1, -1):
    current_grid = level_grids[level_idx]
    u_obs_level, v_obs_level = obs_hierarchy[level_idx]

    print(f"\n=== Optimizing at level {level_idx}: {current_grid.ny} x {current_grid.nx} ===")

    writer = VTIWriter(
        f"{OUTPUT_DIR}/level_{level_idx}",
        base="inverse",
        dx=float(current_grid.dx)
    )

    # Write observations
    u_obs_c = 0.5 * (u_obs_level[:, 1:] + u_obs_level[:, :-1])
    v_obs_c = 0.5 * (v_obs_level[1:] + v_obs_level[:-1])
    write_vti(
        f"{OUTPUT_DIR}/level_{level_idx}/u_obs.vti",
        {'vel': [u_obs_c, v_obs_c]},
        float(current_grid.dx)
    )

    counter = [0]

    # Allocate pinned memory for fast CPU-GPU transfers
    n_params = current_grid.nh
    x_pinned = allocate_pinned(n_params, dtype=np.float64)
    grad_pinned = allocate_pinned(n_params, dtype=np.float64)

    def objective(log_beta_flat):
        """Objective function for L-BFGS-B."""
        # Transfer from (pinned) CPU to GPU
        log_beta = cp.asarray(log_beta_flat.reshape((current_grid.ny, current_grid.nx)), dtype=cp.float32)
        current_grid.beta[:] = cp.exp(log_beta)

        # Reset state
        current_grid.u.fill(0.0)
        current_grid.v.fill(0.0)
        current_grid.H[:] = current_grid.H_prev

        # Forward solve
        restrict_parameters_to_hierarchy(current_grid)
        current_grid.f_H[:,:] = current_grid.H_prev/current_grid.dt + current_grid.smb


        for _ in range(5):
            current_grid.compute_eta_field()
            current_grid.compute_beta_eff_field()
            current_grid.compute_c_eff_field()
            # Restrict frozen fields to entire hierarchy
            restrict_frozen_fields_to_hierarchy(current_grid)
            
            fascd_vcycle_frozen(current_grid, physics.thklim, finest=True)

        # Compute loss
        J = abs_loss(current_grid.u, current_grid.v, u_obs_level, v_obs_level)
        dJdu, dJdv = abs_grad(current_grid.u, current_grid.v, u_obs_level, v_obs_level)

        # Adjoint solve
        current_grid.f_adj_u[:] = -dJdu
        current_grid.f_adj_v[:] = -dJdv
        current_grid.f_adj_H.fill(0.0)
        current_grid.Lambda.fill(0.0)

        adjoint_vcycle(current_grid)

        # Gradient
        grad_beta = current_grid.compute_grad_beta()
        grad_log_beta = current_grid.beta * grad_beta

        # Regularization
        tik_loss, tik_grad = tikhonov_regularization(current_grid.beta)
        tik_loss *= REG_WEIGHT
        tik_grad *= REG_WEIGHT

        total_loss = J + tik_loss
        total_grad = grad_log_beta + tik_grad

        print(f"  Loss: {J:.4f}, Reg: {tik_loss:.4f}, Total: {total_loss:.4f}")

        # Transfer gradient to pinned memory (fast GPU->CPU)
        grad_pinned[:] = total_grad.ravel().get().astype(np.float64)
        return float(total_loss), grad_pinned

    def callback(log_beta_flat):
        """Callback for visualization."""
        log_beta = cp.asarray(log_beta_flat.reshape((current_grid.ny, current_grid.nx)), dtype=cp.float32)
        counter[0] += 1

        u_c = 0.5 * (current_grid.u[:, 1:] + current_grid.u[:, :-1])
        v_c = 0.5 * (current_grid.v[1:] + current_grid.v[:-1])

        writer.write_step(counter[0], counter[0], {
            'log_beta': log_beta,
            'vel': [u_c, v_c]
        })
        writer.write_pvd()

    # Initialize x0 in pinned memory
    x_pinned[:] = cp.log(current_grid.beta).ravel().get().astype(np.float64)
    bounds = [(-6, 5)] * current_grid.nh

    result = fmin_l_bfgs_b(
        objective, x_pinned,
        bounds=bounds,
        callback=callback,
        factr=1e11,
        m=15
    )

    # Update beta with optimized values
    current_grid.beta[:] = cp.exp(cp.array(result[0].reshape((current_grid.ny, current_grid.nx))).astype(cp.float32))
    current_grid.mask.fill(0.0)

    # Save result
    pickle.dump(
        current_grid.beta.get(),
        open(f"{OUTPUT_DIR}/beta_level_{level_idx}.p", 'wb')
    )

    # Prolongate to finer grid for next level
    if level_idx > 0:
        parent = level_grids[level_idx - 1]
        prolongate_cell_centered(current_grid.beta, kernels, H_fine=parent.beta)

print("\nOptimization complete!")
print(f"Results saved to {OUTPUT_DIR}/")
