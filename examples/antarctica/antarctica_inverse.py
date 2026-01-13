"""
Greenland inverse modeling example.

Infers basal friction (beta) from observed surface velocities using
adjoint-based optimization. Run interactively or as a script.
"""

import pickle
import cupy as cp
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from glide import IcePhysics
from glide.io import VTIWriter, write_vti
from glide.physics import abs_loss,abs_grad,huber_loss, huber_grad, tikhonov_regularization
from glide.data import (
    load_bedmachine,
    load_velocity_mosaic,
    load_smb_racmo,
    prepare_grid,
    interpolate_to_grid,
    load_antarctic_velocity
)
from glide.kernels import restrict_vfacet, restrict_hfacet, get_kernels
from glide.solver import fascd_vcycle, adjoint_vcycle, restrict_parameters_to_hierarchy
from glide.kernels import prolongate_cell_centered

# =============================================================================
# Configuration - modify these paths and parameters
# =============================================================================
GEOMETRY_PATH = "./data/BedMachineAntarctica-v3.nc"
SMB_PATH = "./data/smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc"
U_OBS_PATH = "./data/antarctica_ice_velocity_450m_v2.nc"
OUTPUT_DIR = "./inverse_output"

SKIP = 4              # Geometry downsampling factor
DT = 1.0             # Time step (years)
N_LEVELS = 5          # Multigrid levels
REG_WEIGHT = 1e-4     # Tikhonov regularization weight

# Physical constants
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0

# =============================================================================
# Load data
# =============================================================================

kernels = get_kernels()

print("Loading geometry...")
geometry = load_bedmachine(GEOMETRY_PATH, skip=SKIP, thklim=0.1,bbox_pad=[1100,1000,1600,1600])
geometry = prepare_grid(geometry, n_levels=N_LEVELS)

ny, nx = geometry['ny'], geometry['nx']
dx = geometry['dx']
x, y = geometry['x'], geometry['y']

print(f"Grid: {ny} x {nx}, dx = {dx:.1f} m")

print("Loading observed velocities...")
x_vel,y_vel,vx,vy = load_antarctic_velocity(U_OBS_PATH)

u_obs_cell = interpolate_to_grid(vx, x_vel, y_vel, x, y)
v_obs_cell = interpolate_to_grid(vy, x_vel, y_vel, x, y)


# Interpolate to faces
u_obs = cp.zeros((ny, nx + 1), dtype=cp.float32)
u_obs[:, 1:-1] = (u_obs_cell[:, 1:] + u_obs_cell[:, :-1]) / 2.0
v_obs = cp.zeros((ny + 1, nx), dtype=cp.float32)
v_obs[1:-1] = (v_obs_cell[1:] + v_obs_cell[:-1]) / 2.0

smb = load_smb_racmo(SMB_PATH,x,y)
#smb[geometry['surface']==0] = -50
# Compute B (rate factor)
B_scalar = cp.float32(1e-17 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)

# =============================================================================
# Initialize physics
# =============================================================================

print("Initializing physics...")
physics = IcePhysics(ny, nx, dx, n_levels=N_LEVELS, thklim=0.1,calving_rate=0.0,water_drag=1e-5,gl_derivatives=False)
physics.set_geometry(geometry['bed'], geometry['thickness'])
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

for g in level_grids:
    g.dt = cp.float32(DT)

# =============================================================================
# Multi-resolution optimization
# =============================================================================
verbose = False
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

    def objective(log_beta_flat):
        """Objective function for L-BFGS-B."""
        log_beta = cp.array(log_beta_flat.reshape((current_grid.ny, current_grid.nx))).astype(cp.float32)
        current_grid.beta[:] = cp.exp(log_beta)

        # Reset state
        current_grid.u.fill(0.0)
        current_grid.v.fill(0.0)
        current_grid.H[:] = current_grid.H_prev

        # Forward solve
        restrict_parameters_to_hierarchy(current_grid)
        current_grid.f_H[:,:] = current_grid.H_prev/current_grid.dt + current_grid.smb

        if verbose:
            rss_H_init = current_grid.compute_residual(return_fischer_burmeister=True)
            r0 = float(cp.sqrt(
                cp.linalg.norm(current_grid.r_u)**2 +
                cp.linalg.norm(current_grid.r_v)**2 +
                cp.linalg.norm(rss_H_init)**2
            ))
            print(f"  Initial: |r| = {r0:.2e}, "
                  f"|r_u| = {float(cp.linalg.norm(current_grid.r_u)):.2e}, "
                  f"|r_v| = {float(cp.linalg.norm(current_grid.r_v)):.2e}, "
                  f"|rss_H| = {float(cp.linalg.norm(rss_H_init)):.2e}")
        for _ in range(5):
            fascd_vcycle(current_grid, physics.thklim, finest=True)
            if verbose:
                rss_H = current_grid.compute_residual(return_fischer_burmeister=True)
                r_combined = float(cp.sqrt(
                    cp.linalg.norm(current_grid.r_u)**2 +
                    cp.linalg.norm(current_grid.r_v)**2 +
                    cp.linalg.norm(rss_H)**2
                ))
                rel = r_combined / r0 if r0 > 0 else 0.0
                print(f"  V-cycle {counter}: |r|/|r0| = {rel:.2e}, "
                      f"|r_u| = {float(cp.linalg.norm(current_grid.r_u)):.2e}, "
                      f"|r_v| = {float(cp.linalg.norm(current_grid.r_v)):.2e}, "
                      f"|rss_H| = {float(cp.linalg.norm(rss_H)):.2e}")


        # Compute loss
        J = abs_loss(current_grid.u, current_grid.v, u_obs_level, v_obs_level)
        dJdu, dJdv = abs_grad(current_grid.u, current_grid.v, u_obs_level, v_obs_level)

        # Adjoint solve
        current_grid.f_adj_u[:] = -dJdu
        current_grid.f_adj_v[:] = -dJdv
        current_grid.f_adj_H.fill(0.0)
        current_grid.Lambda.fill(0.0)

        adjoint_vcycle(current_grid)
        adjoint_vcycle(current_grid)

        # Gradient
        grad_beta = current_grid.grad_beta = current_grid.compute_grad_beta()
        grad_log_beta = current_grid.beta * grad_beta

        # Regularization
        tik_loss, tik_grad = tikhonov_regularization(current_grid.beta)
        tik_loss *= REG_WEIGHT
        tik_grad *= REG_WEIGHT

        total_loss = J + tik_loss
        total_grad = grad_log_beta + tik_grad

        print(f"  Loss: {J:.4f}, Reg: {tik_loss:.4f}, Total: {total_loss:.4f}")

        return float(total_loss), total_grad.get().ravel().astype(np.float64)

    def callback(log_beta_flat):
        """Callback for visualization."""
        log_beta = cp.array(log_beta_flat.reshape((current_grid.ny, current_grid.nx))).astype(cp.float32)
        counter[0] += 1

        u_c = 0.5 * (current_grid.u[:, 1:] + current_grid.u[:, :-1])
        v_c = 0.5 * (current_grid.v[1:] + current_grid.v[:-1])

        writer.write_step(counter[0], counter[0], {
            'log_beta': log_beta,
            'vel': [u_c, v_c]
        })
        writer.write_pvd()

    # Run optimization
    x0 = cp.log(current_grid.beta).ravel().get().astype(np.float64)
    bounds = [(-6, 5)] * current_grid.nh

    result = fmin_l_bfgs_b(
        objective, x0,
        bounds=bounds,
        callback=callback,
        factr=1e10,
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
