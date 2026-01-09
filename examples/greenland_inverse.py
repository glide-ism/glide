"""
Greenland inverse modeling example.

Infers basal friction (beta) from observed surface velocities using
adjoint-based optimization.

Requirements:
    - BedMachineGreenland-v5.nc (geometry data)
    - Velocity mosaic GeoTIFFs (observed velocities)
    - MAR SMB data (surface mass balance)

Usage:
    python greenland_inverse.py --geometry /path/to/BedMachineGreenland-v5.nc \
        --u-obs /path/to/vx.tif --v-obs /path/to/vy.tif
"""

import argparse
import pickle
import cupy as cp
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from glide import IcePhysics
from glide.io import VTIWriter, write_vti
from glide.physics import huber_loss, huber_grad, tikhonov_regularization
from glide.data import (
    load_bedmachine_greenland,
    load_velocity_mosaic,
    load_smb_mar,
    prepare_greenland_grid,
    interpolate_to_grid
)
from glide.kernels import restrict_vfacet, restrict_hfacet, get_kernels


# Physical constants
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0


def main():
    parser = argparse.ArgumentParser(description="Greenland inverse modeling")
    parser.add_argument("--geometry", required=True, help="Path to BedMachineGreenland-v5.nc")
    parser.add_argument("--u-obs", required=True, help="Path to x-velocity GeoTIFF")
    parser.add_argument("--v-obs", required=True, help="Path to y-velocity GeoTIFF")
    parser.add_argument("--smb", required=True, help="Path to MAR SMB netCDF")
    parser.add_argument("--output", default="inverse_output", help="Output directory")
    parser.add_argument("--skip", type=int, default=6, help="Geometry downsampling factor")
    parser.add_argument("--dt", type=float, default=10.0, help="Time step (years)")
    parser.add_argument("--n-levels", type=int, default=5, help="Multigrid levels")
    parser.add_argument("--reg-weight", type=float, default=1e-4, help="Tikhonov regularization weight")
    args = parser.parse_args()

    kernels = get_kernels()

    # Load geometry
    print("Loading geometry...")
    geometry = load_bedmachine_greenland(args.geometry, skip=args.skip, thklim=0.1)
    geometry = prepare_greenland_grid(geometry, n_levels=args.n_levels)

    ny, nx = geometry['ny'], geometry['nx']
    dx = geometry['dx']
    x, y = geometry['x'], geometry['y']

    print(f"Grid: {ny} x {nx}, dx = {dx:.1f} m")

    # Load observed velocities
    print("Loading observed velocities...")
    vel_data = load_velocity_mosaic(args.u_obs, args.v_obs)
    X, Y = cp.meshgrid(cp.array(x), cp.array(y))

    u_obs_cell = interpolate_to_grid(vel_data['u'], vel_data['x'], vel_data['y'], x, y)
    v_obs_cell = interpolate_to_grid(vel_data['v'], vel_data['x'], vel_data['y'], x, y)

    # Interpolate to faces
    u_obs = cp.zeros((ny, nx + 1), dtype=cp.float32)
    u_obs[:, 1:-1] = (u_obs_cell[:, 1:] + u_obs_cell[:, :-1]) / 2.0
    v_obs = cp.zeros((ny + 1, nx), dtype=cp.float32)
    v_obs[1:-1] = (v_obs_cell[1:] + v_obs_cell[:-1]) / 2.0

    # Load SMB
    print("Loading SMB...")
    smb_data = load_smb_mar(args.smb)
    smb = interpolate_to_grid(smb_data['smb'], smb_data['x'], smb_data['y'], x, y)

    # Compute B (rate factor)
    B_scalar = cp.float32(1e-17 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
    B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)

    # Initialize physics
    print("Initializing physics...")
    physics = IcePhysics(ny, nx, dx, n_levels=args.n_levels, thklim=0.1)
    physics.set_geometry(geometry['bed'], geometry['thickness'])
    physics.set_parameters(B=B, beta=0.01, smb=smb)

    # Multi-resolution optimization: start from coarsest grid
    grid = physics.grid

    # Build observation hierarchy
    obs_hierarchy = [(u_obs, v_obs)]
    current_u, current_v = u_obs, v_obs
    g = grid
    while g.child is not None:
        current_u = restrict_vfacet(current_u, kernels)
        current_v = restrict_hfacet(current_v, kernels)
        obs_hierarchy.append((current_u, current_v))
        g = g.child

    # Start from coarsest
    level_grids = [grid]
    g = grid
    while g.child is not None:
        level_grids.append(g.child)
        g = g.child

    current_grid = level_grids[-1]
    current_obs_idx = len(obs_hierarchy) - 1

    for level_idx in range(len(level_grids) - 1, -1, -1):
        current_grid = level_grids[level_idx]
        u_obs_level, v_obs_level = obs_hierarchy[level_idx]

        print(f"\n=== Optimizing at level {level_idx}: {current_grid.ny} x {current_grid.nx} ===")

        # Set up output writer for this level
        writer = VTIWriter(
            f"{args.output}/level_{level_idx}",
            base="inverse",
            dx=float(current_grid.dx)
        )

        # Write observations
        u_obs_c = 0.5 * (u_obs_level[:, 1:] + u_obs_level[:, :-1])
        v_obs_c = 0.5 * (v_obs_level[1:] + v_obs_level[:-1])
        write_vti(
            f"{args.output}/level_{level_idx}/u_obs.vti",
            {'vel': [u_obs_c, v_obs_c]},
            float(current_grid.dx)
        )

        counter = [0]  # Mutable counter for callback

        def objective(log_beta_flat):
            """Objective function for L-BFGS-B."""
            log_beta = cp.array(log_beta_flat.reshape((current_grid.ny, current_grid.nx))).astype(cp.float32)
            current_grid.beta[:] = cp.exp(log_beta)

            # Reset state
            current_grid.u.fill(0.0)
            current_grid.v.fill(0.0)
            current_grid.H[:] = current_grid.H_prev

            # Forward solve
            from glide.solver import fascd_vcycle, restrict_parameters_to_hierarchy
            restrict_parameters_to_hierarchy(current_grid)

            for _ in range(5):
                fascd_vcycle(current_grid, physics.thklim, finest=True)

            # Compute loss
            J = huber_loss(current_grid.u, current_grid.v, u_obs_level, v_obs_level)
            dJdu, dJdv = huber_grad(current_grid.u, current_grid.v, u_obs_level, v_obs_level)

            # Adjoint solve
            current_grid.f_adj_u[:] = dJdu
            current_grid.f_adj_v[:] = dJdv
            current_grid.f_adj_H.fill(0.0)
            current_grid.Lambda.fill(0.0)

            from glide.solver import adjoint_vcycle
            adjoint_vcycle(current_grid)

            # Gradient
            grad_beta = current_grid.compute_grad_beta().clip(-1.0, 1.0)
            grad_log_beta = current_grid.beta * grad_beta

            # Regularization
            tik_loss, tik_grad = tikhonov_regularization(current_grid.beta)
            tik_loss *= args.reg_weight
            tik_grad *= args.reg_weight

            total_loss = J + tik_loss
            total_grad = -grad_log_beta + tik_grad

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

        # Update beta
        current_grid.beta[:] = cp.exp(cp.array(result[0].reshape((current_grid.ny, current_grid.nx))).astype(cp.float32))
        current_grid.mask.fill(0.0)

        # Save result
        pickle.dump(
            current_grid.beta.get(),
            open(f"{args.output}/beta_level_{level_idx}.p", 'wb')
        )

        # Prolongate to finer grid
        if level_idx > 0:
            from glide.kernels import prolongate_cell_centered
            parent = level_grids[level_idx - 1]
            prolongate_cell_centered(current_grid.beta, kernels, H_fine=parent.beta)

    print("\nOptimization complete!")
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()
