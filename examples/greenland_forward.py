"""
Greenland forward simulation example.

Runs a time-dependent simulation of the Greenland ice sheet using GLIDE.

Requirements:
    - BedMachineGreenland-v5.nc (geometry data)
    - MAR SMB data (surface mass balance)
    - Pre-computed beta field (basal friction)

Usage:
    python greenland_forward.py --geometry /path/to/BedMachineGreenland-v5.nc
"""

import argparse
import pickle
import cupy as cp
import numpy as np

from glide import IcePhysics
from glide.io import VTIWriter, write_vti
from glide.data import (
    load_bedmachine_greenland,
    load_smb_mar,
    prepare_greenland_grid,
    interpolate_to_grid
)


# Physical constants
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0


def main():
    parser = argparse.ArgumentParser(description="Greenland forward simulation")
    parser.add_argument("--geometry", required=True, help="Path to BedMachineGreenland-v5.nc")
    parser.add_argument("--smb", required=True, help="Path to MAR SMB netCDF")
    parser.add_argument("--beta", required=True, help="Path to pickled beta field")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--skip", type=int, default=6, help="Geometry downsampling factor")
    parser.add_argument("--dt", type=float, default=25.0, help="Time step (years)")
    parser.add_argument("--n-steps", type=int, default=200, help="Number of time steps")
    parser.add_argument("--n-levels", type=int, default=5, help="Multigrid levels")
    args = parser.parse_args()

    # Load geometry
    print("Loading geometry...")
    geometry = load_bedmachine_greenland(args.geometry, skip=args.skip, thklim=0.1)
    geometry = prepare_greenland_grid(geometry, n_levels=args.n_levels)

    ny, nx = geometry['ny'], geometry['nx']
    dx = geometry['dx']
    x, y = geometry['x'], geometry['y']

    print(f"Grid: {ny} x {nx}, dx = {dx:.1f} m")

    # Load SMB
    print("Loading SMB...")
    smb_data = load_smb_mar(args.smb)
    smb = interpolate_to_grid(
        smb_data['smb'], smb_data['x'], smb_data['y'],
        x, y
    )

    # Load beta
    print("Loading beta...")
    beta = cp.array(pickle.load(open(args.beta, 'rb')))

    # Compute B (rate factor)
    B_scalar = cp.float32(1e-18 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
    B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)

    # Initialize physics
    print("Initializing physics...")
    physics = IcePhysics(ny, nx, dx, n_levels=args.n_levels, thklim=0.1)
    physics.set_geometry(geometry['bed'], geometry['thickness'])
    physics.set_parameters(B=B, beta=beta, smb=smb)

    # Set up output
    writer = VTIWriter(args.output, base="greenland", dx=dx)
    write_vti(f"{args.output}/bed.vti", {'bed': physics.grid.bed}, dx)

    # Time stepping
    print(f"Running {args.n_steps} time steps of {args.dt} years...")
    t = 0.0

    for step in range(args.n_steps):
        print(f"Step {step}: t = {t:.1f} yr, H_mean = {float(physics.grid.H.mean()):.1f} m")

        # Forward solve
        u, v, H = physics.forward(dt=args.dt, n_vcycles=3, verbose=True)
        t += args.dt

        # Output
        u_c, v_c = physics.get_velocities_cell_centered()
        surface = physics.get_surface()

        writer.write_step(step, t, {
            'thk': H,
            'srf': surface,
            'vel': [u_c, v_c]
        })
        writer.write_pvd()

    print("Done!")


if __name__ == "__main__":
    main()
