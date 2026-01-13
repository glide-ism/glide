"""
Greenland forward simulation example.

Run interactively or execute as a script. Modify the paths and parameters
below to match your setup.
"""

import pickle
import cupy as cp
import numpy as np

from glide import IcePhysics
from glide.io import VTIWriter, write_vti
from glide.data import (
    load_bedmachine,
    load_smb_mar,
    prepare_grid,
    interpolate_to_grid
)

# =============================================================================
# Configuration - modify these paths and parameters
# =============================================================================

GEOMETRY_PATH = "./data/bitterroot.tif"




OUTPUT_DIR = "./output"

SKIP = 6           # Geometry downsampling factor
DT = 10.0          # Time step (years)
N_STEPS = 200      # Number of time steps
N_LEVELS = 5       # Multigrid levels
N_VCYCLES = 3      # V-cycles per time step

# Physical constants
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0

# =============================================================================
# Load data
# =============================================================================

print("Loading geometry...")
import xarray as xr
data = xr.load_dataset(GEOMETRY_PATH)
bed = data.band_data.values.squeeze()[100:-100,100:-100]
srf = bed + 0.1
thk = srf - bed
x = data.x.values[100:-100]
y = data.y.values[100:-100]
ny,nx = srf.shape
dx = x[1]-x[0]

print(f"Grid: {ny} x {nx}, dx = {dx:.1f} m")

print("Loading SMB...")
ela = 1800
smb = 0.5/1000.0*(srf - ela)

print("Loading beta...")
beta = cp.ones_like(thk)*0.1#cp.array(pickle.load(open(BETA_PATH, 'rb')))

# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B_scalar = cp.float32(1e-16 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)

# =============================================================================
# Initialize physics
# =============================================================================

print("Initializing physics...")
physics = IcePhysics(ny, nx, dx, n_levels=N_LEVELS, thklim=0.1,water_drag=1e-6)
physics.set_geometry(bed, thk)
physics.set_parameters(B=B, beta=beta, smb=smb)

# Access the grid hierarchy
grid = physics.grid

# =============================================================================
# Set up output
# =============================================================================

writer = VTIWriter(OUTPUT_DIR, base="bearcreek", dx=dx)
write_vti(f"{OUTPUT_DIR}/bed.vti", {'bed': grid.bed}, dx)

# =============================================================================
# Time stepping
# =============================================================================

print(f"Running {N_STEPS} time steps of {DT} years...")
t = 0.0

for step in range(N_STEPS):
    print(f"Step {step}: t = {t:.1f} yr, H_mean = {float(grid.H.mean()):.1f} m")

    # Forward solve
    u, v, H = physics.forward(dt=DT, n_vcycles=N_VCYCLES, verbose=True)
    t += DT

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

