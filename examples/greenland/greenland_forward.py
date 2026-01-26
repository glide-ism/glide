"""
Greenland forward simulation example.

Run interactively or execute as a script. Modify the paths and parameters
below to match your setup.
"""

import xarray as xr
import pickle
import cupy as cp
import numpy as np

from glide import IcePhysics
from glide.io import VTIWriter, write_vti
from glide.data import (
    load_bedmachine,
    load_smb_mar,
    prepare_grid,
    interpolate_to_grid,
    load_greenland_preprocessed
)

# =============================================================================
# Configuration - modify these paths and parameters
# =============================================================================

OUTPUT_DIR = "./output"

SKIP = 6           # Geometry downsampling factor
DT = 20.0          # Time step (years)
N_STEPS = 50      # Number of time steps
N_LEVELS = 5       # Multigrid levels
N_VCYCLES = 5      # V-cycles per time step

# Physical constants
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0

# =============================================================================
# Load data - from source files
# =============================================================================

"""
GEOMETRY_PATH = "./data/BedMachineGreenland-v5.nc"
SMB_PATH = "./data/MARv3.9-yearly-MIROC5-rcp85-ltm1995-2014.nc"
BETA_PATH = "./inverse_output/beta_level_0.p"

print("Loading geometry...")
geometry = load_bedmachine(GEOMETRY_PATH, skip=SKIP, thklim=0.1)
geometry = prepare_grid(geometry, n_levels=N_LEVELS)

bed = geometry['bed']
thickness = geometry['thickness']

ny, nx = geometry['ny'], geometry['nx']
dx = geometry['dx']
x, y = geometry['x'], geometry['y']

print(f"Grid: {ny} x {nx}, dx = {dx:.1f} m")

print("Loading SMB...")
smb_data = load_smb_mar(SMB_PATH)
smb = interpolate_to_grid(
    smb_data['smb'], smb_data['x'], smb_data['y'],
    x, y
)

print("Loading beta...")
beta = cp.array(pickle.load(open(BETA_PATH, 'rb')))
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
beta = dataset.beta.values
smb = dataset.smb.values
smb -= 2.0
# =============================================================================
# Initialize physics
# =============================================================================

# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B_scalar = cp.float32(1e-18 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)


print("Initializing physics...")
physics = IcePhysics(ny, nx, dx, n_levels=N_LEVELS, thklim=0.1,water_drag=1e-3,gl_sigmoid_c=0.1,gl_derivatives=False,calving_rate=1.0)
physics.set_geometry(bed, thickness)
physics.set_parameters(B=B, beta=beta, smb=smb)

# Access the grid hierarchy
grid = physics.grid

# =============================================================================
# Set up output
# =============================================================================

writer = VTIWriter(OUTPUT_DIR, base="greenland", dx=dx)
write_vti(f"{OUTPUT_DIR}/bed.vti", {'bed': grid.bed}, dx)

# =============================================================================
# Time stepping
# =============================================================================

print(f"Running {N_STEPS} time steps of {DT} years...")
t = 0.0

for step in range(N_STEPS):
    print(f"Step {step}: t = {t:.1f} yr, H_mean = {float(grid.H.mean()):.1f} m")

    # Forward solve
    #grid.u.fill(0)
    #grid.v.fill(0)
    u, v, H = physics.forward_frozen(dt=DT, n_vcycles=N_VCYCLES, verbose=True)
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

