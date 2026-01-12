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
    load_smb_racmo,
    prepare_grid,
    interpolate_to_grid
)

# =============================================================================
# Configuration - modify these paths and parameters
# =============================================================================

GEOMETRY_PATH = "./data/BedMachineAntarctica-v3.nc"
SMB_PATH = "./data/smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc"
BETA_PATH = "./inverse_output/beta_level_0.p"
OUTPUT_DIR = "./output"

SKIP = 4           # Geometry downsampling factor
DT = 10.0          # Time step (years)
N_STEPS = 20      # Number of time steps
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
geometry = load_bedmachine(GEOMETRY_PATH, skip=SKIP, thklim=0.1,bbox_pad=[1100,1000,1600,1600])
geometry = prepare_grid(geometry, n_levels=N_LEVELS)

ny, nx = geometry['ny'], geometry['nx']
dx = geometry['dx']
x, y = geometry['x'], geometry['y']

print(f"Grid: {ny} x {nx}, dx = {dx:.1f} m")

print("Loading SMB...")
smb = load_smb_racmo(SMB_PATH,x,y)

smb[geometry['surface'] == 0] = -50.0

print("Loading beta...")
beta = cp.array(pickle.load(open(BETA_PATH, 'rb')))

# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B_scalar = cp.float32(1e-17 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)

# =============================================================================
# Initialize physics
# =============================================================================

print("Initializing physics...")
physics = IcePhysics(ny, nx, dx, n_levels=N_LEVELS, thklim=0.1,calving_rate=0.0,water_drag=1e-5)
physics.set_geometry(geometry['bed'], geometry['thickness'])
physics.set_parameters(B=B, beta=beta, smb=smb)

# Access the grid hierarchy
grid = physics.grid

# =============================================================================
# Set up output
# =============================================================================

writer = VTIWriter(OUTPUT_DIR, base="antarctica", dx=dx)
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
