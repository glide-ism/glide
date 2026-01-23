"""
Greenland forward simulation example.

Run interactively or execute as a script. Modify the paths and parameters
below to match your setup.
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from glide import IcePhysics
from glide.io import VTIWriter, write_vti

from glide.solver import restrict_frozen_fields_to_hierarchy,fascd_vcycle_frozen

# =============================================================================
# Configuration - modify these paths and parameters
# =============================================================================

N_LEVELS = 6       # Multigrid levels
N_VCYCLES = 20
L = 20000
EXP = 'C'

# Physical constants
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0

# =============================================================================
# Configure Domain
# =============================================================================

base_res = 128

y_factr = 7
x_factr = 7

ny = base_res*y_factr
nx = base_res*x_factr

y_slice = int((y_factr//2  +  1./4) * base_res)
x_slice = slice(x_factr//2*base_res,(x_factr//2 + 1)*base_res,1)

x = cp.linspace(0,x_factr*L,nx,dtype=cp.float32)
y = cp.linspace(0,y_factr*L,ny,dtype=cp.float32)
dx = (x[1] - x[0]).item()

X,Y = cp.meshgrid(x,y)

srf = 1000.0 * cp.ones((ny,nx),dtype=cp.float32) - cp.tan(cp.deg2rad(0.1))*X + 1000
bed = srf - 1000 
thk = srf - bed

if EXP == 'C':
    beta = (1000*cp.sin(2*cp.pi*X/L)*cp.sin(2*cp.pi*Y/L) + 1000)/(RHO_ICE*G)
elif EXP == 'D':
    beta = (1000*cp.sin(2*cp.pi*X/L) + 1000)/(RHO_ICE*G)
else:
    raise NotImplementedError('Only support ISMIP-HOM C and D for now')

smb = cp.zeros_like(thk)

# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B_scalar = cp.float32(1e-16 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)

# =============================================================================
# Initialize physics
# =============================================================================

print("Initializing physics...")
physics = IcePhysics(ny, nx, dx, n_levels=N_LEVELS,eps_reg=cp.float32(1e-6))
physics.set_geometry(bed, thk)
physics.set_parameters(B=B, beta=beta, smb=smb)

# Access the grid hierarchy
grid = physics.grid

writer = VTIWriter('./results/', base=f"ismip-hom-{EXP}-{L}", dx=dx)

# Forward solve
u, v, H = physics.forward_frozen(dt=0.001, n_vcycles=N_VCYCLES, verbose=True)

# Output
u_c, v_c = physics.get_velocities_cell_centered()
surface = physics.get_surface()

writer.write_step(0, 0, {
    'thk': H,
    'srf': surface,
    'vel': [u_c, v_c]
})
writer.write_pvd()
plt.plot(x[x_slice].get() - x[x_slice][0].get(),grid.u[y_slice,x_slice].get())

