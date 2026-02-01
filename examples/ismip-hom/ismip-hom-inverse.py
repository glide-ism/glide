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

from glide.solver import restrict_frozen_fields_to_hierarchy

# =============================================================================
# Configuration - modify these paths and parameters
# =============================================================================

N_LEVELS = 6       # Multigrid levels
N_VCYCLES = 10
L = 5000
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
beta*=5

# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B_scalar = cp.float32(1e-16 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)

# =============================================================================
# Initialize physics
# =============================================================================

print("Initializing physics...")
physics = IcePhysics(ny, nx, dx, n_levels=N_LEVELS,m=1./3.)
physics.set_geometry(bed, thk)
physics.set_parameters(B=B, beta=beta, smb=smb)

physics.set_grid_level(2)
# Access the grid hierarchy
grid = physics.grid

# Forward solve
u, v, H = physics.forward(dt=0.01, n_vcycles=N_VCYCLES, verbose=True,update_geometry=False)

u_obs = cp.array(u)
v_obs = cp.array(v)

beta = cp.ones_like(grid.beta)*grid.beta.mean()
physics.set_parameters(beta=beta)
u, v, H = physics.forward(dt=0.01, n_vcycles=N_VCYCLES, verbose=True)
dJdu = (u - u_obs)
dJdv = (v - v_obs)
dJdH = cp.zeros_like(H)

physics.adjoint(dJdu,dJdv,dJdH,n_vcycles=10)


