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
N_VCYCLES = 20
L = 40000
EXP = 'C'

# Physical constants
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0

# =============================================================================
# Configure Domain
# =============================================================================

base_res = 128

y_factr = 3
x_factr = 3

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

#physics.set_grid_level(2)
# Access the grid hierarchy
grid = physics.grid

# Forward solve
u, v, H = physics.forward(dt=0.01, n_vcycles=N_VCYCLES, verbose=True,update_geometry=False)

u_obs = cp.array(u)
v_obs = cp.array(v)

beta = cp.ones_like(grid.beta)*grid.beta.mean()

u_init = cp.array(u)
v_init = cp.array(v)
H_init = cp.array(H)

physics.set_parameters(beta=beta)
u, v, H = physics.forward(dt=0.01, n_vcycles=N_VCYCLES, verbose=True,update_geometry=False)

#J_0 = 0.5*((u - u_obs)**2).sum() + 0.5*((v - v_obs)**2).sum()
J_0 = (abs(u - u_obs)).sum() + (abs(v - v_obs)).sum()

dJdu = cp.sign(u - u_obs)
dJdv = cp.sign(v - v_obs)
dJdH = cp.zeros_like(H)
physics.adjoint(dJdu,dJdv,dJdH,n_vcycles=10)
grad_beta = grid.compute_grad_beta()

v = cp.random.randn(*grad_beta.shape,dtype=cp.float32)
"""
#v = cp.zeros_like(grad_beta, dtype=cp.float32)

n_comp = 5
kmin, kmax = 1, 3  # integer modes in [1,3]

for _ in range(n_comp):
    kx = cp.random.randint(kmin, kmax + 1)
    ky = cp.random.randint(kmin, kmax + 1)
    phix = 2*cp.pi*cp.random.rand()
    phiy = 2*cp.pi*cp.random.rand()

    v += cp.sin(2*cp.pi*kx*X/L + phix) * cp.sin(2*cp.pi*ky*Y/L + phiy)

v /= cp.sqrt(cp.float32(n_comp))
v -= v.mean()
v /= cp.sqrt((v*v).mean() + 1e-30).astype(cp.float32)
"""        
#v = cp.zeros(grad_beta.shape,dtype=cp.float32)
#v[24,18] = 1.0
eps = cp.float32(1e-2)
grid.beta[:] += v*eps

grid.u[:] = u_init
grid.v[:] = v_init
grid.H[:] = H_init

u1, v1, H1 = physics.forward(dt=0.01, n_vcycles=N_VCYCLES, verbose=True,update_geometry=False)
#J_1 = 0.5*((u1 - u_obs)**2).sum() + 0.5*((v1 - v_obs)**2).sum()
J_1 = (abs(u1 - u_obs)).sum() + (abs(v1 - v_obs)).sum()

grid.beta[:] -= 2*v*eps

grid.u[:] = u_init
grid.v[:] = v_init
grid.H[:] = H_init

u0, v0, H0 = physics.forward(dt=0.01, n_vcycles=N_VCYCLES, verbose=True,update_geometry=False)
#J_1 = 0.5*((u1 - u_obs)**2).sum() + 0.5*((v1 - v_obs)**2).sum()
J_0 = (abs(u0 - u_obs)).sum() + (abs(v0 - v_obs)).sum()

gvp_fd = (J_1 - J_0)/(2*eps)
gvp_ad = (grad_beta*v).sum()

rel_err = abs(gvp_fd - gvp_ad)/abs(gvp_ad)
assert rel_err < 5e-2

