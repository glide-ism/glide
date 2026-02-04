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
from glide.physics import abs_loss

from glide.solver import restrict_frozen_fields_to_hierarchy
from glide.kernels import restrict_vfacet, restrict_hfacet, prolongate_cell_centered, get_kernels

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

#physics.set_grid_level(2)
# Access the grid hierarchy
grid = physics.grid

# Forward solve
u, v, H = physics.forward(dt=0.01, n_vcycles=20, verbose=True,update_geometry=False)

u_obs = cp.array(u)
v_obs = cp.array(v)

beta = cp.ones_like(grid.beta)*grid.beta.mean()
physics.set_parameters(beta=beta)

kernels = get_kernels()
obs_hierarchy = [(u_obs, v_obs)]
current_u, current_v = u_obs, v_obs
g = grid
while g.child is not None:
    current_u = restrict_vfacet(current_u, kernels)
    current_v = restrict_hfacet(current_v, kernels)
    obs_hierarchy.append((current_u, current_v))
    g = g.child

from scipy.optimize import fmin_l_bfgs_b
for level_idx in range(N_LEVELS - 1, -1, -1):
    physics.set_grid_level(level_idx)
    current_grid = physics.grid
    u_obs_level, v_obs_level = obs_hierarchy[level_idx]

    u_ref,v_ref,H_ref = physics.forward(dt=0.01,n_vcycles=20,verbose=True,update_geometry=False)
    u_ref = cp.array(u_ref)
    v_ref = cp.array(v_ref)
    H_ref = cp.array(H_ref)

    def objective(log_beta_flat):

        log_beta = cp.asarray(log_beta_flat.reshape((current_grid.ny, current_grid.nx)), dtype=cp.float32) 
        current_grid.beta[:] = cp.exp(log_beta)

        current_grid.u[:] = u_ref
        current_grid.v[:] = v_ref
        current_grid.H[:] = H_ref
        u, v, H = physics.forward(dt=0.01, n_vcycles=10, verbose=False,update_geometry=False)


        J, dJdu, dJdv = abs_loss(u,v,u_obs_level,v_obs_level)
        dJdH = cp.zeros_like(H)
        
    
        grid.Lambda.fill(0)
        physics.adjoint(dJdu,dJdv,dJdH,n_vcycles=3,verbose=False)
        grad_beta = current_grid.compute_grad_beta()
        grad_log_beta = current_grid.beta*grad_beta

        print(f"Level: {level_idx},  Loss: {J:.4f}")

        return float(J),grad_log_beta.ravel().get().astype(np.float64)

    x_init = cp.log(current_grid.beta).ravel().get().astype(np.float64)
    bounds = [(-6,6)]*current_grid.nh

    result = fmin_l_bfgs_b(
        objective, x_init,
        bounds=bounds,
        factr=1e12,
        maxiter=50,
        m=15
    )

    current_grid.beta[:] = cp.exp(cp.array(result[0].reshape((current_grid.ny, current_grid.nx))).astype(cp.float32))     # Prolongate to finer grid for next level
    if level_idx > 0:
        parent = physics.grids[level_idx - 1]
        prolongate_cell_centered(current_grid.beta, kernels, H_fine=parent.beta)   

