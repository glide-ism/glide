import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from glide import IcePhysics


# =============================================================================
# Configuration - modify these paths and parameters
# =============================================================================

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

# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B_scalar = cp.float32(1e-16 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)

# =============================================================================
# Initialize physics
# =============================================================================

print("Initializing physics...")
physics = IcePhysics(ny, nx, dx, n_levels=1,m=1./3.)
physics.set_geometry(bed, thk)
physics.set_parameters(B=B, beta=beta, smb=smb)

cp.random.seed(0)

# Access the grid hierarchy
grid = physics.grid
grid.set_rhs()
grid.compute_frozen_fields()
grid.vanka_sweep(1000)



grid.mask[:,:] = cp.random.randint(0,2,size=grid.mask.shape).astype(cp.float32)

grid.d_U[:] = cp.random.randn(grid.n_total)
grid.d_u[:,0].fill(0)
grid.d_u[:,-1].fill(0)
grid.d_v[0].fill(0)
grid.d_v[-1].fill(0)
grid.d_H[grid.mask>0.5] = 0

grid.compute_frozen_fields(mode='dual')
grid.compute_jvp()


eps = cp.float32(1e-3)

grid.U[:] += eps * grid.d_U
grid.compute_frozen_fields()
grid.compute_residual()
r1 = cp.array(grid.r)

grid.U[:] -= 2 * eps * grid.d_U
grid.compute_frozen_fields()
grid.compute_residual()
r0 = cp.array(grid.r)

j_fd = (r1 - r0)/(2*eps)

j_u_,j_v_,j_H_ = grid._vec_to_fields(j_fd)

abs_err = cp.linalg.norm(j_fd - grid.j)
rel_err = abs_err / cp.linalg.norm(grid.j)

print(f"Relative norm of jvp versus finite difference: {rel_err}")

assert rel_err < 1e-2



