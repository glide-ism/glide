from glide.backend import xp as cp
import numpy as np
import matplotlib.pyplot as plt

from glide.grid import Grid
from glide.multigrid import Multigrid, FASCDSolver, FASAdjointSolver

L = 20000.0
dt = cp.float32(1.0)

base_res = 64
y_factr = 7
x_factr = 7

ny = base_res*y_factr
nx = base_res*x_factr

x = cp.linspace(0,x_factr*L,nx,dtype=cp.float32)
y = cp.linspace(0,y_factr*L,ny,dtype=cp.float32)
dx = (x[1] - x[0]).item()

X,Y = cp.meshgrid(x,y)

srf = 1000.0 * cp.ones((ny,nx),dtype=cp.float32) - cp.tan(cp.deg2rad(0.1))*X + 10000
bed = srf - 1000 
thk = srf - bed

rho_i = cp.float32(917.0)
g = cp.float32(9.81)
beta = (1000*cp.sin(2*cp.pi*X/L)*cp.sin(2*cp.pi*Y/L) + 1000)/(rho_i * g)

B = cp.ones((ny,nx),dtype=cp.float32)
B.fill((1e-16 ** -(1./3))/(rho_i * g))

mg = Multigrid(6,ny=ny,nx=nx,dx=dx)
grid = mg.levels[0]

mg.geometry.bed.set(bed)
mg.rheology.B.set(B)
mg.sliding.beta.set(beta)
mg.sliding.m.set(1.0)
mg.sliding.u_reg.set(1.0)
mg.state.H.set(thk)
mg.state.H_prev.set(thk)


### Initialize solver
solver = FASCDSolver(mg)

solver.vanka_options.omega.set(0.5)
solver.vanka_options.newton_options.relaxation.set(0.5)
solver.vanka_options.newton_options.steps.set(30)

solver.fas_options.coarsest_steps.set(200)
solver.fas_options.pre_steps.set(10)
solver.fas_options.post_steps.set(50)
solver.fas_options.finest_steps.set(150)
solver.fas_options.maximum_vcycles.set(10)
solver.fas_options.relative_tolerance.set(1e-3)
solver.fas_options.absolute_tolerance.set(0.1)

solver.solve(dt)

u_obs = cp.array(grid.state.u.data)
v_obs = cp.array(grid.state.v.data)

mg.sliding.beta.set(cp.ones_like(beta)*beta.mean())
solver.solve(dt)

u = cp.array(grid.state.u.data)
v = cp.array(grid.state.v.data)
H = cp.array(grid.state.H.data)

J_0 = (abs(u - u_obs)).sum() + (abs(v - v_obs)).sum()

dJdu = cp.sign(u - u_obs)
dJdv = cp.sign(v - v_obs)

grid.adjoint_operators.f_u[:,:] = -dJdu
grid.adjoint_operators.f_v[:,:] = -dJdv

adjoint_solver = FASAdjointSolver(mg)
adjoint_solver.fas_options.coarsest_steps.set(200)
adjoint_solver.fas_options.pre_steps.set(10)
adjoint_solver.fas_options.post_steps.set(50)
adjoint_solver.fas_options.finest_steps.set(150)
adjoint_solver.fas_options.maximum_vcycles.set(10)
adjoint_solver.fas_options.absolute_tolerance.set(cp.float32(0.1))
adjoint_solver.fas_options.relative_tolerance.set(cp.float32(1e-3))
adjoint_solver.vanka_options.newton_options.ssa_damping.set(cp.float32(0.1))
adjoint_solver.vanka_options.omega.set(cp.float32(0.5))
adjoint_solver.solve(dt)

grid.adjoint_operators.compute_gradient_beta()

beta_pert = cp.random.randn(*grid.sliding.beta.data.shape,dtype=cp.float32)

eps = cp.float32(1e-3)
beta_0 = cp.array(grid.sliding.beta.data)
grid.sliding.beta.data[:,:] = beta_0 + eps*beta_pert

grid.state.u.data[:,:] = u
grid.state.v.data[:,:] = v
grid.state.H.data[:,:] = H

solver.solve(dt)
J_1 = (abs(grid.state.u.data[:,:] - u_obs)).sum() + (abs(grid.state.v.data[:,:] - v_obs)).sum()

grid.sliding.beta.data[:,:] = beta_0 - eps*beta_pert

grid.state.u.data[:,:] = u
grid.state.v.data[:,:] = v
grid.state.H.data[:,:] = H

solver.solve(dt)
J_0 = (abs(grid.state.u.data[:,:] - u_obs)).sum() + (abs(grid.state.v.data[:,:] - v_obs)).sum()

gvp_fd = (J_1 - J_0)/(2*eps)
gvp_ad = (grid.sliding.beta.grad*beta_pert).sum()

rel_err = abs(gvp_fd - gvp_ad)/abs(gvp_ad)
print(f"FD: {gvp_fd}, Adj: {gvp_ad}, Rel. Err.: {rel_err}")
assert rel_err < 5e-2

