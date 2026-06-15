from glide.backend import xp as cp
import numpy as np
import matplotlib.pyplot as plt

from glide.grid import Grid

L = 10000.0
dt = cp.float32(1.0)

base_res = 32
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

grid = Grid(ny,nx,dx)

grid.geometry.bed.set(bed)
grid.rheology.B.set(B)
grid.sliding.beta.set(beta)
grid.sliding.m.set(1.0)
grid.sliding.u_reg.set(1.0)
grid.state.H.set(thk)
grid.state.H_prev.set(thk)

grid.forward_operators.set_rhs(dt)
grid.forward_operators.vanka_sweep(dt,4000)

grid.state.mask.data[:,:] = cp.random.randint(0,2,size=(ny,nx)).astype(cp.float32)

grid.forward_operators.var_u[:,:] = cp.random.randn(ny,nx+1,dtype=cp.float32)
grid.forward_operators.var_u[:,0].fill(0)
grid.forward_operators.var_u[:,-1].fill(0)

grid.forward_operators.var_v[:,:] = cp.random.randn(ny+1,nx,dtype=cp.float32)
grid.forward_operators.var_v[0].fill(0)
grid.forward_operators.var_v[-1].fill(0)

grid.forward_operators.var_H[:,:] = cp.random.randn(ny,nx,dtype=cp.float32)
grid.forward_operators.var_H[grid.state.mask.data>0.5] = 0

grid.forward_operators.compute_jvp(dt)

u_0 = cp.array(grid.state.u.data)
v_0 = cp.array(grid.state.v.data)
H_0 = cp.array(grid.state.H.data)

eps = cp.float32(1e-2)

grid.state.u.data[:,:] = u_0 + eps * grid.forward_operators.var_u
grid.state.v.data[:,:] = v_0 + eps * grid.forward_operators.var_v
grid.state.H.data[:,:] = H_0 + eps * grid.forward_operators.var_H

grid.forward_operators.compute_residual(dt)

r1_u = cp.array(grid.forward_operators.r_u)
r1_v = cp.array(grid.forward_operators.r_v)
r1_H = cp.array(grid.forward_operators.r_H)

grid.state.u.data[:,:] = u_0 - eps * grid.forward_operators.var_u
grid.state.v.data[:,:] = v_0 - eps * grid.forward_operators.var_v
grid.state.H.data[:,:] = H_0 - eps * grid.forward_operators.var_H

grid.forward_operators.compute_residual(dt)

r0_u = cp.array(grid.forward_operators.r_u)
r0_v = cp.array(grid.forward_operators.r_v)
r0_H = cp.array(grid.forward_operators.r_H)

jvp_u_fd = (r1_u - r0_u)/(2*eps)
jvp_v_fd = (r1_v - r0_v)/(2*eps)
jvp_H_fd = (r1_H - r0_H)/(2*eps)

abs_err_u = cp.linalg.norm(jvp_u_fd - grid.forward_operators.jvp_u)
abs_err_v = cp.linalg.norm(jvp_v_fd - grid.forward_operators.jvp_v)
abs_err_H = cp.linalg.norm(jvp_H_fd - grid.forward_operators.jvp_H)

rel_err_u = abs_err_u / cp.linalg.norm(grid.forward_operators.jvp_u)
rel_err_v = abs_err_v / cp.linalg.norm(grid.forward_operators.jvp_v)
rel_err_H = abs_err_H / cp.linalg.norm(grid.forward_operators.jvp_H)

print(f"Relative norm of jvp versus finite difference: {rel_err_u:.6f}, {rel_err_v:.6f}, {rel_err_H:.6f}")


