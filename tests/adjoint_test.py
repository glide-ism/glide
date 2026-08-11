import cupy as cp
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

grid.forward_operators.var_v[:,:] = cp.random.randn(ny+1,nx,dtype=cp.float32)

grid.forward_operators.var_ud[:,:] = cp.random.randn(ny,nx+1,dtype=cp.float32)

grid.forward_operators.var_vd[:,:] = cp.random.randn(ny+1,nx,dtype=cp.float32)

grid.forward_operators.var_H[:,:] = cp.random.randn(ny,nx,dtype=cp.float32)

grid.adjoint.lambda_u.data[:,:] = cp.random.randn(ny,nx+1,dtype=cp.float32)

grid.adjoint.lambda_v.data[:,:] = cp.random.randn(ny+1,nx,dtype=cp.float32)

grid.adjoint.lambda_ud.data[:,:] = cp.random.randn(ny,nx+1,dtype=cp.float32)

grid.adjoint.lambda_vd.data[:,:] = cp.random.randn(ny+1,nx,dtype=cp.float32)

grid.adjoint.lambda_H.data[:,:] = cp.random.randn(ny,nx,dtype=cp.float32)


grid.forward_operators.compute_jvp(dt)
grid.adjoint_operators.compute_vjp(dt)

t1 = ((grid.adjoint_operators.vjp_u * grid.forward_operators.var_u).sum() +
         (grid.adjoint_operators.vjp_v * grid.forward_operators.var_v).sum() +
         (grid.adjoint_operators.vjp_ud * grid.forward_operators.var_ud).sum() +
         (grid.adjoint_operators.vjp_vd * grid.forward_operators.var_vd).sum() +
         (grid.adjoint_operators.vjp_H * grid.forward_operators.var_H).sum())

t2 = ((grid.forward_operators.jvp_u * grid.adjoint.lambda_u.data).sum() +
         (grid.forward_operators.jvp_v * grid.adjoint.lambda_v.data).sum() +
         (grid.forward_operators.jvp_ud * grid.adjoint.lambda_ud.data).sum() +
         (grid.forward_operators.jvp_vd * grid.adjoint.lambda_vd.data).sum() +
         (grid.forward_operators.jvp_H * grid.adjoint.lambda_H.data).sum())

print(t1,t2,(t1 - t2)/(0.5*(t1 + t2)))

