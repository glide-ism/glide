import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from glide.grid import Grid
from glide.model import IceDynamics

rho_i = cp.float32(917.0)
g = cp.float32(9.81)
n = 3.0

# ISMIP-HOM A: no-slip bed, 0.5 degree slope, sinusoidal bed topography
def setup_a(X, Y, L, ny, nx):
    srf = 1000.0 * cp.ones((ny,nx),dtype=cp.float32) - cp.tan(cp.deg2rad(0.5))*X + 10000
    bed = srf - 1000 + 500*cp.sin(2*cp.pi*X/L)*cp.sin(2*cp.pi*Y/L)
    # Effectively frozen bed: beta**2 = 1e6 Pa a / m
    beta = 1e6/(rho_i * g) * cp.ones((ny,nx),dtype=cp.float32)
    return srf, bed, beta

# ISMIP-HOM C: 0.1 degree slope, uniform thickness, sinusoidal basal friction
def setup_c(X, Y, L, ny, nx):
    srf = 1000.0 * cp.ones((ny,nx),dtype=cp.float32) - cp.tan(cp.deg2rad(0.1))*X + 10000
    bed = srf - 1000
    beta = (1000*cp.sin(2*cp.pi*X/L)*cp.sin(2*cp.pi*Y/L) + 1000)/(rho_i * g)
    return srf, bed, beta


fig,axs = plt.subplots(nrows=2,ncols=3)

for ax,L in zip(axs.ravel(),[5000,10000,20000,40000,80000,160000]):

    dt = cp.float32(0.1)

    base_res = 128
    y_factr = 5
    x_factr = 5

    ny = base_res*y_factr
    nx = base_res*x_factr

    x = cp.linspace(0,x_factr*L,nx,dtype=cp.float32)
    y = cp.linspace(0,y_factr*L,ny,dtype=cp.float32)

    y_slice = int((y_factr//2  +  1./4) * base_res)
    x_slice = slice(x_factr//2*base_res,(x_factr//2 + 1)*base_res,1)

    dx = (x[1] - x[0]).item()

    X,Y = cp.meshgrid(x,y)

    for name,setup in [('A',setup_a),('C',setup_c)]:

        srf, bed, beta = setup(X, Y, L, ny, nx)
        thk = srf - bed

        B = cp.ones((ny,nx),dtype=cp.float32)
        B.fill((1e-16 ** -(1./3))/(rho_i * g))

        model = IceDynamics(n_levels=6,ny=ny,nx=nx,dx=dx)

        model.mg.geometry.bed.set(bed)
        model.mg.rheology.B.set(B)
        model.mg.sliding.beta.set(beta)
        model.mg.sliding.m.set(1.0)
        model.mg.sliding.u_reg.set(1.0)
        model.mg.state.H.set(thk)
        model.mg.state.H_prev.set(thk)

        ### Set multigrid solver parameters ###
        model.forward_solver.fas_options.set(
                coarsest_steps=300, pre_steps=10,
                post_steps=20, finest_steps=0,
                relative_tolerance=5e-3, absolute_tolerance=1.0,
                report_norms=True)

        print(f"Experiment {name}, L = {L//1000} km")
        model.forward(0.0,dt)

        state = model.mg[0].state
        u_s = (state.u.data + state.ud.data/(n + 1)).get()

        ax.plot(np.linspace(0,1,base_res),u_s[y_slice,x_slice],label=f'{name}')

    ax.legend(title=f'L={L//1000}km')
    ax.set_xlim(0,1)

for ax in axs[:,0]: ax.set_ylabel('surface speed (m/a)')
for ax in axs[1,:]: ax.set_xlabel('x/L')
plt.show()
