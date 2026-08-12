"""
Antarctica inverse modeling example.

Infers basal friction (beta) from observed surface velocities using
adjoint-based optimization. Run interactively or as a script.
"""

import cupy as cp
import numpy as np
import torch
import pyproj

from scipy.ndimage import gaussian_filter

from glide.model import IceDynamics
from glide.data import load_antarctica_preprocessed
from glide.field import Field, GridEntity
from glide.torch import GlideStep
from glide.io import VTIWriter

### Load a dataset (here a preprocessed greenland dataset)
dataset = load_antarctica_preprocessed()

### Initialize grid
# ny and nx must both divide by 2^(n_levels - 1) cleanly!
n_levels = 6
ny,nx,dx = dataset.ny,dataset.nx,dataset.dx
model = IceDynamics(n_levels=n_levels,ny=ny,nx=nx,dx=dx,
        x0=dataset.x[0].item(),y0=dataset.y[0].item(),
        crs=pyproj.CRS("EPSG:3031"))
mg = model.mg

grid = mg.levels[0]
dt = cp.float32(10.0)

### Initialize state
thk = gaussian_filter(dataset.thickness.values,1)
mg.state.H.set(thk)
mg.state.H_prev.set(thk)

### Initialize geometry
bed = gaussian_filter(dataset.bed.values,1)
mg.geometry.bed.set(bed)
mg.geometry.depth.set(np.maximum(-bed,0.0))
mg.geometry.sigmoid_c.set(0.1)
mg.geometry.sigmoid_k.set(3.0)

### Initialize rheology
B = cp.zeros((ny,nx), dtype=cp.float32)
B.fill(1e-17 ** (-1.0 / 3.0) / (917 * 9.81)) 
mg.rheology.B.set(B)
mg.rheology.eps_reg.set(1e-6)
mg.rheology.n.set(3.0)
mg.rheology.H_reg.set(25.0)

n_glen = 3.0

### Initialize sliding
beta = cp.zeros((ny,nx), dtype=cp.float32)
beta.fill(2.5)

mg.sliding.beta.set(beta)
mg.sliding.m.set(1./3.)
mg.sliding.u_reg.set(1.0)
mg.sliding.water_drag.set(1e-4)

### Initialize calving
mg.calving.calving_rate.set(0.0)

### Initialize forcing
smb = dataset.smb.values
smb[dataset.surface.values==0] = -20.0
mg.forcing.smb.set(smb)

### Load velocity data ###
u_obs = cp.array(dataset.vx.values,dtype=cp.float32)
u_obs[cp.isnan(u_obs)] = 0.0
v_obs = cp.array(dataset.vy.values,dtype=cp.float32)
v_obs[cp.isnan(v_obs)] = 0.0

# Build hierarchy of observations
observation_levels = [(u_obs,v_obs)]
for j in range(1,n_levels):
    u_obs_coarse = mg.restrict_cell(observation_levels[-1][0])
    v_obs_coarse = mg.restrict_cell(observation_levels[-1][1])
    observation_levels.append((u_obs_coarse,v_obs_coarse))

### Set multigrid solver parameters ###
model.forward_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10, 
        post_steps=150, finest_steps=0,
        relative_tolerance=1e-2, absolute_tolerance=10.0,
        report_norms=True)

model.forward_solver.vanka_options.omega.set(cp.float32(0.25))
model.forward_solver.vanka_options.newton_options.momentum_damping.set(cp.float32(0.1))
model.forward_solver.vanka_options.newton_options.step_tolerance.set(cp.float32(1e-6))

model.adjoint_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10,
        post_steps=150, finest_steps=0,
        relative_tolerance=1e-2, absolute_tolerance=1e-5, # Note that adjoint var
        report_norms=True)                               # adjoint var is small 
                                                          # in magnitude
model.adjoint_solver.vanka_options.omega.set(cp.float32(0.25))
model.adjoint_solver.vanka_options.newton_options.momentum_damping.set(cp.float32(0.01))
model.adjoint_solver.vanka_options.newton_options.step_tolerance.set(cp.float32(1e-6))

# Thin Pytorch wrapper of a single glide time step
glide_step = GlideStep.apply

t = cp.float32(0.0) # Dummy time, which we don't use here
n_level_epochs = 50

# Index of coarsest grid to start solving inverse problem at
coarsest_level = 2
log_beta = torch.log(torch.tensor(mg[coarsest_level].sliding.beta.data,device='cuda'))

# Solve the inverse problem at progressively coarser levels
for level in range(coarsest_level,-1,-1):
    # Examples of different writing utilities - First writes to vti/pvd

    # This is what we're optimizing - Convert from the initial guess, 
    # either defined above or by prolongation from the coarser state
    log_beta.requires_grad_()

    # These can be differentiated wrt - but we don't in this simple problem
    H_prev = torch.tensor(mg[level].state.H_prev.data,device='cuda')
    bed = torch.tensor(mg[level].geometry.bed.data,device='cuda')
    smb = torch.tensor(mg[level].forcing.smb.data,device='cuda')

    u_obs,v_obs = (torch.as_tensor(t) for t in observation_levels[level])
    u_mask = abs(u_obs) > 0.01
    v_mask = abs(v_obs) > 0.01

    # Standard torch optimization loop (RMSprop works very well here)
    optimizer = torch.optim.RMSprop([log_beta],lr=1e-2)

    # Derived surface velocity fields for output monitoring: with the MOLHO
    # ansatz the surface velocity is u_bar + u_d/(n+1)
    ny_l, nx_l = mg[level].ny, mg[level].nx
    u_s_field = Field(
            data=cp.zeros((ny_l,nx_l+1),dtype=cp.float32),
            grid_entity=GridEntity.VERTICAL_FACET,
            dx=mg[level].dx, grid=mg[level], name='u_s', units='m a^{-1}',
            attrs={'long_name':'Surface velocity (x)'})
    v_s_field = Field(
            data=cp.zeros((ny_l+1,nx_l),dtype=cp.float32),
            grid_entity=GridEntity.HORIZONTAL_FACET,
            dx=mg[level].dx, grid=mg[level], name='v_s', units='m a^{-1}',
            attrs={'long_name':'Surface velocity (y)'})

    # Initialize writer
    vti_writer = VTIWriter(f'inverse/level_{level}/vti', base='antarctica', dx=mg[level].dx,
            static_fields={'U_obs':[u_obs,v_obs]},
            dynamic_fields={'beta':mg[level].sliding.beta,
                            'U':[mg[level].state.u, mg[level].state.v],
                            'U_s':[u_s_field, v_s_field],
                            'xi':mg[level].state.xi}
        )

    vti_writer.initialize(mg[level])
    for j in range(n_level_epochs):
        optimizer.zero_grad()

        # Convert log(beta) to beta
        beta = torch.exp(log_beta)

        # Predict the velocity and thickness at t + dt
        mg.sliding.water_drag.set(1e-5)
        u,v,ud,vd,H,mask = glide_step(t,dt,model,level,H_prev,bed,beta,smb)

        # The observations are surface velocities: with the MOLHO ansatz
        # u_s = u_bar + u_d/(n+1)
        u_s = u + ud/(n_glen + 1.0)
        v_s = v + vd/(n_glen + 1.0)

        # Interpolation from facets (where the model predicts)
        # to cells (where the observations are)
        u_cell = 0.5*(u_s[:,1:] + u_s[:,:-1])
        v_cell = 0.5*(v_s[1:] + v_s[:-1])

        # L1 Objective function, masked by valid data
        J_data = (abs(u_cell - u_obs)*u_mask).mean() + (abs(v_cell - v_obs)*v_mask).mean()

        # Compute first differences
        dx = mg[level].dx
        gy = torch.diff(log_beta,dim=0)/dx
        gx = torch.diff(log_beta,dim=1)/dx
        
        # Tikhonov Regularization
        J_L2 = 1e-4*((gy**2).sum() + (gx**2).sum())*dx**2
        
        # TV Regularization
        gy_ = gy[:,:-1]
        gx_ = gx[:-1]
        eps = 1e-6
        J_L1 = 0*(torch.sqrt(gy_**2 + gx_**2 + eps**2).sum())*dx**2
        
        # Combined objective - elastic net regularization
        J = J_data + J_L1 + J_L2

        # Backpropagate
        mg.sliding.water_drag.set(1e-4)
        J.backward()

        # Update parameter
        optimizer.step()
        log_beta.data[log_beta.data > 3.5] = 3.5
        
        print(f"Level {level}, Iter. {j}/{n_level_epochs} | J: {J.item():.2f}, J_data: {J_data.item():.2f}, J_L1: {J_L1.item():.2f}, J_L2: {J_L2.item():.2f}")
        u_s_field.data[:,:] = mg[level].state.u.data + mg[level].state.ud.data/(n_glen + 1.0)
        v_s_field.data[:,:] = mg[level].state.v.data + mg[level].state.vd.data/(n_glen + 1.0)
        vti_writer.append(mg[level],time=j)
        vti_writer.write_pvd()

    if level>0:
        log_beta = torch.tensor(mg.prolongate_cell(cp.asarray(log_beta.detach()),method='bilinear'))

    beta_xr = mg[level].sliding.beta.to_dataarray()
    beta_xr.to_netcdf(f'./inverse/level_{level}/beta_opt.nc')

