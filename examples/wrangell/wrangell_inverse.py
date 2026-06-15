"""
Mountain glacier forward simulation example, in
which we build a glacier system over the Bitterroot
Mountains in western Montana

Run interactively or execute as a script. Modify the paths and parameters
below to match your setup.
"""
from glide.backend import xp as cp
import torch
import numpy as np
import pyproj

from torch.nn.functional import avg_pool2d, interpolate
from torch.utils.checkpoint import checkpoint

from glide.model import IceDynamics
from glide.io import VTIWriter
from glide.torch import GlideStep
from glide.field import Field,GridEntity

from glare.model import ImprovedTemperatureIndex
from glare.torch import GlareStep

from glide.data import load_wrangell_preprocessed

import xarray as xr

# =============================================================================
# Load data
# =============================================================================

OUTPUT_PATH = './inverse/'

print("Loading geometry...")

N_LEVELS = 6

gridded_data, temperature_anomaly, flightlines = load_wrangell_preprocessed()
crs = pyproj.CRS(gridded_data.spatial_ref.crs_wkt)

factor = 2**N_LEVELS
ny_0,nx_0 = gridded_data.sizes['y'],gridded_data.sizes['x']

ny_target = (ny_0 // factor) * factor
nx_target = (nx_0 // factor) * factor

# Center the subregion
y_start = (ny_0 - ny_target) // 2
x_start = (nx_0 - nx_target) // 2

gridded_data = gridded_data.isel(
        y=slice(y_start,y_start + ny_target),
        x=slice(x_start,x_start + nx_target)
        )

ny,nx = gridded_data.sizes['y'],gridded_data.sizes['x']
dx = (gridded_data.x[1]-gridded_data.x[0]).item()
x0 = gridded_data.x[0].item()
y0 = gridded_data.y[0].item()

### Initialize grid
# ny and nx must both divide by 2^(n_levels - 1) cleanly!
model = IceDynamics(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx,
        x0=x0,y0=y0,
        crs=crs)
mg = model.mg

### Initialize state
mg.state.H.set(0.1)
mg.state.H_prev.set(0.1)

### Initialize geometry
mg.geometry.bed.set(gridded_data.elevation)
mg.geometry.sigmoid_c.set(1.0)
mg.geometry.sigmoid_k.set(5.0)

### Initialize rheology
# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B = 1e-16 ** (-1.0 / 3.0) / (917 * 9.81)
mg.rheology.B.set(B)
mg.rheology.eps_reg.set(1e-5)
mg.rheology.n.set(3.0)

mg.sliding.beta.set(2.0)
mg.sliding.m.set(1./3.)

smb_model = ImprovedTemperatureIndex(ny=ny,nx=nx,nt=12,
        dx=dx,dt=1./12,
        x0=x0,y0=y0,
        crs=crs)

smb_model.grid.insolation.insol_mean.set(gridded_data.monthly_solar_potential_mean)
smb_model.grid.insolation.insol_cos.set(gridded_data.monthly_solar_potential_cos)
smb_model.grid.insolation.insol_sin.set(gridded_data.monthly_solar_potential_sin)
smb_model.grid.temperature.t2m.set(gridded_data.monthly_t2m)
smb_model.grid.precipitation.precip.set(gridded_data.monthly_precip)
smb_model.forward()

### Initialize forcing
mg.forcing.smb.set(smb_model.grid.state.smb.data.mean(axis=0))

### Set multigrid solver parameters ###
model.forward_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10, 
        post_steps=50, finest_steps=50,
        relative_tolerance=1e-2, absolute_tolerance=10.0,
        report_norms=False)

model.adjoint_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10,
        post_steps=50, finest_steps=50,
        relative_tolerance=1e-2, absolute_tolerance=1e-6, # Note that adjoint var
        report_norms=False)                               # adjoint var is small 
                                                          # in magnitude

model.adjoint_solver.vanka_options.newton_options.ssa_damping.set(cp.float32(1.0))

# Thin Pytorch wrapper of a single glide time step
glide_step = GlideStep.apply
glare_step = GlareStep.apply

log_beta = torch.tensor(cp.log(mg[0].sliding.beta.data),
        device='cuda',requires_grad=True)
H_prev = torch.tensor(mg[0].state.H_prev.data,
        device='cuda',requires_grad=False)
bed = torch.tensor(mg[0].geometry.bed.data,
        device='cuda',requires_grad=True)
t2m = torch.tensor(smb_model.grid.temperature.t2m.data,
        device='cuda',requires_grad=False)
precip = torch.tensor(smb_model.grid.precipitation.precip.data,
        device='cuda',requires_grad=False)

def V(x,l,root_a,eig_tol=1e-3):
    K =  root_a * cp.exp(-((x - x[:,None])/l)**2)
    l,v = cp.linalg.eigh(K)
    mask = l > 1e-3*l.max()
    return v[:,mask]*cp.sqrt(l[mask])

Vx = 2*torch.as_tensor(V(mg[0].x_cell,20000,1.0)).to(torch.float32)
Vy = 2*torch.as_tensor(V(mg[0].y_cell,20000,1.0)).to(torch.float32)

precipitation_bias = torch.zeros((Vy.shape[1],Vx.shape[1]),
        device='cuda',dtype=torch.float32,
        requires_grad=True)

log_mf = torch.tensor(cp.log(smb_model.grid.temperature.mf.value),
        device='cuda',requires_grad=True)
log_rf = torch.tensor(cp.log(smb_model.grid.insolation.rf.value),
        device='cuda',requires_grad=True)
alpha_t2m = torch.tensor(2.2,
        device='cuda',requires_grad=False)

WARM_START_PATH = None
if WARM_START_PATH is not None:
    d = torch.load(WARM_START_PATH)
    log_beta = d['log_beta']
    bed = d['bed']
    precipitation_bias = d['precipitation_bias']
    log_mf = d['log_mf']
    log_rf = d['log_rf']


rgi_mask = torch.tensor(gridded_data.rgi_mask.values,device='cuda')
domain_mask = torch.tensor(gridded_data.domain_mask.values,device='cuda')
base_anomaly = temperature_anomaly.sel(time=2012).temp_anomaly.item()

u_obs = torch.tensor(gridded_data.vx.values,
        dtype=torch.float32,device='cuda').nan_to_num().masked_fill(~domain_mask,0.0)
v_obs = torch.tensor(gridded_data.vy.values,
        dtype=torch.float32,device='cuda').nan_to_num().masked_fill(~domain_mask,0.0)
S_obs = torch.tensor(gridded_data.elevation.values,
        dtype=torch.float32,device='cuda')

def compute_smb(smb_model,t2m,t_anomaly,base_anomaly,precip_,mf,rf,domain_mask):
    smb = glare_step(smb_model,t2m + (t_anomaly - base_anomaly),precip_,mf,rf).mean(axis=0)
    smb_ = smb.masked_fill(~domain_mask,-10)
    return smb_

def differentiable_restriction(field,n_times,method='avg'):
    if method=='avg':
        fn = avg_pool2d
    if method=='max':
        fn = max_pool2d
    for _ in range(n_times):
        field = fn(field[None,:,:],(2,2))[0]
    return field

def differentiable_prolongation(field,n_times,grid_entity='cell',method='bilinear'):
    for _ in range(n_times):
        if grid_entity=='cell':
            ny_fine,nx_fine = 2*field.shape[0],2*field.shape[1]
        elif grid_entity=='vfacet':
            ny_fine,nx_fine = 2*field.shape[0],2*(field.shape[1] - 1) + 1
        elif grid_entity=='hfacet':
            ny_fine,nx_fine = 2*(field.shape[0] - 1) + 1,2*field.shape[1]
        field = interpolate(field[None,None,:,:],(ny_fine,nx_fine),mode='bilinear').squeeze()
    return field

MAX_LEVEL = 2
MIN_LEVEL = 0
DT = 20.0
for level in range(MAX_LEVEL,MIN_LEVEL-1,-1):
    model.set_top_level(level)

    
    delta = Field(cp.zeros((mg[level].ny,mg[level].nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=mg[level].dx,
            grid=mg[level])

    srf = Field(cp.zeros((mg[level].ny,mg[level].nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=mg[level].dx,
            grid=mg[level])

    
    mask = torch.tensor(rgi_mask*domain_mask,
            dtype=torch.float32,device='cuda')

    optimizer = torch.optim.Adam([{'params':log_beta,'lr':0.05},
                                     {'params':bed,'lr':10.0},
                                     {'params':precipitation_bias,'lr':0.005},
                                     {'params':log_mf,'lr':0.01},
                                     {'params':log_rf,'lr':0.01}
                                     ],lr=1e-3,betas=(0.5,0.99))
    

    vti_writer = VTIWriter(f'{OUTPUT_PATH}/level_{level}/vti', base='wrangell', dx=mg[level].dx,
            dynamic_fields={'bed':mg[level].geometry.bed,
                            'beta':mg[level].sliding.beta,
                            'thk':mg[level].state.H,
                            'U':[mg[level].state.u,mg[level].state.v],
                            'srf':srf,
                            'delta':delta,
                            'smb':mg[level].forcing.smb}
        )

    vti_writer.initialize(mg[level])
    
    def evaluate_loss(i,compute_gradient=True,write_vti=True):
        optimizer.zero_grad()
        
        mg.state.u.set(0.0,start_level=level)
        mg.state.v.set(0.0,start_level=level)
        mg.state.H.set(0.1,start_level=level)
        mg.state.H_prev.set(0.1,start_level=level)
        mg.state.mask.set(0.0,start_level=level)

        mg.adjoint.lambda_u.set(0.0,start_level=level)
        mg.adjoint.lambda_v.set(0.0,start_level=level)
        mg.adjoint.lambda_H.set(0.0,start_level=level)

        precip_ = precip + (Vy @ precipitation_bias @ Vx.T)[None,:,:]
        mf = torch.exp(log_mf)
        rf = torch.exp(log_rf)

        bed_ = differentiable_restriction(bed,level)
        H_prev_ = differentiable_restriction(H_prev,level)
        log_beta_ = differentiable_restriction(log_beta,level)
        S_obs_ = differentiable_restriction(S_obs,level)

        beta_ = torch.exp(log_beta_)

        t = cp.float32(2012-1000)
        t_end = cp.float32(2012)
        dt = cp.float32(DT)

        
        if i % 10 == 0:
            time_writer = VTIWriter(f'{OUTPUT_PATH}/level_{level}/vti', base='time', dx=mg[level].dx,
                dynamic_fields={'thk':mg[level].state.H,
                                'U':[mg[level].state.u,mg[level].state.v],
                               'smb':mg[level].forcing.smb}
            )
        

        while t < t_end:
            t_anomaly_0 = alpha_t2m*temperature_anomaly.sel(time=int(t)).temp_anomaly.item()
            t_anomaly_1 = alpha_t2m*temperature_anomaly.sel(time=int(t+dt)).temp_anomaly.item()
            t_anomaly = 0.5*(t_anomaly_0 + t_anomaly_1)
            
            smb = checkpoint(compute_smb,smb_model,
                    t2m,t_anomaly,base_anomaly,precip_,
                    mf,rf,domain_mask,use_reentrant=False)

            smb_ = differentiable_restriction(smb,level)

            u,v,H,_ = glide_step(t,dt,model,level,H_prev_,bed_,beta_,smb_)

            t += dt
            H_prev_ = H
            
            if i % 20 == 0:
                time_writer.append(mg[level],time=t)
                time_writer.write_pvd()

        # Surface elevation loss
        S = bed_ + H
        S_ = S

        u = differentiable_prolongation(u,level,grid_entity='vfacet')
        v = differentiable_prolongation(v,level,grid_entity='hfacet')
        H = differentiable_prolongation(H,level,grid_entity='cell')
        S = differentiable_prolongation(S,level,grid_entity='cell')

        J_srf = 2.0*(torch.sqrt((S - S_obs)**2 + 100.0).mean() - 10)
        
        u_pred = (u[:,1:] + u[:,:-1])/2.
        v_pred = (v[1:,:] + v[:-1,:])/2.        
        J_vel = 3.0*(torch.sqrt((u_pred - u_obs)**2 + (v_pred - v_obs)**2 + 100.0).mean() - 10.0)

        # Extent loss
        p_extent = (2./(1+torch.exp(-H/100.0))-1).clip(min=0.001,max=0.999)
        J_extent = -100.0*(mask*torch.log(p_extent)).mean()
         
        # Tikhonov Regularization - bed
        gy = torch.diff(bed,dim=0)/mg[0].dx
        gx = torch.diff(bed,dim=1)/mg[0].dx
        J_tikh_bed = 1e-8*((gy**2).sum() + (gx**2).sum())*mg[0].dx**2

        # Tikhonov Regularization - log_beta
        gy = torch.diff(log_beta,dim=0)/mg[0].dx
        gx = torch.diff(log_beta,dim=1)/mg[0].dx
        J_tikh_beta = 1e-5*((gy**2).sum() + (gx**2).sum())*mg[0].dx**2
                                          
        # L2 Regularization - log_beta
        J_L2_beta = 10*((log_beta - .69)**2).mean()

        # L2 Regularization - smb offset
        J_smb = (precipitation_bias**2).sum() + (log_rf - 2.9)**2 + (log_mf - .6016)**2

        J = J_srf + J_vel + J_extent + J_tikh_bed + J_tikh_beta + J_L2_beta + J_smb

        
        if write_vti:
            delta.data[:,:] = cp.asarray(S_.detach() - S_obs_)
            srf.data[:,:] = cp.asarray(S_.detach())
            vti_writer.append(mg[level],time=i)
            vti_writer.write_pvd()
        

        print(f"{i}, {J.item():.2f}, {J_srf.item():.2f}, {J_vel.item():.2f}, {J_extent.item():.2f}, {J_tikh_bed.item():.2f}, {J_tikh_beta.item():.2f}, {J_L2_beta:.2f}")
        if compute_gradient:
            J.backward()
            bed.grad[~(rgi_mask)] = 0.0
        return J


    max_iters = 250
    for i in range(0,max_iters):
        evaluate_loss(i)
        optimizer.step()
    evaluate_loss(0,write_vti=False,compute_gradient=False)

    u_xr = mg[level].state.u.to_dataarray()
    v_xr = mg[level].state.v.to_dataarray()
    H_xr = mg[level].state.H.to_dataarray()
    bed_xr = mg[level].geometry.bed.to_dataarray()
    beta_xr = mg[level].sliding.beta.to_dataarray()
    smb_xr = mg[level].forcing.smb.to_dataarray()
    ds = xr.merge([u_xr,v_xr,H_xr,bed_xr,beta_xr,smb_xr])
    ds.to_netcdf(f'{OUTPUT_PATH}/level_{level}/inverse_soln.nc')
    torch.save({'log_beta':log_beta,'bed':bed,'precipitation_bias':precipitation_bias,'log_rf':log_rf,'log_mf':log_mf},f'{OUTPUT_PATH}/level_{level}/torch_vars.p')


