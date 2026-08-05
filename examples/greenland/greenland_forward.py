"""
Greenland forward simulation example.

Run interactively or execute as a script. Modify the paths and parameters
below to match your setup.
"""
import cupy as cp
import numpy as np
import pyproj

from scipy.ndimage import gaussian_filter

from glide.model import IceDynamics
from glide.data import load_greenland_preprocessed
from glide.io import ZarrWriter, VTIWriter

### Load a dataset (here a preprocessed greenland dataset)
dataset = load_greenland_preprocessed()

### Initialize grid
# ny and nx must both divide by 2^(n_levels - 1) cleanly!
ny,nx,dx = dataset.ny,dataset.nx,dataset.dx
model = IceDynamics(n_levels=6,ny=ny,nx=nx,dx=dx,
        x0=dataset.x[0].item(),y0=dataset.y[0].item(),
        crs=pyproj.CRS("EPSG:3413"))
mg = model.mg

### Initialize state
thk = gaussian_filter(dataset.thickness.values,1)
mg.state.H.set(thk)
mg.state.H_prev.set(thk)

### Initialize geometry
bed = gaussian_filter(dataset.bed.values,1)
mg.geometry.bed.set(bed)
mg.geometry.depth.set(np.maximum(-bed,0))
mg.geometry.sigmoid_c.set(0.1)
mg.geometry.sigmoid_k.set(3.0)

### Initialize rheology
# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B = cp.zeros((ny,nx), dtype=cp.float32)
B.fill(1e-17 ** (-1.0 / 3.0) / (917 * 9.81)) 
mg.rheology.B.set(B)
mg.rheology.eps_reg.set(1e-6)
mg.rheology.n.set(3.0)

### Initialize sliding
BETA_PATH = None
#BETA_PATH = "./inverse/level_0/beta_opt.nc"
if BETA_PATH:
    import xarray as xr
    beta = cp.array(xr.load_dataarray(BETA_PATH))
else:
    beta = cp.zeros((ny,nx), dtype=cp.float32)
    beta.fill(2.5)

beta[beta>50] = 50

mg.sliding.beta.set(beta)
mg.sliding.m.set(1./3.)
mg.sliding.water_drag.set(1e-4)

### Initialize calving
# Specifies calving velocity for a non-conservative
# calving flux over facets between adjacent floating cells
mg.calving.calving_rate.set(2000.0) 

### Initialize forcing
smb = dataset.smb.values
smb -= 1.0
mg.forcing.smb.set(smb)

### Set multigrid solver parameters ###
model.forward_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10, 
        post_steps=150, finest_steps=0,
        relative_tolerance=1e-2, absolute_tolerance=10.0,
        report_norms=True)

#model.forward_solver.vanka_options.relax_phi.set(cp.float32(0.5))
#model.forward_solver.vanka_options.newton_options.ssa_damping.set(cp.float32(0.1))


# Examples of different writing utilities - First writes to vti/pvd
vti_writer = VTIWriter('forward/vti/', base='greenland', dx=mg[0].dx,
        static_fields={'bed':mg[0].geometry.bed,
                       'beta':mg[0].sliding.beta,},
        dynamic_fields={'H':mg[0].state.H,
                        'U':[mg[0].state.u, mg[0].state.v],
                        'mask':mg[0].state.mask,
                        'xi':mg[0].state.xi}
        )
vti_writer.initialize(mg[0])

# Second writes to zarr archive, which can be converted to netcdf via xarray
zarr_writer = ZarrWriter('forward/example_run.zarr',
        static_fields={'bed':mg[0].geometry.bed,
                       'beta':mg[0].sliding.beta,},
        dynamic_fields={'H':mg[0].state.H,
                        'u':mg[0].state.u,
                        'v':mg[0].state.v,
                        'mask':mg[0].state.mask,}
        )
           
zarr_writer.initialize(mg[0],overwrite=True)

# Run simulation
t = cp.float32(0.0)
t_end = cp.float32(1000.0)
dt = cp.float32(10.0)
while t < t_end:
    print(f"Solving forward problem at t={t} with dt={dt:.2f}")
    model.forward(t,dt)
    t += dt

    # Write
    vti_writer.append(mg[0],time=t)
    vti_writer.write_pvd()
    zarr_writer.append(mg[0],time=t)

# Finalize zarr for fast xarray reading
zarr_writer.consolidate_metadata()

# If you want a netcdf of the simulation, uncomment:
#import xarray as xr
#sim_ds = xr.load_dataset('forward/example_run.zarr')
#sim_ds.to_netcdf('forward/example_run.nc')
