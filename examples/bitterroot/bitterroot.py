"""
Mountain glacier forward simulation example, in
which we build a glacier system over the Bitterroot
Mountains in western Montana

Run interactively or execute as a script. Modify the paths and parameters
below to match your setup.
"""
import cupy as cp
import numpy as np
import pyproj

from glide.model import IceDynamics
from glide.field import Field,GridEntity
from glide.data import load_bitterroot_dem
from glide.io import VTIWriter

# =============================================================================
# Load data
# =============================================================================

print("Loading geometry...")
data = load_bitterroot_dem()
crs = pyproj.CRS(data.spatial_ref.crs_wkt)

bed = data.values.squeeze()[:-1,:-1]
x = data.x.values[:-1]
y = data.y.values[:-1]

n_levels = 5
factor = 2**n_levels
nx_target = (len(x) // factor) * factor
ny_target = (len(y) // factor) * factor

# Center the subregion
x_start = (len(x) - nx_target) // 2
y_start = (len(y) - ny_target) // 2

x_slice = slice(x_start, x_start + nx_target)
y_slice = slice(y_start, y_start + ny_target)

x = x[x_slice]
y = y[y_slice]
bed = bed[y_slice,x_slice]
srf = bed + 0.1
thk = srf - bed

ny,nx = srf.shape
dx = x[1]-x[0]

### Initialize grid
# ny and nx must both divide by 2^(n_levels - 1) cleanly!
model = IceDynamics(n_levels=5,ny=ny,nx=nx,dx=dx,
        x0=x[0],y0=y[0],
        crs=crs,stress_scheme='ssa')
mg = model.mg

### Initialize state
mg.state.H.set(thk)
mg.state.H_prev.set(thk)

### Initialize geometry
mg.geometry.bed.set(bed)

### Initialize rheology
# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B = cp.zeros((ny,nx), dtype=cp.float32)
B.fill(1e-16 ** (-1.0 / 3.0) / (917 * 9.81)) 
mg.rheology.B.set(B)
mg.rheology.eps_reg.set(1e-6)
mg.rheology.n.set(3.0)
mg.rheology.H_reg.set(25.0)

beta = cp.zeros((ny,nx), dtype=cp.float32)
beta.fill(5.0)

mg.sliding.beta.set(beta)
mg.sliding.m.set(1.0)

### Initialize forcing
ela = 1800
smb = 1.0/1000.0*(srf - ela)
mg.forcing.smb.set(smb)

### Set multigrid solver parameters ###
model.forward_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10, 
        post_steps=50, finest_steps=0,
        relative_tolerance=1e-2, absolute_tolerance=10.0,
        report_norms=True)

n_glen = float(mg[0].rheology.n.value)
u_s = Field(
        data=cp.zeros((ny,nx+1),dtype=cp.float32),
        grid_entity=GridEntity.VERTICAL_FACET,
        dx=mg[0].dx, grid=mg[0], name='u_s', units='m a^{-1}',
        attrs={'long_name':'Surface velocity (x)'})
v_s = Field(
        data=cp.zeros((ny+1,nx),dtype=cp.float32),
        grid_entity=GridEntity.HORIZONTAL_FACET,
        dx=mg[0].dx, grid=mg[0], name='v_s', units='m a^{-1}',
        attrs={'long_name':'Surface velocity (y)'})

u_b = Field(
        data=cp.zeros((ny,nx+1),dtype=cp.float32),
        grid_entity=GridEntity.VERTICAL_FACET,
        dx=mg[0].dx, grid=mg[0], name='u_b', units='m a^{-1}',
        attrs={'long_name':'Basal velocity (x)'})
v_b = Field(
        data=cp.zeros((ny+1,nx),dtype=cp.float32),
        grid_entity=GridEntity.HORIZONTAL_FACET,
        dx=mg[0].dx, grid=mg[0], name='v_b', units='m a^{-1}',
        attrs={'long_name':'Basal velocity (y)'})


def update_surface_velocity():
    u_s.data[:,:] = mg[0].state.u.data + mg[0].state.ud.data/(n_glen + 1.0)
    v_s.data[:,:] = mg[0].state.v.data + mg[0].state.vd.data/(n_glen + 1.0)

def update_basal_velocity():
    u_b.data[:,:] = mg[0].state.u.data - mg[0].state.ud.data
    v_b.data[:,:] = mg[0].state.v.data - mg[0].state.vd.data


# Examples of different writing utilities - First writes to vti/pvd
vti_writer = VTIWriter('forward/vti/', base='bitterroot', dx=mg[0].dx,
        static_fields={'bed':mg[0].geometry.bed,
                       'beta':mg[0].sliding.beta,},
        dynamic_fields={'H':mg[0].state.H,
                        'U_s':[u_s, v_s],
                        'U_b':[u_b, v_b],
                        'mask':mg[0].state.mask,}
        )
vti_writer.initialize(mg[0])

# Run simulation
t = cp.float32(0.0)
t_end = cp.float32(2000.0)
dt = cp.float32(25.0)
while t < t_end:
    print(f"Solving forward problem at t={t} with dt={dt:.2f}")
    model.forward(t,dt)
    t += dt

    # Write
    update_surface_velocity()
    update_basal_velocity()
    vti_writer.append(mg[0],time=t)
    vti_writer.write_pvd()
