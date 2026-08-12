"""SSA-mode (runtime-pinned deformational components) verification.

Three checks:
1. Pinning exactness: after a full FAS solve in stress_scheme='ssa', the
   deformational components ud/vd are identically zero on every level -
   no leakage through the smoother, transfers, or tau corrections.
2. Adjoint identity: <w, Jv> == <v, J^T w> with full-space random fields
   (including nonzero var_ud/lambda_ud) at fp32 roundoff, verifying the
   constraint machinery treats the pinned dofs as legitimate identity rows.
3. Physics: SSA-mode and MOLHO-mode solves of a stiff, sliding-dominated
   slab agree closely (MOLHO -> SSA as deformation -> 0), and the
   SSA-mode surface velocity equals its depth-averaged velocity exactly.
"""
import cupy as cp
import numpy as np

from glide.model import IceDynamics
from glide.grid import Grid

cp.random.seed(0)

L = 10000.0
dt = cp.float32(1.0)

base_res = 32
y_factr = 4
x_factr = 4

ny = base_res*y_factr
nx = base_res*x_factr

x = cp.linspace(0,x_factr*L,nx,dtype=cp.float32)
y = cp.linspace(0,y_factr*L,ny,dtype=cp.float32)
dx = (x[1] - x[0]).item()

X,Y = cp.meshgrid(x,y)

srf = 1000.0*cp.ones((ny,nx),dtype=cp.float32) - cp.tan(cp.deg2rad(0.1))*X + 10000
bed = srf - 1000
thk = srf - bed

rho_i = cp.float32(917.0)
g = cp.float32(9.81)
beta = (1000*cp.sin(2*cp.pi*X/L)*cp.sin(2*cp.pi*Y/L) + 1000)/(rho_i*g)

B = cp.ones((ny,nx),dtype=cp.float32)
B.fill((1e-16 ** -(1./3))/(rho_i*g))


def solve(stress_scheme):
    model = IceDynamics(n_levels=4,ny=ny,nx=nx,dx=dx,stress_scheme=stress_scheme)
    mg = model.mg
    mg.state.H.set(thk); mg.state.H_prev.set(thk)
    mg.geometry.bed.set(bed)
    mg.rheology.B.set(B)
    mg.rheology.n.set(3.0)
    mg.rheology.eps_reg.set(1e-6)
    mg.sliding.beta.set(beta)
    mg.sliding.m.set(1.0)
    mg.sliding.u_reg.set(1.0)
    model.forward_solver.fas_options.set(
            coarsest_steps=200, pre_steps=10, post_steps=20, finest_steps=50,
            relative_tolerance=1e-4, absolute_tolerance=1e-2,
            report_norms=False)
    model.forward(0.0,dt,update_geometry=False)
    return model


# --- 1. Pinning exactness through the full multigrid solve
model_ssa = solve('ssa')
for lvl,grid in enumerate(model_ssa.mg.levels):
    ud_max = float(cp.abs(grid.state.ud.data).max())
    vd_max = float(cp.abs(grid.state.vd.data).max())
    print('level %d: max|ud| = %g, max|vd| = %g' % (lvl,ud_max,vd_max))
    assert ud_max == 0.0 and vd_max == 0.0, 'SSA pinning leaked on level %d' % lvl
u_ssa = model_ssa.mg.levels[0].state.u.data.copy()
v_ssa = model_ssa.mg.levels[0].state.v.data.copy()
assert bool(cp.isfinite(u_ssa).all()), 'SSA-mode solve produced non-finite u'

# Surface velocity == depth-averaged velocity when ud == 0
n_glen = 3.0
u_s = u_ssa + model_ssa.mg.levels[0].state.ud.data/(n_glen + 1.0)
assert float(cp.abs(u_s - u_ssa).max()) == 0.0

# --- 2. Full-space adjoint identity in SSA mode
grid = model_ssa.mg.levels[0]
fo = grid.forward_operators
ao = grid.adjoint_operators

grid.state.mask.data[:,:] = cp.random.randint(0,2,size=(ny,nx)).astype(cp.float32)
fo.var_u[:,:] = cp.random.randn(ny,nx+1,dtype=cp.float32)
fo.var_v[:,:] = cp.random.randn(ny+1,nx,dtype=cp.float32)
fo.var_ud[:,:] = cp.random.randn(ny,nx+1,dtype=cp.float32)
fo.var_vd[:,:] = cp.random.randn(ny+1,nx,dtype=cp.float32)
fo.var_H[:,:] = cp.random.randn(ny,nx,dtype=cp.float32)
grid.adjoint.lambda_u.data[:,:] = cp.random.randn(ny,nx+1,dtype=cp.float32)
grid.adjoint.lambda_v.data[:,:] = cp.random.randn(ny+1,nx,dtype=cp.float32)
grid.adjoint.lambda_ud.data[:,:] = cp.random.randn(ny,nx+1,dtype=cp.float32)
grid.adjoint.lambda_vd.data[:,:] = cp.random.randn(ny+1,nx,dtype=cp.float32)
grid.adjoint.lambda_H.data[:,:] = cp.random.randn(ny,nx,dtype=cp.float32)

fo.compute_jvp(dt)
ao.compute_vjp(dt)

t1 = ((ao.vjp_u*fo.var_u).sum() + (ao.vjp_v*fo.var_v).sum() +
      (ao.vjp_ud*fo.var_ud).sum() + (ao.vjp_vd*fo.var_vd).sum() +
      (ao.vjp_H*fo.var_H).sum())
t2 = ((fo.jvp_u*grid.adjoint.lambda_u.data).sum() +
      (fo.jvp_v*grid.adjoint.lambda_v.data).sum() +
      (fo.jvp_ud*grid.adjoint.lambda_ud.data).sum() +
      (fo.jvp_vd*grid.adjoint.lambda_vd.data).sum() +
      (fo.jvp_H*grid.adjoint.lambda_H.data).sum())
rel = float(cp.abs(t1 - t2)/(0.5*cp.abs(t1 + t2)))
print('adjoint identity (ssa mode): <w,Jv> = %g, <v,J^Tw> = %g, rel = %g'
      % (float(t1),float(t2),rel))
assert rel < 1e-4, 'SSA-mode jvp/vjp are not transposes'
grid.state.mask.data.fill(0.0)

# --- 3. MOLHO converges to SSA in the sliding-dominated limit
model_molho = solve('molho')
u_molho = model_molho.mg.levels[0].state.u.data
ud_molho = model_molho.mg.levels[0].state.ud.data
speed = float(cp.abs(u_ssa).max())
def_frac = float(cp.abs(ud_molho).max())/speed
diff = float(cp.abs(u_molho - u_ssa).max())/speed
print('max|u_ssa| = %.2f, deformational fraction = %.3g, '
      'rel. max|u_molho - u_ssa| = %.3g' % (speed,def_frac,diff))
# the schemes may differ by at most the deformational contribution
assert diff < max(4.0*def_frac,1e-3), \
    'SSA-mode deviates from MOLHO by more than the deformational signal'

print('SSA mode: all checks passed')
