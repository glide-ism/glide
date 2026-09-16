"""
Finite-difference check of the bed gradient on a MARINE-BASED geometry.

The bed sits below sea level under grounded ice, so the flotation fraction
xi = 1 - depth/(r H) is strictly between 0 and 1 over the whole domain and
the basal drag beta * xi^p depends on the bed (and on H) through the
effective pressure. That pathway is invisible to grad_test.py (terrestrial
geometry, xi == 1) and was missing from compute_gradient_bed until the
xi terms were added to the drag Jacobians (stress.cu).

Same structure as grad_test.py but with a quadratic velocity misfit (an
L1 misfit adds a few percent of kink error to the finite differences):
adjoint gradient vs central differences along a smooth, long-wavelength
random bed perturbation. Reference numbers (2026-09-13): with the xi terms
the adjoint is within 1.1% of the finite difference at eps = 2 and 4 m;
without them (drag Jacobians blind to the bed) it recovers only 46% of it.
"""
import cupy as cp
import numpy as np

from glide.multigrid import Multigrid, FASCDSolver, FASAdjointSolver

cp.random.seed(0)

L = 20000.0
dt = cp.float32(1.0)

base_res = 64
y_factr = 7
x_factr = 7
ny = base_res * y_factr
nx = base_res * x_factr

x = cp.linspace(0, x_factr * L, nx, dtype=cp.float32)
y = cp.linspace(0, y_factr * L, ny, dtype=cp.float32)
dx = (x[1] - x[0]).item()
X, Y = cp.meshgrid(x, y)

# Grounded marine ice: 1000 m thick on a bed 200-600 m below sea level
# (flotation thickness 218-654 m), gently sloping surface.
thk = 1000.0 * cp.ones((ny, nx), dtype=cp.float32)
bed = -400.0 - 200.0 * cp.sin(2 * cp.pi * X / (3 * L)) * cp.cos(2 * cp.pi * Y / (3 * L)) \
      - cp.tan(cp.deg2rad(0.1)) * X
bed = bed.astype(cp.float32)

rho_i = cp.float32(917.0)
g = cp.float32(9.81)
beta = (1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1000) / (rho_i * g)

B = cp.ones((ny, nx), dtype=cp.float32)
B.fill((1e-16 ** -(1. / 3)) / (rho_i * g))

mg = Multigrid(6, ny=ny, nx=nx, dx=dx)
grid = mg.levels[0]

mg.geometry.bed.set(bed)
mg.geometry.depth.set(-bed)          # signed head deficit: positive under water
mg.geometry.sigmoid_c.set(1.0)
mg.rheology.B.set(B)
mg.sliding.beta.set(beta)
mg.sliding.m.set(1.0)
mg.sliding.u_reg.set(1.0)
mg.sliding.p.set(1.0)
mg.state.H.set(thk)
mg.state.H_prev.set(thk)

solver = FASCDSolver(mg)
solver.vanka_options.newton_options.relaxation.set(0.5)
solver.vanka_options.newton_options.steps.set(30)
solver.fas_options.coarsest_steps.set(200)
solver.fas_options.pre_steps.set(10)
solver.fas_options.post_steps.set(50)
solver.fas_options.finest_steps.set(150)
solver.fas_options.maximum_vcycles.set(10)
solver.fas_options.relative_tolerance.set(cp.float32(1e-5))
solver.fas_options.absolute_tolerance.set(cp.float32(0.01))

solver.solve(dt)
u_obs = cp.array(grid.state.u.data)
v_obs = cp.array(grid.state.v.data)

xi = grid.state.xi.data
print(f"flotation fraction over the domain: min {float(xi.min()):.2f}, max {float(xi.max()):.2f} "
      f"(must be strictly inside (0, 1) for this test to exercise the xi terms)")
assert 0.0 < float(xi.min()) and float(xi.max()) < 1.0

mg.sliding.beta.set(cp.ones_like(beta) * beta.mean())
solver.solve(dt)

u = cp.array(grid.state.u.data)
v = cp.array(grid.state.v.data)
H = cp.array(grid.state.H.data)

dJdu = (u - u_obs)
dJdv = (v - v_obs)
grid.adjoint_operators.f_u[:, :] = -dJdu
grid.adjoint_operators.f_v[:, :] = -dJdv

adjoint_solver = FASAdjointSolver(mg)
adjoint_solver.fas_options.coarsest_steps.set(200)
adjoint_solver.fas_options.pre_steps.set(10)
adjoint_solver.fas_options.post_steps.set(50)
adjoint_solver.fas_options.finest_steps.set(150)
adjoint_solver.fas_options.maximum_vcycles.set(10)
adjoint_solver.fas_options.absolute_tolerance.set(cp.float32(0.1))
adjoint_solver.fas_options.relative_tolerance.set(cp.float32(1e-3))
adjoint_solver.solve(dt)

grid.adjoint_operators.compute_gradient_bed()

# Smooth random bed perturbation of a few LONG-wavelength modes: the
# driving-stress response to a bed change is a slope (difference) operator
# that fades with wavelength, while the effective-pressure response is
# local and one-signed, so long modes weight the xi pathway (roughly a
# third of the gradient-vector product here; short modes hide it).
pert = cp.zeros((ny, nx), dtype=cp.float32)
rng = np.random.default_rng(0)
for _ in range(6):
    kx, ky = rng.uniform(0.3, 1.0, 2)
    px, py = rng.uniform(0, 2 * np.pi, 2)
    pert += cp.float32(rng.normal()) * cp.sin(2 * cp.pi * kx * X / (x_factr * L) + px) \
                                     * cp.sin(2 * cp.pi * ky * Y / (y_factr * L) + py)
pert /= float(pert.std())

eps = cp.float32(2.0)      # metres
bed_0 = cp.array(grid.geometry.bed.data)

solver.fas_options.maximum_vcycles.set(40)
solver.fas_options.relative_tolerance.set(cp.float32(1e-8))
solver.fas_options.absolute_tolerance.set(cp.float32(1e-4))
solver.vanka_options.newton_options.step_tolerance.set(cp.float32(1e-8))


def misfit():
    du = (grid.state.u.data - u_obs).astype(cp.float64)
    dv = (grid.state.v.data - v_obs).astype(cp.float64)
    return float(0.5 * ((du * du).sum() + (dv * dv).sum()))


def solve_at(bed_new):
    mg.geometry.bed.set(bed_new)
    mg.geometry.depth.set(-bed_new)
    grid.state.u.data[:, :] = u
    grid.state.v.data[:, :] = v
    grid.state.H.data[:, :] = H
    solver.solve(dt)
    return misfit()


J_1 = solve_at(bed_0 + eps * pert)
J_0 = solve_at(bed_0 - eps * pert)

gvp_fd = (J_1 - J_0) / (2 * float(eps))
gvp_ad = float((grid.geometry.bed.grad * pert).sum())

rel_err = abs(gvp_fd - gvp_ad) / abs(gvp_ad)
print(f"FD: {gvp_fd}, Adj: {gvp_ad}, Rel. Err.: {rel_err}")
assert rel_err < 2e-2
