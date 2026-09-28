"""
GlideStep must checkpoint the rate factor B.

A thermal model (ThermalModel) changes rheology.B between steps, so the
backward pass of a step has to re-solve with the B that step was integrated
with, not with whatever B is live when backward runs. Test: the gradient of a
misfit on step 1's velocity with respect to beta is computed (a) from step 1
alone, and (b) from step 1 followed by a step 2 run with a 3x softer B that is
left on the model. (b) must reproduce (a); without the checkpoint the step-1
adjoint would be re-solved with step 2's B.
"""
import cupy as cp
import torch

from glide.model import IceDynamics
from glide.torch import GlideStep

cp.random.seed(0)

L = 20000.0
ny = nx = 128
x = cp.linspace(0, 4 * L, nx, dtype=cp.float32)
y = cp.linspace(0, 4 * L, ny, dtype=cp.float32)
dx = (x[1] - x[0]).item()
X, Y = cp.meshgrid(x, y)

rho_i, g = 917.0, 9.81
srf = 11000.0 - cp.tan(cp.deg2rad(0.1)) * X
bed = (srf - 1000.0).astype(cp.float32)
thk = (srf - bed).astype(cp.float32)
beta = ((1000 * cp.sin(2 * cp.pi * X / L) * cp.sin(2 * cp.pi * Y / L) + 1500) / (rho_i * g)).astype(cp.float32)
B1 = cp.full((ny, nx), (1e-16 ** (-1.0 / 3)) / (rho_i * g), dtype=cp.float32)
B2 = (B1 * 3.0 ** (-1.0 / 3)).astype(cp.float32)          # A x 3

model = IceDynamics(n_levels=4, ny=ny, nx=nx, dx=cp.float32(dx))
mg = model.mg
mg.geometry.bed.set(bed)
mg.geometry.depth.set(-bed)
mg.sliding.m.set(1.0)
mg.sliding.u_reg.set(1.0)
mg.forcing.smb.set(cp.zeros((ny, nx), dtype=cp.float32))
for s in (model.forward_solver, model.adjoint_solver):
    s.fas_options.maximum_vcycles.set(20)
    s.fas_options.relative_tolerance.set(cp.float32(1e-6))
    s.fas_options.absolute_tolerance.set(cp.float32(1e-3))

w = cp.random.randn(ny, nx + 1).astype(cp.float32)


def grad(two_steps):
    mg.rheology.B.set(B1)
    mg.state.u.set(cp.zeros((ny, nx + 1), dtype=cp.float32))
    mg.state.v.set(cp.zeros((ny + 1, nx), dtype=cp.float32))
    H0 = torch.tensor(thk, requires_grad=False)
    bed_t = torch.tensor(bed)
    beta_t = torch.tensor(beta, requires_grad=True)
    smb_t = torch.zeros_like(H0)
    u1, v1, ud1, vd1, H1, m1 = GlideStep.apply(0.0, 1.0, model, 0, H0, bed_t, beta_t, smb_t)
    J = (u1 * torch.as_tensor(w)).sum()
    if two_steps:
        mg.rheology.B.set(B2)                        # what a thermal step would do
        u2, *_ = GlideStep.apply(1.0, 1.0, model, 0, H1.detach(), bed_t, beta_t.detach(), smb_t)
        J = J + 0.0 * u2.sum()                       # step 2 on the graph, no gradient to beta
    J.backward()
    return beta_t.grad.clone()


g_a = grad(False)
g_b = grad(True)
rel = float((g_b - g_a).norm() / g_a.norm())
print(f"|grad(step1 + softer step2) - grad(step1)| / |grad(step1)| = {rel:.3e}")
assert rel < 1e-3, "GlideStep did not restore the B its forward used"
print("PASSED")
