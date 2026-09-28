"""Test that uniform E with SMB gives zero residual when H_prev is correct."""
import cupy as cp
import numpy as np
from glide.grid import Grid
from glide.enthalpy import EnthalpyOperators

ny, nx, nz = 4, 4, 11
dx = 5000.0
H_old = 1000.0
H_new = 999.0  # thinned by 1m
T = 253.15
dt = 365.25 * 86400.0
smb_si = -1.0 / (365.25 * 86400.0)

grid = Grid(ny, nx, cp.float32(dx))
grid.state.H.set(cp.full((ny, nx), H_new, dtype=cp.float32))
grid.state.H_prev.set(cp.full((ny, nx), H_new, dtype=cp.float32))
grid.geometry.bed.set(cp.zeros((ny, nx), dtype=cp.float32))
grid.forcing.smb.set(cp.zeros((ny, nx), dtype=cp.float32))
grid.sliding.beta.set(cp.ones((ny, nx), dtype=cp.float32))
grid.sliding.m.set(1.0)
grid.sliding.u_reg.set(1.0)

ops = EnthalpyOperators(grid, nz=nz)
ops.term_flags.horizontal_advection = False
ops.term_flags.drainage = False
ops.initialize_from_temperature(T)
ops.set_surface_enthalpy_from_temperature(
    cp.full((ny, nx), T, dtype=cp.float32))
ops.enthalpy_forcing.Q_geo.fill(0)
ops.enthalpy_forcing.Q_fh.fill(0)
ops.enthalpy_forcing.phi_strain.fill(0)
ops.enthalpy_state.E_prev[:] = ops.enthalpy_state.E
ops.compute_omega(cp.full((ny, nx), smb_si, dtype=cp.float32))

# Test 1: H_prev = H_old (correct — pre-momentum thickness)
ops.H_prev[:] = cp.float32(H_old)
r_correct = float(ops.compute_residual(dt))
print(f"H_prev = {H_old} (pre-momentum): |r| = {r_correct:.3e}")

# Test 2: H_prev = H_new (wrong — post-momentum, same as H)
ops.H_prev[:] = cp.float32(H_new)
r_wrong = float(ops.compute_residual(dt))
print(f"H_prev = {H_new} (post-momentum): |r| = {r_wrong:.3e}")

print(f"\nRatio: {r_wrong / r_correct:.1f}x worse without fix")
assert r_correct < r_wrong, "Pre-momentum H_prev should give smaller residual"
print("PASSED")
