"""
Tests for enthalpy advection-diffusion with analytical steady-state solutions.

Two 1D column problems on a static cold ice slab (no horizontal velocity):

1. Constant sigma_dot  — exponential profile
2. Linear sigma_dot    — Robin / error-function profile

Both keep the base cold (T_bed < T_pmp) so K = K_COLD throughout.
Parameters are consistent with examples/thermal/cold_column.py.
"""
import cupy as cp
import numpy as np
from scipy.special import erf
from glide.grid import Grid
from glide.enthalpy import (
    EnthalpyOperators, C_I, K_I, T_REF, T_MELT, RHO_I, GRAVITY, BETA_CC,
    K_COLD, E_SCALE
)

# ---- Shared parameters (match examples/thermal/cold_column.py) ----
H_ICE = 1000.0
T_SURFACE = 243.15      # K (-30 C)
Q_GEO = 0.04            # W/m^2
SEC_PER_KYR = 1000.0 * 365.25 * 86400.0


def _make_ops(nz=41):
    """Static cold slab with Q_geo at the bed, zero velocity."""
    ny, nx = 4, 4
    grid = Grid(ny, nx, cp.float32(1000.0))
    grid.state.H.set(cp.full((ny, nx), H_ICE, dtype=cp.float32))
    grid.state.H_prev.set(cp.full((ny, nx), H_ICE, dtype=cp.float32))
    grid.geometry.bed.set(cp.zeros((ny, nx), dtype=cp.float32))
    grid.forcing.smb.set(cp.zeros((ny, nx), dtype=cp.float32))
    grid.sliding.beta.set(cp.ones((ny, nx), dtype=cp.float32))
    grid.sliding.m.set(1.0)
    grid.sliding.u_reg.set(1.0)

    ops = EnthalpyOperators(grid, nz=nz)
    ops.term_flags.horizontal_advection = False  # pure column test
    ops.initialize_from_temperature(T_SURFACE)
    ops.set_surface_enthalpy_from_temperature(
        cp.full((ny, nx), T_SURFACE, dtype=cp.float32))
    ops.enthalpy_forcing.Q_geo[:, :] = Q_GEO
    ops.enthalpy_forcing.phi_strain.fill(0)
    ops.enthalpy_forcing.drain_rate.set(0.0)
    ops.enthalpy_velocity.u3d.fill(0)
    ops.enthalpy_velocity.v3d.fill(0)
    ops.enthalpy_velocity.omega.fill(0)
    ops.enthalpy_forcing.Q_fh.fill(0)
    return ops


def _run_to_steady_state(ops, max_steps=5000, conv_tol=1e-6):
    """Adaptive-dt convergence loop (same pattern as Stefan test)."""
    i_col, j_col = 2, 2
    dt = 100.0 * SEC_PER_KYR
    dt_max = 1e6 * SEC_PER_KYR

    for step in range(1, max_steps + 1):
        E_old = ops.enthalpy_state.E[i_col, j_col, :].copy()
        ops.set_rhs()
        ops.column_sweep(dt, 20)
        E_new = ops.enthalpy_state.E[i_col, j_col, :]
        rel = float(cp.max(cp.abs(E_new - E_old))) / (float(cp.max(cp.abs(E_new))) + 1.0)
        if rel < 1e-2:
            dt = min(dt * 1.5, dt_max)
        if rel < conv_tol:
            return step, True
    return max_steps, False


# ====================================================================
# Test 1: Constant sigma_dot  (exponential analytical solution)
#
# Steady-state ODE (constant w = sigma_dot < 0, K = K_COLD):
#
#   rho_i w dE/dsig = (K/h^2) d^2E/dsig^2
#
# with Pe = rho_i w h^2 / K  (Pe < 0 for downward advection).
#
# General solution:  E = A + B exp(Pe sigma)
#
# BCs:
#   E(1)         = E_s              => A + B exp(Pe) = E_s
#   dE/dsig|_0   = -h Q_geo / K    => B Pe = -h Q_geo / K
#
# Therefore:
#   B = -h Q_geo / (K Pe)
#   A = E_s - B exp(Pe)
# ====================================================================
def test_constant_advection_diffusion():
    """
    Constant downward sigma_dot with Neumann bed BC.
    Analytical solution is an exponential profile.
    """
    Pe = -5.0
    nz = 41
    ops = _make_ops(nz=nz)
    sigma = cp.asnumpy(ops.sigma)
    ny, nx = ops.grid.ny, ops.grid.nx

    # Prescribed omega = H * sigma_dot (constant, negative = downward)
    # sigma_dot = Pe * K_COLD / (RHO_I * H^2), omega = H * sigma_dot
    omega_val = Pe * K_COLD / (RHO_I * H_ICE)  # = H * sigma_dot
    ops.enthalpy_velocity.omega.fill(cp.float32(omega_val))

    # Analytical solution
    E_s = C_I * (T_SURFACE - T_REF)
    B = -H_ICE * Q_GEO / (K_COLD * Pe)
    A = E_s - B * np.exp(Pe)
    E_analytical = A + B * np.exp(Pe * sigma)
    T_analytical = E_analytical / C_I + T_REF

    # Verify base stays cold
    T_pmp_bed = T_MELT - BETA_CC * RHO_I * GRAVITY * H_ICE
    assert T_analytical[0] < T_pmp_bed, "Base must stay cold for this test"

    # Run to steady state
    steps, converged = _run_to_steady_state(ops)

    i_col, j_col = 2, 2
    E_final = cp.asnumpy(ops.enthalpy_state.E[i_col, j_col, :]) * E_SCALE
    T_final = E_final / C_I + T_REF

    # Errors (exclude surface Dirichlet node)
    E_err = np.abs(E_final[:-1] - E_analytical[:-1])
    E_rel = np.max(E_err) / np.max(np.abs(E_analytical[:-1]))
    T_err = np.max(np.abs(T_final[:-1] - T_analytical[:-1]))

    print(f"Constant advection-diffusion test (Pe = {Pe:.1f}):")
    print(f"  Converged:     {converged} ({steps} steps)")
    print(f"  T_bed model:   {T_final[0]:.4f} K")
    print(f"  T_bed ref:     {T_analytical[0]:.4f} K")
    print(f"  Max T error:   {T_err:.2e} K")
    print(f"  Max E rel err: {E_rel:.2e}")

    assert converged, f"Did not converge in {steps} steps"
    assert E_rel < 0.02, f"Enthalpy relative error {E_rel:.2e} > 2%"
    print("  PASSED")


# ====================================================================
# Test 2: Linear sigma_dot  (Robin / error-function solution)
#
# sigma_dot(sigma) = -(SMB / h) sigma    (zero at bed, downward)
#
# Steady-state ODE:
#
#   -rho_i (SMB/h) sigma dE/dsig = (K/h^2) d^2E/dsig^2
#
# Let Pe_R = rho_i SMB h / K  (Robin Peclet number).
# Substituting u = dE/dsig and solving:
#
#   u(sigma) = C1 exp(-Pe_R sigma^2 / 2)
#
# with C1 = dE/dsig|_0 = -h Q_geo / K  (Neumann BC).
#
# Integrating:
#
#   E(sigma) = E_s + (h Q_geo / K) sqrt(pi / (2 Pe_R))
#              * [erf(sqrt(Pe_R/2)) - erf(sigma sqrt(Pe_R/2))]
# ====================================================================
def test_robin_advection_diffusion():
    """
    Linear sigma_dot (accumulation-driven) with Neumann bed BC.
    Analytical solution involves the error function.

    NOTE: This test prescribes omega = -SMB*sigma, which implies
    domega/dsig = -SMB != 0.  With constant H, the sigma-space
    continuity equation requires domega/dsig = 0.  The conservative
    form correctly captures this inconsistency via the E*domega/dsig
    term, so the test is skipped until a consistent Robin benchmark
    with evolving H is implemented.
    """
    print("\nRobin advection-diffusion test (Pe_R = 10.0):")
    print("  SKIPPED — inconsistent with conservative form (domega/dsig != 0 at constant H)")
    return
    Pe_R = 10.0
    nz = 41
    ops = _make_ops(nz=nz)
    sigma = cp.asnumpy(ops.sigma)
    ny, nx = ops.grid.ny, ops.grid.nx

    # Prescribed omega = H * sigma_dot = -SMB * sigma per level
    SMB = Pe_R * K_COLD / (RHO_I * H_ICE)
    for k in range(nz):
        sig_k = float(ops.sigma[k])
        ops.enthalpy_velocity.omega[:, :, k] = cp.float32(-SMB * sig_k)

    # Analytical solution
    E_s = C_I * (T_SURFACE - T_REF)
    alpha = np.sqrt(Pe_R / 2.0)
    scale = (H_ICE * Q_GEO / K_COLD) * np.sqrt(np.pi / (2.0 * Pe_R))
    E_analytical = E_s + scale * (erf(alpha) - erf(sigma * alpha))
    T_analytical = E_analytical / C_I + T_REF

    # Verify base stays cold
    T_pmp_bed = T_MELT - BETA_CC * RHO_I * GRAVITY * H_ICE
    assert T_analytical[0] < T_pmp_bed, "Base must stay cold for this test"

    # Run to steady state
    steps, converged = _run_to_steady_state(ops)

    i_col, j_col = 2, 2
    E_final = cp.asnumpy(ops.enthalpy_state.E[i_col, j_col, :])
    T_final = E_final / C_I + T_REF

    # Errors (exclude surface Dirichlet node)
    E_err = np.abs(E_final[:-1] - E_analytical[:-1])
    E_rel = np.max(E_err) / np.max(np.abs(E_analytical[:-1]))
    T_err = np.max(np.abs(T_final[:-1] - T_analytical[:-1]))

    print(f"Robin advection-diffusion test (Pe_R = {Pe_R:.1f}):")
    print(f"  Converged:     {converged} ({steps} steps)")
    print(f"  T_bed model:   {T_final[0]:.4f} K")
    print(f"  T_bed ref:     {T_analytical[0]:.4f} K")
    print(f"  Max T error:   {T_err:.2e} K")
    print(f"  Max E rel err: {E_rel:.2e}")

    assert converged, f"Did not converge in {steps} steps"
    assert E_rel < 0.02, f"Enthalpy relative error {E_rel:.2e} > 2%"
    print("  PASSED")


if __name__ == '__main__':
    test_constant_advection_diffusion()
    test_robin_advection_diffusion()
