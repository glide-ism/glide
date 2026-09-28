"""
Tests for polythermal enthalpy behavior (Stefan problem).

A static ice slab with geothermal heat flux large enough to produce a
temperate basal layer. The Neumann heat flux must persist after the bed
reaches the pressure melting point — excess energy is stored as latent
heat (water content), not clamped to E_pmp.

Parameters are consistent with examples/thermal/stefan.py.
"""
import cupy as cp
import numpy as np
from scipy.optimize import brentq
from glide.grid import Grid
from glide.enthalpy import (
    EnthalpyOperators, C_I, K_I, T_REF, T_MELT, RHO_I, GRAVITY, BETA_CC,
    L_HEAT, K_COLD, K_TEMP, K_TEMP_FACTOR, E_SCALE
)

# ---- Shared parameters (match examples/thermal/stefan.py) ----
H_ICE = 1000.0
T_SURFACE = 223.15     # K (-50 C)
Q_GEO = 0.5            # W/m^2
SEC_PER_KYR = 1000.0 * 365.25 * 86400.0


def _make_ops(nx=4, ny=4, nz=21):
    """Build an EnthalpyOperators for a static slab with Q_geo at the bed."""
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


def _E_pmp_bed():
    T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * H_ICE
    return C_I * (T_pmp - T_REF)


def test_temperate_bed_still_applies_geothermal_flux():
    """
    When E > E_pmp everywhere (uniform temperate state, no gradients),
    the only residual at the bed should be the Neumann heat-flux term
    divided by the half-cell control volume width.
    """
    ny, nx = 4, 4
    dt = 1000.0 * SEC_PER_KYR

    ops = _make_ops(nx=nx, ny=ny)

    E_pmp_bed = _E_pmp_bed()
    E_temperate = (E_pmp_bed + 1000.0) / E_SCALE
    ops.enthalpy_state.E.fill(E_temperate)
    ops.enthalpy_state.E_prev.fill(E_temperate)

    ops.set_rhs()
    ops.compute_residual(dt)
    bed_residual = float(cp.asnumpy(ops.r_E[ny // 2, nx // 2, 0]))
    dsig_half = 0.5 * float(ops.sigma[1] - ops.sigma[0])
    # Conservative form: Q_geo term is -(Q_geo) / (dsig_half * E_SCALE)
    expected = -Q_GEO / (dsig_half * E_SCALE)

    print("Temperate bed geothermal-flux test:")
    print(f"  Bed residual: {bed_residual:.6e}")
    print(f"  Expected:     {expected:.6e}")

    assert np.isclose(bed_residual, expected, rtol=1e-3, atol=1e-2), (
        "Temperate bed should still see the geothermal-flux residual term"
    )


def test_temperate_bed_is_not_clamped_to_pmp():
    """After smoothing from a temperate state, E_bed must stay above E_pmp."""
    dt = 1000.0 * SEC_PER_KYR

    ops = _make_ops()

    E_pmp_bed = _E_pmp_bed()
    E_temperate = (E_pmp_bed + 1000.0) / E_SCALE
    ops.enthalpy_state.E.fill(E_temperate)
    ops.enthalpy_state.E_prev.fill(E_temperate)

    ops.set_rhs()
    ops.column_sweep(dt, 5)
    ops.enthalpy_state.E[:] += ops.smoother_config.omega * ops.delta_E

    E_bed = float(cp.asnumpy(ops.enthalpy_state.E[0, 0, 0])) * E_SCALE
    print("Temperate bed unclamped test:")
    print(f"  E_bed after sweep: {E_bed:.2f}")
    print(f"  E_pmp at bed:      {E_pmp_bed:.2f}")

    assert E_bed > E_pmp_bed, "Basal enthalpy should not be hard-clamped to E_pmp"


def test_steady_state_polythermal_profile():
    """
    Run to steady state with adaptive dt and verify the converged profile
    against the semi-analytical polythermal reference (same as
    examples/thermal/stefan.py).

    Reference: two-zone steady diffusion with Neumann bed BC.
      Cold zone  (sigma > sigma*):  E linear with slope Q_geo*H/K_COLD
      Temperate  (sigma < sigma*):  E linear with slope Q_geo*H/K_TEMP
    """
    nz = 41
    ops = _make_ops(nz=nz)
    sigma = cp.asnumpy(ops.sigma)

    # --- Semi-analytical reference ---
    def T_pmp(sig):
        return T_MELT - BETA_CC * RHO_I * GRAVITY * (1 - sig) * H_ICE

    def E_pmp(sig):
        return C_I * (T_pmp(sig) - T_REF)

    Q_crit = K_I * (T_pmp(0.0) - T_SURFACE) / H_ICE
    assert Q_GEO > Q_crit, "Test requires Q_geo > Q_crit for temperate base"

    def cts_residual(sig_star):
        return Q_GEO * H_ICE * (1.0 - sig_star) / K_I - (T_pmp(sig_star) - T_SURFACE)

    sigma_star = brentq(cts_residual, 0.0, 1.0 - 1e-10)

    E_cts = E_pmp(sigma_star)
    E_analytical = np.zeros_like(sigma)
    for k, sig in enumerate(sigma):
        if sig <= sigma_star:
            E_analytical[k] = E_cts + Q_GEO * H_ICE / K_TEMP * (sigma_star - sig)
        else:
            E_analytical[k] = C_I * (T_SURFACE - T_REF) + Q_GEO * H_ICE / K_COLD * (1.0 - sig)

    T_pmp_profile = np.array([T_pmp(s) for s in sigma])
    T_analytical = np.minimum(E_analytical / C_I + T_REF, T_pmp_profile)
    omega_analytical = np.maximum(E_analytical - np.array([E_pmp(s) for s in sigma]), 0.0) / L_HEAT

    # --- Run to steady state with adaptive dt ---
    i_col, j_col = 2, 2
    dt = 100.0 * SEC_PER_KYR
    dt_max = 1e6 * SEC_PER_KYR
    conv_tol = 1e-5
    max_steps = 5000

    converged = False
    for step in range(1, max_steps + 1):
        E_old = ops.enthalpy_state.E[i_col, j_col, :].copy()
        ops.set_rhs()
        ops.column_sweep(dt, 20)

        E_new = ops.enthalpy_state.E[i_col, j_col, :]
        max_dE = float(cp.max(cp.abs(E_new - E_old)))
        E_scale = float(cp.max(cp.abs(E_new))) + 1.0
        rel_change = max_dE / E_scale

        if rel_change < 1e-2:
            dt = min(dt * 1.5, dt_max)
        if rel_change < conv_tol:
            converged = True
            break

    E_final = cp.asnumpy(ops.enthalpy_state.E[i_col, j_col, :]) * E_SCALE
    T_final = cp.asnumpy(ops.get_temperature()[i_col, j_col, :])
    omega_final = cp.asnumpy(ops.get_water_content()[i_col, j_col, :])

    # Errors (exclude surface Dirichlet node)
    T_err = np.max(np.abs(T_final[:-1] - T_analytical[:-1]))
    E_rel_err = np.max(np.abs(E_final[:-1] - E_analytical[:-1])) / np.max(np.abs(E_analytical[:-1]))
    omega_rel_err = np.abs(omega_final[0] - omega_analytical[0]) / omega_analytical[0]

    print(f"Steady-state polythermal profile test:")
    print(f"  Converged:        {converged} ({step} steps)")
    print(f"  sigma* (ref):     {sigma_star:.4f}")
    print(f"  Max T error:      {T_err:.2e} K")
    print(f"  Max E rel error:  {E_rel_err:.2e}")
    print(f"  omega_bed:        {omega_final[0]:.3e}  (ref: {omega_analytical[0]:.3e}, rel err: {omega_rel_err:.2e})")

    assert converged, f"Did not converge in {max_steps} steps"
    assert T_err < 0.1, f"Temperature error {T_err:.2e} K > 0.1 K"
    assert E_rel_err < 0.02, f"Enthalpy relative error {E_rel_err:.2e} > 2%"
    assert omega_rel_err < 0.02, f"Basal omega relative error {omega_rel_err:.2e} > 2%"
    print("  PASSED")


if __name__ == '__main__':
    test_temperate_bed_still_applies_geothermal_flux()
    test_temperate_bed_is_not_clamped_to_pmp()
    test_steady_state_polythermal_profile()
