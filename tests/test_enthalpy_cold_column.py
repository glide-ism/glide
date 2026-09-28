"""
Test: Static cold ice column with analytical steady-state solution.

Setup:
    - Flat ice slab of uniform thickness H, no velocity (u = v = 0)
    - Geothermal heat flux Q_geo at the base (Neumann BC)
    - Fixed surface temperature T_s (Dirichlet BC)
    - No strain heating, no drainage
    - Cold base: T_bed < T_pmp everywhere

Analytical steady state:
    With no advection and constant diffusivity K_c = k_i / c_i,
    the enthalpy equation reduces to:

        d/dz (K_c dE/dz) = 0

    subject to:
        E(z=s) = c_i (T_s - T_ref)             (surface Dirichlet)
        -K_c dE/dz |_{z=b} = Q_geo             (basal Neumann)

    In sigma coordinates (sigma = (z-b)/h, sigma=0 at bed, sigma=1 at surface):
        E(sigma) = E_s + (Q_geo * h / k_i) * (1 - sigma)

    Equivalently in temperature:
        T(sigma) = T_s + (Q_geo * h / k_i) * (1 - sigma)

    The temperature increases linearly from the surface down to the bed.

Validation:
    Run the column smoother to steady state and compare against analytical.
"""
import cupy as cp
import numpy as np
from glide.grid import Grid
from glide.enthalpy import (
    EnthalpyOperators, C_I, K_I, T_REF, T_MELT, RHO_I, GRAVITY, BETA_CC,
    E_SCALE
)


def test_cold_column():
    # ---- Parameters ----
    ny, nx = 4, 4
    dx = 1000.0         # m (doesn't matter, no horizontal gradients)
    H_ice = 1000.0      # m, ice thickness
    T_surface = 243.15   # K (-30C)
    Q_geo = 0.04         # W/m^2 (typical geothermal)
    nz = 21
    dt = 1000.0 * 365.25 * 86400.0  # 1000 years in seconds
    n_steps = 200        # number of time steps (200 kyr total, >> diffusion timescale)
    n_smooth = 20        # smoothing iterations per step

    # Verify the base stays cold with these parameters
    # dT across slab = Q_geo * H / k_i
    dT = Q_geo * H_ice / K_I
    T_bed_analytical = T_surface + dT
    T_pmp_bed = T_MELT - BETA_CC * RHO_I * GRAVITY * H_ice
    assert T_bed_analytical < T_pmp_bed, (
        f"Bed temperature {T_bed_analytical:.2f} K exceeds pressure melting "
        f"point {T_pmp_bed:.2f} K — test setup is not cold-based!"
    )

    # ---- Grid setup ----
    grid = Grid(ny, nx, cp.float32(dx))
    grid.state.H.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
    grid.state.H_prev.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
    grid.geometry.bed.set(cp.zeros((ny, nx), dtype=cp.float32))
    grid.forcing.smb.set(cp.zeros((ny, nx), dtype=cp.float32))

    # Sliding params (not used, but needed for grid)
    grid.sliding.beta.set(cp.ones((ny, nx), dtype=cp.float32))
    grid.sliding.m.set(1.0)
    grid.sliding.u_reg.set(1.0)

    # ---- Enthalpy operator ----
    ops = EnthalpyOperators(grid, nz=nz)  # uniform sigma for clean comparison
    ops.term_flags.horizontal_advection = False  # pure column test

    # Initialize with surface temperature everywhere
    ops.initialize_from_temperature(T_surface)
    ops.set_surface_enthalpy_from_temperature(
        cp.full((ny, nx), T_surface, dtype=cp.float32))

    # Set geothermal flux
    ops.enthalpy_forcing.Q_geo[:, :] = Q_geo

    # Zero strain heating and velocity
    ops.enthalpy_forcing.phi_strain.fill(0)
    ops.enthalpy_velocity.u3d.fill(0)
    ops.enthalpy_velocity.v3d.fill(0)
    ops.enthalpy_velocity.omega.fill(0)
    ops.enthalpy_forcing.Q_fh.fill(0)

    # ---- Time stepping ----
    for step in range(n_steps):
        ops.set_rhs()
        ops.column_sweep(dt, n_smooth)

    # ---- Compare against analytical ----
    sigma = cp.asnumpy(ops.sigma)
    E_numerical = cp.asnumpy(ops.enthalpy_state.E[ny // 2, nx // 2, :]) * E_SCALE

    # Analytical: E(sigma) = E_surface + (Q_geo * H / k_i) * c_i * (1 - sigma)
    # Since E = c_i * (T - T_ref), and T(sigma) = T_s + dT*(1-sigma):
    E_surface = C_I * (T_surface - T_REF)
    E_analytical = E_surface + C_I * Q_geo * H_ice / K_I * (1 - sigma)

    # Relative error (exclude surface node which is exactly imposed)
    error = np.abs(E_numerical[:-1] - E_analytical[:-1])
    rel_error = error / np.abs(E_analytical[:-1])
    max_rel_error = np.max(rel_error)

    print(f"Cold column test:")
    print(f"  dT across slab (analytical): {dT:.4f} K")
    print(f"  T_bed analytical: {T_bed_analytical:.2f} K")
    print(f"  T_pmp at bed:     {T_pmp_bed:.2f} K")
    print(f"  Max relative error: {max_rel_error:.2e}")
    print(f"  Max absolute error: {np.max(error):.4f} J/(kg)")

    # Compare temperatures for readability
    T_numerical = E_numerical / C_I + T_REF
    T_analytical = E_analytical / C_I + T_REF

    print(f"\n  Sigma | T_num (K) | T_ana (K) | Error (K)")
    print(f"  {'─'*50}")
    for k in range(0, nz, max(1, nz // 10)):
        print(f"  {sigma[k]:5.3f} | {T_numerical[k]:9.4f} | {T_analytical[k]:9.4f} | "
              f"{abs(T_numerical[k] - T_analytical[k]):.4e}")

    assert max_rel_error < 5e-3, (
        f"Cold column test FAILED: max relative error = {max_rel_error:.2e} > 5e-3"
    )
    print(f"\n  PASSED (max relative error = {max_rel_error:.2e})")
    return max_rel_error


def test_cold_column_residual_convergence():
    """
    Verify that the residual decreases monotonically during smoothing,
    confirming the column smoother is a valid iterative solver.
    """
    ny, nx = 4, 4
    dx = 1000.0
    H_ice = 1000.0
    T_surface = 243.15
    Q_geo = 0.04
    nz = 21
    dt = 1000.0 * 365.25 * 86400.0

    grid = Grid(ny, nx, cp.float32(dx))
    grid.state.H.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
    grid.state.H_prev.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
    grid.geometry.bed.set(cp.zeros((ny, nx), dtype=cp.float32))
    grid.forcing.smb.set(cp.zeros((ny, nx), dtype=cp.float32))
    grid.sliding.beta.set(cp.ones((ny, nx), dtype=cp.float32))
    grid.sliding.m.set(1.0)
    grid.sliding.u_reg.set(1.0)

    ops = EnthalpyOperators(grid, nz=nz)
    ops.term_flags.horizontal_advection = False  # pure column test
    ops.initialize_from_temperature(T_surface)
    ops.set_surface_enthalpy_from_temperature(
        cp.full((ny, nx), T_surface, dtype=cp.float32))
    ops.enthalpy_forcing.Q_geo[:, :] = Q_geo
    ops.enthalpy_forcing.phi_strain.fill(0)
    ops.enthalpy_velocity.u3d.fill(0)
    ops.enthalpy_velocity.v3d.fill(0)
    ops.enthalpy_velocity.omega.fill(0)
    ops.enthalpy_forcing.Q_fh.fill(0)

    ops.set_rhs()

    residuals = []
    for sweep in range(50):
        r_norm = float(ops.compute_residual(dt))
        residuals.append(r_norm)
        ops.column_smooth(dt)
        ops.enthalpy_state.E[:] += ops.smoother_config.omega * ops.delta_E

    print(f"\nResidual convergence test:")
    print(f"  Sweep | Residual norm")
    print(f"  {'─'*30}")
    for i in [0, 1, 2, 5, 10, 20, 49]:
        print(f"  {i:5d} | {residuals[i]:.4e}")

    # Residual should decrease significantly
    reduction = residuals[0] / residuals[-1]
    print(f"\n  Total reduction: {reduction:.1f}x")

    assert reduction > 100, (
        f"Residual convergence test FAILED: reduction = {reduction:.1f}x < 100x"
    )
    print(f"  PASSED")


if __name__ == '__main__':
    test_cold_column()
    test_cold_column_residual_convergence()
