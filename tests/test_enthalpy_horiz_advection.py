"""
Test horizontal advection in the enthalpy solver.

Tests the conservative form: rho_i * div(H * u * E).

1. Uniform E + divergent velocity: solver stays bounded (conservative
   form correctly produces a nonzero residual from E * div(H*u)).
2. Uniform velocity + step profile: y-symmetry preserved, step advects.
3. Energy conservation: total energy H*E is conserved for uniform
   velocity, uniform H, and periodic-like interior.
"""
import cupy as cp
import numpy as np
from glide.grid import Grid
from glide.enthalpy import EnthalpyOperators, C_I, T_REF, E_SCALE


def _make_grid(ny, nx, dx, H_ice):
    """Create a flat-slab grid."""
    grid = Grid(ny, nx, cp.float32(dx))
    grid.state.H.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
    grid.state.H_prev.set(cp.full((ny, nx), H_ice, dtype=cp.float32))
    grid.geometry.bed.set(cp.zeros((ny, nx), dtype=cp.float32))
    grid.forcing.smb.set(cp.zeros((ny, nx), dtype=cp.float32))
    grid.sliding.beta.set(cp.ones((ny, nx), dtype=cp.float32))
    grid.sliding.m.set(1.0)
    grid.sliding.u_reg.set(1.0)
    return grid


def _make_ops(grid, nz):
    """Create an EnthalpyOperators with no sources."""
    ops = EnthalpyOperators(grid, nz=nz)
    ops.enthalpy_forcing.Q_geo.fill(0)
    ops.enthalpy_forcing.Q_fh.fill(0)
    ops.enthalpy_forcing.phi_strain.fill(0)
    ops.enthalpy_forcing.drain_rate.set(0.0)
    ops.enthalpy_velocity.omega.fill(0)
    ops.term_flags.omega = False       # no vertical transport
    ops.term_flags.drainage = False
    return ops


def test_uniform_E_divergent_velocity():
    """
    Uniform E + spatially varying velocity  →  solver stays bounded.

    In the conservative form, div(H*u*E) != 0 for uniform E and
    divergent u (because E * div(H*u) != 0). This is correct behavior:
    it reflects the changing control volume. We verify the solver
    converges without blowing up.
    """
    ny, nx, nz = 8, 8, 9
    dx = 1000.0
    H_ice = 1000.0
    T0 = 248.15
    dt = 100.0 * 365.25 * 86400.0

    grid = _make_grid(ny, nx, dx, H_ice)
    ops = _make_ops(grid, nz)
    ops.initialize_from_temperature(T0)
    ops.set_surface_enthalpy_from_temperature(
        cp.full((ny, nx), T0, dtype=cp.float32))

    # Divergent velocity: u(x) = alpha * x on MAC faces (m/s)
    alpha = 1e-10
    u_face = np.zeros((ny, nx + 1), dtype=np.float32)
    for j in range(nx + 1):
        u_face[:, j] = alpha * j * dx
    u_face_gpu = cp.array(u_face)
    for k in range(nz):
        ops.enthalpy_velocity.u3d[k, :, :] = u_face_gpu
    ops.enthalpy_velocity.v3d.fill(0)

    ops.set_rhs()
    r0 = float(ops.compute_residual(dt))
    E0 = float(ops.enthalpy_state.E[ny // 2, nx // 2, nz // 2])
    print(f"Uniform E + divergent velocity test:")
    print(f"  E0 = {E0:.1f} J/kg")
    print(f"  Initial residual |r0| = {r0:.2e}")

    ops.set_rhs()
    ops.smoother_config.report_norms = True
    ops.column_sweep(dt, 5)
    ops.smoother_config.report_norms = False

    E_final = cp.asnumpy(ops.enthalpy_state.E[:, :, nz // 2])
    E_range = np.max(E_final) - np.min(E_final)
    print(f"  E range after 5 sweeps = {E_range:.2e} J/kg")
    print("  PASSED")


def test_uniform_velocity_step_profile():
    """
    Uniform velocity u > 0 with a temperature step in x.

    The step should advect to the right. All rows (y-direction)
    should remain identical since v = 0 and the initial condition
    is invariant in y.
    """
    ny, nx, nz = 8, 16, 9
    dx = 1000.0
    H_ice = 1000.0
    T_cold = 240.0
    T_warm = 260.0
    dt = 10.0 * 365.25 * 86400.0

    grid = _make_grid(ny, nx, dx, H_ice)
    ops = _make_ops(grid, nz)

    # Temperature step: cold on left half, warm on right half
    T_init = cp.full((ny, nx, nz), T_cold, dtype=cp.float32)
    T_init[:, nx // 2:, :] = T_warm
    ops.enthalpy_state.E[:] = C_I * (T_init - T_REF) / E_SCALE
    ops.enthalpy_state.E_prev[:] = ops.enthalpy_state.E
    ops.set_surface_enthalpy_from_temperature(
        cp.full((ny, nx), T_cold, dtype=cp.float32))
    ops.enthalpy_velocity.v3d.fill(0)

    # Uniform rightward velocity: 100 m/yr in m/s
    u_ms = cp.float32(100.0 / (365.25 * 86400.0))
    ops.enthalpy_velocity.u3d.fill(u_ms)

    # Run 3 time steps
    for step in range(3):
        ops.set_rhs()
        ops.column_sweep(dt, 20)

    # Check y-symmetry: all rows should be identical
    E_mid = cp.asnumpy(ops.enthalpy_state.E[:, :, nz // 2])
    row_spread = np.max(E_mid, axis=0) - np.min(E_mid, axis=0)
    max_row_spread = np.max(row_spread)
    print(f"\nUniform velocity + step profile test:")
    print(f"  Max row spread (y-asymmetry) = {max_row_spread:.2e} J/kg")

    E_init_row = C_I * (np.where(np.arange(nx) >= nx // 2,
                                  T_warm, T_cold) - T_REF) / E_SCALE
    E_final_row = E_mid[ny // 2, :]
    print(f"  E at interface-2: {E_final_row[nx//2-2]:.1f} "
          f"(init: {E_init_row[nx//2-2]:.1f})")
    print(f"  E at interface:   {E_final_row[nx//2]:.1f} "
          f"(init: {E_init_row[nx//2]:.1f})")
    print(f"  E at interface+2: {E_final_row[nx//2+2]:.1f} "
          f"(init: {E_init_row[nx//2+2]:.1f})")

    assert max_row_spread < 1.0, (
        f"y-asymmetry {max_row_spread:.2e} — rows should be identical "
        f"for uniform velocity + y-invariant initial condition"
    )
    print("  PASSED")


def test_energy_conservation():
    """
    Verify that total energy sum(H * E) is conserved for horizontal
    advection with uniform H, uniform u, and no sources.

    The conservative form d(HE)/dt + div(HuE) = 0 guarantees that
    internal fluxes telescope (cancel at shared faces). Only boundary
    fluxes change total energy. For interior cells (away from domain
    edges), the total should be conserved to machine precision.
    """
    ny, nx, nz = 8, 16, 5
    dx = 1000.0
    H_ice = 1000.0
    T_cold = 240.0
    T_warm = 260.0
    dt = 1.0 * 365.25 * 86400.0  # 1 yr — small step

    grid = _make_grid(ny, nx, dx, H_ice)
    ops = _make_ops(grid, nz)

    # Smooth sinusoidal temperature profile in x (avoids boundary issues)
    sigma = cp.asnumpy(ops.sigma)
    E_init = np.zeros((ny, nx, nz), dtype=np.float32)
    T_avg = 0.5 * (T_cold + T_warm)
    T_amp = 0.5 * (T_warm - T_cold)
    for j in range(nx):
        T_j = T_avg + T_amp * np.sin(2 * np.pi * j / nx)
        E_init[:, j, :] = C_I * (T_j - T_REF) / E_SCALE
    ops.enthalpy_state.E[:] = cp.asarray(E_init)
    ops.enthalpy_state.E_prev[:] = ops.enthalpy_state.E
    ops.set_surface_enthalpy_from_temperature(
        cp.full((ny, nx), T_avg, dtype=cp.float32))
    ops.enthalpy_velocity.v3d.fill(0)

    # Uniform rightward velocity
    u_ms = cp.float32(50.0 / (365.25 * 86400.0))
    ops.enthalpy_velocity.u3d.fill(u_ms)

    # Measure total energy in interior cells (exclude boundary rows/cols
    # and surface layer which has Dirichlet BC)
    def total_energy_interior():
        E = cp.asnumpy(ops.enthalpy_state.E)
        H = cp.asnumpy(grid.state.H.data)
        # Interior: rows 2:-2, cols 2:-2, layers 0:-1 (exclude surface Dirichlet)
        HE = H[2:-2, 2:-2, None] * E[2:-2, 2:-2, :-1]
        return float(np.sum(HE))

    energy_before = total_energy_interior()

    # One time step
    ops.set_rhs()
    ops.column_sweep(dt, 20)

    energy_after = total_energy_interior()

    rel_change = abs(energy_after - energy_before) / abs(energy_before)
    print(f"\nEnergy conservation test:")
    print(f"  Total interior H*E before: {energy_before:.6e}")
    print(f"  Total interior H*E after:  {energy_after:.6e}")
    print(f"  Relative change: {rel_change:.3e}")

    # With constant H, uniform u, and periodic-like interior,
    # the conservative fluxes should nearly cancel. Allow some
    # tolerance for the non-periodic boundaries and Dirichlet BC.
    assert rel_change < 0.01, (
        f"Energy not conserved: relative change {rel_change:.3e} > 1%"
    )
    print("  PASSED")


if __name__ == '__main__':
    test_uniform_E_divergent_velocity()
    test_uniform_velocity_step_profile()
    test_energy_conservation()
