"""
Multigrid solvers for the SSA ice sheet equations.

Implements FASCD (Full Approximation Scheme with Constrained Descent)
for the forward problem and adjoint V-cycles for gradient computation.
"""

import cupy as cp
from .kernels import (
    restrict_vfacet, restrict_hfacet, restrict_cell_centered,
    restrict_max_pool, prolongate_vfacet, prolongate_hfacet,
    prolongate_cell_centered
)


def restrict_solution(grid, adjoint=False):
    """Restrict solution from grid to child."""
    child = grid.child
    kernels = grid.kernels
    if adjoint:
        restrict_vfacet(grid.lambda_u, kernels, u_coarse=child.lambda_u)
        restrict_hfacet(grid.lambda_v, kernels, v_coarse=child.lambda_v)
        restrict_cell_centered(grid.lambda_H, kernels, f_coarse=child.lambda_H)
    else:
        restrict_vfacet(grid.u, kernels, u_coarse=child.u)
        restrict_hfacet(grid.v, kernels, v_coarse=child.v)
        restrict_cell_centered(grid.H, kernels, f_coarse=child.H)


def restrict_residual(grid, adjoint=False):
    """Restrict residual from grid to child."""
    child = grid.child
    kernels = grid.kernels
    if adjoint:
        restrict_vfacet(grid.r_adj_u, kernels, u_coarse=child.r_adj_u)
        restrict_hfacet(grid.r_adj_v, kernels, v_coarse=child.r_adj_v)
        restrict_cell_centered(grid.r_adj_H, kernels, f_coarse=child.r_adj_H)
    else:
        restrict_vfacet(grid.r_u, kernels, u_coarse=child.r_u)
        restrict_hfacet(grid.r_v, kernels, v_coarse=child.r_v)
        restrict_cell_centered(grid.r_H, kernels, f_coarse=child.r_H)


def restrict_f(grid):
    """Restrict RHS from grid to child."""
    child = grid.child
    kernels = grid.kernels
    restrict_vfacet(grid.f_u, kernels, u_coarse=child.f_u)
    restrict_hfacet(grid.f_v, kernels, v_coarse=child.f_v)
    restrict_cell_centered(grid.f_H, kernels, f_coarse=child.f_H)

def restrict_parameters(grid):
    """Restrict physical parameters from grid to child."""
    child = grid.child
    kernels = grid.kernels
    restrict_cell_centered(grid.bed, kernels, f_coarse=child.bed)
    restrict_cell_centered(grid.B, kernels, f_coarse=child.B)
    restrict_cell_centered(grid.beta, kernels, f_coarse=child.beta)
    restrict_cell_centered(grid.H_prev, kernels, f_coarse=child.H_prev)
    restrict_cell_centered(grid.smb, kernels, f_coarse=child.smb)

def restrict_parameters_to_hierarchy(grid):
    """Recursively restrict parameters through entire hierarchy."""
    if grid.child is not None:
        restrict_parameters(grid)
        restrict_parameters_to_hierarchy(grid.child)


def restrict_frozen_fields(grid):
    """Restrict frozen fields (eta, beta_eff, c_eff) from grid to child."""
    child = grid.child
    kernels = grid.kernels
    restrict_cell_centered(grid.eta, kernels, f_coarse=child.eta)
    restrict_cell_centered(grid.beta_eff, kernels, f_coarse=child.beta_eff)
    restrict_cell_centered(grid.c_eff, kernels, f_coarse=child.c_eff)

def restrict_frozen_fields_to_hierarchy(grid):
    """Recursively restrict frozen fields through entire hierarchy."""
    if grid.child is not None:
        restrict_frozen_fields(grid)
        restrict_frozen_fields_to_hierarchy(grid.child)

def fascd_vcycle(grid, thklim, finest=False):
    """
    FASCD V-cycle for the coupled SSA + mass conservation system.

    Full Approximation Scheme with Constrained Descent handles the
    thickness inequality constraint H >= gamma via an active set method.

    Parameters
    ----------
    grid : Grid
        Finest grid level for this V-cycle
    thklim : float
        Minimum thickness constraint
    finest : bool
        Whether this is the finest level (entry point)
    """
    kernels = grid.kernels

    if finest:
        grid.w[:] = grid.U[:]
        grid.chi[:] = grid.gamma - grid.H

    if grid.child is None:
        # Coarsest level: direct solve
        grid.gamma[:] = grid.w_H + grid.chi[:]
        grid.vanka_sweep(500)
        grid.gamma.fill(thklim)
        return

    # Restrict constraint defect
    restrict_max_pool(grid.chi, kernels, f_coarse=grid.child.chi)

    # Prolongate and compute local constraint adjustment
    prolongate_cell_centered(-grid.child.chi, kernels, H_fine=grid.phi, smooth=False)
    grid.phi[:] += grid.chi

    # Pre-smooth with local constraint
    grid.gamma[:, :] = grid.w_H + grid.phi
    grid.vanka_sweep(20)
    grid.gamma.fill(thklim)

    # Compute coarse grid correction
    grid.y[:] = grid.U - grid.w

    # Restrict solution to child
    restrict_solution(grid)
    grid.child.w[:] = grid.child.U[:]

    # Compute and restrict residual
    grid.compute_residual()
    restrict_residual(grid)

    # Form coarse grid RHS: f_c = F_c(I_h^H u_h) - I_h^H r_h
    grid.child.compute_F()
    grid.child.f[:] = grid.child.F - grid.child.r

    # Recursive call
    fascd_vcycle(grid.child, thklim)

    # Compute coarse correction
    grid.child.z[:] = grid.child.U - grid.child.w

    # Prolongate correction
    prolongate_vfacet(grid.child.z_u, kernels, u_fine=grid.z_u)
    prolongate_hfacet(grid.child.z_v, kernels, v_fine=grid.z_v)
    prolongate_cell_centered(grid.child.z_H, kernels, H_fine=grid.z_H, smooth=False)

    # Apply correction
    grid.z[:] += grid.y
    grid.U[:] = grid.w + grid.z

    # Post-smooth
    grid.gamma[:, :] = grid.w_H + grid.chi
    grid.vanka_sweep(20)

    # Local error-based smoothing
    for _ in range(10):
        grid.compute_residual()
        a = grid.r_H
        b = grid.H - grid.gamma
        rss_H = a + b - cp.sqrt(a**2 + b**2)
        total_error = (abs(grid.r_u[:, 1:]) + abs(grid.r_u[:, :-1]) +
                       abs(grid.r_v[1:]) + abs(grid.r_v[:-1]) + abs(rss_H))
        grid.error_mask[:] = (total_error > 0.01).astype(cp.float32)
        grid.vanka_sweep_local(10)

    grid.gamma.fill(thklim)

def fascd_vcycle_frozen(grid, thklim, finest=False,verbose=False,omega=cp.float32(1.0)):
    """
    FASCD V-cycle for the coupled SSA + mass conservation system.

    Full Approximation Scheme with Constrained Descent handles the
    thickness inequality constraint H >= gamma via an active set method.

    Parameters
    ----------
    grid : Grid
        Finest grid level for this V-cycle
    thklim : float
        Minimum thickness constraint
    finest : bool
        Whether this is the finest level (entry point)
    """
    kernels = grid.kernels

    if finest:
        grid.w[:] = grid.U[:]
        grid.chi[:] = grid.gamma - grid.H

    if grid.child is None:
        # Coarsest level: direct solve
        grid.gamma[:] = grid.w_H + grid.chi[:]
        grid.vanka_sweep(400,frozen=True,n_inner=10,verbose=verbose,omega=omega)
        grid.gamma.fill(thklim)
        return

    # Restrict constraint defect
    restrict_max_pool(grid.chi, kernels, f_coarse=grid.child.chi)

    # Prolongate and compute local constraint adjustment
    prolongate_cell_centered(-grid.child.chi, kernels, H_fine=grid.phi, smooth=False)
    grid.phi[:] += grid.chi

    # Pre-smooth with local constraint
    grid.gamma[:, :] = grid.w_H + grid.phi
    grid.vanka_sweep(10,frozen=True,verbose=verbose,omega=omega)
    grid.gamma.fill(thklim)

    # Compute coarse grid correction
    grid.y[:] = grid.U - grid.w

    # Restrict solution to child
    restrict_solution(grid)
    grid.child.w[:] = grid.child.U[:]

    # Compute and restrict residual
    grid.compute_residual(frozen=True)
    restrict_residual(grid)

    # Form coarse grid RHS: f_c = F_c(I_h^H u_h) - I_h^H r_h
    grid.child.compute_F(frozen=True)
    grid.child.f[:] = grid.child.F - grid.child.r

    # Recursive call
    fascd_vcycle_frozen(grid.child, thklim,verbose=verbose)

    # Compute coarse correction
    grid.child.z[:] = grid.child.U - grid.child.w

    # Prolongate correction
    prolongate_vfacet(grid.child.z_u, kernels, u_fine=grid.z_u)
    prolongate_hfacet(grid.child.z_v, kernels, v_fine=grid.z_v)
    prolongate_cell_centered(grid.child.z_H, kernels, H_fine=grid.z_H, smooth=False)

    # Apply correction
    grid.z[:] += grid.y
    grid.U[:] = grid.w + grid.z

    # Post-smooth
    grid.gamma[:, :] = grid.w_H + grid.chi
    grid.vanka_sweep(10,frozen=True,verbose=verbose,omega=omega)
    grid.gamma.fill(thklim)

def adjoint_vcycle(grid):
    """
    Adjoint V-cycle for computing gradients via reverse-mode AD.

    Solves the linearized adjoint system to compute sensitivities
    of the objective w.r.t. parameters.

    Parameters
    ----------
    grid : Grid
        Finest grid level for this V-cycle
    """
    kernels = grid.kernels

    if grid.child is None:
        # Coarsest level: direct solve
        grid.compute_mask()
        grid.vanka_sweep_adjoint(100, omega=cp.float32(1.0))
        grid.mask.fill(0)
        return

    # Pre-smooth
    grid.compute_mask()
    grid.vanka_sweep_adjoint(10, omega=cp.float32(1.0))
    grid.mask.fill(0)

    # Compute adjoint residual
    grid.r_adj[:] = grid.f_adj[:]
    grid.compute_vjp()
    grid.r_adj[:] -= grid.l

    # Restrict to child
    restrict_solution(grid, adjoint=True)
    restrict_parameters(grid)

    # Set up coarse RHS
    grid.child.f_adj.fill(0.0)
    restrict_vfacet(grid.r_adj_u, kernels, u_coarse=grid.child.f_adj_u)
    restrict_hfacet(grid.r_adj_v, kernels, v_coarse=grid.child.f_adj_v)
    restrict_cell_centered(grid.r_adj_H, kernels, f_coarse=grid.child.f_adj_H)
    grid.child.Lambda.fill(0.0)
    grid.child.Lambda_out.fill(0.0)

    # Recursive call
    adjoint_vcycle(grid.child)

    # Prolongate correction
    grid.z.fill(0.0)
    prolongate_vfacet(grid.child.lambda_u, kernels, u_fine=grid.z_u)
    prolongate_hfacet(grid.child.lambda_v, kernels, v_fine=grid.z_v)
    prolongate_cell_centered(grid.child.lambda_H, kernels, H_fine=grid.z_H, smooth=True)

    # Apply correction
    grid.lambda_u[:] += grid.z_u
    grid.lambda_v[:] += grid.z_v
    grid.lambda_H[:] += grid.z_H

    # Post-smooth
    grid.compute_mask()
    grid.vanka_sweep_adjoint(10, omega=cp.float32(1.0))
    grid.mask.fill(0)
