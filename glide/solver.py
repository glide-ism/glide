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


def restrict_solution(grid):
    """Restrict solution from grid to child."""
    child = grid.child
    kernels = grid.kernels
    restrict_vfacet(grid.u, kernels, u_coarse=child.u)
    restrict_hfacet(grid.v, kernels, v_coarse=child.v)
    restrict_cell_centered(grid.H, kernels, f_coarse=child.H)
    restrict_cell_centered(grid.mask,kernels, f_coarse=child.mask)

def restrict_solution_to_hierarchy(grid):
    """Recursively restrict frozen fields through entire hierarchy."""
    if grid.child is not None:
        restrict_solution(grid)
        restrict_solution_to_hierarchy(grid.child)


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


def restrict_frozen_fields(grid,restrict_adjoint_parameters=False):
    """Restrict frozen fields (eta, beta_eff, c_eff) from grid to child."""
    child = grid.child
    kernels = grid.kernels
    restrict_cell_centered(grid.eta, kernels, f_coarse=child.eta)
    restrict_vfacet(grid.alpha_u, kernels, u_coarse=child.alpha_u)
    restrict_hfacet(grid.alpha_v, kernels, v_coarse=child.alpha_v)
    restrict_cell_centered(grid.c_eff, kernels, f_coarse=child.c_eff)
    restrict_cell_centered(grid.grounded, kernels, f_coarse=child.grounded)
    if restrict_adjoint_parameters:
        restrict_cell_centered(grid.lambda_eta,kernels,f_coarse=child.lambda_eta)
        restrict_vfacet(grid.lambda_alpha_u,kernels,u_coarse=child.lambda_alpha_u)
        restrict_hfacet(grid.lambda_alpha_v,kernels,v_coarse=child.lambda_alpha_v)

def restrict_frozen_fields_to_hierarchy(grid,restrict_adjoint_parameters=False):
    """Recursively restrict frozen fields through entire hierarchy."""
    if grid.child is not None:
        restrict_frozen_fields(grid,restrict_adjoint_parameters=restrict_adjoint_parameters)
        restrict_frozen_fields_to_hierarchy(grid.child,restrict_adjoint_parameters=restrict_adjoint_parameters)

def fascd_vcycle(grid, thklim, finest=False,verbose=False,omega=cp.float32(0.5),pre_steps=10,post_steps=20,coarse_steps=400,newton_iterations=10):
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
        grid.vanka_sweep(coarse_steps,n_inner=newton_iterations,verbose=verbose,omega=omega)
        grid.gamma.fill(thklim)
        return

    # Restrict constraint defect
    restrict_max_pool(grid.chi, kernels, f_coarse=grid.child.chi)

    # Prolongate and compute local constraint adjustment
    prolongate_cell_centered(-grid.child.chi, kernels, H_fine=grid.phi, smooth=False)
    grid.phi[:] += grid.chi

    # Pre-smooth with local constraint
    grid.gamma[:, :] = grid.w_H + grid.phi
    grid.vanka_sweep(pre_steps,n_inner=newton_iterations,verbose=verbose,omega=omega)
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
    fascd_vcycle(grid.child, thklim,verbose=verbose)

    # Compute coarse correction
    grid.child.z[:] = grid.child.U - grid.child.w

    # Prolongate correction
    prolongate_vfacet(grid.child.z_u, kernels, u_fine=grid.z_u, smooth=True)
    prolongate_hfacet(grid.child.z_v, kernels, v_fine=grid.z_v, smooth=True)
    prolongate_cell_centered(grid.child.z_H, kernels, H_fine=grid.z_H, smooth=False)

    # Apply correction
    grid.z[:] += grid.y
    grid.U[:] = grid.w + grid.z

    # Post-smooth
    grid.gamma[:, :] = grid.w_H + grid.chi
    grid.vanka_sweep(post_steps,n_inner=newton_iterations,verbose=verbose,omega=omega)
    grid.gamma.fill(thklim)

def adjoint_vcycle_fas(grid,
                       verbose=False,
                       omega=cp.float32(1.0),
                       pre_steps=10,
                       post_steps=10,
                       coarse_steps=400):
    """
    FAS (Full Approximation Scheme) adjoint V-cycle.

    Solves (possibly nonlinear) adjoint equation:
        N_h(lambda_h) = f_h

    where N_h(lambda_h) is produced by compute_vjp() into grid.l, i.e.
        grid.l := N_h(grid.lambda)

    Notes
    -----
    - For linear operators with consistent restriction/prolongation, FAS collapses to a
      correction scheme (up to algebraic equivalence).
    - For nonlinear or strongly state-dependent operators (e.g. coefficients from a nonlinear
      primal), FAS can be more robust since it explicitly incorporates coarse-grid
      rediscretization error via tau.
    """
    kernels = grid.kernels

    # --- Coarsest level ---
    if grid.child is None:
        grid.vanka_sweep_adjoint(coarse_steps, omega=omega,verbose=verbose)
        return

    # =========================
    # 1) Pre-smooth on fine
    # =========================
    grid.vanka_sweep_adjoint(pre_steps, omega=omega,verbose=verbose)

    # =========================
    # 2) Fine residual: r_h = f_h - N_h(lambda_h)
    # =========================
    # Evaluate N_h(lambda_h) into grid.l
    grid.compute_vjp(use_mask=True)     # requires: uses grid.lambda_* as input, writes grid.l_*
    # Residual storage (you already have r_adj_* arrays)
    grid.r_adj_u[:] = grid.f_adj_u - grid.l_u
    grid.r_adj_v[:] = grid.f_adj_v - grid.l_v
    grid.r_adj_H[:] = grid.f_adj_H - grid.l_H

    # =========================
    # 3) Restrict lambda to coarse as INITIAL GUESS: lambda_2h^0 = R lambda_h
    # =========================
    # (This is the key FAS difference vs CS; you must not start coarse solve from 0 unless you intend to.)
    restrict_vfacet(grid.lambda_u, kernels, u_coarse=grid.child.lambda_u)
    restrict_hfacet(grid.lambda_v, kernels, v_coarse=grid.child.lambda_v)
    restrict_cell_centered(grid.lambda_H, kernels, f_coarse=grid.child.lambda_H)

    grid.child.Lambda0[:] = grid.child.Lambda[:]

    # =========================
    # 4) Build coarse RHS via tau-correction:
    #    f_2h = R f_h + tau_2h
    #    tau_2h = N_2h(R lambda_h) - R N_h(lambda_h)
    # =========================

    # 4a) Compute R f_h
    restrict_vfacet(grid.f_adj_u, kernels, u_coarse=grid.child.f_adj_u)
    restrict_hfacet(grid.f_adj_v, kernels, v_coarse=grid.child.f_adj_v)
    restrict_cell_centered(grid.f_adj_H, kernels, f_coarse=grid.child.f_adj_H)

    # 4b) Compute R N_h(lambda_h) = R l_h
    #     (we already computed l_h above)
    #     store into child scratch arrays (need scratch!)
    #     We'll call them child.Rl_* here.
    restrict_vfacet(grid.l_u, kernels, u_coarse=grid.child.Rl_u)
    restrict_hfacet(grid.l_v, kernels, v_coarse=grid.child.Rl_v)
    restrict_cell_centered(grid.l_H, kernels, f_coarse=grid.child.Rl_H)

    # 4c) Compute N_2h(R lambda_h) on coarse, using the coarse operator:
    #     child.compute_vjp() uses child.lambda_* and writes child.l_*
    grid.child.compute_vjp(use_mask=False)   # writes child.l_* = N_2h(child.lambda_*)  [IMPORTANT]

    # 4d) tau_2h = child.l - child.Rl
    #     then f_2h += tau_2h
    grid.child.f_adj_u[:] += (grid.child.l_u - grid.child.Rl_u)
    grid.child.f_adj_v[:] += (grid.child.l_v - grid.child.Rl_v)
    grid.child.f_adj_H[:] += (grid.child.l_H - grid.child.Rl_H)

    # =========================
    # 5) Recurse on coarse: solve N_2h(lambda_2h) = f_2h
    # =========================
    adjoint_vcycle_fas(grid.child,
                       verbose=verbose,
                       omega=omega,
                       pre_steps=pre_steps,
                       post_steps=post_steps,
                       coarse_steps=coarse_steps)

    # =========================
    # 6) Prolongate CORRECTION (difference) and apply:
    #    lambda_h <- lambda_h + P( lambda_2h - R lambda_h )
    # =========================

    # Form coarse correction delta_2h = lambda_2h(new) - lambda_2h^0
    # Need coarse scratch arrays for delta_* (or reuse existing)
    grid.child.delta_u[:] = grid.child.lambda_u - grid.child.lambda_u0
    grid.child.delta_v[:] = grid.child.lambda_v - grid.child.lambda_v0
    grid.child.delta_H[:] = grid.child.lambda_H - grid.child.lambda_H0

    # Prolongate delta_2h to fine into grid.z_*
    grid.z_u.fill(0.0); grid.z_v.fill(0.0); grid.z_H.fill(0.0)
    prolongate_vfacet(grid.child.delta_u, kernels, u_fine=grid.z_u)
    prolongate_hfacet(grid.child.delta_v, kernels, v_fine=grid.z_v)
    prolongate_cell_centered(grid.child.delta_H, kernels, H_fine=grid.z_H, smooth=True)

    # Apply fine correction
    grid.lambda_u[:] += grid.z_u
    grid.lambda_v[:] += grid.z_v
    grid.lambda_H[:] += grid.z_H

    # =========================
    # 7) Post-smooth
    # =========================
    grid.vanka_sweep_adjoint(post_steps, omega=omega, verbose=verbose)



def adjoint_vcycle(grid,verbose=False,omega=cp.float32(1.0),pre_steps=10,post_steps=10,coarse_steps=400):
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
        grid.vanka_sweep_adjoint(coarse_steps, omega=omega,verbose=verbose)
        return

    # Pre-smooth
    grid.vanka_sweep_adjoint(pre_steps, omega=omega,verbose=verbose)

    # Compute adjoint residual
    grid.r_adj[:] = grid.f_adj[:]
    grid.compute_vjp(use_mask=False)
    grid.r_adj[:] -= grid.l

    restrict_solution(grid)

    # Set up coarse RHS
    grid.child.f_adj.fill(0.0)
    restrict_vfacet(grid.r_adj_u, kernels, u_coarse=grid.child.f_adj_u)
    restrict_hfacet(grid.r_adj_v, kernels, v_coarse=grid.child.f_adj_v)
    restrict_cell_centered(grid.r_adj_H, kernels, f_coarse=grid.child.f_adj_H)

    # Recursive call
    adjoint_vcycle(grid.child,verbose=verbose,omega=omega,pre_steps=pre_steps,post_steps=post_steps,coarse_steps=coarse_steps)

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
    grid.vanka_sweep_adjoint(post_steps, omega=omega, verbose=verbose)
    
