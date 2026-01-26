"""
Core ice physics API.

Provides the IcePhysics class that wraps the forward model and adjoint
computations into a clean interface.
"""

import cupy as cp
from .grid import Grid
from .kernels import get_kernels, restrict_cell_centered
from .solver import (fascd_vcycle, fascd_vcycle_frozen, adjoint_vcycle,
                     restrict_parameters_to_hierarchy, restrict_frozen_fields_to_hierarchy)


# Physical constants
RHO_ICE = 917.0  # kg/m^3
G = 9.81  # m/s^2


class IcePhysics:
    """
    GPU-accelerated shallow shelf approximation ice sheet model.

    Provides forward simulation and adjoint-based gradient computation
    for the coupled momentum + mass conservation system.

    Parameters
    ----------
    ny, nx : int
        Grid dimensions (number of cells)
    dx : float
        Grid spacing in meters
    n_levels : int
        Number of multigrid levels (default 5)
    n : float
        Glen's flow law exponent (default 3.0)
    eps_reg : float
        Strain rate regularization (default 1e-5)
    thklim : float
        Minimum thickness constraint (default 0.1)

    Examples
    --------
    >>> physics = IcePhysics(ny=512, nx=512, dx=1500.0)
    >>> physics.set_geometry(bed, thickness)
    >>> physics.set_parameters(B=B, beta=beta, smb=smb)
    >>> u, v, H = physics.forward(dt=10.0, n_vcycles=3)
    """

    def __init__(self, ny, nx, dx, n_levels=5, n=3.0, eps_reg=1e-5, thklim=0.1, water_drag=1e-3,calving_rate=1.0,gl_sigmoid_c=0.1,gl_derivatives=False):
        self.ny = ny
        self.nx = nx
        self.dx = dx
        self.n_levels = n_levels
        self.n = cp.float32(n)
        self.eps_reg = cp.float32(eps_reg)
        self.thklim = cp.float32(thklim)
        self.water_drag = cp.float32(water_drag)
        self.calving_rate = cp.float32(calving_rate)
        self.gl_sigmoid_c=cp.float32(gl_sigmoid_c)
        self.gl_derivatives=gl_derivatives

        # Load kernels
        self.kernels = get_kernels()

        # Create grid hierarchy
        self._init_hierarchy()

    def _init_hierarchy(self):
        """Initialize the multigrid hierarchy."""
        self.grid = Grid(
            self.ny, self.nx, self.dx, dt=1.0,
            kernels=self.kernels,
            n=self.n, eps_reg=self.eps_reg,
            water_drag=self.water_drag,calving_rate=self.calving_rate,
            gl_sigmoid_c=self.gl_sigmoid_c,gl_derivatives=self.gl_derivatives
        )
        self.grids = [self.grid]

        for _ in range(self.n_levels - 1):
            self.grids.append(self.grids[-1].spawn_child())

    def set_geometry(self, bed, thickness):
        """
        Set the ice sheet geometry.

        Parameters
        ----------
        bed : array_like
            Bed topography (ny, nx), in meters
        thickness : array_like
            Ice thickness (ny, nx), in meters
        """
        self.grid.bed[:] = cp.asarray(bed, dtype=cp.float32)
        self.grid.H[:] = cp.asarray(thickness, dtype=cp.float32)
        self.grid.H_prev[:] = self.grid.H[:]
        self.grid.gamma.fill(self.thklim)

        # Propagate geometry to child grids
        self._propagate_geometry_to_hierarchy()

    def set_parameters(self, B=None, beta=None, smb=None):
        """
        Set physical parameters.

        Parameters
        ----------
        B : array_like, optional
            Rate factor field (ny, nx). If scalar, broadcasts to all cells.
            Units: Pa^(-n) s^(-1) (normalized by rho*g internally)
        beta : array_like, optional
            Basal friction coefficient (ny, nx)
        smb : array_like, optional
            Surface mass balance (ny, nx), in m/yr ice equivalent
        """
        if B is not None:
            B_arr = cp.asarray(B, dtype=cp.float32)
            if B_arr.ndim == 0:
                self.grid.B.fill(float(B_arr))
            else:
                self.grid.B[:] = B_arr

        if beta is not None:
            self.grid.beta[:] = cp.asarray(beta, dtype=cp.float32)

        if smb is not None:
            self.grid.smb[:] = cp.asarray(smb, dtype=cp.float32)

        # Propagate parameters to child grids
        restrict_parameters_to_hierarchy(self.grid)

    def _propagate_geometry_to_hierarchy(self):
        """Propagate geometry (bed, H, H_prev, gamma) to all child grids."""
        for i in range(len(self.grids) - 1):
            parent = self.grids[i]
            child = self.grids[i + 1]
            restrict_cell_centered(parent.bed, self.kernels, f_coarse=child.bed)
            restrict_cell_centered(parent.H, self.kernels, f_coarse=child.H)
            restrict_cell_centered(parent.H_prev, self.kernels, f_coarse=child.H_prev)
            child.gamma.fill(self.thklim)

    def forward(self, dt, n_vcycles=3, verbose=False, update_geometry=True):
        """
        Perform one forward time step.

        Solves the coupled SSA momentum equations and mass conservation
        for the ice velocity and thickness after time dt.

        Parameters
        ----------
        dt : float
            Time step in years
        n_vcycles : int
            Number of multigrid V-cycles (default 3)
        verbose : bool
            Print convergence info

        Returns
        -------
        u : cupy.ndarray
            x-velocity on vertical faces (ny, nx+1), m/yr
        v : cupy.ndarray
            y-velocity on horizontal faces (ny+1, nx), m/yr
        H : cupy.ndarray
            Ice thickness (ny, nx), m
        """
        self.grid.dt = cp.float32(dt)

        # Propagate dt to all levels
        for g in self.grids:
            g.dt = self.grid.dt

        # Set up RHS for mass equation
        self.grid.set_rhs()   #f_H[:, :] = self.grid.H_prev / self.grid.dt + self.grid.smb

        # Compute initial residual for convergence tracking
        if verbose:
            rss_H_init = self.grid.compute_residual(return_fischer_burmeister=True)
            r0 = float(cp.sqrt(
                cp.linalg.norm(self.grid.r_u)**2 +
                cp.linalg.norm(self.grid.r_v)**2 +
                cp.linalg.norm(rss_H_init)**2
            ))
            print(f"  Initial: |r| = {r0:.2e}, "
                  f"|r_u| = {float(cp.linalg.norm(self.grid.r_u)):.2e}, "
                  f"|r_v| = {float(cp.linalg.norm(self.grid.r_v)):.2e}, "
                  f"|rss_H| = {float(cp.linalg.norm(rss_H_init)):.2e}")

        # Solve
        for i in range(n_vcycles):
            fascd_vcycle(self.grid, self.thklim, finest=True)

            if verbose:
                rss_H = self.grid.compute_residual(return_fischer_burmeister=True)
                r_combined = float(cp.sqrt(
                    cp.linalg.norm(self.grid.r_u)**2 +
                    cp.linalg.norm(self.grid.r_v)**2 +
                    cp.linalg.norm(rss_H)**2
                ))
                rel = r_combined / r0 if r0 > 0 else 0.0
                print(f"  V-cycle {i}: |r|/|r0| = {rel:.2e}, "
                      f"|r_u| = {float(cp.linalg.norm(self.grid.r_u)):.2e}, "
                      f"|r_v| = {float(cp.linalg.norm(self.grid.r_v)):.2e}, "
                      f"|rss_H| = {float(cp.linalg.norm(rss_H)):.2e}")

        # Update H_prev for next time step
        if update_geometry:
            self.grid.H_prev[:] = self.grid.H[:]

        return self.grid.u, self.grid.v, self.grid.H

    def forward_frozen(self, dt, n_vcycles=3, verbose=False, update_geometry=True):
        """
        Perform one forward time step using frozen Picard coefficients.

        This variant computes eta, beta_eff, and c_eff once at the finest level
        before starting V-cycles, then restricts these fields to coarser grids.
        This ensures operator consistency across multigrid levels and can improve
        convergence when discontinuous coefficients (grounding line, calving) are
        present.

        Parameters
        ----------
        dt : float
            Time step in years
        n_vcycles : int
            Number of multigrid V-cycles (default 3)
        verbose : bool
            Print convergence info
        update_geometry : bool
            Update H_prev after solve (default True)

        Returns
        -------
        u : cupy.ndarray
            x-velocity on vertical faces (ny, nx+1), m/yr
        v : cupy.ndarray
            y-velocity on horizontal faces (ny+1, nx), m/yr
        H : cupy.ndarray
            Ice thickness (ny, nx), m
        """
        self.grid.dt = cp.float32(dt)

        # Propagate dt to all levels
        for g in self.grids:
            g.dt = self.grid.dt

        # Set up RHS for mass equation
        self.grid.f_H[:, :] = self.grid.H_prev / self.grid.dt + self.grid.smb

        #self.grid.compute_frozen_fields()
        # Compute initial residual for convergence tracking
        if verbose:
            rss_H_init = self.grid.compute_residual(return_fischer_burmeister=True,frozen=False)
            r0 = float(cp.sqrt(
                cp.linalg.norm(self.grid.r_u)**2 +
                cp.linalg.norm(self.grid.r_v)**2 +
                cp.linalg.norm(rss_H_init)**2
            ))
            print(f"  Initial: |r| = {r0:.2e}, "
                  f"|r_u| = {float(cp.linalg.norm(self.grid.r_u)):.2e}, "
                  f"|r_v| = {float(cp.linalg.norm(self.grid.r_v)):.2e}, "
                  f"|rss_H| = {float(cp.linalg.norm(rss_H_init)):.2e}")

        restrict_parameters_to_hierarchy(self.grid)
        self.grid.compute_eta_field()
        self.grid.compute_beta_eff_field()
        self.grid.compute_c_eff_field()
        # Restrict frozen fields to entire hierarchy
        restrict_frozen_fields_to_hierarchy(self.grid)
        # Solve using frozen coefficients
        for i in range(n_vcycles):
            self.grid.compute_eta_field()
            self.grid.compute_beta_eff_field()
            self.grid.compute_c_eff_field(relaxation=0.5)
            # Restrict frozen fields to entire hierarchy
            restrict_frozen_fields_to_hierarchy(self.grid)

            # Run frozen V-cycle
            fascd_vcycle_frozen(self.grid, self.thklim, finest=True)

            if verbose:
                # Use standard residual for convergence check (computes true residual)
                rss_H = self.grid.compute_residual(return_fischer_burmeister=True,frozen=True)
                r_combined = float(cp.sqrt(
                    cp.linalg.norm(self.grid.r_u)**2 +
                    cp.linalg.norm(self.grid.r_v)**2 +
                    cp.linalg.norm(rss_H)**2
                ))
                rel = r_combined / r0 if r0 > 0 else 0.0
                print(f"  V-cycle {i}: |r|/|r0| = {rel:.2e}, "
                      f"|r_u| = {float(cp.linalg.norm(self.grid.r_u)):.2e}, "
                      f"|r_v| = {float(cp.linalg.norm(self.grid.r_v)):.2e}, "
                      f"|rss_H| = {float(cp.linalg.norm(rss_H)):.2e}")

        # Update H_prev for next time step
        if update_geometry:
            self.grid.H_prev[:] = self.grid.H[:]

        return self.grid.u, self.grid.v, self.grid.H

    def adjoint(self, dL_du, dL_dv, dL_dH,n_vcycles=2):
        """
        Compute adjoint (reverse-mode AD) for gradient computation.

        Given gradients of a loss function w.r.t. velocities,
        computes gradients w.r.t. parameters (beta).

        Parameters
        ----------
        dL_du : cupy.ndarray
            Gradient of loss w.r.t. u velocity (ny, nx+1)
        dL_dv : cupy.ndarray
            Gradient of loss w.r.t. v velocity (ny+1, nx)

        Returns
        -------
        grad_beta : cupy.ndarray
            Gradient of loss w.r.t. beta (ny, nx)
        """
        # Set adjoint forcing
        self.grid.f_adj_u[:] = cp.asarray(-dL_du, dtype=cp.float32)
        self.grid.f_adj_v[:] = cp.asarray(-dL_dv, dtype=cp.float32)
        self.grid.f_adj_H[:] = cp.asarray(-dL_dH, dtype=cp.float32)

        self.grid.compute_frozen_fields()
        restrict_frozen_fields_to_hierarchy(self.grid,restrict_adjoint_viscosity=True)

        # Solve adjoint system
        for j in range(n_vcycles):
            adjoint_vcycle(self.grid,frozen=True)
            print(j,cp.linalg.norm((self.grid.l - self.grid.f_adj)))

        # Compute parameter gradient
        #return self.grid.compute_grad_beta()

    def reset_solution(self):
        """Reset velocity fields to zero."""
        self.grid.u.fill(0.0)
        self.grid.v.fill(0.0)
        self.grid.H[:] = self.grid.H_prev[:]

    def get_surface(self):
        """Compute ice surface elevation."""
        base = cp.maximum(self.grid.bed, -RHO_ICE / 1000.0 * self.grid.H)
        return self.grid.H + base

    def get_velocities_cell_centered(self):
        """Return velocities interpolated to cell centers."""
        u_c = 0.5 * (self.grid.u[:, 1:] + self.grid.u[:, :-1])
        v_c = 0.5 * (self.grid.v[1:] + self.grid.v[:-1])
        return u_c, v_c


def huber_loss(u, v, u_obs, v_obs, eps=10.0):
    """
    Compute Huber-like loss for velocity misfit.

    Parameters
    ----------
    u, v : cupy.ndarray
        Model velocities
    u_obs, v_obs : cupy.ndarray
        Observed velocities
    eps : float
        Smoothing parameter

    Returns
    -------
    loss : float
        Loss value
    """
    eps = cp.float32(eps)
    delta_u = u - u_obs
    mask_u = cp.isnan(delta_u)
    delta_u[mask_u] = 0.0
    delta_v = v - v_obs
    mask_v = cp.isnan(delta_v)
    delta_v[mask_v] = 0.0
    return cp.sqrt(delta_u**2 + eps).mean() + cp.sqrt(delta_v**2 + eps).mean()


def huber_grad(u, v, u_obs, v_obs, eps=10.0):
    """
    Compute gradient of Huber-like loss.

    Parameters
    ----------
    u, v : cupy.ndarray
        Model velocities
    u_obs, v_obs : cupy.ndarray
        Observed velocities
    eps : float
        Smoothing parameter

    Returns
    -------
    dL_du, dL_dv : cupy.ndarray
        Gradients w.r.t. velocities
    """
    eps = cp.float32(eps)
    delta_u = u - u_obs
    mask_u = cp.isnan(delta_u)
    delta_u[mask_u] = 0.0
    delta_v = v - v_obs
    mask_v = cp.isnan(delta_v)
    delta_v[mask_v] = 0.0
    n = u.size
    dL_du = delta_u / cp.sqrt(delta_u**2 + eps) / n
    dL_dv = delta_v / cp.sqrt(delta_v**2 + eps) / n
    return dL_du, dL_dv

def abs_loss(u, v, u_obs, v_obs):
    """
    Compute Huber-like loss for velocity misfit.

    Parameters
    ----------
    u, v : cupy.ndarray
        Model velocities
    u_obs, v_obs : cupy.ndarray
        Observed velocities
    eps : float
        Smoothing parameter

    Returns
    -------
    loss : float
        Loss value
    """
    u = u.astype(cp.float64)
    v = v.astype(cp.float64)
    u_obs = u_obs.astype(cp.float64)
    v_obs = v_obs.astype(cp.float64)
    delta_u = u - u_obs
    mask_u = cp.isnan(delta_u)
    delta_u[mask_u] = 0.0
    delta_v = v - v_obs
    mask_v = cp.isnan(delta_v)
    delta_v[mask_v] = 0.0
    return abs(delta_u).mean() + abs(delta_v).mean()
    #return abs(delta_u).sum() + abs(delta_v).sum()


def abs_grad(u, v, u_obs, v_obs, eps=10.0):
    """
    Compute gradient of Huber-like loss.

    Parameters
    ----------
    u, v : cupy.ndarray
        Model velocities
    u_obs, v_obs : cupy.ndarray
        Observed velocities
    eps : float
        Smoothing parameter

    Returns
    -------
    dL_du, dL_dv : cupy.ndarray
        Gradients w.r.t. velocities
    """
    eps = cp.float32(eps)
    delta_u = u - u_obs
    mask_u = cp.isnan(delta_u)
    delta_u[mask_u] = 0.0
    delta_v = v - v_obs
    mask_v = cp.isnan(delta_v)
    delta_v[mask_v] = 0.0
    n = u.size
    return cp.sign(delta_u)/n, cp.sign(delta_v)/n
    #return cp.sign(delta_u), cp.sign(delta_v)


def tikhonov_regularization(field):
    """
    Compute Tikhonov (gradient smoothness) regularization.

    Parameters
    ----------
    field : cupy.ndarray
        2D field to regularize

    Returns
    -------
    loss : float
        Regularization loss
    grad : cupy.ndarray
        Gradient of loss w.r.t. field
    """
    diff_x = field[:, 1:] - field[:, :-1]
    diff_y = field[1:, :] - field[:-1, :]

    loss = 0.5 * (cp.sum(diff_x**2) + cp.sum(diff_y**2))

    grad = cp.zeros_like(field)
    grad[:, 1:-1] -= (field[:, 2:] - 2 * field[:, 1:-1] + field[:, :-2])
    grad[:, 0] -= (field[:, 1] - field[:, 0])
    grad[:, -1] -= (field[:, -2] - field[:, -1])
    grad[1:-1, :] -= (field[2:, :] - 2 * field[1:-1, :] + field[:-2, :])
    grad[0, :] -= (field[1, :] - field[0, :])
    grad[-1, :] -= (field[-2, :] - field[-1, :])

    return float(loss), grad
