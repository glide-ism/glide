"""
Grid hierarchy for multigrid ice sheet solver.

Implements a MAC (marker-and-cell) staggered grid with:
- u velocities on vertical faces: shape (ny, nx+1)
- v velocities on horizontal faces: shape (ny+1, nx)
- H thickness and other scalars on cell centers: shape (ny, nx)
"""

import cupy as cp


class Grid:
    """
    Single level of the multigrid hierarchy.

    Manages state vectors, parameters, and kernel dispatch for one grid level.
    Supports both forward simulation and adjoint-based inverse modeling.

    Parameters
    ----------
    ny, nx : int
        Grid dimensions (number of cells in y and x)
    dx : float
        Grid spacing (assumed isotropic)
    dt : float
        Time step
    kernels : module
        Compiled CUDA kernel module
    parent : Grid, optional
        Parent (finer) grid in hierarchy
    n : float
        Glen's flow law exponent (default 3.0)
    eps_reg : float
        Strain rate regularization (default 1e-5)
    """

    def __init__(self, ny, nx, dx, dt, kernels, parent=None,
                 n=cp.float32(3.0), eps_reg=cp.float32(1e-5)):
        self.parent = parent
        self.child = None
        self.kernels = kernels

        self.dx = cp.float32(dx)
        self.dt = cp.float32(dt)
        self.ny = ny
        self.nx = nx

        # Degrees of freedom
        self.nu = ny * (nx + 1)
        self.nv = (ny + 1) * nx
        self.nU = self.nu + self.nv
        self.nh = ny * nx
        self.n_total = self.nu + self.nv + self.nh

        # Physics parameters
        self.n = cp.float32(n)
        self.eps_reg = cp.float32(eps_reg)

        # Allocate state and work arrays
        self._allocate_arrays()

    def _allocate_arrays(self):
        """Allocate GPU arrays for state, residuals, and work vectors."""
        ny, nx = self.ny, self.nx

        # Previous thickness (for time stepping)
        self.H_prev = cp.zeros((ny, nx), dtype=cp.float32)

        # Primary state vector [u, v, H]
        self.U = cp.zeros(self.n_total, dtype=cp.float32)
        self.u, self.v, self.H = self._vec_to_fields(self.U)

        # Perturbation vector (for JVP)
        self.d_U = cp.zeros(self.n_total, dtype=cp.float32)
        self.d_u, self.d_v, self.d_H = self._vec_to_fields(self.d_U)

        # Update/correction vector
        self.delta_U = cp.zeros(self.n_total, dtype=cp.float32)
        self.delta_u, self.delta_v, self.delta_H = self._vec_to_fields(self.delta_U)

        # Adjoint state vector
        self.Lambda = cp.zeros(self.n_total, dtype=cp.float32)
        self.lambda_u, self.lambda_v, self.lambda_H = self._vec_to_fields(self.Lambda)

        self.Lambda_out = cp.zeros(self.n_total, dtype=cp.float32)
        self.lambda_u_out, self.lambda_v_out, self.lambda_H_out = self._vec_to_fields(self.Lambda_out)

        # RHS vector
        self.f = cp.zeros(self.n_total, dtype=cp.float32)
        self.f_u, self.f_v, self.f_H = self._vec_to_fields(self.f)

        # Adjoint RHS
        self.f_adj = cp.zeros(self.n_total, dtype=cp.float32)
        self.f_adj_u, self.f_adj_v, self.f_adj_H = self._vec_to_fields(self.f_adj)

        # Operator evaluation F(U)
        self.F = cp.zeros(self.n_total, dtype=cp.float32)
        self.F_u, self.F_v, self.F_H = self._vec_to_fields(self.F)

        # Zero vector for residual computation
        self.Z = cp.zeros(self.n_total, dtype=cp.float32)
        self.Z_u, self.Z_v, self.Z_H = self._vec_to_fields(self.Z)

        # Residual vector r = f - F(U)
        self.r = cp.zeros(self.n_total, dtype=cp.float32)
        self.r_u, self.r_v, self.r_H = self._vec_to_fields(self.r)

        # Adjoint residual
        self.r_adj = cp.zeros(self.n_total, dtype=cp.float32)
        self.r_adj_u, self.r_adj_v, self.r_adj_H = self._vec_to_fields(self.r_adj)

        # JVP output
        self.j = cp.zeros(self.n_total, dtype=cp.float32)
        self.j_u, self.j_v, self.j_H = self._vec_to_fields(self.j)

        # VJP output
        self.l = cp.zeros(self.n_total, dtype=cp.float32)
        self.l_u, self.l_v, self.l_H = self._vec_to_fields(self.l)

        # Work vectors for multigrid
        self.w = cp.zeros(self.n_total, dtype=cp.float32)
        self.w_u, self.w_v, self.w_H = self._vec_to_fields(self.w)

        self.y = cp.zeros(self.n_total, dtype=cp.float32)
        self.y_u, self.y_v, self.y_H = self._vec_to_fields(self.y)

        self.z = cp.zeros(self.n_total, dtype=cp.float32)
        self.z_u, self.z_v, self.z_H = self._vec_to_fields(self.z)

        # Constraint work arrays
        self.chi = cp.zeros((ny, nx), dtype=cp.float32)
        self.phi = cp.zeros((ny, nx), dtype=cp.float32)

        # Physical parameters (cell-centered)
        self.bed = cp.zeros((ny, nx), dtype=cp.float32)
        self.beta = cp.zeros((ny, nx), dtype=cp.float32)
        self.B = cp.zeros((ny, nx), dtype=cp.float32)
        self.smb = cp.zeros((ny, nx), dtype=cp.float32)
        self.mask = cp.zeros((ny, nx), dtype=cp.float32)
        self.error_mask = cp.zeros((ny, nx), dtype=cp.float32)
        self.gamma = cp.zeros((ny, nx), dtype=cp.float32)

    def _vec_to_fields(self, x):
        """Create field views into a monolithic state vector."""
        u = x[:self.nu].reshape(self.ny, self.nx + 1)
        v = x[self.nu:self.nU].reshape(self.ny + 1, self.nx)
        H = x[self.nU:].reshape(self.ny, self.nx)
        return u, v, H

    def spawn_child(self):
        """Create a coarser child grid (2x coarsening)."""
        child = Grid(
            self.ny // 2, self.nx // 2,
            self.dx * 2, self.dt,
            self.kernels,
            parent=self,
            n=self.n,
            eps_reg=self.eps_reg
        )
        self.child = child
        return child

    def _kernel_config(self):
        """Return (grid_size, block_size, stride, halo) for kernels."""
        block_size = (16, 16)
        stride = 14
        halo = 1
        grid_size = (self.nx // stride + 1, self.ny // stride + 1)
        return grid_size, block_size, stride, halo

    def compute_residual(self, return_fischer_burmeister=False):
        """Compute residual r = f - F(U)."""
        kernel = self.kernels.ice.get_function('compute_residual')
        grid_size, block_size, stride, halo = self._kernel_config()

        kernel(grid_size, block_size,
               (self.r_u, self.r_v, self.r_H,
                self.u, self.v, self.H,
                self.f_u, self.f_v, self.f_H,
                self.bed, self.B, self.beta,
                self.mask, self.gamma,
                self.n, self.eps_reg,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo))

        if return_fischer_burmeister:
            a = self.r_H
            b = self.H - self.gamma
            return a + b - cp.sqrt(a**2 + b**2)

    def compute_F(self):
        """Compute F(U) (operator evaluation without RHS)."""
        kernel = self.kernels.ice.get_function('compute_residual')
        grid_size, block_size, stride, halo = self._kernel_config()

        kernel(grid_size, block_size,
               (self.F_u, self.F_v, self.F_H,
                self.u, self.v, self.H,
                self.Z_u, self.Z_v, self.Z_H,
                self.bed, self.B, self.beta,
                self.mask, self.gamma,
                self.n, self.eps_reg,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo))

    def compute_jvp(self):
        """Compute Jacobian-vector product J @ d_U."""
        kernel = self.kernels.ice.get_function('compute_jvp')
        grid_size, block_size, stride, halo = self._kernel_config()

        kernel(grid_size, block_size,
               (self.j_u, self.j_v, self.j_H,
                self.u, self.v, self.H,
                self.d_u, self.d_v, self.d_H,
                self.bed, self.B, self.beta,
                self.mask, self.gamma,
                self.n, self.eps_reg,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo))

    def compute_vjp(self):
        """Compute vector-Jacobian product Lambda^T @ J."""
        kernel = self.kernels.ice.get_function('compute_vjp')
        grid_size, block_size, stride, halo = self._kernel_config()

        self.l.fill(0.0)
        kernel(grid_size, block_size,
               (self.l_u, self.l_v, self.l_H,
                self.u, self.v, self.H,
                self.lambda_u, self.lambda_v, self.lambda_H,
                self.bed, self.B, self.beta,
                self.mask, self.gamma,
                self.n, self.eps_reg,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo))

    def vanka_smooth(self, color, omega=cp.float32(0.5), n_inner=1):
        """Apply one Vanka smoother pass (red-black)."""
        kernel = self.kernels.ice.get_function('vanka_smooth')
        grid_size, block_size, stride, halo = self._kernel_config()

        kernel(grid_size, block_size,
               (self.delta_u, self.delta_v, self.delta_H, self.mask,
                self.u, self.v, self.H,
                self.f_u, self.f_v, self.f_H,
                self.bed, self.B, self.beta, self.gamma,
                self.n, self.eps_reg,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo,
                color, n_inner, omega))

    def vanka_smooth_local(self, color, omega=cp.float32(0.5), n_inner=1):
        """Apply Vanka smoother only where error_mask is set."""
        kernel = self.kernels.ice.get_function('vanka_smooth_local')
        grid_size, block_size, stride, halo = self._kernel_config()

        kernel(grid_size, block_size,
               (self.delta_u, self.delta_v, self.delta_H, self.error_mask,
                self.u, self.v, self.H,
                self.f_u, self.f_v, self.f_H,
                self.bed, self.B, self.beta, self.gamma,
                self.n, self.eps_reg,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo,
                color, n_inner, omega))

    def vanka_smooth_adjoint(self, color, omega=cp.float32(0.5)):
        """Apply adjoint Vanka smoother pass."""
        kernel = self.kernels.ice.get_function('vanka_smooth_adjoint')
        grid_size, block_size, stride, halo = self._kernel_config()

        self.Lambda_out[:] = self.Lambda[:]
        kernel(grid_size, block_size,
               (self.lambda_u_out, self.lambda_v_out, self.lambda_H_out,
                self.lambda_u, self.lambda_v, self.lambda_H,
                self.mask,
                self.r_adj_u, self.r_adj_v, self.r_adj_H,
                self.u, self.v, self.H,
                self.bed, self.B, self.beta, self.gamma,
                self.n, self.eps_reg,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo,
                color, omega))
        self.Lambda[:] = self.Lambda_out[:]

    def vanka_sweep(self, n_iter, n_inner=10, omega=cp.float32(0.5)):
        """Perform n_iter red-black Vanka smoothing sweeps."""
        for _ in range(n_iter):
            self.delta_U.fill(0.0)
            self.vanka_smooth(0, omega=cp.float32(1.0), n_inner=n_inner)
            self.vanka_smooth(1, omega=cp.float32(1.0), n_inner=n_inner)
            self.U[:] += omega * self.delta_U

    def vanka_sweep_local(self, n_iter, n_inner=10, omega=cp.float32(0.5)):
        """Perform local Vanka sweeps only where error_mask is set."""
        for _ in range(n_iter):
            self.delta_U.fill(0.0)
            self.vanka_smooth_local(0, omega=cp.float32(1.0), n_inner=n_inner)
            self.vanka_smooth_local(1, omega=cp.float32(1.0), n_inner=n_inner)
            self.U[:] += omega * self.delta_U

    def vanka_sweep_adjoint(self, n_iter, omega=cp.float32(0.5)):
        """Perform n_iter adjoint Vanka smoothing sweeps."""
        for _ in range(n_iter):
            self.r_adj[:] = self.f_adj[:]
            self.compute_vjp()
            self.r_adj[:] -= self.l
            self.vanka_smooth_adjoint(0, omega=omega)

            self.r_adj[:] = self.f_adj[:]
            self.compute_vjp()
            self.r_adj[:] -= self.l
            self.vanka_smooth_adjoint(1, omega=omega)

    def compute_mask(self, tol=1e-1):
        """Compute active set mask for thickness constraints."""
        self.mask[:, :] = (self.H <= (self.gamma + tol)).astype(cp.float32)

    def compute_grad_beta(self):
        """Compute gradient of objective w.r.t. beta via adjoint."""
        kernel = self.kernels.ice.get_function('compute_grad_beta')
        grid_size, block_size, stride, halo = self._kernel_config()

        grad_beta = cp.zeros((self.ny, self.nx), dtype=cp.float32)
        self.compute_mask()

        kernel(grid_size, block_size,
               (grad_beta,
                self.u, self.v, self.H,
                self.lambda_u, self.lambda_v, self.lambda_H,
                self.bed, self.B, self.beta,
                self.mask, self.gamma,
                self.n, self.eps_reg,
                self.dx, self.dt,
                self.ny, self.nx, stride, halo))

        self.mask.fill(0.0)
        return grad_beta
