"""
Core ice physics API.

Provides the IcePhysics class that wraps the forward model and adjoint
computations into a clean interface.
"""

import cupy as cp
from .grid import Grid
from .multigrid import Multigrid, FASCDSolver,FASAdjointSolver
from .enthalpy import EnthalpyOperators, T_MELT

class IceDynamics:
    def __init__(self,mg=None,
            n_levels=None,grid=None,
            ny=None,nx=None,dx=None,
            x0=cp.float32(0.0),y0=cp.float32(0.0),crs=None,
            stress_scheme='molho'):
        # stress_scheme: 'molho' (two-field, shear-resolving) or 'ssa'
        # (deformational components pinned to zero). When an existing mg or
        # grid is supplied, its stress_scheme governs.
        if mg is not None:
            self.mg = mg
        elif grid is not None and n_levels is not None:
            self.mg = Multigrid(n_levels,finest_grid=grid)
        elif ny and nx and dx and n_levels:
            self.mg = Multigrid(n_levels,ny=ny,nx=nx,dx=dx,
                   x0=x0,y0=y0,crs=crs,stress_scheme=stress_scheme)
        else:
            raise ValueError('Must supply either (a) a multigrid object \
                              (b) a grid and number of levels \
                              (c) ny/nx/dx and number of levels')

        self._forward_solver = None
        self._adjoint_solver = None
        self.top_level = 0

        self._post_forward_hooks = []

    @property
    def forward_solver(self):
        if self._forward_solver is None:
            self._forward_solver = FASCDSolver(self.mg)
        return self._forward_solver
    
    @property
    def adjoint_solver(self):
        if self._adjoint_solver is None:
            self._adjoint_solver = FASAdjointSolver(self.mg)
        return self._adjoint_solver

    def set_top_level(self,level):
        self.top_level = level

    def register_post_forward_hook(self,hook):
        self._post_forward_hooks.append(hook)

    def forward(self,t,dt,update_geometry=True):
        self.forward_solver.solve(dt,start_level=self.top_level)
        if update_geometry:
            self.mg.levels[self.top_level].state.H_prev.data[:,:] = (
                self.mg.levels[self.top_level].state.H.data[:,:]
            )
        for f in self._post_forward_hooks:
            f(t+dt)

    def backward(self,t,dt,dJdu=None,dJdv=None,dJdud=None,dJdvd=None,dJdH=None,
            compute_beta_grad=True,compute_bed_grad=True,
            compute_H_prev_grad=True,compute_smb_grad=True):
        if dJdu is not None:
            self.mg.levels[self.top_level].adjoint_operators.f_u[:,:] = -dJdu
        else:
            self.mg.levels[self.top_level].adjoint_operators.f_u.fill(0.0)            
        if dJdud is not None:
            self.mg.levels[self.top_level].adjoint_operators.f_ud[:,:] = -dJdud
        else:
            self.mg.levels[self.top_level].adjoint_operators.f_ud.fill(0.0)            
        if dJdv is not None:
            self.mg.levels[self.top_level].adjoint_operators.f_v[:,:] = -dJdv
        else:
            self.mg.levels[self.top_level].adjoint_operators.f_v.fill(0.0)
        if dJdvd is not None:
            self.mg.levels[self.top_level].adjoint_operators.f_vd[:,:] = -dJdvd
        else:
            self.mg.levels[self.top_level].adjoint_operators.f_vd.fill(0.0)            
        if dJdH is not None:
            self.mg.levels[self.top_level].adjoint_operators.f_H[:,:] = -dJdH
        else:    
            self.mg.levels[self.top_level].adjoint_operators.f_H.fill(0.0)

        converged = self.adjoint_solver.solve(dt,start_level=self.top_level)
        self.mg.levels[self.top_level].adjoint_operators.compute_gradient_beta()
        self.mg.levels[self.top_level].adjoint_operators.compute_gradient_bed()
        self.mg.levels[self.top_level].adjoint_operators.compute_gradient_H_prev(dt)
        self.mg.levels[self.top_level].adjoint_operators.compute_gradient_smb()


        return converged


class ThermalModel:
    """
    Enthalpy solver for coupled momentum/thermal simulations (ported from the
    DIVA-tree ThermalModel; MOLHO / SSA only).

    Wraps EnthalpyOperators into a step(dt) interface that:
    1. Rebuilds the 3D velocity from the momentum solution on the grid
    2. Solves the enthalpy advection-diffusion equation
    3. Feeds the updated rate factor B back to the rheology

    Parameters
    ----------
    grid : Grid
        The grid the momentum solve runs on (shared with IceDynamics).
    nz : int
        Number of sigma levels.
    n_smooth : int
        Maximum number of column smoothing sweeps per time step.
    update_rheology : bool
        If True, update B after each step from the rate factor.
    frictional_heating, strain_heating : bool
        Basal frictional heat flux / deformational heating from the resolved
        vertical shear (zero under SSA).
    rho_i : float
        Ice density (kg/m^3), used by the enthalpy PDE, the pressure-melting
        point and the B unit conversion. Pass the momentum solver's value.
    mg, level : Multigrid, int, optional
        When given, B is pushed with mg.rheology.B.set(B, start_level=level),
        i.e. restricted onto every coarser FAS level; otherwise only grid's
        own B is written (single-grid use), and coarse levels keep a stale B.
    weighting : 'shear' | 'mean'
        How A(sigma) collapses to one B per column (get_arrhenius_factor).
    enhancement : (ny, nx) array or None
        Multiplier on the collapsed A (A_eff <- E A_eff); None = 1.
    thin_B : float or None
        B (glide units) written where H < h_thin instead of the thermal
        value. Those columns are clamped to the surface temperature, i.e.
        cold and stiff (B up to 2x the interior's), and on ice-free cells
        (H = thklim) that stiffness only couples the membrane stencil to
        nothing: next to soft temperate outlets the contrast drove the 1 km
        momentum solve to NaN in the first 25-yr step (Greenland, 2026-09-27)
        while the same B with the thin cells at the isothermal value
        converged. None keeps the thermal value everywhere.
    """

    SEC_PER_YR = 365.25 * 86400.0

    def __init__(self, grid, nz=21, n_smooth=10,
                 update_rheology=True, frictional_heating=True,
                 strain_heating=True, rho_i=917.0,
                 mg=None, level=0, weighting='shear', enhancement=None,
                 thin_B=None):
        self.ops = EnthalpyOperators(grid, nz=nz, rho_i=rho_i)
        self.n_smooth = n_smooth
        self.update_rheology = update_rheology
        self.frictional_heating = frictional_heating
        self.strain_heating = strain_heating
        self.rho_i = rho_i
        self.g = 9.81
        self.mg = mg
        self.level = level
        self.weighting = weighting
        self.enhancement = enhancement
        self.thin_B = thin_B

    def initialize(self, T_surface, T_field=None, Q_geo=None):
        """
        Set initial conditions and boundary data.

        Parameters
        ----------
        T_surface : array-like, shape (ny, nx) or scalar
            Surface temperature in Kelvin (Dirichlet BC).
        T_field : array-like, shape (ny, nx, nz) or scalar, optional
            Initial 3D temperature. Defaults to T_surface everywhere.
        Q_geo : array-like, shape (ny, nx) or scalar, optional
            Geothermal heat flux in W/m^2. Defaults to 0.
        """
        if T_field is None:
            T_field = T_surface
        self.ops.initialize_from_temperature(T_field)
        self.set_surface_temperature(T_surface)
        if Q_geo is not None:
            self.ops.enthalpy_forcing.Q_geo[:] = cp.asarray(
                Q_geo, dtype=cp.float32)

    def set_surface_temperature(self, T_surface):
        """Surface Dirichlet BC from a temperature (K, capped at T_melt)."""
        T = cp.asarray(T_surface, dtype=cp.float32)
        if T.ndim == 0:
            T = cp.full((self.ops.grid.ny, self.ops.grid.nx), T.item(), dtype=cp.float32)
        self.ops.set_surface_enthalpy_from_temperature(T)

    def pre_momentum(self):
        """Snapshot E and H before the momentum step.

        Must be called BEFORE model.forward() so that the conservative
        time derivative rho_i*(H*E - H_prev*E_prev)/dt uses the
        correct pre-step thickness.
        """
        ops = self.ops
        ops.enthalpy_state.E_prev[:] = ops.enthalpy_state.E
        ops.H_prev[:] = ops.grid.state.H.data

    def step(self, dt):
        """
        Advance enthalpy by one time step.

        Call pre_momentum() before the momentum step, then step()
        after. This ensures H_prev captures the pre-momentum thickness.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        """
        ops = self.ops

        # Sync velocity from the momentum solution (m/yr).
        ops.broadcast_velocity()

        # Compute omega BEFORE the m/yr -> m/s velocity conversion.
        # The omega kernel evaluates div(Hu) in m/yr to match the
        # momentum solver's float32 arithmetic exactly -- this avoids
        # catastrophic cancellation from scale-dependent rounding
        # differences between the m/yr and m/s evaluations.
        self._compute_omega(dt)

        sec_per_yr = cp.float32(self.SEC_PER_YR)
        ops.enthalpy_velocity.u3d /= sec_per_yr
        ops.enthalpy_velocity.v3d /= sec_per_yr

        if self.frictional_heating:
            self._compute_frictional_heating()
        if self.strain_heating:
            self._compute_strain_heating()

        # Precompute forcing array (E_prev and H_prev set by pre_momentum)
        ops.set_rhs(snapshot=False)

        # Enthalpy solve
        ops.column_sweep(dt, self.n_smooth)

        if self.update_rheology:
            self.push_rheology()

    def rheology_B(self):
        """B in GLIDE's year-based head units: B_SI / (rho_i g sec_per_yr^{1/n})."""
        B_si = self.ops.get_arrhenius_factor(weighting=self.weighting,
                                             enhancement=self.enhancement)
        B = (B_si / cp.float32(self.B_scale)).astype(cp.float32)
        if self.thin_B is not None:
            thin = self.ops.grid.state.H.data < self.ops.enthalpy_forcing.h_thin.value
            B = cp.where(thin, cp.float32(self.thin_B), B)
        return B

    def push_rheology(self):
        """Write the current B to the momentum solver (all coarser levels too when mg is set)."""
        B = self.rheology_B()
        if self.mg is not None:
            self.mg.rheology.B.set(B, start_level=self.level)
        else:
            self.ops.grid.rheology.B.data[:] = B

    def _drag_coefficient(self):
        """beta * xi^p, the effective drag of stress.cu (xi^p pinned to 0 where xi = 0)."""
        grid = self.ops.grid
        sliding = grid.sliding
        xi = grid.state.xi.data
        p = float(sliding.p.value)
        xip = cp.where(xi > 0.0, cp.power(cp.maximum(xi, 1e-30), p), 0.0)
        return sliding.beta.data * xip

    def _compute_frictional_heating(self):
        """Basal frictional heat flux Q_fh = tau_b . u_b  (W/m^2).

        In GLIDE's head units the basal drag is tau_b/(rho g) = beta xi^p
        (|u|^2 + u_reg)^((m-1)/2) |u| with |u| in m/yr and u_reg in (m/yr)^2
        (stress.cu), so

            Q_fh = (rho g / SEC_PER_YR) beta xi^p (|u|^2 + u_reg)^((m-1)/2) |u|^2.

        Gated by the same xi^p as the momentum drag so floating margins get no
        spurious friction.
        """
        grid = self.ops.grid
        sliding = grid.sliding
        vel = self.ops.enthalpy_velocity
        spy = cp.float32(self.SEC_PER_YR)
        u_bed = vel.u3d[0, :, :] * spy  # m/s -> m/yr, shape (ny, nx+1)
        v_bed = vel.v3d[0, :, :] * spy  # m/s -> m/yr, shape (ny+1, nx)

        u_cell = 0.5 * (u_bed[:, 1:] + u_bed[:, :-1])
        v_cell = 0.5 * (v_bed[1:, :] + v_bed[:-1, :])
        speed2 = u_cell**2 + v_cell**2
        m = float(sliding.m.value)
        u_reg = float(sliding.u_reg.value)
        scale = cp.float32(self.rho_i * self.g / self.SEC_PER_YR)  # head->Pa, /yr->/s
        self.ops.enthalpy_forcing.Q_fh[:] = (
            scale * self._drag_coefficient() * (speed2 + u_reg) ** ((m - 1.0) / 2.0) * speed2)

    def _compute_strain_heating(self):
        """Fill phi_strain [W/m^3] with the deformational (shear) heating.

        With the MOLHO profile u(sigma) = u_b + (u_s - u_b)*psi(sigma),
        psi = 1 - (1 - sigma)^(n+1), the shear strain rate is
        du/dz = (u_s - u_b)/H * (n+1)(1 - sigma)^n, and the shear stress
        tau_xz = tau_b * (1 - sigma) (basal drag distributed linearly), so

            phi_shear(sigma) = tau_b * (u_s - u_b) * (n+1)/H * (1 - sigma)^(n+1),

        peaked at the bed. tau_b is the physical basal drag [Pa]; u_s, u_b are
        cell speeds [m/yr]. Zero under SSA (u_s = u_b). Membrane dissipation is
        neglected.
        """
        grid = self.ops.grid
        ph = self.ops.enthalpy_forcing.phi_strain
        if getattr(grid, 'stress_scheme', 'ssa') != 'molho' or grid.state.ud is None:
            ph.fill(0.0)
            return
        sliding = grid.sliding
        n = float(grid.rheology.n.value)
        m = float(sliding.m.value)
        u_reg = float(sliding.u_reg.value)
        u = grid.state.u.data; v = grid.state.v.data
        ud = grid.state.ud.data; vd = grid.state.vd.data
        ubx = u - ud; uby = v - vd; usx = u + ud / (n + 1.0); usy = v + vd / (n + 1.0)
        ub = cp.hypot(0.5 * (ubx[:, 1:] + ubx[:, :-1]), 0.5 * (uby[1:] + uby[:-1]))
        us = cp.hypot(0.5 * (usx[:, 1:] + usx[:, :-1]), 0.5 * (usy[1:] + usy[:-1]))
        H = cp.maximum(grid.state.H.data, 1.0)
        tau_b = (cp.float32(self.rho_i * self.g) * self._drag_coefficient()
                 * (ub**2 + u_reg) ** ((m - 1.0) / 2.0) * ub)            # Pa
        dU = cp.maximum(us - ub, 0.0) / cp.float32(self.SEC_PER_YR)     # m/s
        pref = tau_b * dU * (n + 1.0) / H                              # W/m^3 at the bed
        for k in range(self.ops.nz):
            s = float(self.ops.sigma[k])
            ph[:, :, k] = pref * (1.0 - s) ** (n + 1.0)

    def _compute_omega(self, dt):
        """Compute omega from the actual thickness change.

        Uses (H_new - H_prev) / dt as the column dH/dt, ensuring exact
        consistency between the conservative enthalpy equation and the
        realized mass balance from the momentum step. Evaluated in m/yr
        (the momentum solver's native scale) so the LF mass flux matches
        its float32 arithmetic; the result is converted to m/s.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        """
        ops = self.ops
        sec_per_yr = cp.float32(self.SEC_PER_YR)
        dt_yr = cp.float32(dt / self.SEC_PER_YR)
        dh_dt_yr = (ops.grid.state.H.data - ops.H_prev) / dt_yr
        ops.compute_omega(dh_dt_yr)
        ops.enthalpy_velocity.omega /= sec_per_yr

    def state_dict(self):
        """Restartable thermal state (host arrays)."""
        return {'E': cp.asnumpy(self.ops.enthalpy_state.E),
                'E_surface': cp.asnumpy(self.ops.enthalpy_forcing.E_surface),
                'Q_geo': cp.asnumpy(self.ops.enthalpy_forcing.Q_geo)}

    def load_state_dict(self, d):
        self.ops.enthalpy_state.E[:] = cp.asarray(d['E'], dtype=cp.float32)
        self.ops.enthalpy_state.E_prev[:] = self.ops.enthalpy_state.E
        self.ops.enthalpy_forcing.E_surface[:] = cp.asarray(d['E_surface'], dtype=cp.float32)
        self.ops.enthalpy_forcing.Q_geo[:] = cp.asarray(d['Q_geo'], dtype=cp.float32)
        self.ops.H_prev[:] = self.ops.grid.state.H.data

    @property
    def B_scale(self):
        """Conversion factor: B_glide = B_SI / B_scale."""
        n = float(self.ops.grid.rheology.n.value)
        return self.rho_i * self.g * self.SEC_PER_YR ** (1.0 / n)

    @property
    def temperature(self):
        """3D temperature field, shape (ny, nx, nz)."""
        return self.ops.get_temperature()

    @property
    def water_content(self):
        """3D water content field, shape (ny, nx, nz)."""
        return self.ops.get_water_content()

    @property
    def enthalpy(self):
        """3D enthalpy field, shape (ny, nx, nz)."""
        return self.ops.enthalpy_state.E

    @property
    def sigma(self):
        """Sigma level positions, shape (nz,)."""
        return self.ops.sigma
