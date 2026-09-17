from dataclasses import dataclass, field, fields
import cupy as cp
import xarray as xr
from cupy.typing import NDArray
from .field import Field, SubgridField, Constant, GridEntity
from .operators import ForwardOperators, AdjointOperators

@dataclass
class State:
    u: Field | None = None
    v: Field | None = None
    ud: Field | None = None
    vd: Field | None = None
    H: Field | None = None
    H_prev: Field | None = None
    phi: Field | None = None
    xi: Field | None = None
    dxi_dH: Field | None = None      # d xi / dH (drag Jacobian); restricted, not recomputed, on coarse adjoint levels
    psi: Field | None = None
    dpsi_dH: Field | None = None     # d psi / dH of the calving flag (transport Jacobian)
    dpsi_dbed: Field | None = None   # d psi / d bed (bed gradient)
    mask: Field | None = None

    def __repr__(self):
        return f'{self.u.compact_string}\n{self.v.compact_string}\n{self.H.compact_string}\n{self.H_prev.compact_string}\n{self.phi.compact_string}\n{self.mask.compact_string}'

@dataclass
class AdjointState:
    lambda_u: Field | None = None
    lambda_v: Field | None = None
    lambda_ud: Field | None = None
    lambda_vd: Field | None = None
    lambda_H: Field | None = None

    def __repr__(self):
        return f'{self.lambda_u.compact_string}\n{self.lambda_v.compact_string}\n{self.lambda_H.compact_string}'

@dataclass
class Geometry:
    bed: SubgridField | None = None
    depth: Field | None = None
    thklim: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.1),
            name='thklim',
            units='m',
            attrs={'long_name':'minimum thickness'})
        )
    sigmoid_c: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.1),
            name='sigmoid_c',
            units='m^{-1}',
            attrs={'long_name':("smoothing factor for sigmoidal \
                                  grounding flag used in driving stress. \
                                  Lower values imply a smoother transition \
                                  from grounded to floating physics")})
        )

    def __repr__(self):
        return f'{self.bed.compact_string}\n{self.thklim}\n{self.sigmoid_c}'

@dataclass
class Rheology:
    B: Field | None = None
    n: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(3.0),
            name='n',
            units='',
            attrs={'long_name':'Glens law n'})
        )
    eps_reg: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(1e-6),
            name='eps_reg',
            units='s^{-2}',
            attrs={'long_name':'Strain invariant squared regularizer'})
        )
    H_reg: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(0.0),
            name='H_reg',
            units='m',
            attrs={'long_name':("Thickness regularizer for the vertical \
                                 shear terms: 1/H^2 -> 1/(H^2 + H_reg^2) \
                                 in the shear invariant, and consistently \
                                 eta/H -> eta*H/(H^2 + H_reg^2) in the \
                                 shear residual (the variational pair - \
                                 see get_sigma_xz_jac), plus a thin-ice \
                                 reactive spring kappa(H)*ud that pins \
                                 the deformational velocity on columns \
                                 thinner than ~H_reg, where the \
                                 regularized shear resistance would \
                                 otherwise vanish and let bare steep \
                                 terrain flow. Zero recovers the \
                                 unregularized model.")})
        )

    def __repr__(self):
        return f'{self.B.compact_string}\n{self.n}\n{self.eps_reg}\n{self.H_reg}'
    
@dataclass
class Sliding:
    beta: Field | None = None
    m: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(1.0),
            name='m',
            units='',
            attrs={'long_name':'Weertman law m'})
        )
    u_reg: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(1.0),
            name='u_reg',
            units='m a^{-1}',
            attrs={'long_name':'Weertman law regularization'})
        )
    water_drag: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(1e-5),
            name='water_drag',
            units='',
            attrs={'long_name':'basal traction exerted by water'})
        )

    p: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(1.0),
            name='p',
            units='',
            attrs={'long_name':("effective pressure exponent: the basal \
                                  traction coefficient is beta * xi^p, with \
                                  xi = N / (rho_i g H) the flotation fraction \
                                  (Leguy et al. 2014). p = 1 assumes full \
                                  hydraulic connection to the ocean; p -> 0 \
                                  confines the drag reduction to the \
                                  grounding line and recovers a grounded flag \
                                  (xi^0 is taken as 0 where xi = 0).")})
        )


    def __repr__(self):
        return f'{self.beta.compact_string}\n{self.m}\n{self.u_reg}\n{self.water_drag}\n{self.p}'

@dataclass
class Calving:
    timescale: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(cp.inf),
            name='timescale',
            units='a',
            attrs={'long_name':("decay timescale of the calving sink: a cell \
                        below the height-above-buoyancy threshold (calving \
                        flag psi = 0) loses ice at the rate H / timescale, \
                        i.e. by a factor 1 / (1 + dt / timescale) per \
                        implicit step, until the active set pins it at \
                        thklim. inf disables calving.")})
        )

    # Height-above-buoyancy calving criterion, hybrid threshold:
    #   ice calves where H - H_f < q H + h0   (psi = 0 there),
    # i.e. within a FRACTION q of its thickness plus an ABSOLUTE margin h0
    # (m) of flotation. Both are cell fields (like sliding.beta) so they can
    # vary along the coast, e.g. under an ocean thermal forcing
    # q = q0 + alpha_q dTF, h0 = h00 + alpha_h dTF; a scalar .set() fills
    # them uniformly. Restricted by averaging. q = h0 = 0 calves exactly the
    # floating ice.
    q: Field | None = None
    h0: Field | None = None
    # Floating-ice (shelf) minimum thickness (m). The calving flag is the
    # single monotone criterion psi = sigmoid(c r (H - H_calve)) with the
    # critical thickness blended between the grounded law H_g = (depth/r +
    # h0)/(1 - q) and the shelf law H_c by a linear ramp in the gap between
    # ice base and bed (see common.cu calving_F): floating ice down to
    # H_c - h0 calves iff the margin is positive, thinner floating ice
    # calves below H_c, so tongues exist only under a negative margin.
    # inf (the default) = the grounded law everywhere (floating ice always
    # calves, the pre-2026-09-15 behaviour).
    H_c: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(cp.inf),
            name='H_c',
            units='m',
            attrs={'long_name':("shelf minimum thickness of the gap-blended "
                                "calving criterion (see common.cu calving_F)")})
        )

    def __repr__(self):
        return f'{self.timescale}\n{self.q.compact_string}\n{self.h0.compact_string}\n{self.H_c}'

@dataclass
class Forcing:
    smb: Field = None
    
    def __repr__(self):
        return f'{self.smb.compact_string}'

class Grid:
    """
    Single level of the multigrid hierarchy.

    Parameters
    ----------
    ny, nx : int
        Grid dimensions (number of cells in y and x)
    dx : float
        Grid spacing (assumed isotropic)
    parent : Grid, optional
        Parent (finer) grid in hierarchy

    Note: If any of the optional dataclasses are passed in,
    the Fields and Constants contained therein are *not*
    copied and will mutate if the original data is 
    mutated externally.  This may or may not be 
    desirable behavior.
    """

    def __init__(self, ny: int, nx: int, dx: cp.float32,
            x0: cp.float32=cp.float32(0.0),
            y0: cp.float32=cp.float32(0.0),
            crs=None,
            parent = None,
            state: State = None,
            adjoint: AdjointState = None,
            geometry: Geometry = None,
            rheology: Rheology = None,
            sliding: Sliding = None,
            calving: Calving = None,
            forcing: Forcing = None,
            stress_scheme: str = 'molho',
            ):

        self.parent = parent
        self.child = None

        # 'molho' solves the two-field shear-resolving model; 'ssa' pins the
        # deformational components (ud, vd) to zero everywhere via identity
        # rows, which reduces the momentum balance exactly to the SSA. The
        # state always carries ud/vd (identically zero under SSA) so all
        # downstream code is scheme-agnostic.
        if stress_scheme not in ('molho', 'ssa'):
            raise ValueError("stress_scheme must be 'molho' or 'ssa'")
        self.stress_scheme = stress_scheme
        self.ssa = stress_scheme == 'ssa'
        
        self.ny = ny
        self.nx = nx
        
        self.dx = cp.float32(dx)

        self.x0 = cp.float32(x0)
        self.y0 = cp.float32(y0)
        self.crs = crs

        self.x_cell = cp.arange(x0,x0 + dx*nx, dx)
        self.y_cell = cp.arange(y0,y0 - dx*ny,-dx)

        self.x_vfacet = cp.arange(x0 - dx/2, x0 - dx/2 + dx*(nx+1), dx) 
        self.y_hfacet = cp.arange(y0 + dx/2, y0 + dx/2 - dx*(ny+1),-dx) 

        # Degrees of freedom
        self.nu = ny * (nx + 1)
        self.nv = (ny + 1) * nx
        self.nh = ny * nx
        self.n_total = self.nu + self.nv + self.nh

        self.state    = state    if state    is not None else self._allocate_state()
        self.geometry = geometry if geometry is not None else self._allocate_geometry()
        self.rheology = rheology if rheology is not None else self._allocate_rheology()
        self.sliding  = sliding  if sliding  is not None else self._allocate_sliding()
        self.calving  = calving  if calving  is not None else self._allocate_calving()
        self.forcing  = forcing  if forcing  is not None else self._allocate_forcing()
        
        # Adjoint fields are initialized lazily
        self._adjoint  = adjoint

        self._forward_operators = None
        self._adjoint_operators = None

    @property
    def forward_operators(self):
        if self._forward_operators is None:
            self._forward_operators = ForwardOperators(self)
        return self._forward_operators

    @property
    def adjoint_operators(self):
        if self._adjoint_operators is None:
            self._adjoint_operators = AdjointOperators(self)
        return self._adjoint_operators

    @property
    def adjoint(self):
        if self._adjoint is None:
            self._adjoint = self._allocate_adjoint_state()
        return self._adjoint

    def _allocate_state(self):
        u = Field(
            data=cp.zeros((self.ny, self.nx+1),dtype=cp.float32),
            grid_entity=GridEntity.VERTICAL_FACET,
            dx=self.dx,
            grid=self,
            name='u',
            units='m a^{-1}',
            attrs={'long_name':'x component of depth-averaged velocity'})
        
        v = Field(
            data=cp.zeros((self.ny+1, self.nx),dtype=cp.float32),
            grid_entity=GridEntity.HORIZONTAL_FACET,
            dx=self.dx,
            grid=self,
            name='v',
            units='m a^{-1}',
            attrs={'long_name':'y component of depth-averaged velocity'})

        ud = Field(
            data=cp.zeros((self.ny, self.nx+1),dtype=cp.float32),
            grid_entity=GridEntity.VERTICAL_FACET,
            dx=self.dx,
            grid=self,
            name='ud',
            units='m a^{-1}',
            attrs={'long_name':'x component of deformation velocity'})
        
        vd = Field(
            data=cp.zeros((self.ny+1, self.nx),dtype=cp.float32),
            grid_entity=GridEntity.HORIZONTAL_FACET,
            dx=self.dx,
            grid=self,
            name='vd',
            units='m a^{-1}',
            attrs={'long_name':'y component of deformation velocity'})



        H = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='H',
            units='m',
            attrs={'long_name':'Ice thickness at t + dt (end of time step)'})
        
        H_prev = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='H_prev',
            units='m',
            attrs={'long_name':'Ice thickness at t (beginning of time step)'})

        phi = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='phi',
            units='',
            attrs={'long_name':'Grounded flag: sigmoid(c z), blend weight in the driving stress'})

        xi = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='xi',
            units='',
            attrs={'long_name':'Flotation fraction N / (rho_i g H), clipped to [0,1]'})

        psi = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='psi',
            units='',
            attrs={'long_name':'Calving flag: sigmoid(c (z - q rho_i/rho_w H)), height-above-buoyancy criterion'})


        mask = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='mask',
            units='',
            attrs={'long_name':'''Active set mask - if unity, thickness is 
                         set to thklim in Dirichlet BC fashion'''})

        dpsi_dH = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='dpsi_dH',
            units='m^{-1}',
            attrs={'long_name':'d psi / dH of the calving flag (unrelaxed), for the calving sink Jacobian'})

        dpsi_dbed = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='dpsi_dbed',
            units='m^{-1}',
            attrs={'long_name':'d psi / d bed of the calving flag, for the bed gradient'})

        dxi_dH = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='dxi_dH',
            units='m^{-1}',
            attrs={'long_name':'d xi / dH of the flotation fraction, (1 - xi)/H where 0 < xi < 1; '
                               'RESTRICTED (not recomputed) on coarse adjoint levels, for the drag Jacobian'})

        return State(u=u,v=v,ud=ud,vd=vd,H=H,H_prev=H_prev,phi=phi,xi=xi,dxi_dH=dxi_dH,psi=psi,dpsi_dH=dpsi_dH,dpsi_dbed=dpsi_dbed,mask=mask)

    def _allocate_adjoint_state(self):
        lambda_u = Field(
            data=cp.zeros((self.ny, self.nx+1),dtype=cp.float32),
            grid_entity=GridEntity.VERTICAL_FACET,
            dx=self.dx,
            grid=self,
            name='lambda_u',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for u'})

        lambda_v = Field(
            data=cp.zeros((self.ny+1, self.nx),dtype=cp.float32),
            grid_entity=GridEntity.HORIZONTAL_FACET,
            dx=self.dx,
            grid=self,
            name='lambda_v',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for v'})

        lambda_ud = Field(
            data=cp.zeros((self.ny, self.nx+1),dtype=cp.float32),
            grid_entity=GridEntity.VERTICAL_FACET,
            dx=self.dx,
            grid=self,
            name='lambda_ud',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for ud'})

        lambda_vd = Field(
            data=cp.zeros((self.ny+1, self.nx),dtype=cp.float32),
            grid_entity=GridEntity.HORIZONTAL_FACET,
            dx=self.dx,
            grid=self,
            name='lambda_vd',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for vd'})


        lambda_H = Field(
            cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='lambda_H',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for H'})

        return AdjointState(lambda_u=lambda_u,lambda_v=lambda_v,lambda_ud=lambda_ud,lambda_vd=lambda_vd,lambda_H=lambda_H)

    def _allocate_geometry(self):
        bed = SubgridField(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='bed',
            units='m',
            attrs={'long_name':'bed elevation (not necessarily the ice base)'})

        depth = SubgridField(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='depth',
            units='m',
            attrs={'long_name':'water depth'})
        return Geometry(bed=bed,depth=depth)

    def _allocate_rheology(self):
        B = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='B',
            units='m',
            attrs={'long_name':'Rheologic prefactor.  B=A^{-1/n}'})

        return Rheology(B=B)

    def _allocate_sliding(self):
        beta = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='beta',
            units='?',
            attrs={'long_name':'Basal sliding coefficient'})

        return Sliding(beta=beta)

    def _allocate_calving(self):
        q = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='q',
            units='',
            attrs={'long_name':("multiplicative height-above-buoyancy calving "
                                "margin: ice calves where H - H_f < q H + h0 "
                                "(calving flag psi = 0). q = h0 = 0 calves "
                                "exactly the floating ice.")})
        h0 = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='h0',
            units='m',
            attrs={'long_name':("additive height-above-buoyancy calving "
                                "margin: ice calves where H - H_f < q H + h0")})
        return Calving(q=q, h0=h0)
    
    def _allocate_forcing(self):
        smb = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='smb',
            units='m a^{-1}',
            attrs={'long_name':'Surface mass balance'})

        return Forcing(smb=smb)

    def spawn_child(self):
        child = Grid(
            self.ny // 2, self.nx // 2,
            self.dx * 2, parent=self
        )
        self.child = child
        return child

    def to_dataset(self,
            fields: dict[str, Field] | None = None,
            *,
            attrs: dict | None = None):
    
        data_vars = {name: fld.to_dataarray() for name,fld in fields.items()}
        ds = xr.Dataset(data_vars=data_vars)
        ds.attrs['dx'] = self.dx
        ds.attrs['crs'] = str(self.crs)
        ds.attrs["spatial_ref"] = self.crs.to_wkt()
        ds.attrs["crs_wkt"] = self.crs.to_wkt()        

        if attrs:
            ds.attrs.update(attrs)

        return ds
        
  
