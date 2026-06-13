from dataclasses import dataclass, field, fields
from glide.backend import xp as cp, NDArray
import xarray as xr
from .field import Field, SubgridField, Constant, GridEntity
from .operators import ForwardOperators, AdjointOperators

@dataclass
class State:
    u: Field | None = None
    v: Field | None = None
    H: Field | None = None
    H_prev: Field | None = None
    phi: Field | None = None
    mask: Field | None = None

    def __repr__(self):
        return f'{self.u.compact_string}\n{self.v.compact_string}\n{self.H.compact_string}\n{self.H_prev.compact_string}\n{self.phi.compact_string}\n{self.mask.compact_string}'

@dataclass
class AdjointState:
    lambda_u: Field | None = None
    lambda_v: Field | None = None
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
    sigmoid_k: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(3.0),
            name='sigmoid_k',
            units='m^{-1}',
            attrs={'long_name':("offset factor for sigmoidal \
                                  grounding flag used in driving stress. \
                                  larger values imply a more asymmetric \
                                  transition from grounded to floating \
                                  physics")})
        )


    def __repr__(self):
        return f'{self.bed.compact_string}\n{self.thklim}\n{self.flotation_reg_driving}'

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
    
    def __repr__(self):
        return f'{self.B.compact_string}\n{self.n}\n{self.eps_reg}'
    
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

    flotation_reg_sliding: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.1),
            name='flotation_reg_sliding',
            units='m',
            attrs={'long_name':("smoothing factor for pseudo-sigmoidal \
                                  grounding flag used in basal stress. \
                                  Larger values imply a smoother transition \
                                  from grounded to floating physics")})
        )


    def __repr__(self):
        return f'{self.beta.compact_string}\n{self.m}\n{self.u_reg}\n{self.water_drag}\n{self.flotation_reg_sliding}'

@dataclass
class Calving:
    calving_rate: Constant = field(
        default_factory = lambda: Constant(
            value=cp.float32(0.0),
            name='calving_rate',
            units='m a^{-1}',
            attrs={'long_name':"The speed at which ice \
                        nonconservatively fluxes through \
                        a facet when both cells are floating"})
        )

    flotation_reg_calving: Constant = field(
        default_factory=lambda: Constant(
            value=cp.float32(0.1),
            name='flotation_reg_calving',
            units='m',
            attrs={'long_name':("smoothing factor for pseudo-sigmoidal \
                                  grounding flag used in calving. \
                                  Larger values imply a smoother transition \
                                  from grounded to floating physics")})
        )

    def __repr__(self):
        return f'{self.calving_rate}\n{self.flotation_reg_calving}'

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
            ):

        self.parent = parent
        self.child = None
        
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
            units='m',
            attrs={'long_name':'Potential Head'})

        mask = Field(
            data=cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='mask',
            units='',
            attrs={'long_name':'''Active set mask - if unity, thickness is 
                         set to thklim in Dirichlet BC fashion'''})

        return State(u=u,v=v,H=H,H_prev=H_prev,phi=phi,mask=mask)

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

        lambda_H = Field(
            cp.zeros((self.ny,self.nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=self.dx,
            grid=self,
            name='lambda_H',
            units='varies with objective fn',
            attrs={'long_name':'Adjoint variable for H'})

        return AdjointState(lambda_u=lambda_u,lambda_v=lambda_v,lambda_H=lambda_H)

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
        return Calving()
    
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
        
  
