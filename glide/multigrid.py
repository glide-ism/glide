import cupy as cp
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from .grid import Grid
from .operators import VankaConfig,NewtonConfig
from .field import LocalOption,BroadcastOption

class Multigrid:
    def __init__(self,n_levels: int,finest_grid=None,
            ny=None,nx=None,dx=None,
            x0=cp.float32(0.0),y0=cp.float32(0.0),crs=None,
            use_fast_math=True):

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = ['common.cu', 'transfer.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()

        self.kernels = cp.RawModule(code=cuda_source, options=options)

        if finest_grid is not None:
            print("Instantiating multigrid from existing grid")
            self.finest_grid = finest_grid
        else:
            print("Instantiating multigrid from new grid")
            self.finest_grid = Grid(ny,nx,dx,x0=x0,y0=y0,crs=crs)

        self.n_levels = n_levels

        if n_levels is not None:
            self.create_grid_hierarchy(n_levels,restrict_fields=True)

        self.state = MGStateManager(self)
        self.geometry = MGGeometryManager(self)
        self.rheology = MGRheologyManager(self)
        self.sliding = MGSlidingManager(self)
        self.calving = MGCalvingManager(self)
        self.forcing = MGForcingManager(self)

        self._adjoint = None

    @property
    def adjoint(self):
        if self._adjoint is None:
            self._adjoint = MGAdjointManager(self)
        return self._adjoint

    def create_grid_hierarchy(self,n_levels,restrict_fields=True):
        self.levels = [self.finest_grid]
        for i in range(1,n_levels):
            coarse_grid = self.create_coarse_grid(self.levels[-1],
                restrict_fields=restrict_fields)
            self.levels.append(coarse_grid)
        return self.levels

    def create_coarse_grid(self,parent_grid,restrict_fields=True):
        child_grid = Grid(
            parent_grid.ny // 2, parent_grid.nx // 2,
            parent_grid.dx * 2, 
            x0=parent_grid.x0 + parent_grid.dx/2,
            y0=parent_grid.y0 - parent_grid.dx/2,
            crs=parent_grid.crs,
            parent=parent_grid
        )
        parent_grid.child = child_grid
        if restrict_fields == True:
            self.restrict_state(parent_grid,child_grid)
            self.restrict_geometry(parent_grid,child_grid)
            self.restrict_rheology(parent_grid,child_grid)
            self.restrict_sliding(parent_grid,child_grid)
            self.restrict_calving(parent_grid,child_grid)
            self.restrict_forcing(parent_grid,child_grid)
        return child_grid

    def restrict_state(self,fine_grid,coarse_grid):
        self.restrict_vfacet(fine_grid.state.u.data,coarse_grid.state.u.data)
        self.restrict_hfacet(fine_grid.state.v.data,coarse_grid.state.v.data)
        self.restrict_vfacet(fine_grid.state.ud.data,coarse_grid.state.ud.data)
        self.restrict_hfacet(fine_grid.state.vd.data,coarse_grid.state.vd.data)
        self.restrict_cell(fine_grid.state.H.data,coarse_grid.state.H.data)
        self.restrict_cell(fine_grid.state.H_prev.data,coarse_grid.state.H_prev.data)
        self.restrict_cell(fine_grid.state.phi.data,coarse_grid.state.phi.data)
        self.restrict_cell(fine_grid.state.xi.data,coarse_grid.state.xi.data)
        self.restrict_cell(fine_grid.state.mask.data,coarse_grid.state.mask.data,method='max')

    def restrict_geometry(self,fine_grid,coarse_grid):
        self.restrict_cell(fine_grid.geometry.bed.data,coarse_grid.geometry.bed.data)
        self.restrict_cell(fine_grid.geometry.depth.data,coarse_grid.geometry.depth.data)
        coarse_grid.geometry.thklim.set(fine_grid.geometry.thklim.value)
        coarse_grid.geometry.sigmoid_c.set(fine_grid.geometry.sigmoid_c.value)
        coarse_grid.geometry.sigmoid_k.set(fine_grid.geometry.sigmoid_k.value)

    def restrict_rheology(self,fine_grid,coarse_grid):
        self.restrict_cell(fine_grid.rheology.B.data,coarse_grid.rheology.B.data)
        coarse_grid.rheology.n.set(fine_grid.rheology.n.value)
        coarse_grid.rheology.eps_reg.set(fine_grid.rheology.eps_reg.value)
        coarse_grid.rheology.H_reg.set(fine_grid.rheology.H_reg.value)
    
    def restrict_sliding(self,fine_grid,coarse_grid):
        self.restrict_cell(fine_grid.sliding.beta.data,coarse_grid.sliding.beta.data)
        coarse_grid.sliding.m.set(fine_grid.sliding.m.value)
        coarse_grid.sliding.u_reg.set(fine_grid.sliding.u_reg.value)
        coarse_grid.sliding.water_drag.set(fine_grid.sliding.water_drag.value)
        coarse_grid.sliding.flotation_reg_sliding.set(fine_grid.sliding.flotation_reg_sliding.value)

    def restrict_calving(self,fine_grid,coarse_grid):
        coarse_grid.calving.calving_rate.set(fine_grid.calving.calving_rate.value)
        coarse_grid.calving.flotation_reg_calving.set(fine_grid.calving.flotation_reg_calving.value)
    
    def restrict_forcing(self,fine_grid,coarse_grid):
        self.restrict_cell(fine_grid.forcing.smb.data,coarse_grid.forcing.smb.data)

    def restrict_residual(self,fine_grid,coarse_grid):
        self.restrict_vfacet(fine_grid.forward_operators.r_u,coarse_grid.forward_operators.r_u)
        self.restrict_hfacet(fine_grid.forward_operators.r_v,coarse_grid.forward_operators.r_v)
        self.restrict_vfacet(fine_grid.forward_operators.r_ud,coarse_grid.forward_operators.r_ud)
        self.restrict_hfacet(fine_grid.forward_operators.r_vd,coarse_grid.forward_operators.r_vd)
        self.restrict_cell(fine_grid.forward_operators.r_H,coarse_grid.forward_operators.r_H)
   
    def restrict_vfacet(self,fine_field,coarse_field=None):
        """Restrict u-velocity (vertical face) field to coarse grid."""
        kernel = self.kernels.get_function('restrict_vfacet')
        ny, nx_plus_1 = fine_field.shape
        nx = nx_plus_1 - 1
        ny_coarse = ny // 2
        nx_coarse = nx // 2

        if coarse_field is None:
            coarse_field = cp.empty((ny_coarse, nx_coarse + 1), dtype=cp.float32)

        total_work = ny_coarse * (nx_coarse + 1)
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (fine_field, coarse_field, ny_coarse, nx_coarse))

        return coarse_field

    def restrict_hfacet(self,fine_field,coarse_field=None):
        """Restrict u-velocity (vertical face) field to coarse grid."""
        kernel = self.kernels.get_function('restrict_hfacet')
        ny_plus_1, nx = fine_field.shape
        ny = ny_plus_1 - 1
        ny_coarse = ny // 2
        nx_coarse = nx // 2

        if coarse_field is None:
            coarse_field = cp.empty((ny_coarse + 1, nx_coarse), dtype=cp.float32)

        total_work = (ny_coarse + 1) * nx_coarse
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (fine_field, coarse_field, ny_coarse, nx_coarse))

        return coarse_field

    def restrict_cell(self,fine_field,coarse_field=None,method='avg'):
        """Restrict u-velocity (vertical face) field to coarse grid."""
        if method == 'avg':
            kernel = self.kernels.get_function('restrict_cell_avg')
        elif method == 'max':
            kernel = self.kernels.get_function('restrict_cell_max')
        elif method == 'min':
            kernel = self.kernels.get_function('restrict_cell_min')
        elif method == 'var':
            kernel = self.kernels.get_function('restrict_cell_var')
        else:
            raise TypeError('Valid restriction methods: [avg,max,min]')

        ny, nx = fine_field.shape
        ny_coarse = ny // 2
        nx_coarse = nx // 2

        if coarse_field is None:
            coarse_field = cp.empty((ny_coarse, nx_coarse), dtype=cp.float32)

        total_work = ny_coarse * nx_coarse
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (fine_field, coarse_field, ny_coarse, nx_coarse))

        return coarse_field

    def prolongate_vfacet(self,coarse_field, fine_field=None, method='injection'):
        """Prolongate u-velocity (vertical face) field to fine grid."""
        if method == 'injection':
            kernel = self.kernels.get_function('prolongate_vfacet_injection')
        elif method == 'bilinear':
            kernel = self.kernels.get_function('prolongate_vfacet_bilinear')
        else:
            raise TypeError('Valid prolongation methods: [injection, bilinear]')

        ny, nx_plus_1 = coarse_field.shape
        nx = nx_plus_1 - 1
        ny_fine = ny * 2
        nx_fine = nx * 2

        if fine_field is None:
            fine_field = cp.empty((ny_fine, nx_fine + 1), dtype=cp.float32)

        total_work = ny_fine * (nx_fine + 1)
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (coarse_field, fine_field, ny_fine, nx_fine))
        return fine_field

    def prolongate_hfacet(self,coarse_field, fine_field=None, method='injection'):
        """Prolongate v-velocity (horizontal face) field to fine grid."""
        if method == 'injection':
            kernel = self.kernels.get_function('prolongate_hfacet_injection')
        elif method == 'bilinear':
            kernel = self.kernels.get_function('prolongate_hfacet_bilinear')
        else:
            raise TypeError('Valid prolongation methods: [injection, bilinear]')

        ny_plus_1, nx = coarse_field.shape
        ny = ny_plus_1 - 1
        ny_fine = ny * 2
        nx_fine = nx * 2

        if fine_field is None:
            fine_field = cp.empty((ny_fine + 1, nx_fine), dtype=cp.float32)

        total_work = (ny_fine + 1) * nx_fine
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (coarse_field, fine_field, ny_fine, nx_fine))
        return fine_field

    def prolongate_cell(self,coarse_field, fine_field=None, method='injection'):
        """Prolongate cell-centered field to fine grid."""
        if method == 'injection':
            kernel = self.kernels.get_function('prolongate_cell_injection')
        elif method == 'bilinear':
            kernel = self.kernels.get_function('prolongate_cell_bilinear')
        else:
            raise TypeError('Valid prolongation methods: [injection, bilinear]')

        ny, nx = coarse_field.shape
        ny_fine = ny * 2
        nx_fine = nx * 2

        if fine_field is None:
            fine_field = cp.empty((ny_fine, nx_fine), dtype=cp.float32)

        total_work = ny_fine * nx_fine
        block_size = 256
        grid_size = (total_work + block_size - 1) // block_size

        kernel((grid_size,), (block_size,),
               (coarse_field, fine_field, ny_fine, nx_fine))
        return fine_field   

    def __getitem__(self,key):
        return self.levels[key]

class HierarchyFieldManager:
    def __init__(self, levels, getter, restrict,name=None):
        self._levels = levels
        self._getter = getter
        self._restrict = restrict
        self._name = name

    def set(self, value, start_level=0):
        finest = self._getter(self._levels[start_level])
        finest.set(value)
        self.restrict_down(start_level)

    def restrict_down(self, start_level):
        for l in range(start_level,len(self._levels) - 1):
            fine = self._getter(self._levels[l])
            coarse = self._getter(self._levels[l + 1])
            self._restrict(fine, coarse)

    def set_level(self, level, value):
        self._getter(self._levels[level].grid).set(value)

class MGStateManager:
    def __init__(self, mg):
        self.mg = mg
        self.u = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.u,
            restrict=lambda f,c: mg.restrict_vfacet(f.data,c.data),
            name="u",
        )

        self.v = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.v,
            restrict=lambda f,c: mg.restrict_hfacet(f.data,c.data),
            name="v",
        )

        self.ud = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.ud,
            restrict=lambda f,c: mg.restrict_vfacet(f.data,c.data),
            name="ud",
        )

        self.vd = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.vd,
            restrict=lambda f,c: mg.restrict_hfacet(f.data,c.data),
            name="vd",
        )

        self.H = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.H,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="H",
        )
        
        self.H_prev = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.H_prev,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="H_prev",
        )

        self.phi = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.phi,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="phi",
        )
        self.xi = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.xi,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="xi",
        )

        self.mask = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.state.mask,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='max'),
            name="mask",
        )

    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].state.__repr__()

class MGAdjointManager:
    def __init__(self, mg):
        self.mg = mg
        self.lambda_u = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.adjoint.lambda_u,
            restrict=lambda f,c: mg.restrict_vfacet(f.data,c.data),
            name="lambda_u",
        )

        self.lambda_v = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.adjoint.lambda_v,
            restrict=lambda f,c: mg.restrict_hfacet(f.data,c.data),
            name="lambda_v",
        )

        self.lambda_ud = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.adjoint.lambda_ud,
            restrict=lambda f,c: mg.restrict_vfacet(f.data,c.data),
            name="lambda_ud",
        )

        self.lambda_vd = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.adjoint.lambda_vd,
            restrict=lambda f,c: mg.restrict_hfacet(f.data,c.data),
            name="lambda_vd",
        )

        self.lambda_H = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.adjoint.lambda_H,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="lambda_H",
        )
        
    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].adjoint.__repr__()

class MGGeometryManager:
    def __init__(self, mg):
        self.mg = mg
        self.bed = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.geometry.bed,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="bed",
        )

        self.depth = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.geometry.depth,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="depth",
        )

        self.sigmoid_c = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.geometry.sigmoid_c,
            restrict=lambda f,c: c.set(f.value),
            name="sigmoid_c",
        )
        
        self.sigmoid_k = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.geometry.sigmoid_k,
            restrict=lambda f,c: c.set(f.value),
            name="sigmoid_k",
        )

        self.thklim = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.geometry.thklim,
            restrict=lambda f,c: c.set(f.value),
            name="thklim",
        )
    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].geometry.__repr__()

class MGRheologyManager:
    def __init__(self, mg):
        self.mg = mg
        self.B = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.rheology.B,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="B",
        )

        self.n = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.rheology.n,
            restrict=lambda f,c: c.set(f.value),
            name="n",
        )

        self.eps_reg = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.rheology.eps_reg,
            restrict=lambda f,c: c.set(f.value),
            name="eps_reg",
        )

        self.H_reg = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.rheology.H_reg,
            restrict=lambda f,c: c.set(f.value),
            name="H_reg",
        )
    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].rheology.__repr__()

class MGSlidingManager:
    def __init__(self, mg):
        self.mg = mg
        self.beta = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.sliding.beta,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="beta",
        )

        self.m = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.sliding.m,
            restrict=lambda f,c: c.set(f.value),
            name="m",
        )
        
        self.u_reg = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.sliding.u_reg,
            restrict=lambda f,c: c.set(f.value),
            name="u_reg",
        )
        
        self.water_drag = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.sliding.water_drag,
            restrict=lambda f,c: c.set(f.value),
            name="water_drag",
        )
        
        self.flotation_reg_sliding = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.sliding.flotation_reg_sliding,
            restrict=lambda f,c: c.set(f.value),
            name="flotation_reg_sliding",
        )

    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].sliding.__repr__()

class MGCalvingManager:
    def __init__(self, mg):
        self.mg = mg
        self.calving_rate = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.calving.calving_rate,
            restrict=lambda f,c: c.set(f.value),
            name="calving_rate",
        )
        
        self.flotation_reg_calving = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.calving.flotation_reg_calving,
            restrict=lambda f,c: c.set(f.value),
            name="flotation_reg_calving",
        )

    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].calving.__repr__()

class MGForcingManager:
    def __init__(self, mg):
        self.mg = mg
        self.smb = HierarchyFieldManager(
            mg.levels,
            getter=lambda g: g.forcing.smb,
            restrict=lambda f,c: mg.restrict_cell(f.data,c.data,method='avg'),
            name="smb",
        )

    def __repr__(self):
        return f'Top-level ({self.mg.n_levels} levels): \n'+self.mg.levels[0].forcing.__repr__()

class FASCDSolver:
    def __init__(self,multigrid):
        self.multigrid = multigrid
        self.levels = [FASCDLevel(grid, FASCDScratch(grid)) for grid in multigrid.levels]

        self._fas_config = FASCDConfig()
        self.fas_options = FASCDOptions(self._fas_config)
        
        self.vanka_options = VankaOptions(
            self.levels,
            getter=lambda lev: lev.grid.forward_operators.vanka_config,
        )

        self.n_levels = len(self.levels)
        self.dt = None

    def solve(self,dt,start_level=0,report_norms=True):
        self.dt = cp.float32(dt)
        
        start_level_ = self.multigrid.levels[start_level]
        start_level_.forward_operators.set_rhs(dt)
        
        ru_init,rv_init,rud_init,rvd_init,rH_init = start_level_.forward_operators.compute_residual(dt,return_norms=True)
        initial_residual_norm = cp.sqrt(ru_init**2 + rv_init**2 + rud_init**2 + rvd_init**2 + rH_init**2)
        relative_residual_norm = cp.float32(1.0)

        if self._fas_config.report_norms:
            print(f"  Initial:   |r0|     = {initial_residual_norm:.2e}, "
                  f"|r_u| = {float(ru_init):.2e}, "
                  f"|r_v| = {float(rv_init):.2e}, "
                  f"|r_ud| = {float(rud_init):.2e}, "
                  f"|r_vd| = {float(rvd_init):.2e}, "
                  f"|r_H| = {float(rH_init):.2e}")

        absolute_residual_norm = initial_residual_norm
        iteration = 0
        while (relative_residual_norm > self._fas_config.relative_tolerance 
                and absolute_residual_norm > self._fas_config.absolute_tolerance
                and iteration < self._fas_config.maximum_vcycles):
            self.vcycle(start_level,finest=True)
            ru,rv,rud,rvd,rH = start_level_.forward_operators.compute_residual(dt,freeze_phi=True,return_norms=True)

            absolute_residual_norm = cp.sqrt(ru**2 + rv**2 + rud**2 + rvd**2 + rH**2)
            relative_residual_norm = absolute_residual_norm / initial_residual_norm
            if self._fas_config.report_norms:
                print(f"  V-cycle {iteration}: |r|/|r0| = {relative_residual_norm:.2e}, "
                      f"|r_u| = {float(ru):.2e}, "
                      f"|r_v| = {float(rv):.2e}, "
                      f"|r_ud| = {float(rud):.2e}, "
                      f"|r_vd| = {float(rvd):.2e}, "
                      f"|r_H| = {float(rH):.2e}")
            iteration += 1

        
    def vcycle(self, l, finest=False):
        """
        FASCD V-cycle for the coupled SSA + mass conservation system.

        Full Approximation Scheme with Constrained Descent handles the
        thickness inequality constraint H >= gamma via an active set method.

        Parameters
        ----------
        l : Current grid level
        """
        coarse = not finest
        mg = self.multigrid
        dt = self.dt
        level = self.levels[l]
        
        if finest:
            level.scratch.w_u[:,:] = level.grid.state.u.data[:,:]
            level.scratch.w_v[:,:] = level.grid.state.v.data[:,:]
            level.scratch.w_ud[:,:] = level.grid.state.ud.data[:,:]
            level.scratch.w_vd[:,:] = level.grid.state.vd.data[:,:]
            level.scratch.w_H[:,:] = level.grid.state.H.data[:,:]
            level.scratch.chi[:,:] = level.grid.geometry.thklim.value - level.grid.state.H.data

        if l == self.n_levels - 1:
            # Coarsest level: direct solve
            level.grid.forward_operators.gamma[:,:] = level.scratch.w_H[:,:] + level.scratch.chi[:,:]
            level.grid.forward_operators.vanka_sweep(self.dt,
                self._fas_config.coarsest_steps,
                freeze_calving=coarse and self._fas_config.freeze_coarse_calving,
                freeze_phi=coarse and self._fas_config.freeze_coarse_phi
            )
            level.grid.forward_operators.gamma.fill(level.grid.geometry.thklim.value)
            return

        next_level = self.levels[l+1]

        # Restrict constraint defect
        mg.restrict_cell(level.scratch.chi, next_level.scratch.chi, method='max')

        # Prolongate and compute local constraint adjustment
        mg.prolongate_cell(-next_level.scratch.chi, level.scratch.phi, method='injection')
        level.scratch.phi[:,:] += level.scratch.chi

        # Pre-smooth with local constraint
        level.grid.forward_operators.gamma[:, :] = level.scratch.w_H + level.scratch.phi
        level.grid.forward_operators.vanka_sweep(self.dt,self._fas_config.pre_steps,
                freeze_calving=coarse and self._fas_config.freeze_coarse_calving,
                freeze_phi=coarse and self._fas_config.freeze_coarse_phi)
        level.grid.forward_operators.gamma.fill(level.grid.geometry.thklim.value)

        # Compute coarse grid correction
        level.scratch.y_u[:,:] = level.grid.state.u.data - level.scratch.w_u
        level.scratch.y_v[:,:] = level.grid.state.v.data - level.scratch.w_v
        level.scratch.y_ud[:,:] = level.grid.state.ud.data - level.scratch.w_ud
        level.scratch.y_vd[:,:] = level.grid.state.vd.data - level.scratch.w_vd
        level.scratch.y_H[:,:] = level.grid.state.H.data - level.scratch.w_H

        # Restrict solution to child
        mg.restrict_state(level.grid,next_level.grid)
        next_level.scratch.w_u[:,:] = next_level.grid.state.u.data[:,:]
        next_level.scratch.w_v[:,:] = next_level.grid.state.v.data[:,:]
        next_level.scratch.w_ud[:,:] = next_level.grid.state.ud.data[:,:]
        next_level.scratch.w_vd[:,:] = next_level.grid.state.vd.data[:,:]
        next_level.scratch.w_H[:,:] = next_level.grid.state.H.data[:,:]

        # Compute and restrict residual
        level.grid.forward_operators.compute_residual(dt, use_mask=False, 
                freeze_calving=coarse and self._fas_config.freeze_coarse_calving,
                freeze_phi=True)
        mg.restrict_residual(level.grid,next_level.grid)

        # Form coarse grid RHS: f_c = F_c(I_h^H u_h) - I_h^H r_h
        next_level.grid.forward_operators.compute_residual(dt, use_mask=False, 
                operator_only=True, 
                freeze_calving=self._fas_config.freeze_coarse_calving,
                freeze_phi=self._fas_config.freeze_coarse_phi)

        next_level.grid.forward_operators.f_u[:,:] = next_level.grid.forward_operators.F_u[:,:] - next_level.grid.forward_operators.r_u[:,:]
        next_level.grid.forward_operators.f_v[:,:] = next_level.grid.forward_operators.F_v[:,:] - next_level.grid.forward_operators.r_v[:,:]
        next_level.grid.forward_operators.f_ud[:,:] = next_level.grid.forward_operators.F_ud[:,:] - next_level.grid.forward_operators.r_ud[:,:]
        next_level.grid.forward_operators.f_vd[:,:] = next_level.grid.forward_operators.F_vd[:,:] - next_level.grid.forward_operators.r_vd[:,:]
        next_level.grid.forward_operators.f_H[:,:] = next_level.grid.forward_operators.F_H[:,:] - next_level.grid.forward_operators.r_H[:,:]

        # Recursive call
        self.vcycle(l+1)

        # Compute coarse correction
        next_level.scratch.z_u[:] = next_level.grid.state.u.data - next_level.scratch.w_u
        next_level.scratch.z_v[:] = next_level.grid.state.v.data - next_level.scratch.w_v
        next_level.scratch.z_ud[:] = next_level.grid.state.ud.data - next_level.scratch.w_ud
        next_level.scratch.z_vd[:] = next_level.grid.state.vd.data - next_level.scratch.w_vd
        next_level.scratch.z_H[:] = next_level.grid.state.H.data - next_level.scratch.w_H

        # Prolongate correction
        mg.prolongate_vfacet(next_level.scratch.z_u,level.scratch.z_u,method='bilinear')
        mg.prolongate_hfacet(next_level.scratch.z_v,level.scratch.z_v,method='bilinear')
        mg.prolongate_vfacet(next_level.scratch.z_ud,level.scratch.z_ud,method='bilinear')
        mg.prolongate_hfacet(next_level.scratch.z_vd,level.scratch.z_vd,method='bilinear')
        mg.prolongate_cell(next_level.scratch.z_H,level.scratch.z_H,method='injection')

        # Apply correction
        level.scratch.z_u[:,:] += level.scratch.y_u[:,:]
        level.scratch.z_v[:,:] += level.scratch.y_v[:,:]
        level.scratch.z_ud[:,:] += level.scratch.y_ud[:,:]
        level.scratch.z_vd[:,:] += level.scratch.y_vd[:,:]
        level.scratch.z_H[:,:] += level.scratch.y_H[:,:]

        level.grid.state.u.data[:,:] = level.scratch.w_u + level.scratch.z_u
        level.grid.state.v.data[:,:] = level.scratch.w_v + level.scratch.z_v
        level.grid.state.ud.data[:,:] = level.scratch.w_ud + level.scratch.z_ud
        level.grid.state.vd.data[:,:] = level.scratch.w_vd + level.scratch.z_vd
        level.grid.state.H.data[:,:] = level.scratch.w_H + level.scratch.z_H

        # Post-smooth
        level.grid.forward_operators.gamma[:, :] = level.scratch.w_H + level.scratch.chi
        level.grid.forward_operators.vanka_sweep(self.dt,self._fas_config.post_steps,
                freeze_calving=coarse and self._fas_config.freeze_coarse_calving,
                freeze_phi=coarse and self._fas_config.freeze_coarse_phi)
        level.grid.forward_operators.gamma.fill(level.grid.geometry.thklim.value)

        if finest:
            level.grid.forward_operators.vanka_sweep(self.dt,
                self._fas_config.finest_steps,
                freeze_phi=False,
                freeze_calving=False)

class FASCDScratch:
    def __init__(self,grid):
        ny,nx = grid.ny,grid.nx
        
        self.w_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.w_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.w_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.w_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.w_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.y_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.y_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.y_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.y_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.y_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.z_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.z_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.z_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.z_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.z_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.chi = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.phi = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

@dataclass
class FASCDLevel:
    grid: Grid
    scratch: FASCDScratch

@dataclass
class FASCDConfig:
    freeze_coarse_calving: bool = True
    freeze_coarse_phi: bool = True
    coarsest_steps: int = 200
    pre_steps: int = 10
    post_steps: int = 20
    finest_steps: int = 50
    maximum_vcycles: int = 10
    relative_tolerance: cp.float32 = cp.float32(1e-3)
    absolute_tolerance: cp.float32 = cp.float32(5.0)
    report_norms: bool = True

def dataclass_field_names(cls):
    return [f.name for f in fields(cls)]

def validate_kwargs(config_cls, kwargs):
    valid = set(dataclass_field_names(config_cls))
    unknown = set(kwargs) - valid
    if unknown:
        raise AttributeError(
            f"Unknown {config_cls.__name__} option(s): {sorted(unknown)}. "
            f"Valid options are: {sorted(valid)}"
        )

class FASCDOptions:
    """
    User-facing wrapper around a single FASCDConfig.
    """

    def __init__(self, config: FASCDConfig):
        self._config = config

        self.options = ['freeze_coarse_calving',
            'freeze_coarse_phi',
            'coarsest_steps',
            'pre_steps',
            'post_steps',
            'finest_steps']

        self.freeze_coarse_calving = LocalOption(
            getter=lambda: self._config.freeze_coarse_calving,
            setter=lambda v: setattr(self._config, "freeze_coarse_calving", v),
            name="freeze_coarse_calving",
        )
        self.freeze_coarse_phi = LocalOption(
            getter=lambda: self._config.freeze_coarse_phi,
            setter=lambda v: setattr(self._config, "freeze_coarse_phi", v),
            name="freeze_coarse_phi",
        )
        self.coarsest_steps = LocalOption(
            getter=lambda: self._config.coarsest_steps,
            setter=lambda v: setattr(self._config, "coarsest_steps", v),
            name="coarsest_steps",
        )
        self.pre_steps = LocalOption(
            getter=lambda: self._config.pre_steps,
            setter=lambda v: setattr(self._config, "pre_steps", v),
            name="pre_steps",
        )
        self.post_steps = LocalOption(
            getter=lambda: self._config.post_steps,
            setter=lambda v: setattr(self._config, "post_steps", v),
            name="post_steps",
        )
        self.finest_steps = LocalOption(
            getter=lambda: self._config.finest_steps,
            setter=lambda v: setattr(self._config, "finest_steps", v),
            name="finest_steps",
        )

        self.maximum_vcycles = LocalOption(
            getter=lambda: self._config.maximum_vcycles,
            setter=lambda v: setattr(self._config, "maximum_vcycles", v),
            name="maximum_vcyles",
        )
        
        self.relative_tolerance = LocalOption(
            getter=lambda: self._config.relative_tolerance,
            setter=lambda v: setattr(self._config, "relative_tolerance", v),
            name="relative_tolerance",
        )
        
        self.absolute_tolerance = LocalOption(
            getter=lambda: self._config.absolute_tolerance,
            setter=lambda v: setattr(self._config, "absolute_tolerance", v),
            name="absolute_tolerance",
        )

        self.report_norms = LocalOption(
            getter=lambda: self._config.absolute_tolerance,
            setter=lambda v: setattr(self._config, "report_norms", v),
            name="report_norms",
        )


    def set(self, **kwargs):
        validate_kwargs(FASCDConfig, kwargs)
        for k, v in kwargs.items():
            setattr(self._config, k, v)

    def __repr__(self):
        return 'fas_options={' + ', '.join([f"{getattr(self,o)}" for o in self.options]) + '}'

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.options))



class VankaOptions:
    """
    Broadcast wrapper around VankaConfig across all solver levels.
    """

    def __init__(self, levels, getter):
        self._levels = levels
        self._getter = getter  # level -> VankaConfig

        self.options = ['omega','relax_phi']

        self.omega = BroadcastOption(self._levels, self._getter, "omega")
        self.relax_phi = BroadcastOption(self._levels, self._getter, "relax_phi")

        self.newton_options = NewtonOptions(
            self._levels,
            getter=lambda lev: self._getter(lev).newton_config,
        )

    def set(self, **kwargs):
        validate_kwargs(VankaConfig, kwargs)
        for lev in self._levels:
            cfg = self._getter(lev)
            for k, v in kwargs.items():
                setattr(cfg, k, v)

    def set_level(self, level_index: int, **kwargs):
        validate_kwargs(VankaConfig, kwargs)
        cfg = self._getter(self._levels[level_index])
        for k, v in kwargs.items():
            setattr(cfg, k, v)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.options))

    def __repr__(self):
        return 'vanka_options={' + ', '.join([f"{getattr(self,o)}" for o in self.options]) + '}'



class NewtonOptions:
    """
    Broadcast wrapper around NewtonConfig across all solver levels.
    """

    def __init__(self, levels, getter):
        self._levels = levels
        self._getter = getter  # level -> NewtonConfig
        self.options = ['steps','relaxation','step_tolerance','ssa_damping','mc_damping']

        self.steps = BroadcastOption(self._levels, self._getter, "steps")
        self.relaxation = BroadcastOption(self._levels, self._getter, "relaxation")
        self.step_tolerance = BroadcastOption(self._levels, self._getter, "step_tolerance")
        self.ssa_damping = BroadcastOption(self._levels, self._getter, "ssa_damping")
        self.mc_damping = BroadcastOption(self._levels, self._getter, "mc_damping")

    def set(self, **kwargs):
        validate_kwargs(NewtonConfig, kwargs)
        for lev in self._levels:
            cfg = self._getter(lev)
            for k, v in kwargs.items():
                setattr(cfg, k, v)

    def set_level(self, level_index: int, **kwargs):
        validate_kwargs(NewtonConfig, kwargs)
        cfg = self._getter(self._levels[level_index])
        for k, v in kwargs.items():
            setattr(cfg, k, v)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.options))

    def __repr__(self):
        return 'newton_options={' + ', '.join([f"{getattr(self,o)}" for o in self.options]) + '}'


class FASAdjointSolver:
    def __init__(self,multigrid):
        self.multigrid = multigrid
        self.levels = [FASAdjointLevel(grid, FASAdjointScratch(grid)) for grid in multigrid.levels]

        self._fas_config = FASAdjointConfig()
        self.fas_options = FASAdjointOptions(self._fas_config)
        
        self.vanka_options = VankaOptions(
            self.levels,
            getter=lambda lev: lev.grid.adjoint_operators.vanka_config,
        )

        self.n_levels = len(self.levels)
        self.dt = None

    def solve(self,dt,start_level=0,report_norms=True,zero_init=True):
        self.dt = cp.float32(dt)
        
        start_level_ = self.multigrid.levels[start_level]

        if zero_init:
            start_level_.adjoint.lambda_u.data.fill(0.0)
            start_level_.adjoint.lambda_v.data.fill(0.0)
            start_level_.adjoint.lambda_ud.data.fill(0.0)
            start_level_.adjoint.lambda_vd.data.fill(0.0)
            start_level_.adjoint.lambda_H.data.fill(0.0)

        ru_init,rv_init,rud_init,rvd_init,rH_init = start_level_.adjoint_operators.compute_residual(dt,return_norms=True)
        initial_residual_norm = cp.sqrt(ru_init**2 + rv_init**2 + rud_init**2 + rvd_init**2 + rH_init**2)
        relative_residual_norm = cp.float32(1.0)

        if self._fas_config.report_norms:
            print(f"  Initial:   |r0|     = {initial_residual_norm:.2e}, "
                  f"|r_u| = {float(ru_init):.2e}, "
                  f"|r_v| = {float(rv_init):.2e}, "
                  f"|r_ud| = {float(rud_init):.2e}, "
                  f"|r_vd| = {float(rvd_init):.2e}, "
                  f"|r_H| = {float(rH_init):.2e}")

        absolute_residual_norm = initial_residual_norm
        iteration = 0

        while (relative_residual_norm > self._fas_config.relative_tolerance
                and absolute_residual_norm > self._fas_config.absolute_tolerance
                and iteration < self._fas_config.maximum_vcycles):
            self.vcycle(start_level,finest=True)
            ru,rv,rud,rvd,rH = start_level_.adjoint_operators.compute_residual(dt,return_norms=True)

            absolute_residual_norm = cp.sqrt(ru**2 + rv**2 + rud**2 + rvd**2 + rH**2)
            relative_residual_norm = absolute_residual_norm / initial_residual_norm
            if self._fas_config.report_norms:
                print(f"  V-cycle {iteration}: |r|/|r0| = {relative_residual_norm:.2e}, "
                      f"|r_u| = {float(ru):.2e}, "
                      f"|r_v| = {float(rv):.2e}, "
                      f"|r_ud| = {float(rud):.2e}, "
                      f"|r_vd| = {float(rvd):.2e}, "
                      f"|r_H| = {float(rH):.2e}")
            iteration += 1
        if iteration < self._fas_config.maximum_vcycles:
            converged = True
        else:
            converged = False
        return converged

    def vcycle(self, l, finest=False):

        coarse = not finest
        mg = self.multigrid
        dt = self.dt
        level = self.levels[l]

        # Coarsest level - smooth to convergence
        if l == self.n_levels - 1:
            level.grid.adjoint_operators.vanka_sweep(dt,
                self._fas_config.coarsest_steps)
            return

        next_level = self.levels[l+1]

        # Pre-smooth
        level.grid.adjoint_operators.vanka_sweep(dt,
            self._fas_config.pre_steps)

        # Restrict forward solution to child - ensures adjoint sees the restriction of the correct forward state
        mg.restrict_vfacet(level.grid.state.u.data,next_level.grid.state.u.data)
        mg.restrict_hfacet(level.grid.state.v.data,next_level.grid.state.v.data)
        mg.restrict_vfacet(level.grid.state.ud.data,next_level.grid.state.ud.data)
        mg.restrict_hfacet(level.grid.state.vd.data,next_level.grid.state.vd.data)
        mg.restrict_cell(level.grid.state.H.data,next_level.grid.state.H.data)
        mg.restrict_cell(level.grid.state.H_prev.data,next_level.grid.state.H_prev.data)
        mg.restrict_cell(level.grid.state.phi.data,next_level.grid.state.phi.data)
        mg.restrict_cell(level.grid.state.mask.data,next_level.grid.state.mask.data,method='max')

        # Restrict adjoint solution to child
        mg.restrict_vfacet(level.grid.adjoint.lambda_u.data,next_level.grid.adjoint.lambda_u.data)
        mg.restrict_hfacet(level.grid.adjoint.lambda_v.data,next_level.grid.adjoint.lambda_v.data)
        mg.restrict_vfacet(level.grid.adjoint.lambda_ud.data,next_level.grid.adjoint.lambda_ud.data)
        mg.restrict_hfacet(level.grid.adjoint.lambda_vd.data,next_level.grid.adjoint.lambda_vd.data)
        mg.restrict_cell(level.grid.adjoint.lambda_H.data,next_level.grid.adjoint.lambda_H.data)

        next_level.scratch.w_lambda_u[:,:] = next_level.grid.adjoint.lambda_u.data[:,:]
        next_level.scratch.w_lambda_v[:,:] = next_level.grid.adjoint.lambda_v.data[:,:]
        next_level.scratch.w_lambda_ud[:,:] = next_level.grid.adjoint.lambda_ud.data[:,:]
        next_level.scratch.w_lambda_vd[:,:] = next_level.grid.adjoint.lambda_vd.data[:,:]
        next_level.scratch.w_lambda_H[:,:] = next_level.grid.adjoint.lambda_H.data[:,:]

        # Compute and restrict adjoint residual
        level.grid.adjoint_operators.compute_residual(dt,use_mask=False)
        mg.restrict_vfacet(level.grid.adjoint_operators.r_u,next_level.grid.adjoint_operators.r_u)
        mg.restrict_hfacet(level.grid.adjoint_operators.r_v,next_level.grid.adjoint_operators.r_v)
        mg.restrict_vfacet(level.grid.adjoint_operators.r_ud,next_level.grid.adjoint_operators.r_ud)
        mg.restrict_hfacet(level.grid.adjoint_operators.r_vd,next_level.grid.adjoint_operators.r_vd)
        mg.restrict_cell(level.grid.adjoint_operators.r_H,next_level.grid.adjoint_operators.r_H)

        next_level.grid.adjoint_operators.compute_vjp(dt,use_mask=False)
        next_level.grid.adjoint_operators.f_u[:,:] = next_level.grid.adjoint_operators.vjp_u[:,:] - next_level.grid.adjoint_operators.r_u[:,:]
        next_level.grid.adjoint_operators.f_v[:,:] = next_level.grid.adjoint_operators.vjp_v[:,:] - next_level.grid.adjoint_operators.r_v[:,:]
        next_level.grid.adjoint_operators.f_ud[:,:] = next_level.grid.adjoint_operators.vjp_ud[:,:] - next_level.grid.adjoint_operators.r_ud[:,:]
        next_level.grid.adjoint_operators.f_vd[:,:] = next_level.grid.adjoint_operators.vjp_vd[:,:] - next_level.grid.adjoint_operators.r_vd[:,:]
        next_level.grid.adjoint_operators.f_H[:,:] = next_level.grid.adjoint_operators.vjp_H[:,:] - next_level.grid.adjoint_operators.r_H[:,:]

        # recursive call
        self.vcycle(l+1)

        # compute coarse_correction
        next_level.scratch.z_lambda_u[:,:] = next_level.grid.adjoint.lambda_u.data[:,:] - next_level.scratch.w_lambda_u[:,:]
        next_level.scratch.z_lambda_v[:,:] = next_level.grid.adjoint.lambda_v.data[:,:] - next_level.scratch.w_lambda_v[:,:]
        next_level.scratch.z_lambda_ud[:,:] = next_level.grid.adjoint.lambda_ud.data[:,:] - next_level.scratch.w_lambda_ud[:,:]
        next_level.scratch.z_lambda_vd[:,:] = next_level.grid.adjoint.lambda_vd.data[:,:] - next_level.scratch.w_lambda_vd[:,:]
        next_level.scratch.z_lambda_H[:,:] = next_level.grid.adjoint.lambda_H.data[:,:] - next_level.scratch.w_lambda_H[:,:]

        mg.prolongate_vfacet(next_level.scratch.z_lambda_u,level.scratch.z_lambda_u,method='bilinear')
        mg.prolongate_hfacet(next_level.scratch.z_lambda_v,level.scratch.z_lambda_v,method='bilinear')
        mg.prolongate_vfacet(next_level.scratch.z_lambda_ud,level.scratch.z_lambda_ud,method='bilinear')
        mg.prolongate_hfacet(next_level.scratch.z_lambda_vd,level.scratch.z_lambda_vd,method='bilinear')
        mg.prolongate_cell(next_level.scratch.z_lambda_H,level.scratch.z_lambda_H,method='bilinear')

        # Apply fine correction
        level.grid.adjoint.lambda_u.data[:,:] += level.scratch.z_lambda_u[:,:]
        level.grid.adjoint.lambda_v.data[:,:] += level.scratch.z_lambda_v[:,:]
        level.grid.adjoint.lambda_ud.data[:,:] += level.scratch.z_lambda_ud[:,:]
        level.grid.adjoint.lambda_vd.data[:,:] += level.scratch.z_lambda_vd[:,:]
        level.grid.adjoint.lambda_H.data[:,:] += level.scratch.z_lambda_H[:,:]

        # Post-smooth
        level.grid.adjoint_operators.vanka_sweep(dt, self._fas_config.post_steps)

        if not coarse:
            level.grid.adjoint_operators.vanka_sweep(dt, self._fas_config.finest_steps)

class FASAdjointScratch:
    def __init__(self,grid):
        ny,nx = grid.ny,grid.nx
        
        self.w_lambda_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.w_lambda_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.w_lambda_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.w_lambda_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.w_lambda_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.z_lambda_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.z_lambda_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.z_lambda_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.z_lambda_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.z_lambda_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

@dataclass
class FASAdjointLevel:
    grid: Grid
    scratch: FASAdjointScratch

@dataclass
class FASAdjointConfig:
    coarsest_steps: int = 200
    pre_steps: int = 10
    post_steps: int = 20
    finest_steps: int = 50
    maximum_vcycles: int = 10
    relative_tolerance: cp.float32 = cp.float32(1e-3)
    absolute_tolerance: cp.float32 = cp.float32(5.0)
    report_norms: bool = True


class FASAdjointOptions:
    """
    User-facing wrapper around a single FASCDConfig.
    """

    def __init__(self, config: FASCDConfig):
        self._config = config

        self.options = ['coarsest_steps',
            'pre_steps',
            'post_steps',
            'finest_steps']

        self.coarsest_steps = LocalOption(
            getter=lambda: self._config.coarsest_steps,
            setter=lambda v: setattr(self._config, "coarsest_steps", v),
            name="coarsest_steps",
        )
        self.pre_steps = LocalOption(
            getter=lambda: self._config.pre_steps,
            setter=lambda v: setattr(self._config, "pre_steps", v),
            name="pre_steps",
        )
        self.post_steps = LocalOption(
            getter=lambda: self._config.post_steps,
            setter=lambda v: setattr(self._config, "post_steps", v),
            name="post_steps",
        )
        self.finest_steps = LocalOption(
            getter=lambda: self._config.finest_steps,
            setter=lambda v: setattr(self._config, "finest_steps", v),
            name="finest_steps",
        )

        self.maximum_vcycles = LocalOption(
            getter=lambda: self._config.maximum_vcycles,
            setter=lambda v: setattr(self._config, "maximum_vcycles", v),
            name="maximum_vcyles",
        )
        
        self.relative_tolerance = LocalOption(
            getter=lambda: self._config.relative_tolerance,
            setter=lambda v: setattr(self._config, "relative_tolerance", v),
            name="relative_tolerance",
        )
        
        self.absolute_tolerance = LocalOption(
            getter=lambda: self._config.absolute_tolerance,
            setter=lambda v: setattr(self._config, "absolute_tolerance", v),
            name="absolute_tolerance",
        )

        self.report_norms = LocalOption(
            getter=lambda: self._config.absolute_tolerance,
            setter=lambda v: setattr(self._config, "report_norms", v),
            name="report_norms",
        )

    def set(self, **kwargs):
        validate_kwargs(FASAdjointConfig, kwargs)
        for k, v in kwargs.items():
            setattr(self._config, k, v)

    def __repr__(self):
        return 'fas_options={' + ', '.join([f"{getattr(self,o)}" for o in self.options]) + '}'

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.options))
