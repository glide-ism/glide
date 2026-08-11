import cupy as cp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

class ForwardOperators:
    def __init__(self,grid,
            use_fast_math=True):

        self.grid = grid

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = ['common.cu', 'viscosity.cu', 'stress.cu', 'flux.cu',
                          'residuals.cu', 'vanka.cu', 'grad.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()

        self.kernels = cp.RawModule(code=cuda_source, options=options)

        self.r_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.r_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.r_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.r_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.r_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.f_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.f_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.f_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.f_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.f_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.F_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.F_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.F_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.F_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.F_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.delta_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.delta_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.delta_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.delta_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.delta_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self._var_u = None
        self._var_v = None
        self._var_ud = None
        self._var_vd = None
        self._var_H = None

        self._jvp_u = None
        self._jvp_v = None
        self._jvp_ud = None
        self._jvp_vd = None
        self._jvp_H = None

        self.gamma = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.gamma.fill(grid.geometry.thklim.value)

        self.vanka_config = VankaConfig()

    @property
    def _kernel_config(self):
        block_size = (16, 16)
        stride = 14
        halo = 1
        grid_size = (self.grid.nx // stride + 1, self.grid.ny // stride + 1)
        return grid_size, block_size, stride, halo

    def compute_residual(self, dt, 
            use_mask=True, 
            operator_only=False, 
            freeze_calving=False, 
            freeze_phi=False,
            return_norms=False):

        kernel = self.kernels.get_function('compute_residual')
        grid_size, block_size, stride, halo = self._kernel_config
  
        grid = self.grid
        state = grid.state
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        if not freeze_phi:
            self.compute_phi()
            self.compute_xi()

        if operator_only:
            out_u = self.F_u
            out_v = self.F_v
            out_ud = self.F_ud
            out_vd = self.F_vd
            out_H = self.F_H
            use_forcing = False
        else:
            out_u = self.r_u
            out_v = self.r_v
            out_ud = self.r_ud
            out_vd = self.r_vd
            out_H = self.r_H
            use_forcing = True

        kernel(grid_size, block_size,
               (out_u, out_v, out_ud, out_vd, out_H,
                state.u.data, state.v.data, state.ud.data, state.vd.data, state.H.data, 
                state.phi.data, state.xi.data, state.mask.data,
                self.f_u, self.f_v, self.f_ud, self.f_vd, self.f_H,
                geometry.bed.data, 
                rheology.B.data, 
                sliding.beta.data,
                self.gamma,
                use_forcing,use_mask,
                rheology.n.value, rheology.eps_reg.value, rheology.H_reg.value,
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, sliding.flotation_reg_sliding.value,
                calving_rate, calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo)) 

        if return_norms:
            return cp.linalg.norm(out_u),cp.linalg.norm(out_v),cp.linalg.norm(out_ud),cp.linalg.norm(out_vd),cp.linalg.norm(out_H)

    def compute_jvp(self, dt, 
            use_mask=True, 
            freeze_calving=False, 
            freeze_phi=False,
            return_norms=False):

        kernel = self.kernels.get_function('compute_jvp')
        grid_size, block_size, stride, halo = self._kernel_config
  
        grid = self.grid
        state = grid.state
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        if not freeze_phi:
            self.compute_phi()
            self.compute_xi()

        kernel(grid_size, block_size,
               (self.jvp_u, self.jvp_v, self.jvp_ud, self.jvp_vd, self.jvp_H,
                state.u.data, state.v.data, state.ud.data, state.vd.data, state.H.data,
                self.var_u, self.var_v, self.var_ud, self.var_vd, self.var_H,
                state.phi.data, state.xi.data, state.mask.data,
                self.f_u, self.f_v, self.f_ud, self.f_vd, self.f_H,
                geometry.bed.data,
                rheology.B.data,
                sliding.beta.data,
                self.gamma,
                use_mask,
                rheology.n.value, rheology.eps_reg.value, rheology.H_reg.value,
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value,
                sliding.water_drag.value, sliding.flotation_reg_sliding.value,
                calving_rate, calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo))

        if return_norms:
            return (cp.linalg.norm(self.jvp_u), cp.linalg.norm(self.jvp_v),
                    cp.linalg.norm(self.jvp_ud), cp.linalg.norm(self.jvp_vd),
                    cp.linalg.norm(self.jvp_H))

    def compute_phi(self, relaxation=cp.float32(0.0)):
        kernel = self.kernels.get_function('compute_grounded')
        grid_size, block_size, stride, halo = self._kernel_config
            
        grid = self.grid
        kernel(grid_size, block_size,
                   (grid.state.phi.data,
                    grid.state.H.data, grid.geometry.depth.data, 
                    grid.geometry.sigmoid_c.value,
                    grid.geometry.sigmoid_k.value,
                    relaxation,
                    grid.ny, grid.nx, 
                    stride, halo))

    def compute_xi(self, relaxation=cp.float32(0.0)):
        kernel = self.kernels.get_function('compute_flotation_fraction')
        grid_size, block_size, stride, halo = self._kernel_config
            
        grid = self.grid
        kernel(grid_size, block_size,
                   (grid.state.xi.data,
                    grid.state.H.data, grid.geometry.depth.data, 
                    grid.geometry.sigmoid_c.value,
                    grid.geometry.sigmoid_k.value,
                    relaxation,
                    grid.ny, grid.nx, 
                    stride, halo))

    def vanka_smooth(self, dt,
            freeze_calving=False,
            freeze_phi=False):

        kernel = self.kernels.get_function('vanka_smooth')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid
        state = grid.state
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        if not freeze_phi:
            self.compute_phi(relaxation=self.vanka_config.relax_phi)
            self.compute_xi(relaxation=self.vanka_config.relax_phi)

        self.delta_u.fill(0.0)
        self.delta_v.fill(0.0)
        self.delta_ud.fill(0.0)
        self.delta_vd.fill(0.0)
        self.delta_H.fill(0.0)
        kernel(grid_size, block_size,
               (self.delta_u, self.delta_v, self.delta_ud, self.delta_vd, self.delta_H, 
                state.mask.data,
                state.u.data, state.v.data, state.ud.data, state.vd.data, state.H.data, 
                state.phi.data, state.xi.data,
                self.f_u, self.f_v, self.f_ud, self.f_vd, self.f_H,
                geometry.bed.data, rheology.B.data, sliding.beta.data,
                self.gamma,
                rheology.n.value, rheology.eps_reg.value, rheology.H_reg.value,
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value,
                sliding.water_drag.value,
                sliding.flotation_reg_sliding.value,
                calving_rate,
                calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo,
                cp.int32(self.vanka_config.newton_config.steps),
                cp.float32(self.vanka_config.newton_config.relaxation),
                cp.float32(self.vanka_config.newton_config.ssa_damping),
                cp.float32(self.vanka_config.newton_config.mc_damping))
        )

    def vanka_sweep(self, dt, n_iter, 
            freeze_calving=False,
            freeze_phi=False):
        for i in range(n_iter):
            self.vanka_smooth(dt,freeze_calving=freeze_calving,freeze_phi=freeze_phi)
            self.grid.state.u.data[:] += self.vanka_config.omega * self.delta_u
            self.grid.state.v.data[:] += self.vanka_config.omega * self.delta_v
            self.grid.state.ud.data[:] += self.vanka_config.omega * self.delta_ud
            self.grid.state.vd.data[:] += self.vanka_config.omega * self.delta_vd
            self.grid.state.H.data[:] += self.vanka_config.omega * self.delta_H
            self.vanka_config.hook_func(i)

    def vanka_dump(self,dt):
        kernel = self.kernels.get_function('vanka_dump')
        grid_size, block_size, stride, halo = self._kernel_config
        grid = self.grid

        J = cp.zeros((self.grid.ny*self.grid.nx,81),dtype=cp.float32)
        r = cp.zeros((self.grid.ny*self.grid.nx,9),dtype=cp.float32)
        kernel(grid_size, block_size,
               (J,r,
                grid.state.u.data, grid.state.v.data, grid.state.ud.data, grid.state.vd.data, grid.state.H.data, grid.state.phi.data, grid.state.xi.data,
                self.f_u, self.f_v, self.f_ud, self.f_vd, self.f_H,
                grid.geometry.bed.data, grid.rheology.B.data, grid.sliding.beta.data, self.gamma,
                grid.rheology.n.value, grid.rheology.eps_reg.value, grid.rheology.H_reg.value, grid.geometry.sigmoid_c.value,
                grid.sliding.m.value, grid.sliding.u_reg.value, grid.sliding.water_drag.value, grid.sliding.flotation_reg_sliding.value,
                grid.calving.calving_rate.value, grid.calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo,
                )
        )

        return J,r
            
    def set_rhs(self,dt):
        self.f_H[:,:] = self.grid.state.H_prev.data/dt + self.grid.forcing.smb.data

    @property
    def var_u(self):
        if self._var_u is None:
            self._var_u = cp.zeros((self.grid.ny,self.grid.nx+1),dtype=cp.float32)
        return self._var_u
    
    @property
    def var_v(self):
        if self._var_v is None:
            self._var_v = cp.zeros((self.grid.ny+1,self.grid.nx),dtype=cp.float32)
        return self._var_v

    @property
    def var_ud(self):
        if self._var_ud is None:
            self._var_ud = cp.zeros((self.grid.ny,self.grid.nx+1),dtype=cp.float32)
        return self._var_ud
    
    @property
    def var_vd(self):
        if self._var_vd is None:
            self._var_vd = cp.zeros((self.grid.ny+1,self.grid.nx),dtype=cp.float32)
        return self._var_vd

    
    @property
    def var_H(self):
        if self._var_H is None:
            self._var_H = cp.zeros((self.grid.ny,self.grid.nx),dtype=cp.float32)
        return self._var_H

    @property
    def jvp_u(self):
        if self._jvp_u is None:
            self._jvp_u = cp.zeros((self.grid.ny,self.grid.nx+1),dtype=cp.float32)
        return self._jvp_u
    
    @property
    def jvp_v(self):
        if self._jvp_v is None:
            self._jvp_v = cp.zeros((self.grid.ny+1,self.grid.nx),dtype=cp.float32)
        return self._jvp_v

    @property
    def jvp_ud(self):
        if self._jvp_ud is None:
            self._jvp_ud = cp.zeros((self.grid.ny,self.grid.nx+1),dtype=cp.float32)
        return self._jvp_ud
    
    @property
    def jvp_vd(self):
        if self._jvp_vd is None:
            self._jvp_vd = cp.zeros((self.grid.ny+1,self.grid.nx),dtype=cp.float32)
        return self._jvp_vd
    
    @property
    def jvp_H(self):
        if self._jvp_H is None:
            self._jvp_H = cp.zeros((self.grid.ny,self.grid.nx),dtype=cp.float32)
        return self._jvp_H

@dataclass
class NewtonConfig:
    steps: int = 30
    relaxation: cp.float32 = cp.float32(0.5)
    ssa_damping: cp.float32 = cp.float32(0.01)
    mc_damping: cp.float32 = cp.float32(1.0)

@dataclass
class VankaConfig:
    omega: cp.float32 = cp.float32(0.5)
    newton_config: NewtonConfig = field(default_factory = lambda: NewtonConfig())
    relax_phi: cp.float32 = cp.float32(0.0)
    hook_interval: int = 1
    hook_func: Callable[[int],None] = field(default_factory = lambda: lambda i:None)


class AdjointOperators:
    def __init__(self,grid,
            use_fast_math=True):

        self.grid = grid

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = ['common.cu', 'viscosity.cu', 'stress.cu', 'flux.cu',
                          'residuals.cu', 'vanka.cu', 'grad.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()

        self.kernels = cp.RawModule(code=cuda_source, options=options)

        self.r_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.r_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.r_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.r_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.r_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.f_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.f_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.f_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.f_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.f_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.vjp_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.vjp_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.vjp_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.vjp_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.vjp_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.delta_lambda_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.delta_lambda_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.delta_lambda_ud = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.delta_lambda_vd = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.delta_lambda_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.gamma = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.gamma.fill(grid.geometry.thklim.value)

        self.vanka_config = VankaConfig()

    @property
    def _kernel_config(self):
        block_size = (16, 16)
        stride = 14
        halo = 1
        grid_size = (self.grid.nx // stride + 1, self.grid.ny // stride + 1)
        return grid_size, block_size, stride, halo

    def compute_residual(self, dt, 
            use_mask=True, 
            freeze_calving=False, 
            return_norms=False):

        kernel = self.kernels.get_function('compute_vjp')
        grid_size, block_size, stride, halo = self._kernel_config
  
        grid = self.grid
        state = grid.state
        adjoint = grid.adjoint
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        self.r_u.fill(0)
        self.r_v.fill(0)
        self.r_ud.fill(0)
        self.r_vd.fill(0)
        self.r_H.fill(0)
        use_forcing=True
        kernel(grid_size, block_size,
               (self.r_u, self.r_v, self.r_ud, self.r_vd, self.r_H,
                state.u.data, state.v.data, state.ud.data, state.vd.data, state.H.data,
                adjoint.lambda_u.data, adjoint.lambda_v.data,
                adjoint.lambda_ud.data, adjoint.lambda_vd.data, adjoint.lambda_H.data,
                state.phi.data, state.xi.data, state.mask.data,
                self.f_u, self.f_v, self.f_ud, self.f_vd, self.f_H,
                geometry.bed.data,
                rheology.B.data,
                sliding.beta.data,
                self.gamma,
                use_forcing, use_mask,
                rheology.n.value, rheology.eps_reg.value, rheology.H_reg.value,
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value,
                sliding.water_drag.value, sliding.flotation_reg_sliding.value,
                calving_rate, calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo))

        if return_norms:
            return (cp.linalg.norm(self.r_u),cp.linalg.norm(self.r_v),
                    cp.linalg.norm(self.r_ud),cp.linalg.norm(self.r_vd),
                    cp.linalg.norm(self.r_H))



    def compute_vjp(self, dt, 
            use_mask=True,
            use_forcing=False,
            freeze_calving=False):

        kernel = self.kernels.get_function('compute_vjp')
        grid_size, block_size, stride, halo = self._kernel_config
  
        grid = self.grid
        state = grid.state
        adjoint = grid.adjoint
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        use_forcing=False
        self.vjp_u.fill(0)
        self.vjp_v.fill(0)
        self.vjp_ud.fill(0)
        self.vjp_vd.fill(0)
        self.vjp_H.fill(0)
        kernel(grid_size, block_size,
               (self.vjp_u, self.vjp_v, self.vjp_ud, self.vjp_vd, self.vjp_H,
                state.u.data, state.v.data, state.ud.data, state.vd.data, state.H.data,
                adjoint.lambda_u.data, adjoint.lambda_v.data,
                adjoint.lambda_ud.data, adjoint.lambda_vd.data, adjoint.lambda_H.data,
                state.phi.data, state.xi.data, state.mask.data,
                self.f_u, self.f_v, self.f_ud, self.f_vd, self.f_H,
                geometry.bed.data,
                rheology.B.data,
                sliding.beta.data,
                self.gamma,
                use_forcing, use_mask,
                rheology.n.value, rheology.eps_reg.value, rheology.H_reg.value,
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value,
                sliding.water_drag.value, sliding.flotation_reg_sliding.value,
                calving_rate, calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo))


    def vanka_smooth(self, dt,
            freeze_calving=False):

        kernel = self.kernels.get_function('vanka_smooth_adjoint')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid
        state = grid.state
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        if freeze_calving:
            calving_rate = cp.float32(0.0)
        else:
            calving_rate = calving.calving_rate.value

        self.delta_lambda_u.fill(0.0)
        self.delta_lambda_v.fill(0.0)
        self.delta_lambda_ud.fill(0.0)
        self.delta_lambda_vd.fill(0.0)
        self.delta_lambda_H.fill(0.0)
        kernel(grid_size, block_size,
               (self.delta_lambda_u, self.delta_lambda_v,
                self.delta_lambda_ud, self.delta_lambda_vd, self.delta_lambda_H,
                state.u.data, state.v.data, state.ud.data, state.vd.data, state.H.data,
                state.phi.data, state.xi.data, state.mask.data,
                self.r_u, self.r_v, self.r_ud, self.r_vd, self.r_H,
                geometry.bed.data, rheology.B.data, sliding.beta.data,
                self.gamma,
                rheology.n.value, rheology.eps_reg.value, rheology.H_reg.value,
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value,
                sliding.water_drag.value,
                sliding.flotation_reg_sliding.value,
                calving_rate,
                calving.flotation_reg_calving.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo,
                cp.float32(self.vanka_config.newton_config.ssa_damping),
                cp.float32(self.vanka_config.newton_config.mc_damping))
        )

    def vanka_sweep(self, dt, n_iter,
            freeze_calving=False):
        for i in range(n_iter):
            self.compute_residual(dt,freeze_calving=freeze_calving)
            self.vanka_smooth(dt,freeze_calving=freeze_calving)
            self.grid.adjoint.lambda_u.data[:] -= self.vanka_config.omega * self.delta_lambda_u
            self.grid.adjoint.lambda_v.data[:] -= self.vanka_config.omega * self.delta_lambda_v
            self.grid.adjoint.lambda_ud.data[:] -= self.vanka_config.omega * self.delta_lambda_ud
            self.grid.adjoint.lambda_vd.data[:] -= self.vanka_config.omega * self.delta_lambda_vd
            self.grid.adjoint.lambda_H.data[:] -= self.vanka_config.omega * self.delta_lambda_H

    def compute_gradient_beta(self):
        kernel = self.kernels.get_function('compute_gradient_beta')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid
        state = grid.state
        adjoint = grid.adjoint
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        sliding.beta.grad.fill(0)
        kernel(grid_size, block_size,
               (sliding.beta.grad,
                state.u.data, state.v.data, state.ud.data, state.vd.data, state.H.data,
                adjoint.lambda_u.data, adjoint.lambda_v.data,
                adjoint.lambda_ud.data, adjoint.lambda_vd.data, adjoint.lambda_H.data,
                state.phi.data, state.xi.data, state.mask.data,
                geometry.bed.data, 
                rheology.B.data, 
                sliding.beta.data,
                self.gamma,
                rheology.n.value, rheology.eps_reg.value, 
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, sliding.flotation_reg_sliding.value,
                calving.calving_rate.value, calving.flotation_reg_calving.value,
                grid.dx, cp.float32(0.0),
                grid.ny, grid.nx, stride, halo)) 

    def compute_gradient_bed(self):
        kernel = self.kernels.get_function('compute_gradient_bed')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid
        state = grid.state
        adjoint = grid.adjoint
        geometry = grid.geometry        
        rheology = grid.rheology
        sliding = grid.sliding
        calving = grid.calving
        forcing = grid.forcing

        geometry.bed.grad.fill(0)
        kernel(grid_size, block_size,
               (geometry.bed.grad,
                state.u.data, state.v.data, state.H.data, 
                adjoint.lambda_u.data, adjoint.lambda_v.data, adjoint.lambda_H.data, 
                state.phi.data, state.xi.data, state.mask.data,
                geometry.bed.data, 
                rheology.B.data, 
                sliding.beta.data,
                self.gamma,
                rheology.n.value, rheology.eps_reg.value, 
                geometry.sigmoid_c.value,
                sliding.m.value, sliding.u_reg.value, 
                sliding.water_drag.value, sliding.flotation_reg_sliding.value,
                calving.calving_rate.value, calving.flotation_reg_calving.value,
                grid.dx, cp.float32(0.0),
                grid.ny, grid.nx, stride, halo)) 


    def compute_gradient_H_prev(self, dt):
        self.grid.state.H_prev.grad[:,:] = -self.grid.adjoint.lambda_H.data[:,:]/dt

    def compute_gradient_smb(self):
        self.grid.forcing.smb.grad[:,:] = -self.grid.adjoint.lambda_H.data[:,:] 
  
    



