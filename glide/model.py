"""
Core ice physics API.

Provides the IcePhysics class that wraps the forward model and adjoint
computations into a clean interface.
"""

import cupy as cp
from .grid import Grid
from .multigrid import Multigrid, FASCDSolver,FASAdjointSolver

class IceDynamics:
    def __init__(self,mg=None,
            n_levels=None,grid=None,
            ny=None,nx=None,dx=None,
            x0=cp.float32(0.0),y0=cp.float32(0.0),crs=None):
        if mg is not None:
            self.mg = mg
        elif grid is not None and n_levels is not None:
            self.mg = Multigrid(n_levels,finest_grid=grid)
        elif ny and nx and dx and n_levels:
            self.mg = Multigrid(n_levels,ny=ny,nx=nx,dx=dx,
                   x0=x0,y0=y0,crs=crs)
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
        




        

