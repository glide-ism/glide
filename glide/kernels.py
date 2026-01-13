"""
CUDA kernel loading and multigrid transfer operators.
"""

import cupy as cp
import numpy as np
from pathlib import Path


def make_physics_params(n=3.0, eps_reg=1e-5, water_drag=0.001, calving_rate=1.0,
                        gl_sigmoid_c=0.1, gl_derivatives=False):
    """
    Create a PhysicsParams array for passing to CUDA kernels.

    The array layout matches the CUDA PhysicsParams struct.

    Parameters
    ----------
    n : float
        Glen's flow law exponent (default 3.0)
    eps_reg : float
        Strain rate regularization (default 1e-5)
    water_drag : float
        Drag coefficient for floating ice (default 0.001)
    calving_rate : float
        Calving rate for mass loss at margins (default 1.0)
    gl_sigmoid_c : float
        Sigmoid sharpness for grounding line transitions (default 0.1)
    gl_derivatives : bool
        Include H derivatives through grounding line sigmoid (default False).
        Enabling improves gradient accuracy at grounding lines but may
        destabilize the solver for large time steps.

    Returns
    -------
    cupy.ndarray
        Parameter array on GPU, passed as pointer to kernels
    """
    # Pack as bytes to match CUDA struct layout (5 floats + 1 int)
    params = np.zeros(6, dtype=np.float32)
    params[0] = n
    params[1] = eps_reg
    params[2] = water_drag
    params[3] = calving_rate
    params[4] = gl_sigmoid_c
    # Reinterpret last slot as int32 for the boolean flag
    params_bytes = params.tobytes()
    params_bytes = params_bytes[:20] + np.array([int(gl_derivatives)], dtype=np.int32).tobytes()
    return cp.asarray(np.frombuffer(params_bytes, dtype=np.float32))


class Kernels:
    """
    Container for compiled CUDA kernel modules.

    Loads ice physics kernels and utility (multigrid) kernels from .cu files.
    """

    def __init__(self):
        cuda_dir = Path(__file__).parent / "cuda"

        with open(cuda_dir / "utility.cu", "r") as f:
            self.util = cp.RawModule(code=f.read())

        # Concatenate ice kernel files in dependency order
        ice_files = ['common.cu', 'viscosity.cu', 'stress.cu', 'flux.cu',
                          'residuals.cu', 'vanka.cu', 'grad.cu']
        ice_source = '\n'.join(
            (cuda_dir / f).read_text() for f in ice_files
             )
        
        self.ice = cp.RawModule(code=ice_source, options=("--use_fast_math",))



# Global kernel instance (lazy-loaded)
_kernels = None


def get_kernels():
    """Get or create the global kernel module instance."""
    global _kernels
    if _kernels is None:
        _kernels = Kernels()
    return _kernels


# Multigrid transfer operators

def restrict_vfacet(u_fine, kernels, u_coarse=None):
    """Restrict u-velocity (vertical face) field to coarse grid."""
    kernel = kernels.util.get_function('restrict_u')
    ny, nx_plus1 = u_fine.shape
    nx = nx_plus1 - 1
    ny_coarse = ny // 2
    nx_coarse = nx // 2

    if u_coarse is None:
        u_coarse = cp.empty((ny_coarse, nx_coarse + 1), dtype=cp.float32)

    total_work = ny_coarse * (nx_coarse + 1)
    block_size = 256
    grid_size = (total_work + block_size - 1) // block_size

    kernel((grid_size,), (block_size,),
           (u_fine, u_coarse, ny_coarse, nx_coarse))
    return u_coarse


def restrict_hfacet(v_fine, kernels, v_coarse=None):
    """Restrict v-velocity (horizontal face) field to coarse grid."""
    kernel = kernels.util.get_function('restrict_v')
    ny_plus1, nx = v_fine.shape
    ny = ny_plus1 - 1
    ny_coarse = ny // 2
    nx_coarse = nx // 2

    if v_coarse is None:
        v_coarse = cp.empty((ny_coarse + 1, nx_coarse), dtype=cp.float32)

    total_work = (ny_coarse + 1) * nx_coarse
    block_size = 256
    grid_size = (total_work + block_size - 1) // block_size

    kernel((grid_size,), (block_size,),
           (v_fine, v_coarse, ny_coarse, nx_coarse))
    return v_coarse


def restrict_cell_centered(f_fine, kernels, f_coarse=None):
    """Restrict cell-centered field to coarse grid (full-weighting)."""
    kernel = kernels.util.get_function('restrict_cell_centered')
    ny, nx = f_fine.shape
    ny_coarse = ny // 2
    nx_coarse = nx // 2

    if f_coarse is None:
        f_coarse = cp.empty((ny_coarse, nx_coarse), dtype=cp.float32)

    total_work = ny_coarse * nx_coarse
    block_size = 256
    grid_size = (total_work + block_size - 1) // block_size

    kernel((grid_size,), (block_size,),
           (f_fine, f_coarse, ny_coarse, nx_coarse))
    return f_coarse


def restrict_max_pool(f_fine, kernels, f_coarse=None):
    """Restrict cell-centered field using max pooling."""
    kernel = kernels.util.get_function('restrict_max_pool')
    ny, nx = f_fine.shape
    ny_coarse = ny // 2
    nx_coarse = nx // 2

    if f_coarse is None:
        f_coarse = cp.empty((ny_coarse, nx_coarse), dtype=cp.float32)

    total_work = ny_coarse * nx_coarse
    block_size = 256
    grid_size = (total_work + block_size - 1) // block_size

    kernel((grid_size,), (block_size,),
           (f_fine, f_coarse, ny_coarse, nx_coarse))
    return f_coarse


def prolongate_vfacet(u_coarse, kernels, u_fine=None, smooth=True):
    """Prolongate u-velocity (vertical face) field to fine grid."""
    if smooth:
        kernel = kernels.util.get_function('prolongate_u_smooth')
    else:
        kernel = kernels.util.get_function('prolongate_u')

    ny, nx_plus1 = u_coarse.shape
    nx = nx_plus1 - 1
    ny_fine = ny * 2
    nx_fine = nx * 2

    if u_fine is None:
        u_fine = cp.empty((ny_fine, nx_fine + 1), dtype=cp.float32)

    total_work = ny_fine * (nx_fine + 1)
    block_size = 256
    grid_size = (total_work + block_size - 1) // block_size

    kernel((grid_size,), (block_size,),
           (u_coarse, u_fine, ny_fine, nx_fine))
    return u_fine


def prolongate_hfacet(v_coarse, kernels, v_fine=None, smooth=True):
    """Prolongate v-velocity (horizontal face) field to fine grid."""
    if smooth:
        kernel = kernels.util.get_function('prolongate_v_smooth')
    else:
        kernel = kernels.util.get_function('prolongate_v')

    ny_plus1, nx = v_coarse.shape
    ny = ny_plus1 - 1
    ny_fine = ny * 2
    nx_fine = nx * 2

    if v_fine is None:
        v_fine = cp.empty((ny_fine + 1, nx_fine), dtype=cp.float32)

    total_work = (ny_fine + 1) * nx_fine
    block_size = 256
    grid_size = (total_work + block_size - 1) // block_size

    kernel((grid_size,), (block_size,),
           (v_coarse, v_fine, ny_fine, nx_fine))
    return v_fine


def prolongate_cell_centered(H_coarse, kernels, H_fine=None, smooth=True):
    """Prolongate cell-centered field to fine grid."""
    if smooth:
        kernel = kernels.util.get_function('prolongate_H_smooth')
    else:
        kernel = kernels.util.get_function('prolongate_cell_centered')

    ny, nx = H_coarse.shape
    ny_fine = ny * 2
    nx_fine = nx * 2

    if H_fine is None:
        H_fine = cp.empty((ny_fine, nx_fine), dtype=cp.float32)

    total_work = ny_fine * nx_fine
    block_size = 256
    grid_size = (total_work + block_size - 1) // block_size

    kernel((grid_size,), (block_size,),
           (H_coarse, H_fine, ny_fine, nx_fine))
    return H_fine
