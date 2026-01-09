# GLIDE

**GPU-accelerated Lightweight Ice Dynamics Engine**

A CUDA-accelerated ice sheet model implementing the shallow shelf approximation (SSA) with support for forward simulation and adjoint-based inverse modeling.

## Features

- **GPU-accelerated**: All computations run on NVIDIA GPUs via CuPy
- **Multigrid solver**: Efficient FASCD (Full Approximation Scheme with Constrained Descent) V-cycles
- **Coupled physics**: Simultaneous solution of momentum and mass conservation
- **Automatic differentiation**: Built-in adjoint for gradient computation
- **Inverse modeling**: Infer basal friction from observed velocities

## Installation

```bash
git clone https://github.com/username/glide.git
cd glide
pip install -e .
```

### Requirements

- Python >= 3.8
- NVIDIA GPU with CUDA support
- CuPy (install for your CUDA version, e.g., `pip install cupy-cuda12x`)

Optional dependencies for data loading:
```bash
pip install netCDF4 rasterio
```

## Quick Start

```python
import cupy as cp
from glide import IcePhysics

# Create model
physics = IcePhysics(ny=512, nx=512, dx=1500.0, n_levels=5)

# Set geometry and parameters
physics.set_geometry(bed, thickness)
physics.set_parameters(B=rate_factor, beta=basal_friction, smb=surface_mass_balance)

# Forward simulation
u, v, H = physics.forward(dt=10.0, n_vcycles=3)

# Adjoint for gradient computation (inverse modeling)
grad_beta = physics.adjoint(dL_du, dL_dv)
```

## Architecture

The core API is structured around:

```
IcePhysics(H_0, parameters, dt) -> (u, v, H_1)
```

and its reverse-mode automatic differentiation.

### Module Structure

```
glide/
├── physics.py     # Core IcePhysics API
├── grid.py        # MAC staggered grid hierarchy
├── solver.py      # Multigrid V-cycle solvers
├── kernels.py     # CUDA kernel loading
├── io.py          # VTI/HDF5 output writers
├── data.py        # Data download utilities
└── cuda/
    ├── ice_kernels.cu      # SSA physics kernels
    └── utility_kernels.cu  # Multigrid transfer operators
```

## Physics

GLIDE solves the vertically-integrated shallow shelf approximation (SSA):

**Momentum balance:**
```
∇·(2ηH(2ε̇ + tr(ε̇)I)) - β·u = ρgH∇s
```

**Mass conservation:**
```
∂H/∂t + ∇·(Hu) = SMB
```

where:
- η: effective viscosity (Glen's flow law)
- H: ice thickness
- u: velocity vector
- β: basal friction coefficient
- s: surface elevation
- SMB: surface mass balance

## Examples

See the `examples/` directory:

- `greenland_forward.py`: Time-dependent Greenland simulation
- `greenland_inverse.py`: Infer basal friction from observed velocities

## License

GPL-3.0. See [LICENSE](LICENSE) for details.
