# GLIDE

**GPU-accelerated Lightweight Ice Dynamics Engine**

A CUDA-accelerated ice sheet model implementing the shallow shelf approximation (SSA) with support for forward simulation and adjoint-based inverse modeling.

## Features

- **GPU-accelerated**: All computations run on the GPU via custom kernels — NVIDIA/CUDA (CuPy) or Apple Silicon/Metal (macmetalpy), selectable at runtime
- **FASCD multigrid solver**: Full Approximation Scheme with Constrained Descent, using a Vanka smoother and Newton linearization to coupled-solve momentum and mass conservation
- **Coupled physics**: Simultaneous solution of velocity and ice thickness, with a sigmoid grounding-line treatment and non-conservative calving
- **Built-in adjoint**: A matching FAS adjoint solver computes gradients with respect to basal friction, bed, initial thickness, and surface mass balance
- **PyTorch integration**: A differentiable `torch.autograd.Function` wraps a single GLIDE time step, so inverse problems can be driven with standard PyTorch optimizers

## Installation

```bash
git clone https://github.com/glide-ism/glide.git
cd glide
pip install -e .
```

To include the differentiable PyTorch step (used by the inverse examples) or the
dependencies needed to run the examples:

```bash
pip install -e ".[torch]"      # adds torch
pip install -e ".[examples]"   # adds torch, matplotlib, polars, rioxarray, shapely
```

### Requirements

- Python >= 3.9
- A GPU backend (see below)
- NumPy, SciPy, xarray, h5py, h5netcdf, zarr, pyproj, geopandas
- gdown (automatic download of preprocessed datasets)

### GPU backend

glide runs on either of two array backends, selected at runtime by the
`GLIDE_BACKEND` environment variable (`cupy` or `macmetalpy`). If unset, it
auto-detects: CuPy when importable, otherwise macmetalpy.

**NVIDIA / CUDA (CuPy)** — runs the `glide/cuda/*.cu` kernels:

```bash
pip install -e ".[cuda13]"     # or cuda12 / cuda11 to match your toolkit
```

**Apple Silicon / Metal (macmetalpy)** — runs the `glide/metal/*.metal` kernels:

```bash
pip install -e ".[metal]"
python -m metalgpu build        # one-time: build the native Metal library (needs cmake)
GLIDE_BACKEND=macmetalpy python your_script.py
```

## Quick Start

The package does not re-export symbols from its top level; import from the
submodules (`glide.model`, `glide.data`, `glide.io`, `glide.torch`).

```python
import cupy as cp
import numpy as np
import pyproj

from glide.model import IceDynamics
from glide.data import load_greenland_preprocessed
from glide.io import ZarrWriter

# Load a preprocessed dataset (auto-downloaded and cached)
dataset = load_greenland_preprocessed()

# Build the model and its multigrid hierarchy.
# ny and nx must both be divisible by 2^(n_levels - 1).
ny, nx, dx = dataset.ny, dataset.nx, dataset.dx
model = IceDynamics(n_levels=6, ny=ny, nx=nx, dx=dx,
                    x0=dataset.x[0].item(), y0=dataset.y[0].item(),
                    crs=pyproj.CRS("EPSG:3413"))
mg = model.mg

# Initialize state and parameters through the field managers
mg.state.H.set(dataset.thickness.values)
mg.state.H_prev.set(dataset.thickness.values)
mg.geometry.bed.set(dataset.bed.values)
mg.geometry.depth.set(np.maximum(-dataset.bed.values, 0))

B = cp.full((ny, nx), 1e-17 ** (-1.0 / 3.0) / (917 * 9.81), dtype=cp.float32)
mg.rheology.B.set(B)
mg.rheology.n.set(3.0)
mg.rheology.eps_reg.set(1e-6)

mg.sliding.beta.set(cp.full((ny, nx), 2.5, dtype=cp.float32))
mg.sliding.m.set(1.0 / 3.0)
mg.forcing.smb.set(dataset.smb.values)

# Configure the solver
model.forward_solver.fas_options.set(
    coarsest_steps=200, pre_steps=10, post_steps=150, finest_steps=0,
    relative_tolerance=1e-2, absolute_tolerance=10.0)

# Time-stepping forward simulation (updates state in place)
writer = ZarrWriter('run.zarr',
                    static_fields={'bed': mg[0].geometry.bed},
                    dynamic_fields={'H': mg[0].state.H,
                                    'u': mg[0].state.u,
                                    'v': mg[0].state.v})
writer.initialize(mg[0], overwrite=True)

t, dt = cp.float32(0.0), cp.float32(25.0)
while t < 1000.0:
    model.forward(t, dt)
    t += dt
    writer.append(mg[0], time=t)
writer.consolidate_metadata()
```

### Gradients

For gradient-based inverse modeling there are two routes:

- **Built-in adjoint**: `model.backward(t, dt, dJdu=..., dJdv=..., dJdH=...)` runs
  the FAS adjoint solver and populates the gradients with respect to `beta`, `bed`,
  `H_prev`, and `smb` on the active level.
- **PyTorch**: `glide.torch.GlideStep` exposes a single time step as a
  differentiable `autograd.Function`, so an objective can be backpropagated and the
  parameters optimized with `torch.optim` (see the inverse examples).

## Architecture

The model is organized around an `IceDynamics` object that owns a `Multigrid`
hierarchy. Per-level fields are grouped into managers (`state`, `geometry`,
`rheology`, `sliding`, `calving`, `forcing`), each value set via a `Field`'s
`.set(...)`. Levels are reached with `mg.levels[i]` or `mg[i]`.

### Module Structure

```
glide/
├── model.py       # IceDynamics — forward()/backward() top-level API
├── multigrid.py   # Multigrid hierarchy, FASCDSolver, FASAdjointSolver, options
├── grid.py        # Per-level Grid and the State/Geometry/Rheology/Sliding/Calving/Forcing dataclasses
├── field.py       # Field/Constant abstractions over CuPy arrays
├── operators.py   # ForwardOperators / AdjointOperators (kernel dispatch)
├── io.py          # write_vti, VTIWriter (VTI/PVD), ZarrWriter
├── data.py        # Dataset download/caching and preprocessing utilities
├── torch.py       # GlideStep — differentiable autograd.Function wrapper
└── cuda/
    ├── common.cu      # Shared device helpers
    ├── stress.cu      # SSA stress / momentum residuals
    ├── viscosity.cu   # Glen's-law effective viscosity
    ├── flux.cu        # Mass-conservation fluxes and calving
    ├── grad.cu        # Adjoint gradient kernels
    ├── residuals.cu   # Coupled residual evaluation
    ├── vanka.cu       # Vanka smoother
    └── transfer.cu    # Multigrid restriction/prolongation
```

## Physics

GLIDE solves the vertically-integrated shallow shelf approximation (SSA) coupled
to mass conservation:

**Momentum balance:**
```
∇·(2ηH(2ε̇ + tr(ε̇)I)) - β·u = ρgH∇s
```

**Mass conservation:**
```
∂H/∂t + ∇·(Hu) = SMB
```

where:
- η: effective viscosity (Glen's flow law, parameters `B`, `n`, `eps_reg`)
- H: ice thickness
- u: velocity vector
- β: basal friction coefficient (sliding law parameters `beta`, `m`, `water_drag`)
- s: surface elevation
- SMB: surface mass balance

The grounded-to-floating transition is handled with a smoothed (sigmoid)
flotation criterion (`geometry.sigmoid_c`, `geometry.sigmoid_k`), and a
non-conservative calving flux is applied between adjacent floating cells
(`calving.calving_rate`).

> Note: driving stress is measured in units of head, so the `ρg` factor is folded
> into the definitions of `beta` and `B` (see the example scripts).

## Data

`glide.data` provides loaders that download and cache preprocessed inputs:

- `load_greenland_preprocessed()`
- `load_antarctica_preprocessed()`
- `load_wrangell_preprocessed()`
- `load_bitterroot_dem()`

plus lower-level utilities for ingesting BedMachine geometry, velocity mosaics,
and MAR/RACMO surface mass balance, and interpolating them onto a model grid.

## Examples

See the `examples/` directory:

- `greenland/greenland_forward.py` — time-dependent Greenland simulation
- `greenland/greenland_inverse.py` — infer basal friction from observed velocities (PyTorch)
- `antarctica/` — Antarctic forward/inverse runs and dataset construction
- `ismip-hom/` — ISMIP-HOM benchmark
- `bitterroot/`, `wrangell/` — mountain-glacier examples and preprocessing

## License

GPL-3.0. See [LICENSE](LICENSE) for details.
