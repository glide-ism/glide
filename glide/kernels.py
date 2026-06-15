"""Backend-agnostic GPU kernel loading and launch.

Hides the difference between the two kernel execution models:

* **CuPy / CUDA** — ``cupy.RawModule`` compiled from ``glide/cuda/*.cu``.
  Kernels take mixed array + scalar arguments and a 2-D ``(grid, block)``
  launch configuration; calls pass straight through.

* **macmetalpy / Metal** — ``macmetalpy.RawKernel`` compiled from
  ``glide/metal/*.metal``.  Metal's RawKernel can only receive ndarray
  *buffers* (no scalar args) and only a flat thread count.  This module
  therefore, for every launch:

    1. splits the argument tuple into ndarray buffers and trailing scalars
       (in every glide kernel all arrays come first, then all scalars),
    2. packs the scalars into a single ``float32`` "params" buffer, appended
       as the last buffer — kernels read ``params[k]`` (``(int)`` / ``!= 0``
       for ints/bools) in the same order the scalars appear in the call,
    3. dispatches ``prod(grid) * prod(block)`` threads.  This is the original
       CUDA thread count, which always covers the kernel's output domain; the
       MSL kernels bounds-check ``[[thread_position_in_grid]]`` and the extra
       threads return early.

Call sites use the identical ``library.get_function(name)(grid, block, args)``
signature on both backends.
"""

import math
from pathlib import Path

import numpy as np

from .backend import xp, BACKEND

__all__ = ["KernelLibrary"]

_PKG = Path(__file__).parent


def _prod(dims):
    if isinstance(dims, (tuple, list)):
        return int(math.prod(dims))
    return int(dims)


class _CudaKernel:
    """Pass-through wrapper around a ``cupy`` raw kernel function."""

    __slots__ = ("_fn",)

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, grid, block, args):
        self._fn(grid, block, args)


class _MetalKernel:
    """Adapts macmetalpy's buffers-only RawKernel to the CUDA call shape."""

    __slots__ = ("_kernel",)

    def __init__(self, source, name):
        self._kernel = xp.RawKernel(source, name)

    def __call__(self, grid, block, args):
        buffers = []
        scalars = []
        for a in args:
            if isinstance(a, xp.ndarray):
                buffers.append(a)
            else:
                scalars.append(float(a))
        params = xp.asarray(np.asarray(scalars, dtype=np.float32))
        total = _prod(grid) * _prod(block)
        self._kernel(int(total), buffers + [params])


class KernelLibrary:
    """Compile a set of kernel source files for the active backend.

    Parameters
    ----------
    stems : sequence of str
        Source-file base names (without extension), in dependency order.
        Resolved to ``glide/cuda/<stem>.cu`` for cupy or
        ``glide/metal/<stem>.metal`` for macmetalpy.
    use_fast_math : bool
        Passed through as ``--use_fast_math`` to the CUDA compiler (Metal
        enables fast-math by default, so it is ignored there).
    """

    def __init__(self, stems, use_fast_math=True):
        if BACKEND == "cupy":
            source = "\n".join(
                (_PKG / "cuda" / f"{s}.cu").read_text() for s in stems
            )
            options = ("--use_fast_math",) if use_fast_math else ()
            self._module = xp.RawModule(code=source, options=options)
            self._metal_source = None
        else:
            self._module = None
            self._metal_source = "\n".join(
                (_PKG / "metal" / f"{s}.metal").read_text() for s in stems
            )
        self._cache = {}

    def get_function(self, name):
        kern = self._cache.get(name)
        if kern is None:
            if self._module is not None:
                kern = _CudaKernel(self._module.get_function(name))
            else:
                kern = _MetalKernel(self._metal_source, name)
            self._cache[name] = kern
        return kern
