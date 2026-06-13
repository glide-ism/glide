"""Array backend selection for glide.

glide's compute core can run on two GPU array backends:

* **cupy**       — NVIDIA CUDA (loads the ``glide/cuda/*.cu`` kernels)
* **macmetalpy** — Apple Metal  (loads the ``glide/metal/*.metal`` kernels)

The backend is chosen by the ``GLIDE_BACKEND`` environment variable
(``cupy`` or ``macmetalpy``).  If unset, it auto-detects: CuPy is used when it
is importable (i.e. a CUDA toolkit/GPU is present), otherwise macmetalpy.

Everything else in glide imports the array module from here as::

    from glide.backend import xp as cp

so the same source runs on either backend.
"""

import os

__all__ = ["xp", "BACKEND", "NDArray", "asnumpy"]


def _load(name):
    name = name.strip().lower()
    if name == "cupy":
        import cupy as mod
        return mod, "cupy"
    if name in ("macmetalpy", "metal"):
        import macmetalpy as mod
        return mod, "macmetalpy"
    raise ValueError(
        f"Unknown GLIDE_BACKEND={name!r}; expected 'cupy' or 'macmetalpy'."
    )


_requested = os.environ.get("GLIDE_BACKEND", "").strip()
if _requested:
    xp, BACKEND = _load(_requested)
else:
    try:
        import cupy as xp  # noqa: F401
        BACKEND = "cupy"
    except Exception:
        import macmetalpy as xp  # noqa: F401
        BACKEND = "macmetalpy"


# Type alias used for array annotations across the package.  CuPy ships a
# typing helper; macmetalpy does not, so fall back to the concrete ndarray.
if BACKEND == "cupy":
    try:
        from cupy.typing import NDArray
    except Exception:  # pragma: no cover - very old cupy
        NDArray = xp.ndarray
else:
    NDArray = xp.ndarray


def _patch_macmetalpy():
    """Work around a macmetalpy bug in ``linalg._get_np``.

    macmetalpy.linalg defines ``_get_np`` twice; the second (shadowing)
    definition recurses into itself when an array is GPU-resident
    (``_np_data is None``) instead of synchronizing — so ``cp.linalg.norm`` on
    any kernel-output array hits ``RecursionError``.  We reinstall a correct
    version that syncs and returns the host view.
    """
    try:
        from macmetalpy import linalg as _linalg
        from macmetalpy._metal_backend import MetalBackend

        def _get_np(a):
            np_data = a._np_data
            if np_data is not None:
                return np_data
            MetalBackend().synchronize()
            return a._get_view()

        _linalg._get_np = _get_np
    except Exception:
        pass

    # macmetalpy.random.randn(*shape) lacks CuPy's `dtype=` kwarg, which glide's
    # tests/examples pass.  Wrap it to accept and apply dtype.
    try:
        from macmetalpy import random as _random
        _orig_randn = _random.randn

        def _randn(*shape, dtype=None):
            out = _orig_randn(*shape)
            if dtype is not None:
                out = out.astype(dtype)
            return out

        _random.randn = _randn
        xp.random.randn = _randn
    except Exception:
        pass

    # macmetalpy ndarray lacks numpy/cupy-style __format__, so f"{x:.2e}" on a
    # 0-d array raises.  Format 0-d arrays as their scalar value.
    try:
        def _format(self, spec):
            if getattr(self, "ndim", None) == 0:
                return format(float(self), spec)
            return str(self) if spec == "" else format(str(self), spec)

        xp.ndarray.__format__ = _format
    except Exception:
        pass

    # macmetalpy's `nextafter` ufunc emits MSL `nextafter()`, which the Metal
    # shading language does not provide (compile error: "use of undeclared
    # identifier 'nextafter'").  Post-process the elementwise shader to use an
    # emulated `metal_nextafter` (bit-increment toward the target).
    try:
        from macmetalpy import _kernel_cache, _kernels

        _helper = (
            "\ninline float metal_nextafter(float a, float b) {\n"
            "    if (isnan(a) || isnan(b)) return a + b;\n"
            "    if (a == b) return b;\n"
            "    if (a == 0.0f) { uint i = 1u; if (b < 0.0f) i |= 0x80000000u; return as_type<float>(i); }\n"
            "    uint i = as_type<uint>(a);\n"
            "    bool toward_larger_mag = (b > a) == (a > 0.0f);\n"
            "    if (toward_larger_mag) i += 1u; else i -= 1u;\n"
            "    return as_type<float>(i);\n"
            "}\n"
        )
        _orig_elementwise = _kernels.elementwise_shader

        def _patched_elementwise(metal_type, fast_math=False):
            src = _orig_elementwise(metal_type, fast_math=fast_math)
            if "nextafter(" in src and "metal_nextafter" not in src:
                src = src.replace("kernel void nextafter_op",
                                  _helper + "\nkernel void nextafter_op", 1)
                src = src.replace("out[id] = nextafter(",
                                  "out[id] = metal_nextafter(")
            return src

        _kernels.elementwise_shader = _patched_elementwise
        _kernel_cache._GENERATORS["elementwise"] = _patched_elementwise
        _kernel_cache.KernelCache().clear()
    except Exception:
        pass


if BACKEND == "macmetalpy":
    _patch_macmetalpy()


def asnumpy(a):
    """Return a NumPy array view/copy of a backend array.

    Mirrors ``cupy.asnumpy``; macmetalpy arrays expose ``.get()`` instead.
    Plain NumPy / scalar inputs pass through.
    """
    import numpy as np

    if BACKEND == "cupy":
        return xp.asnumpy(a)
    if isinstance(a, xp.ndarray):
        return a.get()
    return np.asarray(a)
