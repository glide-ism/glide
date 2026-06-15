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
    """Apply in-memory work-arounds for upstream macmetalpy bugs.

    All four are still present in the latest macmetalpy and are patched at
    runtime (not by editing site-packages) so they survive a reinstall of
    either glide or macmetalpy:

    1. ``linalg._get_np`` is defined twice; the shadowing definition recurses
       into itself when an array is GPU-resident (``_np_data is None``) instead
       of synchronizing, so ``cp.linalg.norm`` on a kernel-output array hits
       ``RecursionError``.  Reinstall a correct version.
    2. ``random.randn`` lacks CuPy's ``dtype=`` kwarg that glide passes.
    3. ndarray has no ``__format__``, so f-string formatting of a 0-d array
       (e.g. ``f"{x:.2e}"``) raises ``TypeError``.
    4. The ``nextafter`` ufunc emits MSL ``nextafter()``, which Metal does not
       provide — that broken kernel poisons the whole element-wise GPU library
       (CPU fallback below the GPU threshold, ``Segmentation fault: 11`` above
       it).  Inject a correct ``nextafter`` into every shader header instead.
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
    # Shading Language does not provide (compile error: "use of undeclared
    # identifier 'nextafter'").  Upstream this one broken kernel poisons the
    # whole element-wise GPU library: small arrays silently fall back to CPU,
    # while arrays above macmetalpy's GPU threshold crash with
    # ``Segmentation fault: 11`` (the library fails to compile and the null
    # result is dereferenced instead of raising).
    #
    # The fix is to define a correct float ``nextafter`` (ULP bit-step) in the
    # ``_MSL_HEADER`` that macmetalpy prepends to every generated shader.  We
    # patch the header in-memory for *all three* code generators —
    # ``_kernels`` (element-wise), ``_kernel_cache`` (cached inline) and
    # ``_fusion`` (fused ops) — because each emits ``nextafter()`` through its
    # own header.  Doing it here, in glide, keeps the fix alive across a
    # ``pip install``/reinstall of either package (unlike editing macmetalpy's
    # site-packages files directly, which any reinstall silently wipes).
    _nextafter_def = (
        "// Injected by glide: Metal stdlib has no nextafter(); ULP bit-step.\n"
        "inline float nextafter(float from, float to) {\n"
        "    if (isnan(from) || isnan(to)) return NAN;\n"
        "    if (from == to) return to;\n"
        "    if (from == 0.0f) return copysign(as_type<float>(1), to);\n"
        "    int i = as_type<int>(from);\n"
        "    i += ((to > from) == (from > 0.0f)) ? 1 : -1;\n"
        "    return as_type<float>(i);\n"
        "}\n"
    )
    _anchor = "using namespace metal;\n"
    import importlib

    for _modname in ("_kernels", "_kernel_cache", "_fusion"):
        try:
            _mod = importlib.import_module(f"macmetalpy.{_modname}")
        except Exception:
            continue
        _hdr = getattr(_mod, "_MSL_HEADER", None)
        if not isinstance(_hdr, str) or "nextafter" in _hdr:
            # Header missing, or upstream already provides nextafter -> leave it.
            continue
        if _anchor in _hdr:
            _mod._MSL_HEADER = _hdr.replace(_anchor, _anchor + _nextafter_def, 1)
        else:
            _mod._MSL_HEADER = _nextafter_def + _hdr

    # Invalidate any shader sources cached (singleton dict + lru_cache'd inline
    # generators) before the header was patched, so they regenerate with it.
    try:
        from macmetalpy import _kernel_cache as _kc, _kernels as _kn

        _kc.KernelCache().clear()
        for _m in (_kc, _kn):
            for _name in dir(_m):
                _fn = getattr(_m, _name, None)
                if hasattr(_fn, "cache_clear"):
                    _fn.cache_clear()
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
