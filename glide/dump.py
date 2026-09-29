"""
Save and replay single forward solves.

`save_solve_state` writes everything one FASCDSolver.solve needs to be
repeated in isolation -- the start level's state, geometry, rheology,
sliding, calving and forcing fields, the time step, and the FAS / Vanka /
Newton settings -- to one .npz. `load_solve_state` rebuilds an IceDynamics
whose finest level is that grid (with the same number of levels below it)
and applies the saved settings, so

    model, dt, meta = load_solve_state(path)
    model.forward_solver.solve(dt)

repeats the solve; change any option in between to experiment. The solver
writes these files itself when FASCDConfig.dump_dir is set (solves that end
unconverged or non-finite).
"""
import json
from dataclasses import asdict, fields

import cupy as cp
import numpy as np

_CELL_FIELDS = {
    'state': ('H', 'H_prev', 'mask', 'phi', 'psi', 'xi', 'dxi_dH', 'dpsi_dH', 'dpsi_dbed'),
    'geometry': ('bed', 'depth'),
    'rheology': ('B',),
    'sliding': ('beta',),
    'calving': ('q', 'h0'),
    'forcing': ('smb',),
}
_FACET_FIELDS = ('u', 'v', 'ud', 'vd')
_CONSTANTS = {
    'geometry': ('thklim', 'sigmoid_c'),
    'rheology': ('n', 'eps_reg', 'H_reg'),
    'sliding': ('m', 'u_reg', 'water_drag', 'p', 'u0', 'N_scale_H', 'N_floor_H'),
    'calving': ('timescale', 'H_c'),
}


def _plain(v):
    if isinstance(v, (cp.ndarray, np.ndarray, np.generic, cp.generic)):
        return float(v)
    return v


def snapshot(grid):
    """Device copies of the fields a solve starts from (cheap; kept in memory
    until the solve ends)."""
    snap = {}
    for group, names in _CELL_FIELDS.items():
        g = getattr(grid, group)
        for n in names:
            f = getattr(g, n, None)
            if f is not None and f.data is not None:
                snap[f'{group}.{n}'] = f.data.copy()
    for n in _FACET_FIELDS:
        f = getattr(grid.state, n, None)
        if f is not None and f.data is not None:
            snap[f'state.{n}'] = f.data.copy()
    return snap


def save_solve_state(path, grid, snap, dt, solver, start_level, info=None):
    """Write `snap` (from `snapshot` at the start of the solve) plus the grid
    description, constants and solver settings to `path` (.npz)."""
    arrays = {k: cp.asnumpy(v) for k, v in snap.items()}
    consts = {}
    for group, names in _CONSTANTS.items():
        g = getattr(grid, group)
        for n in names:
            consts[f'{group}.{n}'] = _plain(getattr(g, n).value)
    vc = grid.forward_operators.vanka_config
    nc = vc.newton_config
    meta = {
        'dt': float(dt),
        'ny': int(grid.ny), 'nx': int(grid.nx), 'dx': float(grid.dx),
        'x0': float(grid.x0), 'y0': float(grid.y0),
        'n_levels': int(solver.n_levels - start_level),
        'stress_scheme': 'ssa' if grid.ssa else 'molho',
        'constants': consts,
        'fas': {k: _plain(v) for k, v in asdict(solver._fas_config).items()},
        'vanka': {'omega': _plain(vc.omega), 'relax_phi': _plain(vc.relax_phi)},
        'newton': {f.name: _plain(getattr(nc, f.name)) for f in fields(nc)},
        'info': info or {},
    }
    np.savez(path, meta=json.dumps(meta), **arrays)
    return path


def load_solve_state(path, n_levels=None, use_fast_math=True):
    """Rebuild a model from a save_solve_state file. Returns (model, dt, meta).
    `n_levels` overrides the saved hierarchy depth (1 = smoothing only)."""
    from .model import IceDynamics
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['meta']))
    nl = meta['n_levels'] if n_levels is None else int(n_levels)
    model = IceDynamics(n_levels=nl, ny=meta['ny'], nx=meta['nx'], dx=cp.float32(meta['dx']),
                        x0=cp.float32(meta['x0']), y0=cp.float32(meta['y0']),
                        stress_scheme=meta['stress_scheme'])
    mg = model.mg
    for key, val in meta['constants'].items():
        group, name = key.split('.')
        getattr(getattr(mg.levels[0], group), name).set(val)
        for lev in mg.levels[1:]:
            getattr(getattr(lev, group), name).set(val)
    for key in d.files:
        if key == 'meta':
            continue
        group, name = key.split('.')
        manager = getattr(getattr(mg, group), name)
        manager.set(cp.asarray(d[key], dtype=cp.float32), start_level=0)
    fs = model.forward_solver
    fas = dict(meta['fas'])
    fas['dump_dir'] = None                     # a replay does not re-dump
    for k, v in fas.items():
        setattr(fs._fas_config, k, v)
    for lev in fs.levels:
        vc = lev.grid.forward_operators.vanka_config
        vc.omega = cp.float32(meta['vanka']['omega'])
        vc.relax_phi = cp.float32(meta['vanka']['relax_phi'])
        for k, v in meta['newton'].items():
            if v is not None:
                setattr(vc.newton_config, k, type(getattr(vc.newton_config, k) or 0.0)(v)
                        if not isinstance(v, bool) else v)
    model.set_top_level(0)
    return model, cp.float32(meta['dt']), meta
