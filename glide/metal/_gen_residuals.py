"""Generator: translate cuda/residuals.cu kernels to metal/residuals.metal.

The CUDA kernels use a 16x16 shared-memory `eta_local` viscosity tile with a
1-cell halo (block/stride/halo machinery).  Under macmetalpy's flat Metal
dispatch there is no threadgroup/shared memory, so we:

  * drop the tile + __syncthreads + is_active gate,
  * map a flat thread id to (i, j) over the (ny+1) x (nx+1) facet superset,
  * replace every `eta_local[bi+di][bj+dj]` read with an on-demand
    `eta_at(i+di, j+dj, ...)` / `eta_dual_at(...)` call (viscosity.metal),
  * pass scalar kernel args through a trailing `params` buffer.

Run from the repo root:  python glide/metal/_gen_residuals.py
"""
import re
from pathlib import Path

CU = Path("glide/cuda/residuals.cu").read_text()


def body_inside(text, marker):
    """Return code inside the first `{...}` block at/after `marker`."""
    si = text.index(marker)
    bi = text.index('{', si)
    depth = 0
    for k in range(bi, len(text)):
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
            if depth == 0:
                return text[bi + 1:k]
    raise RuntimeError("unbalanced braces")


def _off(token):
    t = (token or '').replace(' ', '')
    return f' {t[0]} {t[1:]}' if t else ''


def subst_eta(s, dual, dbufs='d_u, d_v'):
    fn = 'eta_dual_at' if dual else 'eta_at'
    extra = f'u, v, {dbufs}, B' if dual else 'u, v, B'

    def repl(m):
        ii = 'i' + _off(m.group(1))
        jj = 'j' + _off(m.group(2))
        return f'{fn}({ii}, {jj}, {extra}, n, eps_reg, dx, ny, nx)'

    return re.sub(r'eta_local\[bi\s*([+-]\s*\d+)?\]\[bj\s*([+-]\s*\d+)?\]', repl, s)


# scatter helpers for compute_vjp: each shared-tile atomicAdd becomes a direct
# global scatter, with the original flush-stage boundary guards relocated here.
SCATTER_HELPERS = r'''
inline void scatter_u(device atomic_uint* vjp_u, int gy, int gx, float val, int ny, int nx) {
    if (gy >= 0 && gy < ny && gx > 0 && gx < nx)
        atomic_add_float(&vjp_u[gy * (nx + 1) + gx], val);
}
inline void scatter_v(device atomic_uint* vjp_v, int gy, int gx, float val, int ny, int nx) {
    if (gy > 0 && gy < ny && gx >= 0 && gx < nx)
        atomic_add_float(&vjp_v[gy * nx + gx], val);
}
inline void scatter_H(device atomic_uint* vjp_H, device const float* mask,
                      int gy, int gx, float val, int ny, int nx) {
    if (gy >= 0 && gy < ny && gx >= 0 && gx < nx) {
        if (get_cell(mask, gy, gx, ny, nx) > 0.0f) val = 0.0f;
        atomic_add_float(&vjp_H[gy * nx + gx], val);
    }
}
'''


def subst_scatter(s):
    # atomicAdd(&s_adj_X[bi+di][bj+dj], VAL);  ->  scatter_X(vjp_X[, mask], i+di, j+dj, VAL, ny, nx);
    pat = re.compile(
        r'atomicAdd\(\s*&\s*s_adj_([uvH])\[bi([^\]]*)\]\[bj([^\]]*)\]\s*,\s*(.+?)\);',
        re.S)

    def repl(m):
        comp, oi, oj, val = m.group(1), m.group(2), m.group(3), m.group(4).strip()
        gy = 'i' + _off(oi)
        gx = 'j' + _off(oj)
        extra = 'mask, ' if comp == 'H' else ''
        return f'scatter_{comp}(vjp_{comp}, {extra}{gy}, {gx}, {val}, ny, nx);'

    return pat.sub(repl, s)


def kernel_src(name, between_start, between_end):
    return CU[CU.index(between_start):CU.index(between_end)]


# ---- per-kernel buffer/param layouts (order must match operators.py calls) ----

RESIDUAL_BUFS = ['r_u', 'r_v', 'r_H', 'u', 'v', 'H', 'phi', 'mask',
                 'f_u', 'f_v', 'f_H', 'bed', 'B', 'beta', 'gamma']
RESIDUAL_OUT = {'r_u', 'r_v', 'r_H'}
RESIDUAL_PARAMS = [
    ('use_forcing', 'bool'), ('use_mask', 'bool'),
    ('n', 'float'), ('eps_reg', 'float'), ('flotation_reg_driving', 'float'),
    ('m', 'float'), ('u_reg', 'float'), ('water_drag', 'float'),
    ('flotation_reg_sliding', 'float'),
    ('calving_rate', 'float'), ('flotation_reg_calving', 'float'),
    ('dx', 'float'), ('dt', 'float'), ('ny', 'int'), ('nx', 'int'),
]

JVP_BUFS = ['jvp_u', 'jvp_v', 'jvp_H', 'u', 'v', 'H', 'd_u', 'd_v', 'd_H',
            'phi', 'mask', 'f_u', 'f_v', 'f_H', 'bed', 'B', 'beta', 'gamma']
JVP_OUT = {'jvp_u', 'jvp_v', 'jvp_H'}
JVP_PARAMS = [
    ('use_mask', 'bool'),
    ('n', 'float'), ('eps_reg', 'float'), ('flotation_reg_driving', 'float'),
    ('m', 'float'), ('u_reg', 'float'), ('water_drag', 'float'),
    ('flotation_reg_sliding', 'float'),
    ('calving_rate', 'float'), ('flotation_reg_calving', 'float'),
    ('dx', 'float'), ('dt', 'float'), ('ny', 'int'), ('nx', 'int'),
]


def make_kernel(name, bufs, outs, params, body, atomic_outs=()):
    lines = [f'kernel void {name}(']
    sig = []
    for k, b in enumerate(bufs):
        if b in atomic_outs:
            q = 'device atomic_uint*'
        elif b in outs:
            q = 'device float*'
        else:
            q = 'device const float*'
        sig.append(f'    {q} {b} [[buffer({k})]]')
    sig.append(f'    device const float* params [[buffer({len(bufs)})]]')
    sig.append('    uint tid [[thread_position_in_grid]]')
    lines.append(',\n'.join(sig))
    lines.append(')\n{')
    for idx, (pname, ptype) in enumerate(params):
        if ptype == 'bool':
            lines.append(f'    bool {pname} = params[{idx}] != 0.0f;')
        elif ptype == 'int':
            lines.append(f'    int {pname} = (int)params[{idx}];')
        else:
            lines.append(f'    float {pname} = params[{idx}];')
    lines.append('')
    lines.append('    int i = (int)tid / (nx + 1);')
    lines.append('    int j = (int)tid % (nx + 1);')
    lines.append('    if (i > ny || j > nx) return;')
    lines.append('    {')
    lines.append(body)
    lines.append('    }')
    lines.append('}')
    return '\n'.join(lines)


VJP_BUFS = ['vjp_u', 'vjp_v', 'vjp_H', 'u', 'v', 'H', 'lambda_u', 'lambda_v',
            'lambda_H', 'phi', 'mask', 'f_u', 'f_v', 'f_H', 'bed', 'B', 'beta', 'gamma']
VJP_ATOMIC = {'vjp_u', 'vjp_v', 'vjp_H'}
VJP_PARAMS = [
    ('use_forcing', 'bool'), ('use_mask', 'bool'),
    ('n', 'float'), ('eps_reg', 'float'), ('flotation_reg_driving', 'float'),
    ('m', 'float'), ('u_reg', 'float'), ('water_drag', 'float'),
    ('flotation_reg_sliding', 'float'),
    ('calving_rate', 'float'), ('flotation_reg_calving', 'float'),
    ('dx', 'float'), ('dt', 'float'), ('ny', 'int'), ('nx', 'int'),
]


def main():
    res_kern = kernel_src('compute_residual', 'void compute_residual', 'void compute_jvp')
    res_body = subst_eta(body_inside(res_kern, 'if ( is_active )'), dual=False)
    residual = make_kernel('compute_residual', RESIDUAL_BUFS, RESIDUAL_OUT,
                           RESIDUAL_PARAMS, res_body)

    jvp_kern = kernel_src('compute_jvp', 'void compute_jvp', 'void compute_vjp')
    jvp_body = subst_eta(body_inside(jvp_kern, 'if ( is_active )'), dual=True)
    jvp = make_kernel('compute_jvp', JVP_BUFS, JVP_OUT, JVP_PARAMS, jvp_body)

    # compute_vjp: eta perturbation direction is (lambda_u, lambda_v); shared-tile
    # scatter becomes direct global scatter via scatter_u/v/H.
    vjp_kern = CU[CU.index('void compute_vjp'):]
    vjp_body = body_inside(vjp_kern, 'if ( is_active )')
    vjp_body = subst_eta(vjp_body, dual=True, dbufs='lambda_u, lambda_v')
    vjp_body = subst_scatter(vjp_body)
    assert 's_adj_' not in vjp_body, "unconverted s_adj_ scatter remains"
    # has_cell/has_u/has_v are declared outside is_active in the original; add them.
    vjp_body = (
        "    bool has_cell = i >= 0 && i <  ny && j >= 0 && j <  nx;\n"
        "    bool has_u    = i >= 0 && i <  ny && j >= 0 && j <= nx;\n"
        "    bool has_v    = i >= 0 && i <= ny && j >= 0 && j <  nx;\n"
        + vjp_body)
    vjp = make_kernel('compute_vjp', VJP_BUFS, set(), VJP_PARAMS, vjp_body,
                      atomic_outs=VJP_ATOMIC)

    header = (
        "// Auto-generated from cuda/residuals.cu by _gen_residuals.py.\n"
        "// Flat one-thread-per-(i,j) port; eta_local tile -> eta_at/eta_dual_at;\n"
        "// compute_vjp shared-tile scatter -> direct global scatter_u/v/H.\n"
        "// Relies on common/viscosity/stress/flux .metal (concatenated first).\n\n"
    )
    out = header + SCATTER_HELPERS + "\n" + residual + "\n\n" + jvp + "\n\n" + vjp + "\n"
    Path("glide/metal/residuals.metal").write_text(out)
    print("wrote residuals.metal; kernels:", re.findall(r'kernel void (\w+)', out))


if __name__ == "__main__":
    main()
