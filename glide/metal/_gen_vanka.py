"""Generator: translate cuda/vanka.cu to metal/vanka.metal.

Helpers (lu_5x5_solve / mat5x5_* / build_5x5_vanka) operate on thread-local 5x5
systems; build_5x5_vanka's shared `eta_local` tile becomes on-demand eta_at()
calls.  The three kernels are flattened to one-thread-per-cell; the additive
Vanka scatter (atomicAdd onto shared facets) uses the CAS atomic_add_float
helper, so delta_u/delta_v (and the adjoint lambda outputs) are bound as
`device atomic_uint*` (the host passes pre-zeroed float32 buffers).

Run from repo root:  python glide/metal/_gen_vanka.py
"""
import re
from pathlib import Path

CU = Path("glide/cuda/vanka.cu").read_text()


def body_inside(text, marker):
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
    raise RuntimeError("unbalanced")


def subst_eta(s):
    def repl(m):
        di = (m.group(1) or '').replace(' ', '')
        dj = (m.group(2) or '').replace(' ', '')
        ii = f'i {di[0]} {di[1:]}' if di else 'i'
        jj = f'j {dj[0]} {dj[1:]}' if dj else 'j'
        return f'eta_at({ii}, {jj}, u, v, B, n, eps_reg, dx, ny, nx)'
    return re.sub(r'eta_local\[bi\s*([+-]\s*\d+)?\]\[bj\s*([+-]\s*\d+)?\]', repl, s)


# ---- hand-written ported helpers (thread-local linear algebra) --------------
HELPERS = r'''
// 5x5 LU solve (Doolittle, no pivot) on thread-local arrays.
inline void lu_5x5_solve(thread const float* A, thread const float* b, thread float* x)
{
    float LU[5][5];
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            LU[i][j] = A[i * 5 + j];
    for (int k = 0; k < 5; k++) {
        float inv_diag = 1.0f / LU[k][k];
        for (int i = k + 1; i < 5; i++) {
            LU[i][k] *= inv_diag;
            for (int j = k + 1; j < 5; j++)
                LU[i][j] -= LU[i][k] * LU[k][j];
        }
    }
    float y[5];
    y[0] = b[0];
    y[1] = b[1] - LU[1][0]*y[0];
    y[2] = b[2] - LU[2][0]*y[0] - LU[2][1]*y[1];
    y[3] = b[3] - LU[3][0]*y[0] - LU[3][1]*y[1] - LU[3][2]*y[2];
    y[4] = b[4] - LU[4][0]*y[0] - LU[4][1]*y[1] - LU[4][2]*y[2] - LU[4][3]*y[3];
    x[4] = y[4] / LU[4][4];
    x[3] = (y[3] - LU[3][4]*x[4]) / LU[3][3];
    x[2] = (y[2] - LU[2][3]*x[3] - LU[2][4]*x[4]) / LU[2][2];
    x[1] = (y[1] - LU[1][2]*x[2] - LU[1][3]*x[3] - LU[1][4]*x[4]) / LU[1][1];
    x[0] = (y[0] - LU[0][1]*x[1] - LU[0][2]*x[2] - LU[0][3]*x[3] - LU[0][4]*x[4]) / LU[0][0];
}
'''

BUILD_SIG = r'''
inline void build_5x5_vanka(
    thread float* J, thread float* r,
    float u_l, float u_r, float v_t, float v_b, float H_c,
    device const float* u, device const float* v, device const float* H,
    device const float* phi, device const float* bed,
    device const float* B, device const float* beta, device const float* gamma,
    float n, float eps_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding,
    float calving_rate, float flotation_reg_calving,
    float dx, float dt, int ny, int nx, int i, int j)
{'''


def make_build():
    body = body_inside(CU[CU.index('void build_5x5_vanka'):CU.index('extern "C"')],
                       'void build_5x5_vanka')
    body = subst_eta(body)
    return BUILD_SIG + body + '}\n'


# ---- kernels ----------------------------------------------------------------
def fix_body(body):
    body = subst_eta(body)
    body = body.replace('u, v, H, eta_local, phi,', 'u, v, H, phi,')
    body = body.replace('ny, nx, i, j, bi, bj);', 'ny, nx, i, j);')
    body = re.sub(r'\bfmaxf\(', 'fmax(', body)
    body = re.sub(r'\bfminf\(', 'fmin(', body)
    body = re.sub(r'atomicAdd\(\&', 'atomic_add_float(&', body)
    return body


def make_kernel(name, bufs, atomic_outs, params, body, W='nx',
                guard='if (i < 0 || i >= ny || j < 0 || j >= nx) return;'):
    lines = [f'kernel void {name}(']
    sig = []
    out_set = set(atomic_outs)
    for k, (b, kind) in enumerate(bufs):
        if b in out_set:
            q = 'device atomic_uint*'
        elif kind == 'out':
            q = 'device float*'
        else:
            q = 'device const float*'
        sig.append(f'    {q} {b} [[buffer({k})]]')
    sig.append(f'    device const float* params [[buffer({len(bufs)})]]')
    sig.append('    uint tid [[thread_position_in_grid]]')
    lines.append(',\n'.join(sig))
    lines.append(')\n{')
    for idx, (pname, ptype) in enumerate(params):
        cast = '(int)' if ptype == 'int' else ''
        decl = 'int' if ptype == 'int' else 'float'
        lines.append(f'    {decl} {pname} = {cast}params[{idx}];')
    lines.append('')
    lines.append(f'    int i = (int)tid / ({W});')
    lines.append(f'    int j = (int)tid % ({W});')
    lines.append(f'    {guard}')
    lines.append('    {')
    lines.append(body)
    lines.append('    }')
    lines.append('}')
    return '\n'.join(lines)


# buffer layouts: (name, 'in'|'out'); atomic outputs listed separately
SMOOTH_BUFS = [('delta_u', 'out'), ('delta_v', 'out'), ('delta_H', 'out'),
               ('mask', 'out'), ('u', 'in'), ('v', 'in'), ('H', 'in'),
               ('phi', 'in'), ('f_u', 'in'), ('f_v', 'in'), ('f_H', 'in'),
               ('bed', 'in'), ('B', 'in'), ('beta', 'in'), ('gamma', 'in')]
SMOOTH_ATOMIC = ['delta_u', 'delta_v']
SMOOTH_PARAMS = [('n', 'f'), ('eps_reg', 'f'), ('flotation_reg_driving', 'f'),
                 ('m', 'f'), ('u_reg', 'f'), ('water_drag', 'f'),
                 ('flotation_reg_sliding', 'f'), ('calving_rate', 'f'),
                 ('flotation_reg_calving', 'f'), ('dx', 'f'), ('dt', 'f'),
                 ('ny', 'int'), ('nx', 'int'), ('stride', 'int'), ('halo', 'int'),
                 ('newton_steps', 'int'), ('relaxation', 'f'),
                 ('ssa_damping', 'f'), ('mc_damping', 'f')]

ADJ_BUFS = [('lambda_u_out', 'out'), ('lambda_v_out', 'out'), ('lambda_H_out', 'out'),
            ('u', 'in'), ('v', 'in'), ('H', 'in'), ('phi', 'in'), ('mask', 'in'),
            ('r_adj_u', 'in'), ('r_adj_v', 'in'), ('r_adj_H', 'in'),
            ('bed', 'in'), ('B', 'in'), ('beta', 'in'), ('gamma', 'in')]
ADJ_ATOMIC = ['lambda_u_out', 'lambda_v_out', 'lambda_H_out']
ADJ_PARAMS = [('n', 'f'), ('eps_reg', 'f'), ('flotation_reg_driving', 'f'),
              ('m', 'f'), ('u_reg', 'f'), ('water_drag', 'f'),
              ('flotation_reg_sliding', 'f'), ('calving_rate', 'f'),
              ('flotation_reg_calving', 'f'), ('dx', 'f'), ('dt', 'f'),
              ('ny', 'int'), ('nx', 'int'), ('stride', 'int'), ('halo', 'int'),
              ('ssa_damping', 'f'), ('mc_damping', 'f')]

DUMP_BUFS = [('J_array', 'out'), ('r_array', 'out'), ('u', 'in'), ('v', 'in'),
             ('H', 'in'), ('phi', 'in'), ('f_u', 'in'), ('f_v', 'in'),
             ('f_H', 'in'), ('bed', 'in'), ('B', 'in'), ('beta', 'in'),
             ('gamma', 'in')]
DUMP_PARAMS = SMOOTH_PARAMS[:15]  # n..halo (no newton/relaxation/damping)


def main():
    smooth = CU[CU.index('void vanka_smooth('):CU.index('void vanka_smooth_adjoint')]
    adj = CU[CU.index('void vanka_smooth_adjoint'):CU.index('void vanka_dump')]
    dump = CU[CU.index('void vanka_dump'):]

    k_smooth = make_kernel('vanka_smooth', SMOOTH_BUFS, SMOOTH_ATOMIC, SMOOTH_PARAMS,
                           fix_body(body_inside(smooth, 'if ( is_active )')))
    k_adj = make_kernel('vanka_smooth_adjoint', ADJ_BUFS, ADJ_ATOMIC, ADJ_PARAMS,
                        fix_body(body_inside(adj, 'if ( is_active )')))
    k_dump = make_kernel('vanka_dump', DUMP_BUFS, [], DUMP_PARAMS,
                         fix_body(body_inside(dump, 'if ( is_active )')))

    header = (
        "// Auto-generated from cuda/vanka.cu by _gen_vanka.py.\n"
        "// Flat one-thread-per-cell; eta_local tile -> eta_at; additive Vanka\n"
        "// scatter via atomic_add_float (delta_u/v, lambda_* are atomic_uint).\n"
        "// Relies on common/viscosity/stress/flux .metal (concatenated first).\n"
    )
    out = (header + HELPERS + "\n" + make_build() + "\n\n"
           + k_smooth + "\n\n" + k_adj + "\n\n" + k_dump + "\n")
    Path("glide/metal/vanka.metal").write_text(out)
    print("wrote vanka.metal; kernels:", re.findall(r'kernel void (\w+)', out))


if __name__ == "__main__":
    main()
