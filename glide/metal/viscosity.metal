// =====================================================================
// VISCOSITY (Metal port of viscosity.cu)
//
// Port notes:
//   * The two shared-tile `populate_viscosity(...)` fills become the
//     value-returning helpers `eta_at(i,j,...)` (float) and
//     `eta_dual_at(i,j,...)` (DualFloat).  Callers that used to read a
//     cached `eta_local[bi+di][bj+dj]` now call eta_at(i+di, j+dj, ...).
//   * `populate_grounded` and the commented-out `compute_phi` were dead
//     code in the original (referenced a removed 3-arg get_grounded) and
//     are dropped.
//   * `compute_grounded` is ported to the flat one-thread-per-cell model.
//     params: [0]=sigmoid_c [1]=sigmoid_k [2]=relaxation [3]=ny [4]=nx
//             ([5]=stride [6]=halo, unused).
//   * Relies on common.metal (DualFloat, accessors, dpow) concatenated first.
// =====================================================================

kernel void compute_grounded(
    device float* grounded     [[buffer(0)]],
    device const float* H      [[buffer(1)]],
    device const float* depth  [[buffer(2)]],
    device const float* params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    float sigmoid_c = params[0];
    float sigmoid_k = params[1];
    float relaxation_parameter = params[2];
    int ny = (int)params[3];
    int nx = (int)params[4];

    int i = (int)tid / nx;
    int j = (int)tid % nx;
    if (i < 0 || i >= ny || j < 0 || j >= nx) return;

    float H_c = get_cell(H,i,j,ny,nx);
    float depth_c = get_cell(depth,i,j,ny,nx);
    float grounded_old = grounded[i * nx + j];
    grounded[i * nx + j] = (1.0f - relaxation_parameter) * get_grounded(H_c,depth_c,sigmoid_c, sigmoid_k) + relaxation_parameter * grounded_old;
}

/*==================================================
  ================ VISCOSITY =======================
  ==================================================*/

// DualFloat viscosity at cell (i,j) -- replaces the dual populate_viscosity.
inline DualFloat eta_dual_at(
    int i, int j,
    device const float* u,
    device const float* v,
    device const float* d_u,
    device const float* d_v,
    device const float* B,
    float n, float eps_reg, float dx,
    int ny, int nx){

    float dx_inv = 1.0f/dx;
    float glen_exp = (1.0f - n)/(2.0f * n);

    DualFloat u_l = get_vfacet(u, d_u, i, j, ny, nx);
    DualFloat u_r = get_vfacet(u, d_u, i, j + 1, ny, nx);
    DualFloat v_t = get_hfacet(v, d_v, i, j, ny, nx);
    DualFloat v_b = get_hfacet(v, d_v, i + 1, j, ny, nx);

    DualFloat dudx = (u_r - u_l)*dx_inv;
    DualFloat dvdy = (v_t - v_b)*dx_inv;

    float tl_mask = (i > 0 && j > 0);
    DualFloat u_tl = get_vfacet(u, d_u, i - 1, j, ny, nx);
    DualFloat v_lt = get_hfacet(v, d_v, i, j - 1, ny, nx);
    DualFloat eps_xy_tl = 0.5f*((u_tl - u_l)*dx_inv + (v_t - v_lt)*dx_inv)*tl_mask;

    float tr_mask = (i > 0 && j < (nx - 1));
    DualFloat u_tr = get_vfacet(u, d_u, i - 1, j + 1, ny, nx);
    DualFloat v_rt = get_hfacet(v, d_v, i, j + 1, ny, nx);
    DualFloat eps_xy_tr = 0.5f*((u_tr - u_r)*dx_inv + (v_rt - v_t)*dx_inv)*tr_mask;

    float bl_mask = (i < (ny - 1) && j > 0);
    DualFloat u_bl = get_vfacet(u, d_u, i + 1, j, ny, nx);
    DualFloat v_lb = get_hfacet(v, d_v, i + 1, j - 1, ny, nx);
    DualFloat eps_xy_bl = 0.5f*((u_l - u_bl)*dx_inv + (v_b - v_lb)*dx_inv)*bl_mask;

    float br_mask = (i < (ny - 1) && j < (nx - 1));
    DualFloat u_br = get_vfacet(u, d_u, i + 1, j + 1, ny, nx);
    DualFloat v_rb = get_hfacet(v, d_v, i + 1, j + 1, ny, nx);
    DualFloat eps_xy_br = 0.5f*((u_r - u_br)*dx_inv + (v_rb - v_b)*dx_inv)*br_mask;

    DualFloat eps_xy2_bar = 0.25f*(eps_xy_tl*eps_xy_tl + eps_xy_tr*eps_xy_tr + eps_xy_bl*eps_xy_bl + eps_xy_br*eps_xy_br);

    DualFloat eps_II_c = dudx*dudx + dvdy*dvdy + dudx*dvdy + eps_xy2_bar + eps_reg;

    return 0.5f*get_cell(B,i,j,ny,nx)*dpow(eps_II_c,glen_exp);
}

// Scalar viscosity at cell (i,j) -- replaces the float populate_viscosity.
inline float eta_at(
    int i, int j,
    device const float* u,
    device const float* v,
    device const float* B,
    float n, float eps_reg, float dx,
    int ny, int nx){

    float dx_inv = 1.0f/dx;
    float glen_exp = (1.0f - n)/(2.0f * n);

    float u_l = get_vfacet(u, i, j, ny, nx);
    float u_r = get_vfacet(u, i, j + 1, ny, nx);
    float v_t = get_hfacet(v, i, j, ny, nx);
    float v_b = get_hfacet(v, i + 1, j, ny, nx);

    float dudx = (u_r - u_l)*dx_inv;
    float dvdy = (v_t - v_b)*dx_inv;

    float tl_mask = (i > 0 && j > 0);
    float u_tl = get_vfacet(u, i - 1, j, ny, nx);
    float v_lt = get_hfacet(v, i, j - 1, ny, nx);
    float eps_xy_tl = 0.5f*((u_tl - u_l)*dx_inv + (v_t - v_lt)*dx_inv)*tl_mask;

    float tr_mask = (i > 0 && j < (nx - 1));
    float u_tr = get_vfacet(u, i - 1, j + 1, ny, nx);
    float v_rt = get_hfacet(v, i, j + 1, ny, nx);
    float eps_xy_tr = 0.5f*((u_tr - u_r)*dx_inv + (v_rt - v_t)*dx_inv)*tr_mask;

    float bl_mask = (i < (ny - 1) && j > 0);
    float u_bl = get_vfacet(u, i + 1, j, ny, nx);
    float v_lb = get_hfacet(v, i + 1, j - 1, ny, nx);
    float eps_xy_bl = 0.5f*((u_l - u_bl)*dx_inv + (v_b - v_lb)*dx_inv)*bl_mask;

    float br_mask = (i < (ny - 1) && j < (nx - 1));
    float u_br = get_vfacet(u, i + 1, j + 1, ny, nx);
    float v_rb = get_hfacet(v, i + 1, j + 1, ny, nx);
    float eps_xy_br = 0.5f*((u_r - u_br)*dx_inv + (v_rb - v_b)*dx_inv)*br_mask;

    float eps_xy2_bar = 0.25f*(eps_xy_tl*eps_xy_tl + eps_xy_tr*eps_xy_tr + eps_xy_bl*eps_xy_bl + eps_xy_br*eps_xy_br);

    float eps_II_c = dudx*dudx + dvdy*dvdy + dudx*dvdy + eps_xy2_bar + eps_reg;

    return 0.5f*get_cell(B,i,j,ny,nx)*pow(eps_II_c,glen_exp);
}

/*==================================================
  ========== Viscosity-Thickness Product ===========
  ==================================================*/

struct EtaHCellStencil {
    float eta;
    float H;
};

struct EtaHCellStencilDual{
    DualFloat eta;
    DualFloat H;

    EtaHCellStencil get_primals() const {
        return {eta.v,H.v};
    }

    EtaHCellStencil get_diffs() const {
        return {eta.d,H.d};
    }
};

struct EtaHCellJacobian {
    float res;
    float d_eta;
    float d_H;

    float apply_jvp(const thread EtaHCellStencil& dot) const {
        return d_eta * dot.eta + d_H * dot.H;
    }
};

inline EtaHCellJacobian get_eta_H_cell_jac(EtaHCellStencil s) {
    EtaHCellJacobian jac;
    jac.res = s.H * s.eta;
    jac.d_eta = s.H;
    jac.d_H = s.eta;

    return jac;
}

inline DualFloat get_eta_H_cell_dual(EtaHCellStencilDual s) {
    EtaHCellJacobian jac = get_eta_H_cell_jac(s.get_primals());
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

struct EtaHVertexStencil {
    float eta_tl, eta_tr, eta_bl, eta_br;
    float H_tl, H_tr, H_bl, H_br;
};

struct EtaHVertexStencilDual {
    DualFloat eta_tl, eta_tr, eta_bl, eta_br;
    DualFloat H_tl, H_tr, H_bl, H_br;

    EtaHVertexStencil get_primals() const {
        return {eta_tl.v,eta_tr.v,eta_bl.v,eta_br.v,H_tl.v,H_tr.v,H_bl.v,H_br.v};
    }

    EtaHVertexStencil get_diffs() const {
        return {eta_tl.d,eta_tr.d,eta_bl.d,eta_br.d,H_tl.d,H_tr.d,H_bl.d,H_br.d};
    }
};

struct EtaHVertexJacobian {
    float res;
    float d_eta_tl, d_eta_tr, d_eta_bl, d_eta_br;
    float d_H_tl, d_H_tr, d_H_bl, d_H_br;

    float apply_jvp(const thread EtaHVertexStencil& dot) const {
        return d_eta_tl * dot.eta_tl + d_H_tl * dot.H_tl +
               d_eta_tr * dot.eta_tr + d_H_tr * dot.H_tr +
               d_eta_bl * dot.eta_bl + d_H_bl * dot.H_bl +
               d_eta_br * dot.eta_br + d_H_br * dot.H_br;
    }
};

inline EtaHVertexJacobian get_eta_H_vertex_jac(EtaHVertexStencil s) {
    EtaHVertexJacobian jac;
    jac.res = 0.25f*(s.eta_tl * s.H_tl + s.eta_tr * s.H_tr + s.eta_bl * s.H_bl + s.eta_br * s.H_br);

    jac.d_eta_tl = 0.25f*s.H_tl;
    jac.d_eta_tr = 0.25f*s.H_tr;
    jac.d_eta_bl = 0.25f*s.H_bl;
    jac.d_eta_br = 0.25f*s.H_br;

    jac.d_H_tl = 0.25f*s.eta_tl;
    jac.d_H_tr = 0.25f*s.eta_tr;
    jac.d_H_bl = 0.25f*s.eta_bl;
    jac.d_H_br = 0.25f*s.eta_br;

    return jac;
}

inline DualFloat get_eta_H_vertex_dual(EtaHVertexStencilDual s) {
    EtaHVertexJacobian jac = get_eta_H_vertex_jac(s.get_primals());
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}
