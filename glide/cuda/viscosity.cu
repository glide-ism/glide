/*==================================================
  ================ VISCOSITY =======================
  ==================================================*/
template <int H, int W>
__device__ void populate_viscosity(
    DualFloat (&eta_local)[H][W],
    int bi, int bj,
    int i, int j,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ d_u,
    const float* __restrict__ d_v,
    const float* __restrict__ B,
    float n, float eps_reg, float dx,
    int ny, int nx){

    float dx_inv = 1.0f/dx;
    float glen_exp = (1.0f - n)/(2.0f * n);

    // Cell viscosity
    DualFloat u_l = get_vfacet(u, d_u, i, j, ny, nx);
    DualFloat u_r = get_vfacet(u, d_u, i, j + 1, ny, nx);
    DualFloat v_t = get_hfacet(v, d_v, i, j, ny, nx);
    DualFloat v_b = get_hfacet(v, d_v, i + 1, j, ny, nx);

    DualFloat dudx = (u_r - u_l)*dx_inv;
    DualFloat dvdy = (v_t - v_b)*dx_inv;

    float tl_mask = i > 0 && j > 0;
    DualFloat u_tl = get_vfacet(u, d_u, i - 1, j, ny, nx);
    DualFloat v_lt = get_hfacet(v, d_v, i, j - 1, ny, nx);
    DualFloat eps_xy_tl = 0.5f*((u_tl - u_l)*dx_inv + (v_t - v_lt)*dx_inv)*tl_mask;

    float tr_mask = i > 0 && j < (nx - 1);
    DualFloat u_tr = get_vfacet(u, d_u, i - 1, j + 1, ny, nx);
    DualFloat v_rt = get_hfacet(v, d_v, i, j + 1, ny, nx);
    DualFloat eps_xy_tr = 0.5f*((u_tr - u_r)*dx_inv + (v_rt - v_t)*dx_inv)*tr_mask;

    float bl_mask = i < (ny - 1) && j > 0;
    DualFloat u_bl = get_vfacet(u, d_u, i + 1, j, ny, nx);
    DualFloat v_lb = get_hfacet(v, d_v, i + 1, j - 1, ny, nx);
    DualFloat eps_xy_bl = 0.5f*((u_l - u_bl)*dx_inv + (v_b - v_lb)*dx_inv)*bl_mask;

    float br_mask = i < (ny - 1) && j < (nx - 1);
    DualFloat u_br = get_vfacet(u, d_u, i + 1, j + 1, ny, nx);
    DualFloat v_rb = get_hfacet(v, d_v, i + 1, j + 1, ny, nx);
    DualFloat eps_xy_br = 0.5f*((u_r - u_br)*dx_inv + (v_rb - v_b)*dx_inv)*br_mask;

    DualFloat eps_xy2_bar = 0.25f*(eps_xy_tl*eps_xy_tl + eps_xy_tr*eps_xy_tr + eps_xy_bl*eps_xy_bl + eps_xy_br*eps_xy_br);

    DualFloat eps_II_c = dudx*dudx + dvdy*dvdy + dudx*dvdy + eps_xy2_bar + eps_reg;

    DualFloat eta = 0.5f*get_cell(B,i,j,ny,nx)*__powf(eps_II_c,glen_exp);

    eta_local[bi][bj] = eta;

    __syncthreads();

}

template <int H, int W>
__device__ void populate_viscosity(
    float (&eta_local)[H][W],
    int bi, int bj,
    int i, int j,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ B,
    float n, float eps_reg, float dx,
    int ny, int nx){

    float dx_inv = 1.0f/dx;
    float glen_exp = (1.0f - n)/(2.0f * n);

    // Cell viscosity
    float u_l = get_vfacet(u, i, j, ny, nx);
    float u_r = get_vfacet(u, i, j + 1, ny, nx);
    float v_t = get_hfacet(v, i, j, ny, nx);
    float v_b = get_hfacet(v, i + 1, j, ny, nx);

    float dudx = (u_r - u_l)*dx_inv;
    float dvdy = (v_t - v_b)*dx_inv;

    float tl_mask = i > 0 && j > 0;
    float u_tl = get_vfacet(u, i - 1, j, ny, nx);
    float v_lt = get_hfacet(v, i, j - 1, ny, nx);
    float eps_xy_tl = 0.5f*((u_tl - u_l)*dx_inv + (v_t - v_lt)*dx_inv)*tl_mask;

    float tr_mask = i > 0 && j < (nx - 1);
    float u_tr = get_vfacet(u, i - 1, j + 1, ny, nx);
    float v_rt = get_hfacet(v, i, j + 1, ny, nx);
    float eps_xy_tr = 0.5f*((u_tr - u_r)*dx_inv + (v_rt - v_t)*dx_inv)*tr_mask;

    float bl_mask = i < (ny - 1) && j > 0;
    float u_bl = get_vfacet(u, i + 1, j, ny, nx);
    float v_lb = get_hfacet(v, i + 1, j - 1, ny, nx);
    float eps_xy_bl = 0.5f*((u_l - u_bl)*dx_inv + (v_b - v_lb)*dx_inv)*bl_mask;

    float br_mask = i < (ny - 1) && j < (nx - 1);
    float u_br = get_vfacet(u, i + 1, j + 1, ny, nx);
    float v_rb = get_hfacet(v, i + 1, j + 1, ny, nx);
    float eps_xy_br = 0.5f*((u_r - u_br)*dx_inv + (v_rb - v_b)*dx_inv)*br_mask;

    float eps_xy2_bar = 0.25f*(eps_xy_tl*eps_xy_tl + eps_xy_tr*eps_xy_tr + eps_xy_bl*eps_xy_bl + eps_xy_br*eps_xy_br);

    float eps_II_c = dudx*dudx + dvdy*dvdy + dudx*dvdy + eps_xy2_bar + eps_reg;

    eta_local[bi][bj] = 0.5f*get_cell(B,i,j,ny,nx)*__powf(eps_II_c,glen_exp);

    __syncthreads();

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

    __device__ __forceinline__
    EtaHCellStencil get_primals() const {
        return {eta.v,H.v};
    }

    __device__ __forceinline__
    EtaHCellStencil get_diffs() const {
        return {eta.d,H.d};
    }
};

struct EtaHCellJacobian {
    float res;
    float d_eta;
    float d_H;

    __device__ __forceinline__
    float apply_jvp(const EtaHCellStencil& dot) const {
        return d_eta * dot.eta + d_H * dot.H;

    }
};

__device__ __forceinline__
EtaHCellJacobian get_eta_H_cell_jac(EtaHCellStencil s) {
    EtaHCellJacobian jac;
    jac.res = s.H * s.eta;
    jac.d_eta = s.H;
    jac.d_H = s.eta;

    return jac;
}

__device__ __forceinline__
DualFloat get_eta_H_cell_dual(EtaHCellStencilDual s) {
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

    __device__ __forceinline__
    EtaHVertexStencil get_primals() const {
        return {eta_tl.v,eta_tr.v,eta_bl.v,eta_br.v,H_tl.v,H_tr.v,H_bl.v,H_br.v};
    }

    __device__ __forceinline__
    EtaHVertexStencil get_diffs() const {
        return {eta_tl.d,eta_tr.d,eta_bl.d,eta_br.d,H_tl.d,H_tr.d,H_bl.d,H_br.d};
    }
};

struct EtaHVertexJacobian {
    float res;
    float d_eta_tl, d_eta_tr, d_eta_bl, d_eta_br;
    float d_H_tl, d_H_tr, d_H_bl, d_H_br;

    __device__ __forceinline__
    float apply_jvp(const EtaHVertexStencil& dot) const {
        return d_eta_tl * dot.eta_tl + d_H_tl * dot.H_tl +
               d_eta_tr * dot.eta_tr + d_H_tr * dot.H_tr +
               d_eta_bl * dot.eta_bl + d_H_bl * dot.H_bl +
               d_eta_br * dot.eta_br + d_H_br * dot.H_br;

    }
};

__device__ __forceinline__
EtaHVertexJacobian get_eta_H_vertex_jac(EtaHVertexStencil s) {
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

__device__ __forceinline__
DualFloat get_eta_H_vertex_dual(EtaHVertexStencilDual s) {
    EtaHVertexJacobian jac = get_eta_H_vertex_jac(s.get_primals());
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}
