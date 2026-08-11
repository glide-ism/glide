template <int H, int W>
__device__ void populate_grounded(
    float (&grounded_local)[H][W],
    int bi, int bj,
    int i, int j,
    const float* __restrict__ thk,
    const float* __restrict__ bed,
    float sigmoid_c,
    int ny, int nx){

    float H_c = get_cell(thk,i,j,ny,nx);
    float bed_c = get_cell(bed,i,j,ny,nx);
    grounded_local[bi][bj] = get_grounded(H_c,bed_c,sigmoid_c);
}

extern "C" __global__
void compute_grounded(
    float* __restrict__ grounded,
    const float* __restrict__ H,
    const float* __restrict__ depth,
    float sigmoid_c,
    float sigmoid_k,
    float relaxation_parameter,
    int ny, int nx,
    int stride, int halo
    )
{
    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);    

    if (i < 0 || i >= ny || j<0 || j >= nx) return;

    float H_c = get_cell(H,i,j,ny,nx);
    float depth_c = get_cell(depth,i,j,ny,nx);
    float grounded_old = grounded[i * nx + j];
    grounded[i * nx + j] = (1.0f - relaxation_parameter) * get_grounded(H_c,depth_c,sigmoid_c, sigmoid_k) + relaxation_parameter * grounded_old;
}

extern "C" __global__
void compute_flotation_fraction(
    float* __restrict__ xi,
    const float* __restrict__ H,
    const float* __restrict__ depth,
    float sigmoid_c,
    float sigmoid_k,
    float relaxation_parameter,
    int ny, int nx,
    int stride, int halo
    )
{
    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);    

    if (i < 0 || i >= ny || j<0 || j >= nx) return;

    float H_c = get_cell(H,i,j,ny,nx);
    float depth_c = get_cell(depth,i,j,ny,nx);
    float xi_old = xi[i * nx + j];

    float xi_new = 1.0f - fminf(1.0905*depth_c/H_c,1.0f);

    xi[i * nx + j] = (1.0f - relaxation_parameter) * xi_new + relaxation_parameter * xi_old;
}


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


}

// Two-field (MOLHO) viscosity: the depth-averaged squared invariant picks up
// the deformational membrane terms (weighted by K_1 = int phi^2 dsigma) and
// the vertical shear term (weighted by K_2 = c_1*S_1/4, which contains the
// plug/no-slip limit renormalization).
template <int TH, int TW>
__device__ void populate_viscosity(
    float (&eta_local)[TH][TW],
    int bi, int bj,
    int i, int j,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ ud,
    const float* __restrict__ vd,
    const float* __restrict__ H,
    const float* __restrict__ B,
    float n, float eps_reg, float H_reg, float dx,
    int ny, int nx){

    float dx_inv = 1.0f/dx;
    float glen_exp = (1.0f - n)/(2.0f * n);

    float K_1 = 1.0f / (2.0f * n + 3.0f);
    float S_1 = (n + 2.0f) * (n + 2.0f) / (2.0f * n + 1.0f);
    float c_1 = __powf(1.0f / (n + 2.0f),2.0f * n / (n + 1.0f)) * (2.0f * n + 1.0f);
    float K_2 = c_1 * S_1 / 4.0f;

    // Mean-velocity membrane terms
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

    float eps_II_c = dudx*dudx + dvdy*dvdy + dudx*dvdy + eps_xy2_bar;

    // Deformational membrane terms, same gathers on (ud,vd)
    float ud_l = get_vfacet(ud, i, j, ny, nx);
    float ud_r = get_vfacet(ud, i, j + 1, ny, nx);
    float vd_t = get_hfacet(vd, i, j, ny, nx);
    float vd_b = get_hfacet(vd, i + 1, j, ny, nx);

    float duddx = (ud_r - ud_l)*dx_inv;
    float dvddy = (vd_t - vd_b)*dx_inv;

    float ud_tl = get_vfacet(ud, i - 1, j, ny, nx);
    float vd_lt = get_hfacet(vd, i, j - 1, ny, nx);
    float epsd_xy_tl = 0.5f*((ud_tl - ud_l)*dx_inv + (vd_t - vd_lt)*dx_inv)*tl_mask;

    float ud_tr = get_vfacet(ud, i - 1, j + 1, ny, nx);
    float vd_rt = get_hfacet(vd, i, j + 1, ny, nx);
    float epsd_xy_tr = 0.5f*((ud_tr - ud_r)*dx_inv + (vd_rt - vd_t)*dx_inv)*tr_mask;

    float ud_bl = get_vfacet(ud, i + 1, j, ny, nx);
    float vd_lb = get_hfacet(vd, i + 1, j - 1, ny, nx);
    float epsd_xy_bl = 0.5f*((ud_l - ud_bl)*dx_inv + (vd_b - vd_lb)*dx_inv)*bl_mask;

    float ud_br = get_vfacet(ud, i + 1, j + 1, ny, nx);
    float vd_rb = get_hfacet(vd, i + 1, j + 1, ny, nx);
    float epsd_xy_br = 0.5f*((ud_r - ud_br)*dx_inv + (vd_rb - vd_b)*dx_inv)*br_mask;

    float epsd_xy2_bar = 0.25f*(epsd_xy_tl*epsd_xy_tl + epsd_xy_tr*epsd_xy_tr + epsd_xy_bl*epsd_xy_bl + epsd_xy_br*epsd_xy_br);

    float epsd_II_c = duddx*duddx + dvddy*dvddy + duddx*dvddy + epsd_xy2_bar;

    // Vertical shear: interpolate squared facet values, cell's own H only.
    // 1/H^2 is regularized to 1/(H^2 + H_reg^2), consistent with the
    // eta*H/(H^2 + H_reg^2) form of the shear residual.
    float H_c = get_cell(H,i,j,ny,nx);
    float shear2_c = 0.5f*(ud_l*ud_l + ud_r*ud_r + vd_t*vd_t + vd_b*vd_b)/(H_c*H_c + H_reg*H_reg);

    float eps_II_bar = eps_II_c + K_1 * epsd_II_c + K_2 * shear2_c + eps_reg;

    eta_local[bi][bj] = 0.5f*get_cell(B,i,j,ny,nx)*__powf(eps_II_bar,glen_exp);


}

// Two-field (MOLHO) dual viscosity for the JVP: directional derivative in
// all five state directions (d_u, d_v, d_ud, d_vd, d_H), including the H
// dependence of the shear invariant through 1/(H^2 + H_reg^2).
template <int TH, int TW>
__device__ void populate_viscosity(
    DualFloat (&eta_local)[TH][TW],
    int bi, int bj,
    int i, int j,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ ud,
    const float* __restrict__ vd,
    const float* __restrict__ H,
    const float* __restrict__ d_u,
    const float* __restrict__ d_v,
    const float* __restrict__ d_ud,
    const float* __restrict__ d_vd,
    const float* __restrict__ d_H,
    const float* __restrict__ B,
    float n, float eps_reg, float H_reg, float dx,
    int ny, int nx){

    float dx_inv = 1.0f/dx;
    float glen_exp = (1.0f - n)/(2.0f * n);

    float K_1 = 1.0f / (2.0f * n + 3.0f);
    float S_1 = (n + 2.0f) * (n + 2.0f) / (2.0f * n + 1.0f);
    float c_1 = __powf(1.0f / (n + 2.0f),2.0f * n / (n + 1.0f)) * (2.0f * n + 1.0f);
    float K_2 = c_1 * S_1 / 4.0f;

    // Mean-velocity membrane terms
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

    DualFloat eps_II_c = dudx*dudx + dvdy*dvdy + dudx*dvdy + eps_xy2_bar;

    // Deformational membrane terms
    DualFloat ud_l = get_vfacet(ud, d_ud, i, j, ny, nx);
    DualFloat ud_r = get_vfacet(ud, d_ud, i, j + 1, ny, nx);
    DualFloat vd_t = get_hfacet(vd, d_vd, i, j, ny, nx);
    DualFloat vd_b = get_hfacet(vd, d_vd, i + 1, j, ny, nx);

    DualFloat duddx = (ud_r - ud_l)*dx_inv;
    DualFloat dvddy = (vd_t - vd_b)*dx_inv;

    DualFloat ud_tl = get_vfacet(ud, d_ud, i - 1, j, ny, nx);
    DualFloat vd_lt = get_hfacet(vd, d_vd, i, j - 1, ny, nx);
    DualFloat epsd_xy_tl = 0.5f*((ud_tl - ud_l)*dx_inv + (vd_t - vd_lt)*dx_inv)*tl_mask;

    DualFloat ud_tr = get_vfacet(ud, d_ud, i - 1, j + 1, ny, nx);
    DualFloat vd_rt = get_hfacet(vd, d_vd, i, j + 1, ny, nx);
    DualFloat epsd_xy_tr = 0.5f*((ud_tr - ud_r)*dx_inv + (vd_rt - vd_t)*dx_inv)*tr_mask;

    DualFloat ud_bl = get_vfacet(ud, d_ud, i + 1, j, ny, nx);
    DualFloat vd_lb = get_hfacet(vd, d_vd, i + 1, j - 1, ny, nx);
    DualFloat epsd_xy_bl = 0.5f*((ud_l - ud_bl)*dx_inv + (vd_b - vd_lb)*dx_inv)*bl_mask;

    DualFloat ud_br = get_vfacet(ud, d_ud, i + 1, j + 1, ny, nx);
    DualFloat vd_rb = get_hfacet(vd, d_vd, i + 1, j + 1, ny, nx);
    DualFloat epsd_xy_br = 0.5f*((ud_r - ud_br)*dx_inv + (vd_rb - vd_b)*dx_inv)*br_mask;

    DualFloat epsd_xy2_bar = 0.25f*(epsd_xy_tl*epsd_xy_tl + epsd_xy_tr*epsd_xy_tr + epsd_xy_bl*epsd_xy_bl + epsd_xy_br*epsd_xy_br);

    DualFloat epsd_II_c = duddx*duddx + dvddy*dvddy + duddx*dvddy + epsd_xy2_bar;

    // Vertical shear, including the H leg through 1/(H^2 + H_reg^2)
    DualFloat H_c = get_cell(H,d_H,i,j,ny,nx);
    DualFloat den = H_c*H_c + H_reg*H_reg;
    DualFloat inv_den = {1.0f/den.v, -den.d/(den.v*den.v)};
    DualFloat shear2_c = 0.5f*(ud_l*ud_l + ud_r*ud_r + vd_t*vd_t + vd_b*vd_b)*inv_den;

    DualFloat eps_II_bar = eps_II_c + K_1*epsd_II_c + K_2*shear2_c + eps_reg;

    eta_local[bi][bj] = 0.5f*get_cell(B,i,j,ny,nx)*__powf(eps_II_bar,glen_exp);
}

// Two-field (MOLHO) viscosity for the VJP: eta.d is the directional
// derivative in the lambda direction over the FOUR VELOCITY FIELDS ONLY
// (the H direction is deliberately excluded — the velocity stress block is
// handled by the self-adjointness trick, while the eta(H) chain is
// transposed explicitly). d(eta)/d(H_c) is emitted separately into
// deta_dH_local for those explicit H-column scatters.
template <int TH, int TW>
__device__ void populate_viscosity_vjp(
    DualFloat (&eta_local)[TH][TW],
    float (&deta_dH_local)[TH][TW],
    int bi, int bj,
    int i, int j,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ ud,
    const float* __restrict__ vd,
    const float* __restrict__ H,
    const float* __restrict__ lam_u,
    const float* __restrict__ lam_v,
    const float* __restrict__ lam_ud,
    const float* __restrict__ lam_vd,
    const float* __restrict__ B,
    float n, float eps_reg, float H_reg, float dx,
    int ny, int nx){

    float dx_inv = 1.0f/dx;
    float glen_exp = (1.0f - n)/(2.0f * n);

    float K_1 = 1.0f / (2.0f * n + 3.0f);
    float S_1 = (n + 2.0f) * (n + 2.0f) / (2.0f * n + 1.0f);
    float c_1 = __powf(1.0f / (n + 2.0f),2.0f * n / (n + 1.0f)) * (2.0f * n + 1.0f);
    float K_2 = c_1 * S_1 / 4.0f;

    // Mean-velocity membrane terms (direction = lambda_u, lambda_v)
    DualFloat u_l = get_vfacet(u, lam_u, i, j, ny, nx);
    DualFloat u_r = get_vfacet(u, lam_u, i, j + 1, ny, nx);
    DualFloat v_t = get_hfacet(v, lam_v, i, j, ny, nx);
    DualFloat v_b = get_hfacet(v, lam_v, i + 1, j, ny, nx);

    DualFloat dudx = (u_r - u_l)*dx_inv;
    DualFloat dvdy = (v_t - v_b)*dx_inv;

    float tl_mask = i > 0 && j > 0;
    DualFloat u_tl = get_vfacet(u, lam_u, i - 1, j, ny, nx);
    DualFloat v_lt = get_hfacet(v, lam_v, i, j - 1, ny, nx);
    DualFloat eps_xy_tl = 0.5f*((u_tl - u_l)*dx_inv + (v_t - v_lt)*dx_inv)*tl_mask;

    float tr_mask = i > 0 && j < (nx - 1);
    DualFloat u_tr = get_vfacet(u, lam_u, i - 1, j + 1, ny, nx);
    DualFloat v_rt = get_hfacet(v, lam_v, i, j + 1, ny, nx);
    DualFloat eps_xy_tr = 0.5f*((u_tr - u_r)*dx_inv + (v_rt - v_t)*dx_inv)*tr_mask;

    float bl_mask = i < (ny - 1) && j > 0;
    DualFloat u_bl = get_vfacet(u, lam_u, i + 1, j, ny, nx);
    DualFloat v_lb = get_hfacet(v, lam_v, i + 1, j - 1, ny, nx);
    DualFloat eps_xy_bl = 0.5f*((u_l - u_bl)*dx_inv + (v_b - v_lb)*dx_inv)*bl_mask;

    float br_mask = i < (ny - 1) && j < (nx - 1);
    DualFloat u_br = get_vfacet(u, lam_u, i + 1, j + 1, ny, nx);
    DualFloat v_rb = get_hfacet(v, lam_v, i + 1, j + 1, ny, nx);
    DualFloat eps_xy_br = 0.5f*((u_r - u_br)*dx_inv + (v_rb - v_b)*dx_inv)*br_mask;

    DualFloat eps_xy2_bar = 0.25f*(eps_xy_tl*eps_xy_tl + eps_xy_tr*eps_xy_tr + eps_xy_bl*eps_xy_bl + eps_xy_br*eps_xy_br);

    DualFloat eps_II_c = dudx*dudx + dvdy*dvdy + dudx*dvdy + eps_xy2_bar;

    // Deformational membrane terms (direction = lambda_ud, lambda_vd)
    DualFloat ud_l = get_vfacet(ud, lam_ud, i, j, ny, nx);
    DualFloat ud_r = get_vfacet(ud, lam_ud, i, j + 1, ny, nx);
    DualFloat vd_t = get_hfacet(vd, lam_vd, i, j, ny, nx);
    DualFloat vd_b = get_hfacet(vd, lam_vd, i + 1, j, ny, nx);

    DualFloat duddx = (ud_r - ud_l)*dx_inv;
    DualFloat dvddy = (vd_t - vd_b)*dx_inv;

    DualFloat ud_tl = get_vfacet(ud, lam_ud, i - 1, j, ny, nx);
    DualFloat vd_lt = get_hfacet(vd, lam_vd, i, j - 1, ny, nx);
    DualFloat epsd_xy_tl = 0.5f*((ud_tl - ud_l)*dx_inv + (vd_t - vd_lt)*dx_inv)*tl_mask;

    DualFloat ud_tr = get_vfacet(ud, lam_ud, i - 1, j + 1, ny, nx);
    DualFloat vd_rt = get_hfacet(vd, lam_vd, i, j + 1, ny, nx);
    DualFloat epsd_xy_tr = 0.5f*((ud_tr - ud_r)*dx_inv + (vd_rt - vd_t)*dx_inv)*tr_mask;

    DualFloat ud_bl = get_vfacet(ud, lam_ud, i + 1, j, ny, nx);
    DualFloat vd_lb = get_hfacet(vd, lam_vd, i + 1, j - 1, ny, nx);
    DualFloat epsd_xy_bl = 0.5f*((ud_l - ud_bl)*dx_inv + (vd_b - vd_lb)*dx_inv)*bl_mask;

    DualFloat ud_br = get_vfacet(ud, lam_ud, i + 1, j + 1, ny, nx);
    DualFloat vd_rb = get_hfacet(vd, lam_vd, i + 1, j + 1, ny, nx);
    DualFloat epsd_xy_br = 0.5f*((ud_r - ud_br)*dx_inv + (vd_rb - vd_b)*dx_inv)*br_mask;

    DualFloat epsd_xy2_bar = 0.25f*(epsd_xy_tl*epsd_xy_tl + epsd_xy_tr*epsd_xy_tr + epsd_xy_bl*epsd_xy_bl + epsd_xy_br*epsd_xy_br);

    DualFloat epsd_II_c = duddx*duddx + dvddy*dvddy + duddx*dvddy + epsd_xy2_bar;

    // Vertical shear: H is primal-only here (no H direction)
    float H_c = get_cell(H,i,j,ny,nx);
    float den = H_c*H_c + H_reg*H_reg;
    DualFloat shear2_c = 0.5f*(ud_l*ud_l + ud_r*ud_r + vd_t*vd_t + vd_b*vd_b)/den;

    DualFloat eps_II_bar = eps_II_c + K_1*epsd_II_c + K_2*shear2_c + eps_reg;

    DualFloat eta = 0.5f*get_cell(B,i,j,ny,nx)*__powf(eps_II_bar,glen_exp);
    eta_local[bi][bj] = eta;

    // d(eta_c)/d(H_c) = glen_exp * eta / E2 * dE2/dH,
    // dE2/dH = -K_2 * shear2 * 2H/(H^2 + H_reg^2)
    deta_dH_local[bi][bj] = glen_exp * eta.v / eps_II_bar.v *
        (-K_2 * shear2_c.v * 2.0f * H_c / den);
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


