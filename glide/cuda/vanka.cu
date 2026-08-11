//#pragma once
// In-place Doolittle LU (no pivoting) for small dense N x N systems,
// generalizing the previous hard-coded 5x5 solver. Instantiate with
// N = 5 (drop-in replacement) or N = 9 (two-field patch).
//
// Differences from the 5x5 version, all deliberate:
//   * Factors A in place: assemble the patch matrix directly into a
//     caller-owned scratch array and pass it here. This removes the
//     25/81-entry copy and halves peak live state during setup.
//   * The diagonal of the factored A stores 1/U[k][k], not U[k][k].
//     Triangular solves then multiply instead of divide (FP32 divides
//     are multi-instruction on GA102). Any transpose-solve you write
//     against these factors must respect this convention.
//   * Factorization and solve are split so the same factors can serve
//     the forward solve and the adjoint's transpose solve:
//     A^T x = b is U^T (forward, lower-tri, multiply by the stored
//     reciprocal diagonals) followed by L^T (backward, unit diagonal).
//   * b is consumed as scratch by the forward substitution, which
//     eliminates the separate y[] array.

template <int N>
__device__ __forceinline__ void lu_factor(float* __restrict__ A)
{
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        const float inv_diag = 1.0f / A[k * N + k];
        A[k * N + k] = inv_diag;  // (k,k) is final after this step; store reciprocal
        #pragma unroll
        for (int i = k + 1; i < N; ++i) {
            const float lik = A[i * N + k] * inv_diag;
            A[i * N + k] = lik;
            #pragma unroll
            for (int j = k + 1; j < N; ++j) {
                A[i * N + j] = fmaf(-lik, A[k * N + j], A[i * N + j]);
            }
        }
    }
}

// Symmetric Jacobi equilibration of an N x N patch system: s_i =
// 1/sqrt(|A_ii|), floored against the row's infinity norm so a degenerate
// diagonal cannot produce unbounded scale factors. Rescales A <- S A S and
// b <- S b in place; the solution of the original system is x = S y where
// y solves the scaled one (the caller multiplies by s after the solve).
// The patch rows span shear (eta*H/(H^2+H_reg^2)), membrane (eta*H/dx^2),
// drag, and transport (1/dt) scales - many decades apart at margins -
// which the no-pivot fp32 LU cannot otherwise absorb. Symmetric scaling
// preserves the symmetry of the stress block and commutes with
// transposition ((S A S)^T = S A^T S), so the forward and adjoint
// smoothers share identical machinery.
template <int N>
__device__ __forceinline__ void equilibrate(float* __restrict__ A,
                                            float* __restrict__ b,
                                            float* __restrict__ s)
{
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        float rmax = 0.0f;
        #pragma unroll
        for (int j = 0; j < N; ++j) rmax = fmaxf(rmax, fabsf(A[i*N + j]));
        s[i] = rsqrtf(fmaxf(fabsf(A[i*N + i]), fmaxf(1e-6f*rmax, 1e-30f)));
    }
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) A[i*N + j] *= s[i]*s[j];
        b[i] *= s[i];
    }
    // Pivot floor for the no-pivot factorization: healthy rows have a
    // scaled diagonal of +-1 and are untouched; only rows whose diagonal
    // is degenerate relative to their own row norm are lifted.
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        A[i*N + i] = copysignf(fmaxf(fabsf(A[i*N + i]), 1e-3f), A[i*N + i]);
    }
}

// Solve A x = b using factors produced by lu_factor<N>. b is overwritten
// (it holds y after forward substitution). x may not alias A.
template <int N>
__device__ __forceinline__ void lu_solve_factored(const float* __restrict__ A,
                                                  float* __restrict__ b,
                                                  float* __restrict__ x)
{
    // Forward: L y = b  (unit lower triangular), y stored into b.
    #pragma unroll
    for (int i = 1; i < N; ++i) {
        float s = b[i];
        #pragma unroll
        for (int j = 0; j < i; ++j) {
            s = fmaf(-A[i * N + j], b[j], s);
        }
        b[i] = s;
    }
    // Backward: U x = y  (diagonal entries hold reciprocals).
    #pragma unroll
    for (int i = N - 1; i >= 0; --i) {
        float s = b[i];
        #pragma unroll
        for (int j = i + 1; j < N; ++j) {
            s = fmaf(-A[i * N + j], x[j], s);
        }
        x[i] = s * A[i * N + i];
    }
}

// ============================================================
// LU Solve for 5x5 Systems (Vanka smoother)
// ============================================================
__device__ void lu_5x5_solve(
    const float* A,  // 25 entries: full 5x5 row-major
    const float* b,  // 5 entries
    float* x)        // 5 entries (output)
{
    float LU[5][5];

    #pragma unroll
    for (int i = 0; i < 5; i++) {
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            LU[i][j] = A[i * 5 + j];
        }
    }

    // LU factorization (Doolittle, no pivoting)
    #pragma unroll
    for (int k = 0; k < 5; k++) {
        float inv_diag = 1.0f / LU[k][k];
        #pragma unroll
        for (int i = k + 1; i < 5; i++) {
            LU[i][k] *= inv_diag;
            #pragma unroll
            for (int j = k + 1; j < 5; j++) {
                LU[i][j] -= LU[i][k] * LU[k][j];
            }
        }
    }

    // Forward solve: L*y = b
    float y[5];
    y[0] = b[0];
    y[1] = b[1] - LU[1][0]*y[0];
    y[2] = b[2] - LU[2][0]*y[0] - LU[2][1]*y[1];
    y[3] = b[3] - LU[3][0]*y[0] - LU[3][1]*y[1] - LU[3][2]*y[2];
    y[4] = b[4] - LU[4][0]*y[0] - LU[4][1]*y[1] - LU[4][2]*y[2] - LU[4][3]*y[3];

    // Backward solve: U*x = y
    x[4] = y[4] / LU[4][4];
    x[3] = (y[3] - LU[3][4]*x[4]) / LU[3][3];
    x[2] = (y[2] - LU[2][3]*x[3] - LU[2][4]*x[4]) / LU[2][2];
    x[1] = (y[1] - LU[1][2]*x[2] - LU[1][3]*x[3] - LU[1][4]*x[4]) / LU[1][1];
    x[0] = (y[0] - LU[0][1]*x[1] - LU[0][2]*x[2] - LU[0][3]*x[3] - LU[0][4]*x[4]) / LU[0][0];
}

__device__ __forceinline__
void mat5x5_mat(const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C)
{
    #pragma unroll
    for (int i = 0; i < 5; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 5; ++j)
        {
            float sum = 0.0;
            #pragma unroll
            for (int k = 0; k < 5; ++k)
            {
                sum += A[5*i + k] * B[5*k + j];
            }
            C[5*i + j] = sum;
        }
    }
}

__device__ __forceinline__
void mat5x5_vec(const float* __restrict__ A,
                const float* __restrict__ x,
                float* __restrict__ y)
{
    #pragma unroll
    for (int i = 0; i < 5; ++i)
    {
        double sum = 0.0;
        #pragma unroll
        for (int j = 0; j < 5; ++j)
        {
            sum += A[5*i + j] * x[j];
        }
        y[i] = sum;
    }
}
	
template <int height, int width>
__device__ void build_5x5_vanka(
    float* __restrict__ J,
    float* __restrict__ r,
    float u_l, float u_r,
    float v_t, float v_b,
    float H_c,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float (&eta_local)[height][width], 
    const float* __restrict__ phi,
    const float* __restrict__ xi,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float n, float eps_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding, 
    float calving_rate, float flotation_reg_calving,
    float dx, float dt,
    int ny, int nx,
    int i, int j,
    int bi, int bj)
{
    float dx_inv = 1.0f/dx;

    for (int k=0;k<25;k++) J[k] = 0.0f;
    for (int k=0;k<5;k++) r[k] = 0.0f;

    float phi_c = get_cell(phi,i,j,ny,nx);
    float phi_l = get_cell(phi,i,j-1,ny,nx);
    float phi_r = get_cell(phi,i,j+1,ny,nx);
    float phi_t = get_cell(phi,i-1,j,ny,nx);
    float phi_b = get_cell(phi,i+1,j,ny,nx);

    // Mass Conservation Assembly
    {
    // Standard Mass Conservation: dH/dt + div(q) - smb = 0
    J[24] = 1.0f / dt;
    r[4] += H_c/dt;

    // X-Fluxes
    float H_l = get_cell(H,i,j-1,ny,nx);
    HorizontalFluxJacobian j_l = get_horizontal_flux_jac({u_l, H_l, H_c}, i, j, ny, nx);
    J[20] -= j_l.d_u   * dx_inv;
    J[24] -= j_l.d_H_r * dx_inv;
    r[4]  -= j_l.res   * dx_inv;


    FacetCalvingJacobian j_calve_l = get_facet_calving_jac({H_c,H_l,phi_c,phi_l,calving_rate,flotation_reg_calving},i,j,ny,nx);
    J[24] += j_calve_l.d_H_this * dx_inv;
    r[4] += j_calve_l.res*dx_inv;

    float H_r = get_cell(H,i,j+1,ny,nx);
    HorizontalFluxJacobian j_r = get_horizontal_flux_jac({u_r, H_c, H_r}, i, j+1, ny, nx);
    J[21] += j_r.d_u   * dx_inv;
    J[24] += j_r.d_H_l * dx_inv;
    r[4]  += j_r.res   * dx_inv;
    
    FacetCalvingJacobian j_calve_r = get_facet_calving_jac({H_c,H_r,phi_c,phi_r,calving_rate,flotation_reg_calving},i,j+1,ny,nx);
    J[24] += j_calve_r.d_H_this * dx_inv;
    r[4] += j_calve_r.res * dx_inv;

    // Y-Fluxes (Vertical in grid coordinates)
    float H_t = get_cell(H,i-1,j,ny,nx);
    VerticalFluxJacobian j_t = get_vertical_flux_jac({v_t, H_t, H_c}, i, j, ny, nx);
    J[22] += j_t.d_v   * dx_inv;
    J[24] += j_t.d_H_b * dx_inv;
    r[4]  += j_t.res   * dx_inv;

    FacetCalvingJacobian j_calve_t = get_facet_calving_jac({H_c,H_t,phi_c,phi_t,calving_rate,flotation_reg_calving},i,j,ny,nx);
    J[24] += j_calve_t.d_H_this * dx_inv;
    r[4] += j_calve_t.res * dx_inv;

    float H_b = get_cell(H,i+1,j,ny,nx);
    VerticalFluxJacobian j_b = get_vertical_flux_jac({v_b, H_c, H_b}, i+1, j, ny, nx);
    J[23] -= j_b.d_v   * dx_inv;
    J[24] -= j_b.d_H_t * dx_inv;
    r[4]  -= j_b.res   * dx_inv;

    FacetCalvingJacobian j_calve_b = get_facet_calving_jac({H_c,H_b,phi_c,phi_b,calving_rate,flotation_reg_calving},i+1,j,ny,nx);
    J[24] += j_calve_b.d_H_this * dx_inv;
    r[4] += j_calve_b.res * dx_inv;
    }
    
    {
    float eta_c = eta_local[bi][bj];
    EtaHCellJacobian eta_H_c = get_eta_H_cell_jac({eta_c,H_c});
    
    // Compute the contribution of sigma_xx at the center to both the left and right u-residuals (since it is used by both)
    SigmaNormalJacobian sigma_xx_c = get_sigma_xx_jac({u_l,u_r,v_t,v_b,eta_H_c.res},dx_inv,i,j,ny,nx);
    
    r[0] += sigma_xx_c.res * dx_inv;
    J[0] += sigma_xx_c.d_u_l * dx_inv;
    J[1] += sigma_xx_c.d_u_r * dx_inv;
    J[2] += sigma_xx_c.d_v_t * dx_inv;
    J[3] += sigma_xx_c.d_v_b * dx_inv;
    J[4] += sigma_xx_c.d_eta_H * eta_H_c.d_H * dx_inv;
    
    r[1] -= sigma_xx_c.res * dx_inv;
    J[5] -= sigma_xx_c.d_u_l * dx_inv;
    J[6] -= sigma_xx_c.d_u_r * dx_inv;
    J[7] -= sigma_xx_c.d_v_t * dx_inv;
    J[8] -= sigma_xx_c.d_v_b * dx_inv;
    J[9] -= sigma_xx_c.d_eta_H * eta_H_c.d_H * dx_inv;

    SigmaNormalJacobian sigma_yy_c = get_sigma_yy_jac({u_l,u_r,v_t,v_b,eta_H_c.res},dx_inv,i,j,ny,nx);
    r[2]  -= sigma_yy_c.res * dx_inv;
    J[10] -= sigma_yy_c.d_u_l * dx_inv;
    J[11] -= sigma_yy_c.d_u_r * dx_inv;
    J[12] -= sigma_yy_c.d_v_t * dx_inv;
    J[13] -= sigma_yy_c.d_v_b * dx_inv;
    J[14] -= sigma_yy_c.d_eta_H * eta_H_c.d_H * dx_inv;

    r[3]  += sigma_yy_c.res * dx_inv;
    J[15] += sigma_yy_c.d_u_l * dx_inv;
    J[16] += sigma_yy_c.d_u_r * dx_inv;
    J[17] += sigma_yy_c.d_v_t * dx_inv;
    J[18] += sigma_yy_c.d_v_b * dx_inv;
    J[19] += sigma_yy_c.d_eta_H * eta_H_c.d_H * dx_inv;
    }

    // Compute the contribution of sigma_xx from the left cell to the left u-residual
    {
    float eta_l  = eta_local[bi][bj - 1];
    float H_l    = get_cell(H,i,j-1,ny,nx);
    EtaHCellJacobian eta_H_l = get_eta_H_cell_jac({eta_l,H_l});

    float u_ll   = get_vfacet(u,i,j-1,ny,nx);
    float v_lt   = get_hfacet(v,i,j-1,ny,nx);
    float v_lb   = get_hfacet(v,i+1,j-1,ny,nx);
    SigmaNormalJacobian sigma_xx_l = get_sigma_xx_jac({u_ll,u_l,v_lt,v_lb,eta_H_l.res},dx_inv,i,j - 1,ny,nx);
    r[0] -= sigma_xx_l.res * dx_inv;
    J[0] -= sigma_xx_l.d_u_r * dx_inv;
    }

    // Compute the contribution of sigma_xx from the right cell to the right u-residual
    {
    float eta_r  = eta_local[bi][bj + 1];
    float H_r    = get_cell(H,i,j+1,ny,nx);
    EtaHCellJacobian eta_H_r = get_eta_H_cell_jac({eta_r,H_r});

    float u_rr   = get_vfacet(u,i,j+2,ny,nx);
    float v_rt   = get_hfacet(v,i,j+1,ny,nx);
    float v_rb   = get_hfacet(v,i+1,j+1,ny,nx);
    SigmaNormalJacobian sigma_xx_r = get_sigma_xx_jac({u_r,u_rr,v_rt,v_rb,eta_H_r.res},dx_inv,i,j + 1,ny,nx);
    r[1] += sigma_xx_r.res * dx_inv;
    J[6] += sigma_xx_r.d_u_l * dx_inv;
    }

    // Compute the contribution of sigma_yy from the top cell to the top v-residual
    {
    float eta_t  = eta_local[bi - 1][bj];
    float H_t    = get_cell(H,i-1,j,ny,nx);
    EtaHCellJacobian eta_H_t = get_eta_H_cell_jac({eta_t,H_t});

    float u_tl   = get_vfacet(u,i-1,j,ny,nx);
    float u_tr   = get_vfacet(u,i-1,j+1,ny,nx);
    float v_tt   = get_hfacet(v,i-1,j,ny,nx);
    SigmaNormalJacobian sigma_yy_t = get_sigma_yy_jac({u_tl,u_tr,v_tt,v_t,eta_H_t.res},dx_inv,i - 1,j,ny,nx);
    r[2] += sigma_yy_t.res * dx_inv;
    J[12] += sigma_yy_t.d_v_b * dx_inv;
    }

    // Compute the contribution of sigma_yy from the bottom cell to the bottom v-residual
    {
    float eta_b  = eta_local[bi + 1][bj];
    float H_b    = get_cell(H,i + 1,j,ny,nx);
    EtaHCellJacobian eta_H_b = get_eta_H_cell_jac({eta_b,H_b});

    float u_bl   = get_vfacet(u,i+1,j,ny,nx);
    float u_br   = get_vfacet(u,i+1,j+1,ny,nx);
    float v_bb   = get_hfacet(v,i+2,j,ny,nx);
    SigmaNormalJacobian sigma_yy_b = get_sigma_yy_jac({u_bl,u_br,v_b,v_bb,eta_H_b.res},dx_inv,i + 1,j,ny,nx);
    r[3] -= sigma_yy_b.res * dx_inv;
    J[18] -= sigma_yy_b.d_v_t * dx_inv;
    }
    
    
    // Compute the contribution of sigma_xy from the top-left corner to the left u-residual and top v-residual
    {
    float eta_tl = eta_local[bi - 1][bj - 1];
    float eta_t  = eta_local[bi - 1][bj];
    float eta_l  = eta_local[bi][bj - 1];
    float eta_c  = eta_local[bi][bj];
    
    float H_tl   = get_cell(H,i-1,j-1,ny,nx);
    float H_t    = get_cell(H,i-1,j,ny,nx);
    float H_l    = get_cell(H,i,j-1,ny,nx);
    
    EtaHVertexJacobian eta_H_tl = get_eta_H_vertex_jac({eta_tl,eta_t,eta_l,eta_c,H_tl,H_t,H_l,H_c});
    
    float u_tl = get_vfacet(u,i-1,j,ny,nx);
    float v_lt = get_hfacet(v,i,j-1,ny,nx);
    
    SigmaShearJacobian sigma_xy_tl = get_sigma_xy_jac({u_tl,u_l,v_lt,v_t,eta_H_tl.res},dx_inv,i,j,ny,nx);
    r[0] += sigma_xy_tl.res * dx_inv;
    J[0] += sigma_xy_tl.d_u_b * dx_inv;
    J[4] += sigma_xy_tl.d_eta_H * eta_H_tl.d_H_br * dx_inv;

    r[2] -= sigma_xy_tl.res * dx_inv;
    J[12] -= sigma_xy_tl.d_v_r * dx_inv;
    J[14] -= sigma_xy_tl.d_eta_H * eta_H_tl.d_H_br * dx_inv;
    }

    // Compute the contribution of sigma_xy from the top-right corner to the right u-residual and top v-residual
    {
    float eta_t  = eta_local[bi - 1][bj];
    float eta_tr = eta_local[bi - 1][bj + 1];
    float eta_c  = eta_local[bi][bj];
    float eta_r  = eta_local[bi][bj + 1];
    
    float H_t    = get_cell(H,i-1,j,ny,nx);
    float H_tr   = get_cell(H,i-1,j+1,ny,nx);
    float H_r    = get_cell(H,i,j+1,ny,nx);
    
    EtaHVertexJacobian eta_H_tr = get_eta_H_vertex_jac({eta_t,eta_tr,eta_c,eta_r,H_t,H_tr,H_c,H_r});
    
    float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
    float v_rt = get_hfacet(v,i,j+1,ny,nx);
    
    SigmaShearJacobian sigma_xy_tr = get_sigma_xy_jac({u_tr,u_r,v_t,v_rt,eta_H_tr.res},dx_inv,i,j+1,ny,nx);
    r[1] += sigma_xy_tr.res * dx_inv;
    J[6] += sigma_xy_tr.d_u_b * dx_inv;
    J[9] += sigma_xy_tr.d_eta_H * eta_H_tr.d_H_bl * dx_inv;

    r[2] += sigma_xy_tr.res * dx_inv;
    J[12] += sigma_xy_tr.d_v_l * dx_inv;
    J[14] += sigma_xy_tr.d_eta_H * eta_H_tr.d_H_bl * dx_inv;
    }

    // Compute the contribution of sigma_xy from the bottom-left corner to the left u-residual and bottom v-residual
    {
    float eta_l  = eta_local[bi][bj - 1];
    float eta_c  = eta_local[bi][bj];
    float eta_bl = eta_local[bi + 1][bj - 1];
    float eta_b  = eta_local[bi + 1][bj];
    
    float H_l    = get_cell(H,i,j-1,ny,nx);
    float H_bl   = get_cell(H,i+1,j-1,ny,nx);
    float H_b    = get_cell(H,i+1,j,ny,nx);

    EtaHVertexJacobian eta_H_bl = get_eta_H_vertex_jac({eta_l,eta_c,eta_bl,eta_b,H_l,H_c,H_bl,H_b});
    
    float u_bl   = get_vfacet(u,i+1,j,ny,nx);
    float v_lb   = get_hfacet(v,i+1,j-1,ny,nx);
    SigmaShearJacobian sigma_xy_bl = get_sigma_xy_jac({u_l,u_bl,v_lb,v_b,eta_H_bl.res},dx_inv,i + 1,j,ny,nx);
    r[0] -= sigma_xy_bl.res * dx_inv;
    J[0] -= sigma_xy_bl.d_u_t * dx_inv;
    J[4] -= sigma_xy_bl.d_eta_H * eta_H_bl.d_H_tr * dx_inv;

    r[3] -= sigma_xy_bl.res * dx_inv;
    J[18] -= sigma_xy_bl.d_v_r * dx_inv;
    J[19] -= sigma_xy_bl.d_eta_H * eta_H_bl.d_H_tr * dx_inv;
    }

    // Compute the contribution of sigma_xy from the bottom-right corner to the right u-residual and bottom v-residual
    {
    float eta_c  = eta_local[bi][bj];
    float eta_r  = eta_local[bi][bj + 1];
    float eta_b  = eta_local[bi + 1][bj];
    float eta_br = eta_local[bi + 1][bj + 1];
    
    float H_r    = get_cell(H,i,j+1,ny,nx);
    float H_b    = get_cell(H,i+1,j,ny,nx);
    float H_br   = get_cell(H,i+1,j+1,ny,nx);

    EtaHVertexJacobian eta_H_br = get_eta_H_vertex_jac({eta_c,eta_r,eta_b,eta_br,H_c,H_r,H_b,H_br});
    
    float u_br   = get_vfacet(u,i+1,j+1,ny,nx);
    float v_rb   = get_hfacet(v,i+1,j+1,ny,nx);
    SigmaShearJacobian sigma_xy_br = get_sigma_xy_jac({u_r,u_br,v_b,v_rb,eta_H_br.res},dx_inv,i + 1,j + 1,ny,nx);
    r[1] -= sigma_xy_br.res * dx_inv;
    J[6] -= sigma_xy_br.d_u_t * dx_inv;
    J[9] -= sigma_xy_br.d_eta_H * eta_H_br.d_H_tl * dx_inv;

    r[3] += sigma_xy_br.res * dx_inv;
    J[18] += sigma_xy_br.d_v_l * dx_inv;
    J[19] += sigma_xy_br.d_eta_H * eta_H_br.d_H_tl * dx_inv;
    }
    
    
    // Basal shear stress for left momentum
    {
    float u_ll   = get_vfacet(u,i,j-1,ny,nx);
    float v_tl   = get_hfacet(v,i,j-1,ny,nx);
    float v_bl   = get_hfacet(v,i+1,j-1,ny,nx);

    float H_l    = get_cell(H,i,j-1,ny,nx);
    float beta_l = get_cell(beta,i,j-1,ny,nx);
    float beta_c = get_cell(beta,i,j,ny,nx);
    float xi_l = get_cell(xi,i,j-1,ny,nx);
    float xi_c = get_cell(xi,i,j,ny,nx);

    TauBxJacobian tau_bx_l = get_tau_bx_jac({u_l,u_ll,u_r,v_tl,v_t,v_bl,v_b,H_l,H_c,xi_l,xi_c,beta_l,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
    r[0] += tau_bx_l.res;
    J[0] += tau_bx_l.d_u_c;
    J[1] += tau_bx_l.d_u_r;
    J[2] += tau_bx_l.d_v_tr;
    J[3] += tau_bx_l.d_v_br;
    J[4] += tau_bx_l.d_H_r;
    }

    // Basal shear stress for right momentum
    {
    float u_rr   = get_vfacet(u,i,j+2,ny,nx);
    float v_tr   = get_hfacet(v,i,j+1,ny,nx);
    float v_br   = get_hfacet(v,i+1,j+1,ny,nx);
    
    float H_r    = get_cell(H,i,j+1,ny,nx);
    float beta_c = get_cell(beta,i,j,ny,nx);
    float beta_r = get_cell(beta,i,j+1,ny,nx);
    float xi_c = get_cell(xi,i,j,ny,nx);
    float xi_r = get_cell(xi,i,j+1,ny,nx);

    TauBxJacobian tau_bx_r = get_tau_bx_jac({u_r,u_l,u_rr,v_t,v_tr,v_b,v_br,H_c,H_r,xi_c,xi_r,beta_c,beta_r,m,u_reg,water_drag,flotation_reg_sliding});
    r[1] += tau_bx_r.res;
    J[5] += tau_bx_r.d_u_l;
    J[6] += tau_bx_r.d_u_c;
    J[7] += tau_bx_r.d_v_tl;
    J[8] += tau_bx_r.d_v_bl;
    J[9] += tau_bx_r.d_H_l;
    }

    // Basal shear stress for top momentum
    {
    float v_tt = get_hfacet(v,i-1,j,ny,nx);
    float u_tl = get_vfacet(u,i-1,j,ny,nx);
    float u_tr = get_vfacet(u,i-1,j+1,ny,nx);

    float H_t    = get_cell(H,i-1,j,ny,nx);
    float beta_t = get_cell(beta,i-1,j,ny,nx);
    float beta_c = get_cell(beta,i,j,ny,nx);
    float xi_t = get_cell(xi,i-1,j,ny,nx);
    float xi_c = get_cell(xi,i,j,ny,nx);

    TauByJacobian tau_by_t = get_tau_by_jac({v_t,v_tt,v_b,u_tl,u_tr,u_l,u_r,H_t,H_c,xi_t,xi_c,beta_t,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
    r[2]  += tau_by_t.res;
    J[12] += tau_by_t.d_v_c;
    J[13] += tau_by_t.d_v_b;
    J[10] += tau_by_t.d_u_bl;
    J[11] += tau_by_t.d_u_br;
    J[14] += tau_by_t.d_H_b;
    }

    // Basal shear stress for bottom momentum
    {
    float v_bb = get_hfacet(v,i+2,j,ny,nx);
    float u_bl = get_vfacet(u,i+1,j,ny,nx);
    float u_br = get_vfacet(u,i+1,j+1,ny,nx);

    float H_b    = get_cell(H,i+1,j,ny,nx);
    float beta_c = get_cell(beta,i,j,ny,nx);
    float beta_b = get_cell(beta,i+1,j,ny,nx);
    float xi_c = get_cell(xi,i,j,ny,nx);
    float xi_b = get_cell(xi,i+1,j,ny,nx);

    TauByJacobian tau_by_b = get_tau_by_jac({v_b,v_t,v_bb,u_l,u_r,u_bl,u_br,H_c,H_b,xi_c,xi_b,beta_c,beta_b,m,u_reg,water_drag,flotation_reg_sliding});
    r[3]  += tau_by_b.res;
    J[18] += tau_by_b.d_v_c;
    J[17] += tau_by_b.d_v_t;
    J[15] += tau_by_b.d_u_tl;
    J[16] += tau_by_b.d_u_tr;
    J[19] += tau_by_b.d_H_t;
    }
    
    // Driving stress for left momentum (u)
    {
    float H_l    = get_cell(H,i,j-1,ny,nx);
    float bed_l  = get_cell(bed,i,j-1,ny,nx);
    float bed_c  = get_cell(bed,i,j,ny,nx);
    TauDxJacobian tau_dx_l = get_tau_dx_jac({H_l,H_c,bed_l,bed_c,phi_l,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);
    r[0] -= tau_dx_l.res;
    J[4] -= tau_dx_l.d_H_r;
    }

    // Driving stress for right momentum (u)
    {
    float H_r    = get_cell(H,i,j+1,ny,nx);
    float bed_c  = get_cell(bed,i,j,ny,nx);
    float bed_r  = get_cell(bed,i,j+1,ny,nx);
    TauDxJacobian tau_dx_r = get_tau_dx_jac({H_c,H_r,bed_c,bed_r,phi_c,phi_r,flotation_reg_driving},dx_inv,i,j+1,ny,nx);
    r[1] -= tau_dx_r.res;
    J[9] -= tau_dx_r.d_H_l;
    }

    // Driving stress for top momentum (v)
    {
    float H_t    = get_cell(H,i-1,j,ny,nx);
    float bed_t  = get_cell(bed,i-1,j,ny,nx);
    float bed_c  = get_cell(bed,i,j,ny,nx);
    TauDyJacobian tau_dy_t = get_tau_dy_jac({H_t,H_c,bed_t,bed_c,phi_t,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);
    r[2]  -= tau_dy_t.res;
    J[14] -= tau_dy_t.d_H_b;
    }

    // Driving stress for bottom momentum (v)
    {
    float H_b    = get_cell(H,i+1,j,ny,nx);
    float bed_c  = get_cell(bed,i,j,ny,nx);
    float bed_b  = get_cell(bed,i+1,j,ny,nx);
    TauDyJacobian tau_dy_b = get_tau_dy_jac({H_c,H_b,bed_c,bed_b,phi_c,phi_b,flotation_reg_driving},dx_inv,i+1,j,ny,nx);
    r[3]  -= tau_dy_b.res;
    J[19] -= tau_dy_b.d_H_t;
    }
}

template <int height, int width>
__device__ void build_9x9_vanka(
    float* __restrict__ J,
    float* __restrict__ r,
    float u_l, float u_r,
    float v_t, float v_b,
    float ud_l, float ud_r,
    float vd_t, float vd_b,
    float H_c,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ ud,
    const float* __restrict__ vd,
    const float* __restrict__ H,
    const float (&eta_local)[height][width], 
    const float* __restrict__ phi,
    const float* __restrict__ xi,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float n, float eps_reg, float H_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding,
    float calving_rate, float flotation_reg_calving,
    float dx, float dt,
    int ny, int nx,
    int i, int j,
    int bi, int bj)
{
    float dx_inv = 1.0f/dx;

    for (int k=0;k<81;k++) J[k] = 0.0f;
    for (int k=0;k<25;k++) r[k] = 0.0f;

    float phi_c = get_cell(phi,i,j,ny,nx);
    float phi_l = get_cell(phi,i,j-1,ny,nx);
    float phi_r = get_cell(phi,i,j+1,ny,nx);
    float phi_t = get_cell(phi,i-1,j,ny,nx);
    float phi_b = get_cell(phi,i+1,j,ny,nx);

    float K_1 = 1.0f / (2.0f * n + 3.0f);
    float S_1 = (n + 2.0f) * (n + 2.0f) / (2.0f * n + 1.0f);
    float c_1 = __powf(1.0f / (n + 2.0f),2.0f * n / (n + 1.0f)) * (2.0f * n + 1.0f);

    // Mass Conservation Assembly
    {
    // Standard Mass Conservation: dH/dt + div(q) - smb = 0
    J[80] = 1.0f / dt;
    r[8] += H_c/dt;

    // X-Fluxes
    float H_l = get_cell(H,i,j-1,ny,nx);
    HorizontalFluxJacobian j_l = get_horizontal_flux_jac({u_l, H_l, H_c}, i, j, ny, nx);
    J[76] -= j_l.d_u   * dx_inv;
    J[80] -= j_l.d_H_r * dx_inv;
    r[8]  -= j_l.res   * dx_inv;


    FacetCalvingJacobian j_calve_l = get_facet_calving_jac({H_c,H_l,phi_c,phi_l,calving_rate,flotation_reg_calving},i,j,ny,nx);
    J[80] += j_calve_l.d_H_this * dx_inv;
    r[8] += j_calve_l.res*dx_inv;

    float H_r = get_cell(H,i,j+1,ny,nx);
    HorizontalFluxJacobian j_r = get_horizontal_flux_jac({u_r, H_c, H_r}, i, j+1, ny, nx);
    J[77] += j_r.d_u   * dx_inv;
    J[80] += j_r.d_H_l * dx_inv;
    r[8]  += j_r.res   * dx_inv;
    
    FacetCalvingJacobian j_calve_r = get_facet_calving_jac({H_c,H_r,phi_c,phi_r,calving_rate,flotation_reg_calving},i,j+1,ny,nx);
    J[80] += j_calve_r.d_H_this * dx_inv;
    r[8] += j_calve_r.res * dx_inv;

    // Y-Fluxes (Vertical in grid coordinates)
    float H_t = get_cell(H,i-1,j,ny,nx);
    VerticalFluxJacobian j_t = get_vertical_flux_jac({v_t, H_t, H_c}, i, j, ny, nx);
    J[78] += j_t.d_v   * dx_inv;
    J[80] += j_t.d_H_b * dx_inv;
    r[8]  += j_t.res   * dx_inv;

    FacetCalvingJacobian j_calve_t = get_facet_calving_jac({H_c,H_t,phi_c,phi_t,calving_rate,flotation_reg_calving},i,j,ny,nx);
    J[80] += j_calve_t.d_H_this * dx_inv;
    r[8] += j_calve_t.res * dx_inv;

    float H_b = get_cell(H,i+1,j,ny,nx);
    VerticalFluxJacobian j_b = get_vertical_flux_jac({v_b, H_c, H_b}, i+1, j, ny, nx);
    J[79] -= j_b.d_v   * dx_inv;
    J[80] -= j_b.d_H_t * dx_inv;
    r[8]  -= j_b.res   * dx_inv;

    FacetCalvingJacobian j_calve_b = get_facet_calving_jac({H_c,H_b,phi_c,phi_b,calving_rate,flotation_reg_calving},i+1,j,ny,nx);
    J[80] += j_calve_b.d_H_this * dx_inv;
    r[8] += j_calve_b.res * dx_inv;
    }
    
    {
    float eta_c = eta_local[bi][bj];
    EtaHCellJacobian eta_H_c = get_eta_H_cell_jac({eta_c,H_c});
    
    // Compute the contribution of sigma_xx at the center to both the left and right u-residuals (since it is used by both)
    SigmaNormalJacobian sigma_xx_c = get_sigma_xx_jac({u_l,u_r,v_t,v_b,eta_H_c.res},dx_inv,i,j,ny,nx);
    
    r[4] += sigma_xx_c.res * dx_inv;
    J[40] += sigma_xx_c.d_u_l * dx_inv;
    J[41] += sigma_xx_c.d_u_r * dx_inv;
    J[42] += sigma_xx_c.d_v_t * dx_inv;
    J[43] += sigma_xx_c.d_v_b * dx_inv;
    J[44] += sigma_xx_c.d_eta_H * eta_H_c.d_H * dx_inv;
    
    r[5] -= sigma_xx_c.res * dx_inv;
    J[49] -= sigma_xx_c.d_u_l * dx_inv;
    J[50] -= sigma_xx_c.d_u_r * dx_inv;
    J[51] -= sigma_xx_c.d_v_t * dx_inv;
    J[52] -= sigma_xx_c.d_v_b * dx_inv;
    J[53] -= sigma_xx_c.d_eta_H * eta_H_c.d_H * dx_inv;

    // Compute the contribution of sigma_xx at the center to both the left and right u-residuals (since it is used by both)
    SigmaNormalJacobian sigmad_xx_c = get_sigma_xx_jac({ud_l,ud_r,vd_t,vd_b,eta_H_c.res},dx_inv,i,j,ny,nx);
    
    r[0] += K_1 * sigmad_xx_c.res * dx_inv;
    J[0] += K_1 * sigmad_xx_c.d_u_l * dx_inv;
    J[1] += K_1 * sigmad_xx_c.d_u_r * dx_inv;
    J[2] += K_1 * sigmad_xx_c.d_v_t * dx_inv;
    J[3] += K_1 * sigmad_xx_c.d_v_b * dx_inv;
    J[8] += K_1 * sigmad_xx_c.d_eta_H * eta_H_c.d_H * dx_inv;
    
    r[1] -= K_1 * sigmad_xx_c.res * dx_inv;
    J[9] -= K_1 * sigmad_xx_c.d_u_l * dx_inv;
    J[10] -= K_1 * sigmad_xx_c.d_u_r * dx_inv;
    J[11] -= K_1 * sigmad_xx_c.d_v_t * dx_inv;
    J[12] -= K_1 * sigmad_xx_c.d_v_b * dx_inv;
    J[17] -= K_1 * sigmad_xx_c.d_eta_H * eta_H_c.d_H * dx_inv;

    SigmaNormalJacobian sigma_yy_c = get_sigma_yy_jac({u_l,u_r,v_t,v_b,eta_H_c.res},dx_inv,i,j,ny,nx);
    r[6]  -= sigma_yy_c.res * dx_inv;
    J[58] -= sigma_yy_c.d_u_l * dx_inv;
    J[59] -= sigma_yy_c.d_u_r * dx_inv;
    J[60] -= sigma_yy_c.d_v_t * dx_inv;
    J[61] -= sigma_yy_c.d_v_b * dx_inv;
    J[62] -= sigma_yy_c.d_eta_H * eta_H_c.d_H * dx_inv;

    r[7]  += sigma_yy_c.res * dx_inv;
    J[67] += sigma_yy_c.d_u_l * dx_inv;
    J[68] += sigma_yy_c.d_u_r * dx_inv;
    J[69] += sigma_yy_c.d_v_t * dx_inv;
    J[70] += sigma_yy_c.d_v_b * dx_inv;
    J[71] += sigma_yy_c.d_eta_H * eta_H_c.d_H * dx_inv;

    SigmaNormalJacobian sigmad_yy_c = get_sigma_yy_jac({ud_l,ud_r,vd_t,vd_b,eta_H_c.res},dx_inv,i,j,ny,nx);
    r[2]  -= K_1 * sigmad_yy_c.res * dx_inv;
    J[18] -= K_1 * sigmad_yy_c.d_u_l * dx_inv;
    J[19] -= K_1 * sigmad_yy_c.d_u_r * dx_inv;
    J[20] -= K_1 * sigmad_yy_c.d_v_t * dx_inv;
    J[21] -= K_1 * sigmad_yy_c.d_v_b * dx_inv;
    J[26] -= K_1 * sigmad_yy_c.d_eta_H * eta_H_c.d_H * dx_inv;

    r[3]  += K_1 * sigmad_yy_c.res * dx_inv;
    J[27] += K_1 * sigmad_yy_c.d_u_l * dx_inv;
    J[28] += K_1 * sigmad_yy_c.d_u_r * dx_inv;
    J[29] += K_1 * sigmad_yy_c.d_v_t * dx_inv;
    J[30] += K_1 * sigmad_yy_c.d_v_b * dx_inv;
    J[35] += K_1 * sigmad_yy_c.d_eta_H * eta_H_c.d_H * dx_inv;
    }

    // Compute the contribution of sigma_xx from the left cell to the left u-residual
    {
    float eta_l  = eta_local[bi][bj - 1];
    float H_l    = get_cell(H,i,j-1,ny,nx);
    EtaHCellJacobian eta_H_l = get_eta_H_cell_jac({eta_l,H_l});

    float u_ll   = get_vfacet(u,i,j-1,ny,nx);
    float v_lt   = get_hfacet(v,i,j-1,ny,nx);
    float v_lb   = get_hfacet(v,i+1,j-1,ny,nx);
    SigmaNormalJacobian sigma_xx_l = get_sigma_xx_jac({u_ll,u_l,v_lt,v_lb,eta_H_l.res},dx_inv,i,j - 1,ny,nx);
    r[4] -= sigma_xx_l.res * dx_inv;
    J[40] -= sigma_xx_l.d_u_r * dx_inv;

    float ud_ll   = get_vfacet(ud,i,j-1,ny,nx);
    float vd_lt   = get_hfacet(vd,i,j-1,ny,nx);
    float vd_lb   = get_hfacet(vd,i+1,j-1,ny,nx);
    SigmaNormalJacobian sigmad_xx_l = get_sigma_xx_jac({ud_ll,ud_l,vd_lt,vd_lb,eta_H_l.res},dx_inv,i,j - 1,ny,nx);
    r[0] -= K_1 * sigmad_xx_l.res * dx_inv;
    J[0] -= K_1 * sigmad_xx_l.d_u_r * dx_inv;
    }

    // Compute the contribution of sigma_xx from the right cell to the right u-residual
    {
    float eta_r  = eta_local[bi][bj + 1];
    float H_r    = get_cell(H,i,j+1,ny,nx);
    EtaHCellJacobian eta_H_r = get_eta_H_cell_jac({eta_r,H_r});

    float u_rr   = get_vfacet(u,i,j+2,ny,nx);
    float v_rt   = get_hfacet(v,i,j+1,ny,nx);
    float v_rb   = get_hfacet(v,i+1,j+1,ny,nx);
    SigmaNormalJacobian sigma_xx_r = get_sigma_xx_jac({u_r,u_rr,v_rt,v_rb,eta_H_r.res},dx_inv,i,j + 1,ny,nx);
    r[5] += sigma_xx_r.res * dx_inv;
    J[50] += sigma_xx_r.d_u_l * dx_inv;

    float ud_rr   = get_vfacet(ud,i,j+2,ny,nx);
    float vd_rt   = get_hfacet(vd,i,j+1,ny,nx);
    float vd_rb   = get_hfacet(vd,i+1,j+1,ny,nx);
    SigmaNormalJacobian sigmad_xx_r = get_sigma_xx_jac({ud_r,ud_rr,vd_rt,vd_rb,eta_H_r.res},dx_inv,i,j + 1,ny,nx);
    r[1] += K_1 * sigmad_xx_r.res * dx_inv;
    J[10] += K_1 * sigmad_xx_r.d_u_l * dx_inv;
    }

    // Compute the contribution of sigma_yy from the top cell to the top v-residual
    {
    float eta_t  = eta_local[bi - 1][bj];
    float H_t    = get_cell(H,i-1,j,ny,nx);
    EtaHCellJacobian eta_H_t = get_eta_H_cell_jac({eta_t,H_t});

    float u_tl   = get_vfacet(u,i-1,j,ny,nx);
    float u_tr   = get_vfacet(u,i-1,j+1,ny,nx);
    float v_tt   = get_hfacet(v,i-1,j,ny,nx);
    SigmaNormalJacobian sigma_yy_t = get_sigma_yy_jac({u_tl,u_tr,v_tt,v_t,eta_H_t.res},dx_inv,i - 1,j,ny,nx);
    r[6] += sigma_yy_t.res * dx_inv;
    J[60] += sigma_yy_t.d_v_b * dx_inv;

    float ud_tl   = get_vfacet(ud,i-1,j,ny,nx);
    float ud_tr   = get_vfacet(ud,i-1,j+1,ny,nx);
    float vd_tt   = get_hfacet(vd,i-1,j,ny,nx);
    SigmaNormalJacobian sigmad_yy_t = get_sigma_yy_jac({ud_tl,ud_tr,vd_tt,vd_t,eta_H_t.res},dx_inv,i - 1,j,ny,nx);
    r[2] += K_1 * sigmad_yy_t.res * dx_inv;
    J[20] += K_1 * sigmad_yy_t.d_v_b * dx_inv;
    }

    // Compute the contribution of sigma_yy from the bottom cell to the bottom v-residual
    {
    float eta_b  = eta_local[bi + 1][bj];
    float H_b    = get_cell(H,i + 1,j,ny,nx);
    EtaHCellJacobian eta_H_b = get_eta_H_cell_jac({eta_b,H_b});

    float u_bl   = get_vfacet(u,i+1,j,ny,nx);
    float u_br   = get_vfacet(u,i+1,j+1,ny,nx);
    float v_bb   = get_hfacet(v,i+2,j,ny,nx);
    SigmaNormalJacobian sigma_yy_b = get_sigma_yy_jac({u_bl,u_br,v_b,v_bb,eta_H_b.res},dx_inv,i + 1,j,ny,nx);
    r[7] -= sigma_yy_b.res * dx_inv;
    J[70] -= sigma_yy_b.d_v_t * dx_inv;

    float ud_bl   = get_vfacet(ud,i+1,j,ny,nx);
    float ud_br   = get_vfacet(ud,i+1,j+1,ny,nx);
    float vd_bb   = get_hfacet(vd,i+2,j,ny,nx);
    SigmaNormalJacobian sigmad_yy_b = get_sigma_yy_jac({ud_bl,ud_br,vd_b,vd_bb,eta_H_b.res},dx_inv,i + 1,j,ny,nx);
    r[3] -= K_1 * sigmad_yy_b.res * dx_inv;
    J[30] -= K_1 * sigmad_yy_b.d_v_t * dx_inv;

    }
    
    
    // Compute the contribution of sigma_xy from the top-left corner to the left u-residual and top v-residual
    {
    float eta_tl = eta_local[bi - 1][bj - 1];
    float eta_t  = eta_local[bi - 1][bj];
    float eta_l  = eta_local[bi][bj - 1];
    float eta_c  = eta_local[bi][bj];
    
    float H_tl   = get_cell(H,i-1,j-1,ny,nx);
    float H_t    = get_cell(H,i-1,j,ny,nx);
    float H_l    = get_cell(H,i,j-1,ny,nx);
    
    EtaHVertexJacobian eta_H_tl = get_eta_H_vertex_jac({eta_tl,eta_t,eta_l,eta_c,H_tl,H_t,H_l,H_c});
    
    float u_tl = get_vfacet(u,i-1,j,ny,nx);
    float v_lt = get_hfacet(v,i,j-1,ny,nx);
    
    SigmaShearJacobian sigma_xy_tl = get_sigma_xy_jac({u_tl,u_l,v_lt,v_t,eta_H_tl.res},dx_inv,i,j,ny,nx);
    r[4] += sigma_xy_tl.res * dx_inv;
    J[40] += sigma_xy_tl.d_u_b * dx_inv;
    J[44] += sigma_xy_tl.d_eta_H * eta_H_tl.d_H_br * dx_inv;

    r[6] -= sigma_xy_tl.res * dx_inv;
    J[60] -= sigma_xy_tl.d_v_r * dx_inv;
    J[62] -= sigma_xy_tl.d_eta_H * eta_H_tl.d_H_br * dx_inv;

    float ud_tl = get_vfacet(ud,i-1,j,ny,nx);
    float vd_lt = get_hfacet(vd,i,j-1,ny,nx);
    
    SigmaShearJacobian sigmad_xy_tl = get_sigma_xy_jac({ud_tl,ud_l,vd_lt,vd_t,eta_H_tl.res},dx_inv,i,j,ny,nx);
    r[0] += K_1 * sigmad_xy_tl.res * dx_inv;
    J[0] += K_1 * sigmad_xy_tl.d_u_b * dx_inv;
    J[8] += K_1 * sigmad_xy_tl.d_eta_H * eta_H_tl.d_H_br * dx_inv;

    r[2] -= K_1 * sigmad_xy_tl.res * dx_inv;
    J[20] -= K_1 * sigmad_xy_tl.d_v_r * dx_inv;
    J[26] -= K_1 * sigmad_xy_tl.d_eta_H * eta_H_tl.d_H_br * dx_inv;
    }


    // Compute the contribution of sigma_xy from the top-right corner to the right u-residual and top v-residual
    {
    float eta_t  = eta_local[bi - 1][bj];
    float eta_tr = eta_local[bi - 1][bj + 1];
    float eta_c  = eta_local[bi][bj];
    float eta_r  = eta_local[bi][bj + 1];
    
    float H_t    = get_cell(H,i-1,j,ny,nx);
    float H_tr   = get_cell(H,i-1,j+1,ny,nx);
    float H_r    = get_cell(H,i,j+1,ny,nx);
    
    EtaHVertexJacobian eta_H_tr = get_eta_H_vertex_jac({eta_t,eta_tr,eta_c,eta_r,H_t,H_tr,H_c,H_r});
    
    float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
    float v_rt = get_hfacet(v,i,j+1,ny,nx);
    
    SigmaShearJacobian sigma_xy_tr = get_sigma_xy_jac({u_tr,u_r,v_t,v_rt,eta_H_tr.res},dx_inv,i,j+1,ny,nx);
    r[5] += sigma_xy_tr.res * dx_inv;
    J[50] += sigma_xy_tr.d_u_b * dx_inv;
    J[53] += sigma_xy_tr.d_eta_H * eta_H_tr.d_H_bl * dx_inv;

    r[6] += sigma_xy_tr.res * dx_inv;
    J[60] += sigma_xy_tr.d_v_l * dx_inv;
    J[62] += sigma_xy_tr.d_eta_H * eta_H_tr.d_H_bl * dx_inv;

    float ud_tr = get_vfacet(ud,i-1,j+1,ny,nx);
    float vd_rt = get_hfacet(vd,i,j+1,ny,nx);
    
    SigmaShearJacobian sigmad_xy_tr = get_sigma_xy_jac({ud_tr,ud_r,vd_t,vd_rt,eta_H_tr.res},dx_inv,i,j+1,ny,nx);
    r[1] += K_1 * sigmad_xy_tr.res * dx_inv;
    J[10] += K_1 * sigmad_xy_tr.d_u_b * dx_inv;
    J[17] += K_1 * sigmad_xy_tr.d_eta_H * eta_H_tr.d_H_bl * dx_inv;

    r[2] += K_1 * sigmad_xy_tr.res * dx_inv;
    J[20] += K_1 * sigmad_xy_tr.d_v_l * dx_inv;
    J[26] += K_1 * sigmad_xy_tr.d_eta_H * eta_H_tr.d_H_bl * dx_inv;

    }

    // Compute the contribution of sigma_xy from the bottom-left corner to the left u-residual and bottom v-residual
    {
    float eta_l  = eta_local[bi][bj - 1];
    float eta_c  = eta_local[bi][bj];
    float eta_bl = eta_local[bi + 1][bj - 1];
    float eta_b  = eta_local[bi + 1][bj];
    
    float H_l    = get_cell(H,i,j-1,ny,nx);
    float H_bl   = get_cell(H,i+1,j-1,ny,nx);
    float H_b    = get_cell(H,i+1,j,ny,nx);

    EtaHVertexJacobian eta_H_bl = get_eta_H_vertex_jac({eta_l,eta_c,eta_bl,eta_b,H_l,H_c,H_bl,H_b});
    
    float u_bl   = get_vfacet(u,i+1,j,ny,nx);
    float v_lb   = get_hfacet(v,i+1,j-1,ny,nx);
    SigmaShearJacobian sigma_xy_bl = get_sigma_xy_jac({u_l,u_bl,v_lb,v_b,eta_H_bl.res},dx_inv,i + 1,j,ny,nx);
    r[4] -= sigma_xy_bl.res * dx_inv;
    J[40] -= sigma_xy_bl.d_u_t * dx_inv;
    J[44] -= sigma_xy_bl.d_eta_H * eta_H_bl.d_H_tr * dx_inv;

    r[7] -= sigma_xy_bl.res * dx_inv;
    J[70] -= sigma_xy_bl.d_v_r * dx_inv;
    J[71] -= sigma_xy_bl.d_eta_H * eta_H_bl.d_H_tr * dx_inv;

    float ud_bl   = get_vfacet(ud,i+1,j,ny,nx);
    float vd_lb   = get_hfacet(vd,i+1,j-1,ny,nx);
    SigmaShearJacobian sigmad_xy_bl = get_sigma_xy_jac({ud_l,ud_bl,vd_lb,vd_b,eta_H_bl.res},dx_inv,i + 1,j,ny,nx);
    r[0] -= K_1 * sigmad_xy_bl.res * dx_inv;
    J[0] -= K_1 * sigmad_xy_bl.d_u_t * dx_inv;
    J[8] -= K_1 * sigmad_xy_bl.d_eta_H * eta_H_bl.d_H_tr * dx_inv;

    r[3] -= K_1 * sigmad_xy_bl.res * dx_inv;
    J[30] -= K_1 * sigmad_xy_bl.d_v_r * dx_inv;
    J[35] -= K_1 * sigmad_xy_bl.d_eta_H * eta_H_bl.d_H_tr * dx_inv;
    }

    // Compute the contribution of sigma_xy from the bottom-right corner to the right u-residual and bottom v-residual
    {
    float eta_c  = eta_local[bi][bj];
    float eta_r  = eta_local[bi][bj + 1];
    float eta_b  = eta_local[bi + 1][bj];
    float eta_br = eta_local[bi + 1][bj + 1];
    
    float H_r    = get_cell(H,i,j+1,ny,nx);
    float H_b    = get_cell(H,i+1,j,ny,nx);
    float H_br   = get_cell(H,i+1,j+1,ny,nx);

    EtaHVertexJacobian eta_H_br = get_eta_H_vertex_jac({eta_c,eta_r,eta_b,eta_br,H_c,H_r,H_b,H_br});
    
    float u_br   = get_vfacet(u,i+1,j+1,ny,nx);
    float v_rb   = get_hfacet(v,i+1,j+1,ny,nx);
    SigmaShearJacobian sigma_xy_br = get_sigma_xy_jac({u_r,u_br,v_b,v_rb,eta_H_br.res},dx_inv,i + 1,j + 1,ny,nx);
    r[5] -= sigma_xy_br.res * dx_inv;
    J[50] -= sigma_xy_br.d_u_t * dx_inv;
    J[53] -= sigma_xy_br.d_eta_H * eta_H_br.d_H_tl * dx_inv;

    r[7] += sigma_xy_br.res * dx_inv;
    J[70] += sigma_xy_br.d_v_l * dx_inv;
    J[71] += sigma_xy_br.d_eta_H * eta_H_br.d_H_tl * dx_inv;

    float ud_br   = get_vfacet(ud,i+1,j+1,ny,nx);
    float vd_rb   = get_hfacet(vd,i+1,j+1,ny,nx);
    SigmaShearJacobian sigmad_xy_br = get_sigma_xy_jac({ud_r,ud_br,vd_b,vd_rb,eta_H_br.res},dx_inv,i + 1,j + 1,ny,nx);
    r[1] -= K_1 * sigmad_xy_br.res * dx_inv;
    J[10] -= K_1 * sigmad_xy_br.d_u_t * dx_inv;
    J[17] -= K_1 * sigmad_xy_br.d_eta_H * eta_H_br.d_H_tl * dx_inv;

    r[3] += K_1 * sigmad_xy_br.res * dx_inv;
    J[30] += K_1 * sigmad_xy_br.d_v_l * dx_inv;
    J[35] += K_1 * sigmad_xy_br.d_eta_H * eta_H_br.d_H_tl * dx_inv;
    }

    
    // Vertical shear for left momentum
    {
    float eta_l = eta_local[bi][bj-1];
    float eta_c = eta_local[bi][bj];
    float H_l = get_cell(H,i,j-1,ny,nx);
    float H_c = get_cell(H,i,j,ny,nx);
    float B_l = get_cell(B,i,j-1,ny,nx);
    float B_c = get_cell(B,i,j,ny,nx);

    SigmaVertXZJacobian sigmad_xz_l = get_sigma_xz_jac({ud_l,eta_l,eta_c,H_l,H_c},c_1,S_1,H_reg,i,j,ny,nx);

    r[0] += sigmad_xz_l.res;
    J[0] += sigmad_xz_l.d_u_c;
    J[0] += get_sigma_vert_dvisc(ud_l,eta_l,eta_c,H_l,H_c,B_l,B_c,c_1,S_1,n,H_reg);
    J[8] += sigmad_xz_l.d_H_r;
    }

    // Vertical shear for right momentum
    {
    float eta_c = eta_local[bi][bj];
    float eta_r = eta_local[bi][bj+1];
    float H_c = get_cell(H,i,j,ny,nx);
    float H_r = get_cell(H,i,j+1,ny,nx);
    float B_c = get_cell(B,i,j,ny,nx);
    float B_r = get_cell(B,i,j+1,ny,nx);

    SigmaVertXZJacobian sigmad_xz_r = get_sigma_xz_jac({ud_r,eta_c,eta_r,H_c,H_r},c_1,S_1,H_reg,i,j,ny,nx);

    r[1] += sigmad_xz_r.res;
    J[10] += sigmad_xz_r.d_u_c;
    J[10] += get_sigma_vert_dvisc(ud_r,eta_c,eta_r,H_c,H_r,B_c,B_r,c_1,S_1,n,H_reg);
    J[17] += sigmad_xz_r.d_H_l;
    }

    // Vertical shear for top momentum
    {
    float eta_t = eta_local[bi-1][bj];
    float eta_c = eta_local[bi][bj];
    float H_t = get_cell(H,i-1,j,ny,nx);
    float H_c = get_cell(H,i,j,ny,nx);
    float B_t = get_cell(B,i-1,j,ny,nx);
    float B_c = get_cell(B,i,j,ny,nx);

    SigmaVertYZJacobian sigmad_yz_t = get_sigma_yz_jac({vd_t,eta_t,eta_c,H_t,H_c},c_1,S_1,H_reg,i,j,ny,nx);

    r[2] += sigmad_yz_t.res;
    J[20] += sigmad_yz_t.d_v_c;
    J[20] += get_sigma_vert_dvisc(vd_t,eta_t,eta_c,H_t,H_c,B_t,B_c,c_1,S_1,n,H_reg);
    J[26] += sigmad_yz_t.d_H_b;
    }

    // Vertical shear for bottom momentum
    {
    float eta_c = eta_local[bi][bj];
    float eta_b = eta_local[bi+1][bj];
    float H_c = get_cell(H,i,j,ny,nx);
    float H_b = get_cell(H,i+1,j,ny,nx);
    float B_c = get_cell(B,i,j,ny,nx);
    float B_b = get_cell(B,i+1,j,ny,nx);

    SigmaVertYZJacobian sigmad_yz_b = get_sigma_yz_jac({vd_b,eta_c,eta_b,H_c,H_b},c_1,S_1,H_reg,i,j,ny,nx);

    r[3] += sigmad_yz_b.res;
    J[30] += sigmad_yz_b.d_v_c;
    J[30] += get_sigma_vert_dvisc(vd_b,eta_c,eta_b,H_c,H_b,B_c,B_b,c_1,S_1,n,H_reg);
    J[35] += sigmad_yz_b.d_H_t;
    }
    
    
    // Basal shear stress for left momentum
    {

    float ub_l = u_l - ud_l;
    float ub_r = u_r - ud_r;
    float vb_t = v_t - vd_t;
    float vb_b = v_b - vd_b;

    float u_ll  = get_vfacet(u,i,j-1,ny,nx);
    float ud_ll = get_vfacet(ud,i,j-1,ny,nx);
    float ub_ll = u_ll - ud_ll; 

    float v_tl  = get_hfacet(v,i,j-1,ny,nx);
    float vd_tl = get_hfacet(vd,i,j-1,ny,nx);
    float vb_tl = v_tl - vd_tl;

    float v_bl  = get_hfacet(v,i+1,j-1,ny,nx);
    float vd_bl = get_hfacet(vd,i+1,j-1,ny,nx);
    float vb_bl = v_bl - vd_bl; 

    float H_l    = get_cell(H,i,j-1,ny,nx);
    float beta_l = get_cell(beta,i,j-1,ny,nx);
    float beta_c = get_cell(beta,i,j,ny,nx);
    float xi_l = get_cell(xi,i,j-1,ny,nx);
    float xi_c = get_cell(xi,i,j,ny,nx);

    TauBxJacobian tau_bx_l = get_tau_bx_jac({ub_l,ub_ll,ub_r,vb_tl,vb_t,vb_bl,vb_b,H_l,H_c,xi_l,xi_c,beta_l,beta_c,m,u_reg,water_drag,flotation_reg_sliding});

    // Residual for averaged component
    r[4] += tau_bx_l.res;

    // Vertical cross terms
    J[36] -= tau_bx_l.d_u_c;
    J[37] -= tau_bx_l.d_u_r;
    J[38] -= tau_bx_l.d_v_tr;
    J[39] -= tau_bx_l.d_v_br;

    // Horizontal cross terms
    J[40] += tau_bx_l.d_u_c;
    J[41] += tau_bx_l.d_u_r;
    J[42] += tau_bx_l.d_v_tr;
    J[43] += tau_bx_l.d_v_br;

    // thickness dependence (currently ignored)
    J[44] += tau_bx_l.d_H_r;

    // Residual for deformational component
    r[0]  -= tau_bx_l.res;

    // Vertical cross terms
    J[0]  += tau_bx_l.d_u_c;
    J[1]  += tau_bx_l.d_u_r;
    J[2]  += tau_bx_l.d_v_tr;
    J[3]  += tau_bx_l.d_v_br;
    
    // Horizontal cross terms
    J[4]  -= tau_bx_l.d_u_c;
    J[5]  -= tau_bx_l.d_u_r;
    J[6]  -= tau_bx_l.d_v_tr;
    J[7]  -= tau_bx_l.d_v_br;

    // thickness dependence
    J[8]  -= tau_bx_l.d_H_r;
    }

    // Basal shear stress for right momentum
    {
    float ub_l = u_l - ud_l;
    float ub_r = u_r - ud_r;
    float vb_t = v_t - vd_t;
    float vb_b = v_b - vd_b;

    float u_rr   = get_vfacet(u,i,j+2,ny,nx);
    float ud_rr  = get_vfacet(ud,i,j+2,ny,nx);
    float ub_rr  = u_rr - ud_rr;

    float v_tr   = get_hfacet(v,i,j+1,ny,nx);
    float vd_tr  = get_hfacet(vd,i,j+1,ny,nx);
    float vb_tr  = v_tr - vd_tr;
    
    float v_br   = get_hfacet(v,i+1,j+1,ny,nx);
    float vd_br  = get_hfacet(vd,i+1,j+1,ny,nx);
    float vb_br  = v_br - vd_br;
    
    float H_r    = get_cell(H,i,j+1,ny,nx);
    float beta_c = get_cell(beta,i,j,ny,nx);
    float beta_r = get_cell(beta,i,j+1,ny,nx);
    float xi_c = get_cell(xi,i,j,ny,nx);
    float xi_r = get_cell(xi,i,j+1,ny,nx);

    TauBxJacobian tau_bx_r = get_tau_bx_jac({ub_r,ub_l,ub_rr,vb_t,vb_tr,vb_b,vb_br,H_c,H_r,xi_c,xi_r,beta_c,beta_r,m,u_reg,water_drag,flotation_reg_sliding});
    r[5] += tau_bx_r.res;
    
    J[45] -= tau_bx_r.d_u_l;
    J[46] -= tau_bx_r.d_u_c;
    J[47] -= tau_bx_r.d_v_tl;
    J[48] -= tau_bx_r.d_v_bl;
    
    J[49] += tau_bx_r.d_u_l;
    J[50] += tau_bx_r.d_u_c;
    J[51] += tau_bx_r.d_v_tl;
    J[52] += tau_bx_r.d_v_bl;
    J[53] += tau_bx_r.d_H_l;
    
    r[1]  -= tau_bx_r.res;
    J[9]  += tau_bx_r.d_u_l;
    J[10] += tau_bx_r.d_u_c;
    J[11] += tau_bx_r.d_v_tl;
    J[12] += tau_bx_r.d_v_bl;
    
    J[13] -= tau_bx_r.d_u_l;
    J[14] -= tau_bx_r.d_u_c;
    J[15] -= tau_bx_r.d_v_tl;
    J[16] -= tau_bx_r.d_v_bl;

    J[17] -= tau_bx_r.d_H_l;
    }

    // Basal shear stress for top momentum
    {
    float ub_l = u_l - ud_l;
    float ub_r = u_r - ud_r;
    float vb_t = v_t - vd_t;
    float vb_b = v_b - vd_b;

    float v_tt = get_hfacet(v,i-1,j,ny,nx);
    float vd_tt = get_hfacet(vd,i-1,j,ny,nx);
    float vb_tt = v_tt - vd_tt;

    float u_tl = get_vfacet(u,i-1,j,ny,nx);
    float ud_tl = get_vfacet(ud,i-1,j,ny,nx);
    float ub_tl = u_tl - ud_tl;
    
    float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
    float ud_tr = get_vfacet(ud,i-1,j+1,ny,nx);
    float ub_tr = u_tr - ud_tr;

    float H_t    = get_cell(H,i-1,j,ny,nx);
    float beta_t = get_cell(beta,i-1,j,ny,nx);
    float beta_c = get_cell(beta,i,j,ny,nx);
    float xi_t = get_cell(xi,i-1,j,ny,nx);
    float xi_c = get_cell(xi,i,j,ny,nx);

    TauByJacobian tau_by_t = get_tau_by_jac({vb_t,vb_tt,vb_b,ub_tl,ub_tr,ub_l,ub_r,H_t,H_c,xi_t,xi_c,beta_t,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
    r[6]  += tau_by_t.res;
    
    J[54] -= tau_by_t.d_u_bl;
    J[55] -= tau_by_t.d_u_br;
    J[56] -= tau_by_t.d_v_c;
    J[57] -= tau_by_t.d_v_b;
    
    J[58] += tau_by_t.d_u_bl;
    J[59] += tau_by_t.d_u_br;
    J[60] += tau_by_t.d_v_c;
    J[61] += tau_by_t.d_v_b;
    
    J[62] += tau_by_t.d_H_b;
    
    r[2]  -= tau_by_t.res;

    J[18] += tau_by_t.d_u_bl;
    J[19] += tau_by_t.d_u_br;
    J[20] += tau_by_t.d_v_c;
    J[21] += tau_by_t.d_v_b;
    
    J[22] -= tau_by_t.d_u_bl;
    J[23] -= tau_by_t.d_u_br;
    J[24] -= tau_by_t.d_v_c;
    J[25] -= tau_by_t.d_v_b;

    J[26] -= tau_by_t.d_H_b;
    }

    // Basal shear stress for bottom momentum
    {
    float ub_l = u_l - ud_l;
    float ub_r = u_r - ud_r;
    float vb_t = v_t - vd_t;
    float vb_b = v_b - vd_b;

    float v_bb = get_hfacet(v,i+2,j,ny,nx);
    float vd_bb = get_hfacet(vd,i+2,j,ny,nx);
    float vb_bb = v_bb - vd_bb;

    float u_bl = get_vfacet(u,i+1,j,ny,nx);
    float ud_bl = get_vfacet(ud,i+1,j,ny,nx);
    float ub_bl = u_bl - ud_bl;

    float u_br = get_vfacet(u,i+1,j+1,ny,nx);
    float ud_br = get_vfacet(ud,i+1,j+1,ny,nx);
    float ub_br = u_br - ud_br;

    float H_b    = get_cell(H,i+1,j,ny,nx);
    float beta_c = get_cell(beta,i,j,ny,nx);
    float beta_b = get_cell(beta,i+1,j,ny,nx);
    float xi_c = get_cell(xi,i,j,ny,nx);
    float xi_b = get_cell(xi,i+1,j,ny,nx);

    TauByJacobian tau_by_b = get_tau_by_jac({vb_b,vb_t,vb_bb,ub_l,ub_r,ub_bl,ub_br,H_c,H_b,xi_c,xi_b,beta_c,beta_b,m,u_reg,water_drag,flotation_reg_sliding});
    r[7]  += tau_by_b.res;

    J[63] -= tau_by_b.d_u_tl;
    J[64] -= tau_by_b.d_u_tr;
    J[65] -= tau_by_b.d_v_t;
    J[66] -= tau_by_b.d_v_c;

    J[67] += tau_by_b.d_u_tl;
    J[68] += tau_by_b.d_u_tr;
    J[69] += tau_by_b.d_v_t;
    J[70] += tau_by_b.d_v_c;
    J[71] += tau_by_b.d_H_t;
    
    r[3]  -= tau_by_b.res;

    J[27] += tau_by_b.d_u_tl;
    J[28] += tau_by_b.d_u_tr;
    J[29] += tau_by_b.d_v_t;
    J[30] += tau_by_b.d_v_c;

    J[31] -= tau_by_b.d_u_tl;
    J[32] -= tau_by_b.d_u_tr;
    J[33] -= tau_by_b.d_v_t;
    J[34] -= tau_by_b.d_v_c;

    J[35] -= tau_by_b.d_H_t;
    }
    
    // Driving stress for left momentum (u)
    {
    float H_l    = get_cell(H,i,j-1,ny,nx);
    float bed_l  = get_cell(bed,i,j-1,ny,nx);
    float bed_c  = get_cell(bed,i,j,ny,nx);
    TauDxJacobian tau_dx_l = get_tau_dx_jac({H_l,H_c,bed_l,bed_c,phi_l,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);
    r[4] -= tau_dx_l.res;
    J[44] -= tau_dx_l.d_H_r;
    }

    // Driving stress for right momentum (u)
    {
    float H_r    = get_cell(H,i,j+1,ny,nx);
    float bed_c  = get_cell(bed,i,j,ny,nx);
    float bed_r  = get_cell(bed,i,j+1,ny,nx);
    TauDxJacobian tau_dx_r = get_tau_dx_jac({H_c,H_r,bed_c,bed_r,phi_c,phi_r,flotation_reg_driving},dx_inv,i,j+1,ny,nx);
    r[5] -= tau_dx_r.res;
    J[53] -= tau_dx_r.d_H_l;
    }

    // Driving stress for top momentum (v)
    {
    float H_t    = get_cell(H,i-1,j,ny,nx);
    float bed_t  = get_cell(bed,i-1,j,ny,nx);
    float bed_c  = get_cell(bed,i,j,ny,nx);
    TauDyJacobian tau_dy_t = get_tau_dy_jac({H_t,H_c,bed_t,bed_c,phi_t,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);
    r[6]  -= tau_dy_t.res;
    J[62] -= tau_dy_t.d_H_b;
    }

    // Driving stress for bottom momentum (v)
    {
    float H_b    = get_cell(H,i+1,j,ny,nx);
    float bed_c  = get_cell(bed,i,j,ny,nx);
    float bed_b  = get_cell(bed,i+1,j,ny,nx);
    TauDyJacobian tau_dy_b = get_tau_dy_jac({H_c,H_b,bed_c,bed_b,phi_c,phi_b,flotation_reg_driving},dx_inv,i+1,j,ny,nx);
    r[7]  -= tau_dy_b.res;
    J[71] -= tau_dy_b.d_H_t;
    }
}

extern "C" __global__
void vanka_smooth(
    float* __restrict__ delta_u,
    float* __restrict__ delta_v,
    float* __restrict__ delta_ud,
    float* __restrict__ delta_vd,
    float* __restrict__ delta_H,
    float* __restrict__ mask,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ ud,
    const float* __restrict__ vd,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ xi,
    const float* __restrict__ f_u,
    const float* __restrict__ f_v,
    const float* __restrict__ f_ud,
    const float* __restrict__ f_vd,
    const float* __restrict__ f_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float n, float eps_reg, float H_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding,
    float calving_rate, float flotation_reg_calving,
    float dx, float dt,
    int ny, int nx, int stride, int halo,
    int newton_steps, float relaxation,
    float ssa_damping, float mc_damping
    )
{
    const int bny = 16;
    const int bnx = 16;

    int bi = threadIdx.y;
    int bj = threadIdx.x;

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    __shared__ float eta_local[bny][bnx];

    populate_viscosity(eta_local, bi, bj, i, j, u, v, ud, vd, H, B, n, eps_reg, H_reg, dx, ny, nx);
    __syncthreads();

    if (i < 0 || i >= ny || j<0 || j >= nx) return;

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active ) {
	float dx_inv = 1.0f/dx;

	float masked = get_cell(mask, i, j, ny, nx);
	float u_l = get_vfacet(u, i, j, ny, nx);
	float u_r = get_vfacet(u, i, j + 1, ny, nx);
	float v_t = get_hfacet(v, i, j, ny, nx);
	float v_b = get_hfacet(v, i + 1, j, ny, nx);

	float ud_l = get_vfacet(ud, i, j, ny, nx);
	float ud_r = get_vfacet(ud, i, j + 1, ny, nx);
	float vd_t = get_hfacet(vd, i, j, ny, nx);
	float vd_b = get_hfacet(vd, i + 1, j, ny, nx);

	float H_c = get_cell(H, i, j, ny, nx);
	float thklim = get_cell(gamma,i,j,ny,nx);

	float c_u_l = 0.0f;
	float c_u_r = 0.0f;
	float c_v_t = 0.0f;
	float c_v_b = 0.0f;

	float c_ud_l = 0.0f;
	float c_ud_r = 0.0f;
	float c_vd_t = 0.0f;
	float c_vd_b = 0.0f;

	float c_H_c = 0.0f;

	float rnorm = 1.0f;
	float rnorm0 = -1.0f;
	float tol = 0.000001f;
	int k = 0;

	float J[81] = {0};
        float r[9] = {0};

	while (k<newton_steps && rnorm>tol){
            
	    build_9x9_vanka(J, r,
		    u_l, u_r, v_t, v_b,
		    ud_l, ud_r, vd_t, vd_b, H_c,
		    u, v, ud, vd, H, eta_local, phi, xi,
                    bed, B, beta, gamma,
		    n, eps_reg, H_reg, flotation_reg_driving,
                    m, u_reg, water_drag, flotation_reg_sliding,
		    calving_rate, flotation_reg_calving,
                    dx, dt, ny, nx, i, j, bi, bj);
            

	    r[0] -= get_vfacet(f_ud,i,j,ny,nx);
	    r[1] -= get_vfacet(f_ud,i,j+1,ny,nx);
	    r[2] -= get_hfacet(f_vd,i,j,ny,nx);
	    r[3] -= get_hfacet(f_vd,i+1,j,ny,nx);
	    r[4] -= get_vfacet(f_u,i,j,ny,nx);
	    r[5] -= get_vfacet(f_u,i,j+1,ny,nx);
	    r[6] -= get_hfacet(f_v,i,j,ny,nx);
	    r[7] -= get_hfacet(f_v,i+1,j,ny,nx);
	    r[8] -= get_cell(f_H,i,j,ny,nx);

            // Velocity rows are sign-definite (elliptic), so damping is
            // relative (dimensionless, Levenberg-style): each diagonal is
            // stiffened by a fraction of itself, meaning the same thing in
            // every patch regardless of local scales. The transport row is
            // NOT sign-definite (its diagonal 1/dt + flux terms can vanish
            // or change sign at outflow margins for large dt), so it gets
            // additive pseudo-transient continuation in physical time
            // units: mc_damping = 1/dtau [1/a], dominant when dt is large
            // and negligible when dt is small.
            J[0]  *= (1.0f + ssa_damping);
            J[10] *= (1.0f + ssa_damping);
            J[20] *= (1.0f + ssa_damping);
            J[30] *= (1.0f + ssa_damping);
            J[40] *= (1.0f + ssa_damping);
            J[50] *= (1.0f + ssa_damping);
            J[60] *= (1.0f + ssa_damping);
            J[70] *= (1.0f + ssa_damping);
            J[80] += mc_damping;

	    if (j == 0) {
	    	for(int k=0; k<9; ++k) J[0*9 + k] = 0.0f;
	    	for(int k=0; k<9; ++k) J[k*9 + 0] = 0.0f;
	    	for(int k=0; k<9; ++k) J[4*9 + k] = 0.0f;
	    	for(int k=0; k<9; ++k) J[k*9 + 4] = 0.0f;
                J[0]  = 1.0f;
                r[0] = ud_l;

		J[40] = 1.0f;
		r[4] = u_l;
	    }

	    if (j == (nx - 1)) {
	    	for(int k=0; k<9; ++k) J[1*9 + k] = 0.0f;
	    	for(int k=0; k<9; ++k) J[k*9 + 1] = 0.0f;
	    	for(int k=0; k<9; ++k) J[5*9 + k] = 0.0f;
	    	for(int k=0; k<9; ++k) J[k*9 + 5] = 0.0f;
                J[10] = 1.0f;
                r[1] = ud_r;

		J[50] = 1.0f;
		r[5] = u_r;
	    }

	    if (i == 0) {
	    	for(int k=0; k<9; ++k) J[2*9 + k] = 0.0f;
	    	for(int k=0; k<9; ++k) J[k*9 + 2] = 0.0f;
	    	for(int k=0; k<9; ++k) J[6*9 + k] = 0.0f;
	    	for(int k=0; k<9; ++k) J[k*9 + 6] = 0.0f;
		J[20] = 1.0f;
		r[2] = vd_t;
		J[60] = 1.0f;
		r[6] = v_t;
	    }

	    if (i == (ny-1)) {
	    	for(int k=0; k<9; ++k) J[3*9 + k] = 0.0f;
	    	for(int k=0; k<9; ++k) J[k*9 + 3] = 0.0f;
	    	for(int k=0; k<9; ++k) J[7*9 + k] = 0.0f;
	    	for(int k=0; k<9; ++k) J[k*9 + 7] = 0.0f;
		J[30] = 1.0f;
		r[3] = vd_b;
		J[70] = 1.0f;
		r[7] = v_b;
	    }
	    

	    if ((H_c - dt*r[8]) <= (thklim)) {
		// Active set constraint: Force H = thklim
		masked = 1.0f;
		for(int k=0; k<9; ++k) J[8*9 + k] = 0.0f;
		for(int k=0; k<9; ++k) J[k*9 + 8] = 0.0f;
		J[80] = 1.0f;
		r[8] = H_c - thklim;
	    } else {
	        masked = 0.0f;
	    
	    }
	    

	    float delta_x[9] = {0};
	    float s_eq[9];
	    equilibrate<9>(J, r, s_eq);

	    // Dimensionless convergence measure: the equilibrated patch
	    // residual, relative to its value on the first Newton iteration
	    float rn = r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + r[3]*r[3] + r[4]*r[4] + r[5]*r[5] + r[6]*r[6] + r[7]*r[7] + r[8]*r[8];
	    if (rnorm0 < 0.0f) rnorm0 = fmaxf(rn, 1e-30f);
	    rnorm = rn / rnorm0;

            lu_factor<9>(J);
	    lu_solve_factored<9>(J,r,delta_x);
	    #pragma unroll
	    for (int a = 0; a < 9; ++a) delta_x[a] *= s_eq[a];
	    //relaxation = 0.5f;

	    float y_ud_l = -relaxation*delta_x[0] - c_ud_l;
	    float t_ud_l = ud_l + y_ud_l;
	    c_ud_l = (t_ud_l - ud_l) - y_ud_l;
	    ud_l = t_ud_l;
	    
	    float y_ud_r = -relaxation*delta_x[1] - c_ud_r;
	    float t_ud_r = ud_r + y_ud_r;
	    c_ud_r = (t_ud_r - ud_r) - y_ud_r;
	    ud_r = t_ud_r;

	    float y_vd_t = -relaxation*delta_x[2] - c_vd_t;
	    float t_vd_t = vd_t + y_vd_t;
	    c_vd_t = (t_vd_t - vd_t) - y_vd_t;
	    vd_t = t_vd_t;
	    
	    float y_vd_b = -relaxation*delta_x[3] - c_vd_b;
	    float t_vd_b = vd_b + y_vd_b;
	    c_vd_b = (t_vd_b - vd_b) - y_vd_b;
	    vd_b = t_vd_b;

	    float y_u_l = -relaxation*delta_x[4] - c_u_l;
	    float t_u_l = u_l + y_u_l;
	    c_u_l = (t_u_l - u_l) - y_u_l;
	    u_l = t_u_l;
	    
	    float y_u_r = -relaxation*delta_x[5] - c_u_r;
	    float t_u_r = u_r + y_u_r;
	    c_u_r = (t_u_r - u_r) - y_u_r;
	    u_r = t_u_r;

	    float y_v_t = -relaxation*delta_x[6] - c_v_t;
	    float t_v_t = v_t + y_v_t;
	    c_v_t = (t_v_t - v_t) - y_v_t;
	    v_t = t_v_t;
	    
	    float y_v_b = -relaxation*delta_x[7] - c_v_b;
	    float t_v_b = v_b + y_v_b;
	    c_v_b = (t_v_b - v_b) - y_v_b;
	    v_b = t_v_b;

	    float y_H_c = -relaxation*delta_x[8] - c_H_c;
	    float t_H_c = H_c + y_H_c;
	    c_H_c = (t_H_c - H_c) - y_H_c;
	    H_c = t_H_c;

	    H_c = fmaxf(H_c,thklim);
	    k++;
            
        }
	
	float ud_l_prev = get_vfacet(ud, i, j, ny, nx);
	float ud_r_prev = get_vfacet(ud, i, j + 1, ny, nx);
	float vd_t_prev = get_hfacet(vd, i, j, ny, nx);
	float vd_b_prev = get_hfacet(vd, i + 1, j, ny, nx);
	float u_l_prev = get_vfacet(u, i, j, ny, nx);
	float u_r_prev = get_vfacet(u, i, j + 1, ny, nx);
	float v_t_prev = get_hfacet(v, i, j, ny, nx);
	float v_b_prev = get_hfacet(v, i + 1, j, ny, nx);
	float H_c_prev = get_cell(H, i, j, ny, nx);

	atomicAdd(&delta_ud[i * (nx + 1) + j],       0.5f*(ud_l - ud_l_prev));
	atomicAdd(&delta_ud[i * (nx + 1) + j + 1],   0.5f*(ud_r - ud_r_prev));
	atomicAdd(&delta_vd[i * nx + j],             0.5f*(vd_t - vd_t_prev));
	atomicAdd(&delta_vd[(i + 1) * nx + j ],      0.5f*(vd_b - vd_b_prev));
	atomicAdd(&delta_u[i * (nx + 1) + j],       0.5f*(u_l - u_l_prev));
	atomicAdd(&delta_u[i * (nx + 1) + j + 1],   0.5f*(u_r - u_r_prev));
	atomicAdd(&delta_v[i * nx + j],             0.5f*(v_t - v_t_prev));
	atomicAdd(&delta_v[(i + 1) * nx + j ],      0.5f*(v_b - v_b_prev));
	delta_H[i * nx + j]           = (H_c - H_c_prev);
	mask[i * nx + j]              = masked;
    }
}

extern "C" __global__
void vanka_smooth_adjoint(
    float* __restrict__ lambda_u_out,
    float* __restrict__ lambda_v_out,
    float* __restrict__ lambda_ud_out,
    float* __restrict__ lambda_vd_out,
    float* __restrict__ lambda_H_out,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ ud,
    const float* __restrict__ vd,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ xi,
    const float* __restrict__ mask,
    const float* __restrict__ r_adj_u,
    const float* __restrict__ r_adj_v,
    const float* __restrict__ r_adj_ud,
    const float* __restrict__ r_adj_vd,
    const float* __restrict__ r_adj_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float n, float eps_reg, float H_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding,
    float calving_rate, float flotation_reg_calving,
    float dx, float dt,
    int ny, int nx, int stride, int halo,
    float ssa_damping, float mc_damping
    )
{
    const int bny = 16;
    const int bnx = 16;

    int bi = threadIdx.y;
    int bj = threadIdx.x;

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    __shared__ float eta_local[bny][bnx];

    populate_viscosity(eta_local, bi, bj, i, j, u, v, ud, vd, H, B, n, eps_reg, H_reg, dx, ny, nx);

    __syncthreads();

    if (i < 0 || i >= ny || j<0 || j >= nx) return;

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active ) {
	float dx_inv = 1.0f/dx;

	float masked = get_cell(mask, i, j, ny, nx);

	float u_l = get_vfacet(u, i, j, ny, nx);
	float u_r = get_vfacet(u, i, j + 1, ny, nx);
	float v_t = get_hfacet(v, i, j, ny, nx);
	float v_b = get_hfacet(v, i + 1, j, ny, nx);
	float ud_l = get_vfacet(ud, i, j, ny, nx);
	float ud_r = get_vfacet(ud, i, j + 1, ny, nx);
	float vd_t = get_hfacet(vd, i, j, ny, nx);
	float vd_b = get_hfacet(vd, i + 1, j, ny, nx);
	float H_c = get_cell(H, i, j, ny, nx);

	float J[81] = {0};
	float rhs[9] = {0};
	// Note that the adjoint assembles a forward problem rhs, but it's
	// discarded.
	build_9x9_vanka(J, rhs,
		u_l, u_r, v_t, v_b,
		ud_l, ud_r, vd_t, vd_b, H_c,
		u, v, ud, vd, H, eta_local, phi, xi,
		bed, B, beta, gamma,
		n, eps_reg, H_reg, flotation_reg_driving,
		m, u_reg, water_drag, flotation_reg_sliding,
		calving_rate, flotation_reg_calving,
		dx, dt, ny, nx, i, j, bi, bj);

	// Damping matches the forward smoother: relative on the
	// sign-definite velocity rows, additive physical-time PTC on the
	// transport row (see vanka_smooth)
	J[0]  *= (1.0f + ssa_damping);
        J[10] *= (1.0f + ssa_damping);
        J[20] *= (1.0f + ssa_damping);
        J[30] *= (1.0f + ssa_damping);
        J[40] *= (1.0f + ssa_damping);
        J[50] *= (1.0f + ssa_damping);
        J[60] *= (1.0f + ssa_damping);
        J[70] *= (1.0f + ssa_damping);
        J[80] += mc_damping;

        rhs[0] = get_vfacet(r_adj_ud, i, j, ny, nx);
        rhs[1] = get_vfacet(r_adj_ud, i, j+1, ny, nx);
        rhs[2] = get_hfacet(r_adj_vd, i, j, ny, nx);
        rhs[3] = get_hfacet(r_adj_vd, i+1, j, ny, nx);
        rhs[4] = get_vfacet(r_adj_u, i, j, ny, nx);
        rhs[5] = get_vfacet(r_adj_u, i, j+1, ny, nx);
        rhs[6] = get_hfacet(r_adj_v, i, j, ny, nx);
        rhs[7] = get_hfacet(r_adj_v, i+1, j, ny, nx);
        rhs[8] = get_cell(r_adj_H, i, j, ny, nx);

	if (j == 0) {
	    for(int k=0; k<9; ++k) J[0*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 0] = 0.0f;
	    for(int k=0; k<9; ++k) J[4*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 4] = 0.0f;
	    J[0]  = 1.0f;
	    J[40] = 1.0f;
	}

	if (j == (nx - 1)) {
	    for(int k=0; k<9; ++k) J[1*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 1] = 0.0f;
	    for(int k=0; k<9; ++k) J[5*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 5] = 0.0f;
	    J[10] = 1.0f;
	    J[50] = 1.0f;
	}

	if (i == 0) {
	    for(int k=0; k<9; ++k) J[2*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 2] = 0.0f;
	    for(int k=0; k<9; ++k) J[6*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 6] = 0.0f;
	    J[20] = 1.0f;
	    J[60] = 1.0f;
	}

	if (i == (ny-1)) {
	    for(int k=0; k<9; ++k) J[3*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 3] = 0.0f;
	    for(int k=0; k<9; ++k) J[7*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 7] = 0.0f;
	    J[30] = 1.0f;
	    J[70] = 1.0f;
	}

	if (masked > 0.5) {
	    // Active set constraint: Force H = thklim
	    for(int k=0; k<9; ++k) J[8*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 8] = 0.0f;
	    J[80] = 1.0f;
	}

        // Equilibrate before transposing: (S J S)^T = S J^T S, so the
        // adjoint solve uses the same scale factors as the forward patch
        float s_eq[9];
        equilibrate<9>(J, rhs, s_eq);

        float J_T[81];
        #pragma unroll
        for(int r=0; r<9; ++r) {
            #pragma unroll
            for(int c=0; c<9; ++c) {
                J_T[r*9 + c] = J[c*9 + r];
            }
        }

	float delta_lambda[9] = {0};
	lu_factor<9>(J_T);
	lu_solve_factored<9>(J_T,rhs,delta_lambda);
	#pragma unroll
	for (int a = 0; a < 9; ++a) delta_lambda[a] *= s_eq[a];

	atomicAdd(&lambda_ud_out[i * (nx + 1) + j],      0.5f*delta_lambda[0]);
	atomicAdd(&lambda_ud_out[i * (nx + 1) + j + 1],  0.5f*delta_lambda[1]);
	atomicAdd(&lambda_vd_out[i * nx + j],            0.5f*delta_lambda[2]);
	atomicAdd(&lambda_vd_out[(i + 1) * nx + j ],     0.5f*delta_lambda[3]);
	atomicAdd(&lambda_u_out[i * (nx + 1) + j],      0.5f*delta_lambda[4]);
	atomicAdd(&lambda_u_out[i * (nx + 1) + j + 1],  0.5f*delta_lambda[5]);
	atomicAdd(&lambda_v_out[i * nx + j],            0.5f*delta_lambda[6]);
	atomicAdd(&lambda_v_out[(i + 1) * nx + j ],     0.5f*delta_lambda[7]);
	atomicAdd(&lambda_H_out[i * nx + j],                 delta_lambda[8]);
    }
}


extern "C" __global__
void vanka_dump(
    float* __restrict__ J_array,
    float* __restrict__ r_array,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ ud,
    const float* __restrict__ vd,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ xi,
    const float* __restrict__ f_u,
    const float* __restrict__ f_v,
    const float* __restrict__ f_ud,
    const float* __restrict__ f_vd,
    const float* __restrict__ f_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float n, float eps_reg, float H_reg, float flotation_reg_driving,
    float m, float u_reg, float water_drag, float flotation_reg_sliding,
    float calving_rate, float flotation_reg_calving,
    float dx, float dt,
    int ny, int nx, int stride, int halo)
{
    const int bny = 16;
    const int bnx = 16;

    int bi = threadIdx.y;
    int bj = threadIdx.x;

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    __shared__ float eta_local[bny][bnx];

    populate_viscosity(eta_local, bi, bj, i, j, u, v, ud, vd, H, B, n, eps_reg, H_reg, dx, ny, nx);
    __syncthreads();

    if (i < 0 || i >= ny || j<0 || j >= nx) return;

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active ) {
	float dx_inv = 1.0f/dx;

	float ud_l = get_vfacet(ud, i, j, ny, nx);
	float ud_r = get_vfacet(ud, i, j + 1, ny, nx);
	float vd_t = get_hfacet(vd, i, j, ny, nx);
	float vd_b = get_hfacet(vd, i + 1, j, ny, nx);

	float u_l = get_vfacet(u, i, j, ny, nx);
	float u_r = get_vfacet(u, i, j + 1, ny, nx);
	float v_t = get_hfacet(v, i, j, ny, nx);
	float v_b = get_hfacet(v, i + 1, j, ny, nx);
	float H_c = get_cell(H, i, j, ny, nx);
	float thklim = get_cell(gamma,i,j,ny,nx);

	float J[81] = {0};
        float r[9] = {0};

        build_9x9_vanka(J, r,
	    u_l, u_r, v_t, v_b,
	    ud_l, ud_r, vd_t, vd_b, H_c,
	    u, v, ud, vd, H, eta_local, phi, xi,
	    bed, B, beta, gamma,
	    n, eps_reg, H_reg, flotation_reg_driving,
	    m, u_reg, water_drag, flotation_reg_sliding,
	    calving_rate, flotation_reg_calving,
	    dx, dt, ny, nx, i, j, bi, bj);
	
	r[0] -= get_vfacet(f_ud,i,j,ny,nx);
	r[1] -= get_vfacet(f_ud,i,j+1,ny,nx);
	r[2] -= get_hfacet(f_vd,i,j,ny,nx);
	r[3] -= get_hfacet(f_vd,i+1,j,ny,nx);
	r[4] -= get_vfacet(f_u,i,j,ny,nx);
	r[5] -= get_vfacet(f_u,i,j+1,ny,nx);
	r[6] -= get_hfacet(f_v,i,j,ny,nx);
	r[7] -= get_hfacet(f_v,i+1,j,ny,nx);
	r[8] -= get_cell(f_H,i,j,ny,nx);

	if (j == 0) {
	    for(int k=0; k<9; ++k) J[0*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 0] = 0.0f;
	    for(int k=0; k<9; ++k) J[4*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 4] = 0.0f;
	    J[0]  = 1.0f;
	    r[0] = ud_l;

	    J[40] = 1.0f;
	    r[4] = u_l;
	}

	if (j == (nx - 1)) {
	    for(int k=0; k<9; ++k) J[1*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 1] = 0.0f;
	    for(int k=0; k<9; ++k) J[5*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 5] = 0.0f;
	    J[10] = 1.0f;
	    r[1] = ud_r;

	    J[50] = 1.0f;
	    r[5] = u_r;
	}

	if (i == 0) {
	    for(int k=0; k<9; ++k) J[2*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 2] = 0.0f;
	    for(int k=0; k<9; ++k) J[6*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 6] = 0.0f;
	    J[20] = 1.0f;
	    r[2] = vd_t;
	    J[60] = 1.0f;
	    r[6] = v_t;
	}

	if (i == (ny-1)) {
	    for(int k=0; k<9; ++k) J[3*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 3] = 0.0f;
	    for(int k=0; k<9; ++k) J[7*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 7] = 0.0f;
	    J[30] = 1.0f;
	    r[3] = vd_b;
	    J[70] = 1.0f;
	    r[7] = v_b;
	}
	

	if ((H_c - dt*r[8]) <= (thklim)) {
	    // Active set constraint: Force H = thklim
	    for(int k=0; k<9; ++k) J[8*9 + k] = 0.0f;
	    for(int k=0; k<9; ++k) J[k*9 + 8] = 0.0f;
	    J[80] = 1.0f;
	    r[8] = H_c - thklim;
	}

        for(int k=0; k<81; ++k) J_array[81*(i * nx + j) + k] = J[k]; 
        for(int k=0; k<9; ++k)  r_array[9*(i * nx + j) + k]  = r[k]; 
    }
}

