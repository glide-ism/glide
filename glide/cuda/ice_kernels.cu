// =====================================================================
// SAFE ARRAY ACCESS HELPERS
// =====================================================================
struct DualFloat {
    float v; // Primal value
    float d; // Derivative/Perturbation component

    // Addition: (u + v, du + dv)
    __device__ __forceinline__ friend DualFloat operator+(DualFloat a, DualFloat b) {
        return {a.v + b.v, a.d + b.d};
    }

    // Subtraction: (u - v, du - dv)
    __device__ __forceinline__ friend DualFloat operator-(DualFloat a, DualFloat b) {
        return {a.v - b.v, a.d - b.d};
    }

    // Multiplication: (u * v, u * dv + v * du)
    __device__ __forceinline__ friend DualFloat operator*(DualFloat a, DualFloat b) {
        return {a.v * b.v, __fmaf_rn(a.v, b.d, a.d * b.v)};
    }

    // Multiplication by Scalar: (u * s, du * s)
    __device__ __forceinline__ friend DualFloat operator*(DualFloat a, float s) {
        return {a.v * s, a.d * s};
    }

    __device__ __forceinline__ friend DualFloat operator*(float s, DualFloat a) {
        return {a.v * s, a.d * s};
    }
    
    // Addition with Scalar: (u + s, du)
    __device__ __forceinline__ friend DualFloat operator+(DualFloat a, float s) {
	return {a.v + s, a.d};
    }

    // Commutative version: (s + u, du)
    __device__ __forceinline__ friend DualFloat operator+(float s, DualFloat a) {
	return {s + a.v, a.d};
    }

    // Subtraction with Scalar
    __device__ __forceinline__ friend DualFloat operator-(DualFloat a, float s) {
	return {a.v - s, a.d};
    }

    // Division by Scalar: (u / s, du / s)
    __device__ __forceinline__ friend DualFloat operator/(DualFloat a, float s) {
	float inv_s = 1.0f / s; // Compiler will likely use RCP
	return {a.v * inv_s, a.d * inv_s};
    }

};


__device__ __forceinline__ DualFloat __powf(DualFloat u, float p) {
    // High-performance hardware intrinsic pow
    float val = __powf(u.v, p);
    
    // d/dx(u^p) = p * u^(p-1) * du
    // If u.v is zero, derivative is technically singular; eps_reg handles this.
    float deriv = p * __powf(u.v, p - 1.0f) * u.d;
    
    return {val, deriv};
}

__device__ __forceinline__ float sigmoid(const float z, const float c) {
   float scaled_z = fminf(fmaxf(c*z,-10.0f),10.0f);
   return 1.0f/(1.0f + __expf(-scaled_z));
}

__device__ __forceinline__ float get_vfacet(const float* __restrict__ u, int i, int j, int ny, int nx) {
    //if (i < 0 || i >= ny || j < 0 || j > nx) return 0.0f;
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx),0);
    return u[i * (nx + 1) + j];
}

__device__ __forceinline__ DualFloat get_vfacet(const float* __restrict__ u, const float* __restrict__ du, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx),0);
    int idx = i * (nx + 1) + j;
    return {u[idx],du[idx]};
}

__device__ __forceinline__ float get_hfacet(const float* __restrict__ v, int i, int j, int ny, int nx) {
    //if (i < 0 || i > ny || j < 0 || j >= nx) return 0.0f;
    i = max(min(i,ny),0);
    j = max(min(j,nx - 1),0);
    return v[i * nx + j];
}

__device__ __forceinline__ DualFloat get_hfacet(const float* __restrict__ v, const float* __restrict__ dv, int i, int j, int ny, int nx) {
    i = max(min(i,ny),0);
    j = max(min(j,nx - 1),0);
    int idx = i * nx + j;
    return {v[idx],dv[idx]};
}

__device__ __forceinline__ float get_cell(const float* __restrict__ arr, int i, int j, int ny, int nx) {
    //if (i < 0 || i >= ny || j < 0 || j >= nx) return 0.0f;
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx - 1),0);
    return arr[i * nx + j];
}

__device__ __forceinline__ DualFloat get_cell(const float* __restrict__ arr, const float* __restrict__ darr, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx - 1),0);
    int idx = i * nx + j;
    return {arr[idx],darr[idx]};
}

__device__ __forceinline__ float get_masked_cell(const float* __restrict__ arr, const float* __restrict__ mask, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx - 1),0);
    int idx = i * nx + j;
    return arr[idx]*(1.0f - mask[idx]);
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


// =====================================================================
// LU SOLVE (5x5) WITH PIVOTING
// =====================================================================

__device__ void lu_5x5_solve_pivot(float A[25], const float b[5], float x[5])
{
    float rhs[5] = {b[0], b[1], b[2], b[3], b[4]};
    int piv[5] = {0, 1, 2, 3, 4};
    
    for (int k = 0; k < 4; k++) {
        float max_val = fabsf(A[piv[k] * 5 + k]);
        int max_idx = k;
        for (int ii = k + 1; ii < 5; ii++) {
            float val = fabsf(A[piv[ii] * 5 + k]);
            if (val > max_val) {
                max_val = val;
                max_idx = ii;
            }
        }
        
        int tmp = piv[k];
        piv[k] = piv[max_idx];
        piv[max_idx] = tmp;
        
        float pivot = A[piv[k] * 5 + k];
        if (fabsf(pivot) < 1e-14f) continue;
        
        for (int ii = k + 1; ii < 5; ii++) {
            float factor = A[piv[ii] * 5 + k] / pivot;
            A[piv[ii] * 5 + k] = factor;
            for (int jj = k + 1; jj < 5; jj++) {
                A[piv[ii] * 5 + jj] -= factor * A[piv[k] * 5 + jj];
            }
        }
    }
    
    float y[5];
    for (int ii = 0; ii < 5; ii++) {
        y[ii] = rhs[piv[ii]];
        for (int jj = 0; jj < ii; jj++) {
            y[ii] -= A[piv[ii] * 5 + jj] * y[jj];
        }
    }
    
    for (int ii = 4; ii >= 0; ii--) {
        x[ii] = y[ii];
        for (int jj = ii + 1; jj < 5; jj++) {
            x[ii] -= A[piv[ii] * 5 + jj] * x[jj];
        }
        float diag = A[piv[ii] * 5 + ii];
        if (fabsf(diag) > 1e-14f) {
            x[ii] /= diag;
        } else {
            x[ii] = 0.0f;
        }
    }
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

/*=======================================================
  ================== Normal Stress ======================
 ========================================================*/ 
// Stencil items that require differentiation
struct SigmaNormalStencil {
    float u_l, u_r, v_t, v_b;
    float eta_H;
};

struct SigmaNormalStencilDual {
    DualFloat u_l, u_r, v_t, v_b;
    DualFloat eta_H;

    __device__ __forceinline__
    SigmaNormalStencil get_primals() const {
        return {u_l.v,u_r.v,v_t.v,v_b.v,eta_H.v};
    }

    __device__ __forceinline__
    SigmaNormalStencil get_diffs() const {
        return {u_l.d,u_r.d,v_t.d,v_b.d,eta_H.d};
    }
};

// Return type for sigma_xx, 
// containing residual and jacobian row
struct SigmaNormalJacobian {
    float res;
    float d_u_l, d_u_r, d_v_t, d_v_b;
    float d_eta_H;

    __device__ __forceinline__
    float apply_jvp(const SigmaNormalStencil& dot) const {
        return d_u_l * dot.u_l +
	       d_u_r * dot.u_r +
	       d_v_t * dot.v_t + 
	       d_v_b * dot.v_b +
	       d_eta_H * dot.eta_H;
    }

};

__device__ __forceinline__
SigmaNormalJacobian get_sigma_xx_jac(
    SigmaNormalStencil s,
    float dx_inv,
    int i, int j,  // Defined on cells - the i,j for the cell
    int ny, int nx) {

    SigmaNormalJacobian jac= {0};

    if (j < 0 || j >= nx) {
	return jac;
    } 

    float eps_xx = (2.0f*(s.u_r - s.u_l)*dx_inv + (s.v_t - s.v_b)*dx_inv); 
    float jac_prefactor = 2.0f * s.eta_H * dx_inv;

    jac.res = 2.0f * s.eta_H * eps_xx;
    jac.d_u_l = -2.0f * jac_prefactor;
    jac.d_u_r =  2.0f * jac_prefactor;
    jac.d_v_t =  jac_prefactor;
    jac.d_v_b = -jac_prefactor;
    jac.d_eta_H = 2.0f * eps_xx; 
    return jac;
}

__device__ __forceinline__
DualFloat get_sigma_xx_dual(
    SigmaNormalStencilDual s, 
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    SigmaNormalJacobian jac = get_sigma_xx_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

__device__ __forceinline__
SigmaNormalJacobian get_sigma_yy_jac(
    SigmaNormalStencil s,
    float dx_inv,
    int i, int j,  // Defined on cells - the i,j for the cell
    int ny, int nx) {

    SigmaNormalJacobian jac= {0};

    // No normal stress on out-of-domain cells
    if (i < 0 || i >= ny) {
	return jac;
    } 

    float eps_yy = ((s.u_r - s.u_l)*dx_inv + 2.0f*(s.v_t - s.v_b)*dx_inv); 
    float jac_prefactor = 2.0f * s.eta_H * dx_inv;

    jac.res = 2.0f * s.eta_H * eps_yy;
    jac.d_u_l = -jac_prefactor;
    jac.d_u_r =  jac_prefactor;
    jac.d_v_t =  2.0f*jac_prefactor;
    jac.d_v_b = -2.0f*jac_prefactor;
    jac.d_eta_H = 2.0f * eps_yy; 
    return jac;
}

__device__ __forceinline__
DualFloat get_sigma_yy_dual(
    SigmaNormalStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    SigmaNormalJacobian jac = get_sigma_yy_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*======================================================
  ==================== Shear Stress ====================
  ======================================================*/

// Stencil items that require differentiation
struct SigmaShearStencil {
    float u_t, u_b, v_l, v_r;
    float eta_H;
};

struct SigmaShearStencilDual {
    DualFloat u_t, u_b, v_l, v_r;
    DualFloat eta_H;

    __device__ __forceinline__
    SigmaShearStencil get_primals() const {
        return {u_t.v,u_b.v,v_l.v,v_r.v,eta_H.v};
    }

    __device__ __forceinline__
    SigmaShearStencil get_diffs() const {
        return {u_t.d,u_b.d,v_l.d,v_r.d,eta_H.d};
    }

};

// Return type for sigma_xx, 
// containing residual and jacobian row
struct SigmaShearJacobian {
    float res;
    float d_u_t, d_u_b, d_v_l, d_v_r;
    float d_eta_H;

    __device__ __forceinline__
    float apply_jvp(const SigmaShearStencil& dot) const {
        return d_u_t * dot.u_t +
	       d_u_b * dot.u_b +
	       d_v_l * dot.v_l + 
	       d_v_r * dot.v_r +
	       d_eta_H * dot.eta_H;
    }

};


__device__ __forceinline__
SigmaShearJacobian get_sigma_xy_jac(
    SigmaShearStencil s,
    float dx_inv,
    int i, int j, // defined on vertices, the i,j for the vertex
    int ny, int nx) {

    SigmaShearJacobian jac = {0};
    // No shear on boundary vertices
    if (i <= 0 || i >= ny || j <= 0 || j >= nx) {
        return jac;
    }
    
    float eps_xy = 0.5f*((s.u_t - s.u_b)*dx_inv + (s.v_r - s.v_l)*dx_inv);
    float jac_prefactor = s.eta_H * dx_inv;

    jac.res = 2.0f * s.eta_H * eps_xy;
    jac.d_u_t = jac_prefactor;
    jac.d_u_b = -jac_prefactor;
    jac.d_v_l = -jac_prefactor;
    jac.d_v_r = jac_prefactor;
    jac.d_eta_H = 2.0f * eps_xy;
    return jac;

}

__device__ __forceinline__
DualFloat get_sigma_xy_dual(
    SigmaShearStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    SigmaShearJacobian jac = get_sigma_xy_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*=========================================================
  ================== Basal Shear Stress ===================
  =========================================================*/

struct TauBxStencil {
    float u;
    float H_l, H_r;
    float bed_l, bed_r;
    float beta_l, beta_r;
    float water_drag;
};

struct TauBxStencilDual {
    DualFloat u;
    DualFloat H_l, H_r;
    float bed_l, bed_r;
    float beta_l, beta_r;
    float water_drag;

    __device__ __forceinline__
    TauBxStencil get_primals() const {
        return {u.v,H_l.v,H_r.v,bed_l,bed_r,beta_l,beta_r,water_drag};
    }
    
    __device__ __forceinline__
    TauBxStencil get_diffs() const {
        return {u.d,H_l.d,H_r.d,0.0f,0.0f,0.0f,0.0f,0.0f};
    }

};

struct TauBxJacobian {
    float res;
    float d_u;
    float d_H_l, d_H_r;
    float d_beta_l, d_beta_r;

    __device__ __forceinline__
    float apply_jvp(const TauBxStencil& dot) const {
        return d_u * dot.u +
	       d_H_l * dot.H_l +
	       d_H_r * dot.H_r; 
    }
};

__device__ __forceinline__
TauBxJacobian get_tau_bx_jac(
   TauBxStencil s )
{
    TauBxJacobian jac = {0};

    float grounded_l = sigmoid(s.bed_l + 0.917f*s.H_l,1.0f);
    float grounded_r = sigmoid(s.bed_r + 0.917f*s.H_r,1.0f);

    float beta_eff_l = grounded_l*s.beta_l + (1.0f - grounded_l)*s.water_drag;
    float beta_eff_r = grounded_r*s.beta_r + (1.0f - grounded_r)*s.water_drag;

    float beta_eff = 0.5f*(beta_eff_l + beta_eff_r);

    jac.res = -beta_eff * s.u;
    jac.d_u = -beta_eff;
    jac.d_beta_l = -0.5f*grounded_l*s.u;
    jac.d_beta_r = -0.5f*grounded_r*s.u;

    return jac;
}

__device__ __forceinline__
DualFloat get_tau_bx_dual(TauBxStencilDual s) {
    TauBxJacobian jac = get_tau_bx_jac(s.get_primals());
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

struct TauByStencil {
    float v;
    float H_t, H_b;
    float bed_t, bed_b;
    float beta_t, beta_b;
    float water_drag;
};

struct TauByStencilDual {
    DualFloat v;
    DualFloat H_t, H_b;
    float bed_t, bed_b;
    float beta_t, beta_b;
    float water_drag;

    __device__ __forceinline__
    TauByStencil get_primals() const {
        return {v.v,H_t.v,H_b.v,bed_t,bed_b,beta_t,beta_b,water_drag};
    }
    
    __device__ __forceinline__
    TauByStencil get_diffs() const {
        return {v.d,H_t.d,H_b.d,0.0f,0.0f,0.0f,0.0f,0.0f};
    }

};


struct TauByJacobian {
    float res;
    float d_v;
    float d_H_t, d_H_b;
    float d_beta_t, d_beta_b;

    __device__ __forceinline__
    float apply_jvp(const TauByStencil& dot) const {
        return d_v * dot.v +
	       d_H_t * dot.H_t +
	       d_H_b * dot.H_b; 
    }
};

__device__ __forceinline__
TauByJacobian get_tau_by_jac(
   TauByStencil s) {
    TauByJacobian jac = {0};

    float grounded_t = sigmoid(s.bed_t + 0.917f*s.H_t,1.0f);
    float grounded_b = sigmoid(s.bed_b + 0.917f*s.H_b,1.0f);

    float beta_eff_t = grounded_t*s.beta_t + (1.0f - grounded_t)*s.water_drag;
    float beta_eff_b = grounded_b*s.beta_b + (1.0f - grounded_b)*s.water_drag;

    float beta_eff = 0.5f*(beta_eff_t + beta_eff_b);

    jac.res = -beta_eff * s.v;
    jac.d_v = -beta_eff;
    jac.d_beta_t = -0.5f*grounded_t*s.v;
    jac.d_beta_b = -0.5f*grounded_b*s.v;

    return jac;
}

__device__ __forceinline__
DualFloat get_tau_by_dual(TauByStencilDual s) {
    TauByJacobian jac = get_tau_by_jac(s.get_primals());
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*=========================================================
  ==================== Driving Stress =====================
  =========================================================*/

struct TauDxStencil {
    float H_l, H_r;
    float bed_l, bed_r;
};

struct TauDxStencilDual {
    DualFloat H_l, H_r;
    float bed_l, bed_r;

    __device__ __forceinline__
    TauDxStencil get_primals() const {
        return {H_l.v,H_r.v,bed_l,bed_r};
    }
    
    __device__ __forceinline__
    TauDxStencil get_diffs() const {
        return {H_l.d,H_r.d,0.0f,0.0f};
    }

};

struct TauDxJacobian {
    float res;
    float d_H_l, d_H_r;

    __device__ __forceinline__
    float apply_jvp(const TauDxStencil& dot) const {
        return d_H_l * dot.H_l +
	       d_H_r * dot.H_r; 
    }

};  

__device__ __forceinline__
TauDxJacobian get_tau_dx_jac(
    TauDxStencil s,
    float dx_inv,
    int i, int j,  // Defined on facets
    int ny, int nx) {
    
    TauDxJacobian jac = {0};

    // No driving stress on boundaries
    if (j <= 0 || j >= nx) {
        return jac;
    }

    float H_avg = 0.5f*(s.H_l + s.H_r);

    float base_l = fmaxf(s.bed_l,-0.917f*s.H_l);
    float base_r = fmaxf(s.bed_r,-0.917f*s.H_r);

    float dbase_dH_l = s.bed_l > -0.917f*s.H_l ? 0.0f : -0.917f;
    float dbase_dH_r = s.bed_r > -0.917f*s.H_r ? 0.0f : -0.917f;

    float S_l = base_l + s.H_l;
    float S_r = base_r + s.H_r;

    jac.res = H_avg * (S_r - S_l) * dx_inv;

    jac.d_H_l = 0.5f*(S_r - S_l)*dx_inv - H_avg*(1.0f + dbase_dH_l)*dx_inv;
    jac.d_H_r = 0.5f*(S_r - S_l)*dx_inv + H_avg*(1.0f + dbase_dH_r)*dx_inv;
    //jac.d_H_l = 0.0f*(S_r - S_l)*dx_inv - H_avg*(1.0f + dbase_dH_l)*dx_inv;
    //jac.d_H_r = 0.0f*(S_r - S_l)*dx_inv + H_avg*(1.0f + dbase_dH_r)*dx_inv;
    return jac; 
}

__device__ __forceinline__
DualFloat get_tau_dx_dual(
    TauDxStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    TauDxJacobian jac = get_tau_dx_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}


struct TauDyStencil {
    float H_t, H_b;
    float bed_t, bed_b;
};

struct TauDyStencilDual {
    DualFloat H_t, H_b;
    float bed_t, bed_b;

    __device__ __forceinline__
    TauDyStencil get_primals() const {
        return {H_t.v,H_b.v,bed_t,bed_b};
    }
    
    __device__ __forceinline__
    TauDyStencil get_diffs() const {
        return {H_t.d,H_b.d,0.0f,0.0f};
    }

};

struct TauDyJacobian {
    float res;
    float d_H_t, d_H_b;

    __device__ __forceinline__
    float apply_jvp(const TauDyStencil& dot) const {
        return d_H_t * dot.H_t +
	       d_H_b * dot.H_b; 
    }
};  

__device__ __forceinline__
TauDyJacobian get_tau_dy_jac(
    TauDyStencil s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    
    TauDyJacobian jac = {0};
    if (i <= 0 || i >= ny) {
        return jac;
    }

    float H_avg = 0.5f*(s.H_t + s.H_b);

    float base_t = fmaxf(s.bed_t,-0.917f*s.H_t);
    float base_b = fmaxf(s.bed_b,-0.917f*s.H_b);

    float dbase_dH_t = s.bed_t > -0.917f*s.H_t ? 0.0f : -0.917f;
    float dbase_dH_b = s.bed_b > -0.917f*s.H_b ? 0.0f : -0.917f;

    float S_t = base_t + s.H_t;
    float S_b = base_b + s.H_b;

    jac.res = H_avg * (S_t - S_b) * dx_inv;

    jac.d_H_t = 0.5f*(S_t - S_b)*dx_inv + H_avg*(1.0f + dbase_dH_t)*dx_inv;
    jac.d_H_b = 0.5f*(S_t - S_b)*dx_inv - H_avg*(1.0f + dbase_dH_b)*dx_inv;
    //jac.d_H_t = 0.0f*(S_t - S_b)*dx_inv + H_avg*(1.0f + dbase_dH_t)*dx_inv;
    //jac.d_H_b = 0.0f*(S_t - S_b)*dx_inv - H_avg*(1.0f + dbase_dH_b)*dx_inv;
    return jac; 

}

__device__ __forceinline__
DualFloat get_tau_dy_dual(
    TauDyStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    TauDyJacobian jac = get_tau_dy_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*=========================================================
  ====================== Mass Flux ========================
  =========================================================*/

struct HorizontalFluxStencil {
    float u;
    float H_l, H_r;
};

struct HorizontalFluxStencilDual {
    DualFloat u;
    DualFloat H_l, H_r;

    __device__ __forceinline__
    HorizontalFluxStencil get_primals() const {
        return {u.v,H_l.v,H_r.v};
    }

    __device__ __forceinline__
    HorizontalFluxStencil get_diffs() const {
        return {u.d,H_l.d,H_r.d};
    }
};

struct HorizontalFluxJacobian {
    float res;
    float d_u;
    float d_H_l, d_H_r;

    __device__ __forceinline__
    float apply_jvp(const HorizontalFluxStencil& dot) const {
        return d_u * dot.u +
	       d_H_l * dot.H_l +
	       d_H_r * dot.H_r; 
    }

};  

__device__
HorizontalFluxJacobian get_horizontal_flux_jac(
    HorizontalFluxStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {
 
    HorizontalFluxJacobian jac = {0};

    // No flux on boundaries
    if (j <= 0 || j >= nx) {
	return jac;
    }
    
    float H_avg = 0.5f*(s.H_l + s.H_r);
    //float u_mag = sqrtf(s.u * s.u + 10.0f);//fabsf(s.u);
    //float u_sign = s.u / u_mag;//copysignf(1.0f, s.u);
    float u_mag = fabsf(s.u);
    float u_sign = copysignf(1.0f, s.u);
    jac.res = H_avg*s.u - 0.5f*u_mag*(s.H_r - s.H_l);

    jac.d_H_l = 0.5f*(s.u + u_mag);
    jac.d_H_r = 0.5f*(s.u - u_mag);
    jac.d_u   = H_avg - 0.5f*u_sign*(s.H_r - s.H_l); 
    return jac;
}

__device__ __forceinline__
DualFloat get_horizontal_flux_dual(
    HorizontalFluxStencilDual s, 
    int i, int j,
    int ny, int nx) {
    HorizontalFluxJacobian jac = get_horizontal_flux_jac(s.get_primals(),i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}


struct VerticalFluxStencil {
    float v;
    float H_t, H_b;
};

struct VerticalFluxStencilDual {
    DualFloat v;
    DualFloat H_t, H_b;

    __device__ __forceinline__
    VerticalFluxStencil get_primals() const {
        return {v.v,H_t.v,H_b.v};
    }

    __device__ __forceinline__
    VerticalFluxStencil get_diffs() const {
        return {v.d,H_t.d,H_b.d};
    }
};

struct VerticalFluxJacobian {
    float res;
    float d_v;
    float d_H_t, d_H_b;

    __device__ __forceinline__
    float apply_jvp(const VerticalFluxStencil& dot) const {
        return d_v * dot.v +
	       d_H_t * dot.H_t +
	       d_H_b * dot.H_b; 
    }

};  

__device__
VerticalFluxJacobian get_vertical_flux_jac(
    VerticalFluxStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {
 
    VerticalFluxJacobian jac = {0};

    // No flux on boundaries
    if (i <= 0 || i >= ny) {
	return jac;
    }
    
    float H_avg = 0.5f*(s.H_t + s.H_b);
    //float v_mag = sqrtf(s.v * s.v + 10.0f);//fabsf(s.v);
    //float v_sign = s.v / v_mag;//copysignf(1.0f, s.v);
    float v_mag = fabsf(s.v);
    float v_sign = copysignf(1.0f, s.v);
    jac.res = H_avg*s.v - 0.5f*v_mag*(s.H_t - s.H_b);

    jac.d_H_t = 0.5f*(s.v - v_mag);
    jac.d_H_b = 0.5f*(s.v + v_mag);
    jac.d_v   = H_avg - 0.5f*v_sign*(s.H_t - s.H_b); 
    return jac;
}

__device__ __forceinline__
DualFloat get_vertical_flux_dual(
    VerticalFluxStencilDual s,
    int i, int j,
    int ny, int nx) {
    VerticalFluxJacobian jac = get_vertical_flux_jac(s.get_primals(),i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*==============================================
  ==========  CALVING ==========================
  =============================================*/

struct CellCalvingStencil {
    float H;
    float bed;
    float calving_rate;
};

struct CellCalvingStencilDual {
    DualFloat H;
    float bed;
    float calving_rate;	

    __device__ __forceinline__
    CellCalvingStencil get_primals() const {
        return {H.v,bed,calving_rate};
    }

    __device__ __forceinline__
    CellCalvingStencil get_diffs() const {
        return {H.d,0.0f,0.0f};
    }
};

struct CellCalvingJacobian {
    float res;
    float d_H;

    __device__ __forceinline__
    float apply_jvp(const CellCalvingStencil& dot) const {
        return d_H * dot.H;
    }

};  

__device__
CellCalvingJacobian get_cell_calving_jac(
    CellCalvingStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {
 
    CellCalvingJacobian jac = {0};

    float grounded = sigmoid(s.bed + 0.917f*s.H,1.0f);
    jac.res = -s.calving_rate*(1.0f-grounded)*s.H;

    jac.d_H = -s.calving_rate*(1.0f-grounded);
    return jac;
}

__device__ __forceinline__
DualFloat get_cell_calving_dual(
    CellCalvingStencilDual s,
    int i, int j,
    int ny, int nx) {
    CellCalvingJacobian jac = get_cell_calving_jac(s.get_primals(),i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*=========================================================
  ================= Residual Computation ==================
  =========================================================*/


extern "C" __global__
void compute_residual(
    float* __restrict__ r_u,
    float* __restrict__ r_v,
    float* __restrict__ r_H,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ f_u,
    const float* __restrict__ f_v,
    const float* __restrict__ f_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ mask,
    const float* __restrict__ gamma,
    float n, float eps_reg, 
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
    
    if (i > ny || j > nx) return;

    populate_viscosity(eta_local, bi, bj, i, j, u, v, B, n, eps_reg, dx, ny, nx);

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active ) {
	float dx_inv = 1.0f/dx;
	bool has_cell = i >= 0 && i <  ny && j >= 0 && j <  nx;
	bool has_u    = i >= 0 && i <  ny && j >= 0 && j <= nx;
	bool has_v    = i >= 0 && i <= ny && j >= 0 && j <  nx;

	if (has_cell){


	    float H_c      = get_cell(H,i,j,ny,nx);
	    float f_H_c    = get_cell(f_H,i,j,ny,nx);

	    float rH = H_c/dt - f_H_c;

	    float bed_c = get_cell(bed,i,j,ny,nx);
	    CellCalvingJacobian j_calve = get_cell_calving_jac({H_c,bed_c,1.0f},i, j, ny, nx);
	    rH -= j_calve.res;

	    float H_l = get_cell(H,i,j-1,ny,nx);
	    float u_l = get_vfacet(u,i,j,ny,nx);
	    HorizontalFluxJacobian j_l = get_horizontal_flux_jac({u_l,H_l,H_c}, i, j, ny, nx);
	    rH -= j_l.res*dx_inv;

	    float H_r = get_cell(H,i,j+1,ny,nx);
	    float u_r = get_vfacet(u,i,j+1,ny,nx);
	    HorizontalFluxJacobian j_r = get_horizontal_flux_jac({u_r,H_c,H_r}, i, j + 1, ny, nx);
            rH += j_r.res*dx_inv;

	    float H_t = get_cell(H,i-1,j,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    VerticalFluxJacobian j_t = get_vertical_flux_jac({v_t,H_t,H_c}, i, j, ny, nx);
	    rH += j_t.res*dx_inv;

	    float H_b = get_cell(H,i+1,j,ny,nx);
	    float v_b = get_hfacet(v,i+1,j,ny,nx);
	    VerticalFluxJacobian j_b = get_vertical_flux_jac({v_b,H_c,H_b}, i + 1, j, ny, nx);
            rH -= j_b.res*dx_inv;

	    float masked = get_cell(mask,i,j,ny,nx);
	    float thklim = get_cell(gamma,i,j,ny,nx);
            r_H[i * nx + j] = (1.0f - masked) * rH + masked * (H_c - thklim);
	}

	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	
	if (has_u){

	    float f_u_l = get_vfacet(f_u,i,j,ny,nx);
	    float ru_l = -f_u_l;

	    {
	    float eta_c = eta_local[bi][bj];
	    float H_c = get_cell(H,i,j,ny,nx);
	    EtaHCellJacobian eta_H_c = get_eta_H_cell_jac({eta_c,H_c});

            float u_l = get_vfacet(u,i,j,ny,nx);
	    float u_r = get_vfacet(u,i,j+1,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    float v_b = get_hfacet(v,i+1,j,ny,nx);
            SigmaNormalJacobian sigma_xx_c = get_sigma_xx_jac({u_l,u_r,v_t,v_b,eta_H_c.res},dx_inv,i,j,ny,nx);
            
	    ru_l += sigma_xx_c.res * dx_inv;
	    }

	    {
	    float eta_l  = eta_local[bi][bj - 1];
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    EtaHCellJacobian eta_H_l = get_eta_H_cell_jac({eta_l,H_l});

            float u_l    = get_vfacet(u,i,j,ny,nx);
	    float u_ll   = get_vfacet(u,i,j-1,ny,nx);
	    float v_lt   = get_hfacet(v,i,j-1,ny,nx);
	    float v_lb   = get_hfacet(v,i+1,j-1,ny,nx);
            SigmaNormalJacobian sigma_xx_l = get_sigma_xx_jac({u_ll,u_l,v_lt,v_lb,eta_H_l.res},dx_inv,i,j - 1,ny,nx);

	    ru_l -= sigma_xx_l.res * dx_inv;
	    }
	    
	    {
	    float eta_tl = eta_local[bi - 1][bj - 1];
	    float eta_t  = eta_local[bi - 1][bj];
	    float eta_l  = eta_local[bi][bj - 1];
	    float eta_c  = eta_local[bi][bj];
	    
	    float H_tl   = get_cell(H,i-1,j-1,ny,nx);
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
            
	    EtaHVertexJacobian eta_H_tl = get_eta_H_vertex_jac({eta_tl,eta_t,eta_l,eta_c,H_tl,H_t,H_l,H_c});
	    
	    float u_tl = get_vfacet(u,i-1,j,ny,nx);
	    float u_l = get_vfacet(u,i,j,ny,nx);
	    float v_lt = get_hfacet(v,i,j-1,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    
	    SigmaShearJacobian sigma_xy_tl = get_sigma_xy_jac({u_tl,u_l,v_lt,v_t,eta_H_tl.res},dx_inv,i,j,ny,nx);

	    ru_l += sigma_xy_tl.res * dx_inv;
	    }

	    {
	    float eta_l  = eta_local[bi][bj - 1];
	    float eta_c  = eta_local[bi][bj];
	    float eta_bl = eta_local[bi + 1][bj - 1];
	    float eta_b  = eta_local[bi + 1][bj];
	    
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float H_bl   = get_cell(H,i+1,j-1,ny,nx);
	    float H_b    = get_cell(H,i+1,j,ny,nx);

	    EtaHVertexJacobian eta_H_bl = get_eta_H_vertex_jac({eta_l,eta_c,eta_bl,eta_b,H_l,H_c,H_bl,H_b});
	    
	    float u_l    = get_vfacet(u,i,j,ny,nx);
	    float u_bl   = get_vfacet(u,i+1,j,ny,nx);
	    float v_lb   = get_hfacet(v,i+1,j-1,ny,nx);
	    float v_b    = get_hfacet(v,i+1,j,ny,nx);
	    SigmaShearJacobian sigma_xy_bl = get_sigma_xy_jac({u_l,u_bl,v_lb,v_b,eta_H_bl.res},dx_inv,i + 1,j,ny,nx);
    
	    ru_l -= sigma_xy_bl.res * dx_inv;
	    }
	
            {    
            float u_l    = get_vfacet(u,i,j,ny,nx);
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    TauBxJacobian tau_bx = get_tau_bx_jac({u_l,H_l,H_c,bed_l,bed_c,beta_l,beta_c,0.001f});
	    ru_l += tau_bx.res;
	    }

	    {
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    TauDxJacobian tau_dx = get_tau_dx_jac({H_l,H_c,bed_l,bed_c},dx_inv,i,j,ny,nx);
	    ru_l -= tau_dx.res;
	    }
	    r_u[i * (nx + 1) + j] = ru_l;
	}
	
	if (has_v){

	    float f_v_t = get_hfacet(f_v,i,j,ny,nx);
	    float rv_t = -f_v_t;

	    {
	    float eta_t = eta_local[bi - 1][bj];
	    float H_t  = get_cell(H,i-1,j,ny,nx);
	    EtaHCellJacobian eta_H_t = get_eta_H_cell_jac({eta_t,H_t});
	    
	    float u_tl = get_vfacet(u,i-1,j,ny,nx);
	    float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
	    float v_tt = get_hfacet(v,i-1,j,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    SigmaNormalJacobian sigma_yy_t = get_sigma_yy_jac({u_tl,u_tr,v_tt,v_t,eta_H_t.res},dx_inv,i-1,j,ny,nx);
            rv_t += sigma_yy_t.res * dx_inv;
	    }
            
	    {
	    float eta_c = eta_local[bi][bj];
	    float H_c = get_cell(H,i,j,ny,nx);
	    EtaHCellJacobian eta_H_c = get_eta_H_cell_jac({eta_c,H_c});

            float u_l = get_vfacet(u,i,j,ny,nx);
	    float u_r = get_vfacet(u,i,j+1,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    float v_b = get_hfacet(v,i+1,j,ny,nx);
            SigmaNormalJacobian sigma_yy_c = get_sigma_yy_jac({u_l,u_r,v_t,v_b,eta_H_c.res},dx_inv,i,j,ny,nx);
	    rv_t -= sigma_yy_c.res * dx_inv;
	    }
	    
	    {
	    float eta_tl = eta_local[bi - 1][bj - 1];
	    float eta_t  = eta_local[bi - 1][bj];
	    float eta_l  = eta_local[bi][bj - 1];
	    float eta_c  = eta_local[bi][bj];
	    
	    float H_tl   = get_cell(H,i-1,j-1,ny,nx);
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
            
	    EtaHVertexJacobian eta_H_tl = get_eta_H_vertex_jac({eta_tl,eta_t,eta_l,eta_c,H_tl,H_t,H_l,H_c});
	    
	    float u_tl = get_vfacet(u,i-1,j,ny,nx);
	    float u_l = get_vfacet(u,i,j,ny,nx);
	    float v_lt = get_hfacet(v,i,j-1,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    
	    SigmaShearJacobian sigma_xy_tl = get_sigma_xy_jac({u_tl,u_l,v_lt,v_t,eta_H_tl.res},dx_inv,i,j,ny,nx);

	    rv_t -= sigma_xy_tl.res * dx_inv;
	    }

	    {
	    float eta_t  = eta_local[bi - 1][bj];
	    float eta_tr = eta_local[bi - 1][bj + 1];
	    float eta_c  = eta_local[bi][bj];
	    float eta_r = eta_local[bi][bj + 1];

	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_tr = get_cell(H,i-1,j+1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float H_r = get_cell(H,i,j+1,ny,nx);

	    EtaHVertexJacobian eta_H_tr = get_eta_H_vertex_jac({eta_t,eta_tr,eta_c,eta_r,H_t,H_tr,H_c,H_r});

	    float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
	    float u_r = get_vfacet(u,i,j+1,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    float v_rt = get_hfacet(v,i,j+1,ny,nx);
	    SigmaShearJacobian sigma_xy_tr = get_sigma_xy_jac({u_tr,u_r,v_t,v_rt,eta_H_tr.res},dx_inv,i,j+1,ny,nx);
	    rv_t += sigma_xy_tr.res * dx_inv;
	    }

	    {
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_t = get_cell(bed,i-1,j,ny,nx);
	    float bed_c = get_cell(bed,i,j,ny,nx);
	    float beta_t = get_cell(beta,i-1,j,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);

	    TauByJacobian tau_by = get_tau_by_jac({v_t,H_t,H_c,bed_t,bed_c,beta_t,beta_c,0.001f});
	    rv_t += tau_by.res;
	    }

	    {
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_t = get_cell(bed,i-1,j,ny,nx);
	    float bed_c = get_cell(bed,i,j,ny,nx);

	    TauDyJacobian tau_dy = get_tau_dy_jac({H_t,H_c,bed_t,bed_c},dx_inv,i,j,ny,nx);
	    rv_t -= tau_dy.res;
	    }

	    r_v[i * nx + j] = rv_t;
	}
    }
}


/*=========================================================
  ==================== JVP Computation ====================
  =========================================================*/

extern "C" __global__
void compute_jvp(
    float* __restrict__ jvp_u,
    float* __restrict__ jvp_v,
    float* __restrict__ jvp_H,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ d_u,
    const float* __restrict__ d_v,
    const float* __restrict__ d_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ mask,
    const float* __restrict__ gamma,
    float n, float eps_reg, 
    float dx, float dt,
    int ny, int nx, int stride, int halo) 
{
    const int bny = 16;
    const int bnx = 16;

    int bi = threadIdx.y;
    int bj = threadIdx.x;

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    __shared__ DualFloat eta_local[bny][bnx];
    
    if (i > ny || j > nx) return;

    populate_viscosity(eta_local, bi, bj, i, j, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active ) {
	float dx_inv = 1.0f/dx;
	bool has_cell = i >= 0 && i <  ny && j >= 0 && j <  nx;
	bool has_u    = i >= 0 && i <  ny && j >= 0 && j <= nx;
	bool has_v    = i >= 0 && i <= ny && j >= 0 && j <  nx;

	if (has_cell){
	    DualFloat H_c      = get_cell(H,d_H,i,j,ny,nx);

	    float d_rH = H_c.d/dt;

	    float bed_c = get_cell(bed,i,j,ny,nx);
	    DualFloat calve = get_cell_calving_dual({H_c,bed_c,1.0f},i, j, ny, nx);
	    d_rH -= calve.d;

	    DualFloat H_l = get_cell(H,d_H,i,j-1,ny,nx);
	    DualFloat u_l = get_vfacet(u,d_u,i,j,ny,nx);
	    DualFloat q_l = get_horizontal_flux_dual({u_l,H_l,H_c}, i, j, ny, nx);
	    d_rH -= q_l.d*dx_inv;

	    DualFloat H_r = get_cell(H,d_H,i,j+1,ny,nx);
	    DualFloat u_r = get_vfacet(u,d_u,i,j+1,ny,nx);
	    DualFloat q_r = get_horizontal_flux_dual({u_r,H_c,H_r}, i, j + 1, ny, nx);
            d_rH += q_r.d*dx_inv;

	    DualFloat H_t = get_cell(H,d_H,i-1,j,ny,nx);
	    DualFloat v_t = get_hfacet(v,d_v,i,j,ny,nx);
	    DualFloat q_t = get_vertical_flux_dual({v_t,H_t,H_c}, i, j, ny, nx);
	    d_rH += q_t.d*dx_inv;

	    DualFloat H_b = get_cell(H,d_H,i+1,j,ny,nx);
	    DualFloat v_b = get_hfacet(v,d_v,i+1,j,ny,nx);
	    DualFloat q_b = get_vertical_flux_dual({v_b,H_c,H_b}, i + 1, j, ny, nx);
            d_rH -= q_b.d*dx_inv;

	    float masked = get_cell(mask,i,j,ny,nx);
            jvp_H[i * nx + j] = (1.0f - masked) * d_rH + masked * H_c.d;

	}

	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	
	if (has_u){
	    float d_ru_l = 0.0f;

	    {
	    DualFloat eta_c = eta_local[bi][bj];
	    DualFloat H_c = get_cell(H,d_H,i,j,ny,nx);
	    DualFloat eta_H_c = get_eta_H_cell_dual({eta_c,H_c});

            DualFloat u_l = get_vfacet(u,d_u,i,j,ny,nx);
	    DualFloat u_r = get_vfacet(u,d_u,i,j+1,ny,nx);
	    DualFloat v_t = get_hfacet(v,d_v,i,j,ny,nx);
	    DualFloat v_b = get_hfacet(v,d_v,i+1,j,ny,nx);
	    DualFloat sigma_xx_c = get_sigma_xx_dual({u_l,u_r,v_t,v_b,eta_H_c},dx_inv,i,j,ny,nx);
             
	    d_ru_l += sigma_xx_c.d*dx_inv;
	    }

	    {
	    DualFloat eta_l  = eta_local[bi][bj - 1];
	    DualFloat H_l    = get_cell(H,d_H,i,j-1,ny,nx);
	    DualFloat eta_H_l = get_eta_H_cell_dual({eta_l,H_l});

            DualFloat u_l    = get_vfacet(u,d_u,i,j,ny,nx);
	    DualFloat u_ll   = get_vfacet(u,d_u,i,j-1,ny,nx);
	    DualFloat v_lt   = get_hfacet(v,d_v,i,j-1,ny,nx);
	    DualFloat v_lb   = get_hfacet(v,d_v,i+1,j-1,ny,nx);
            DualFloat sigma_xx_l = get_sigma_xx_dual({u_ll,u_l,v_lt,v_lb,eta_H_l},dx_inv,i,j-1,ny,nx);
	    
	    d_ru_l -= sigma_xx_l.d * dx_inv;
	    }
	    
	    {
	    DualFloat eta_tl = eta_local[bi - 1][bj - 1];
	    DualFloat eta_t  = eta_local[bi - 1][bj];
	    DualFloat eta_l  = eta_local[bi][bj - 1];
	    DualFloat eta_c  = eta_local[bi][bj];
	    
	    DualFloat H_tl   = get_cell(H,d_H,i-1,j-1,ny,nx);
	    DualFloat H_t    = get_cell(H,d_H,i-1,j,ny,nx);
	    DualFloat H_l    = get_cell(H,d_H,i,j-1,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
            
	    DualFloat eta_H_tl = get_eta_H_vertex_dual({eta_tl,eta_t,eta_l,eta_c,H_tl,H_t,H_l,H_c});
	    
	    DualFloat u_tl = get_vfacet(u,d_u,i-1,j,ny,nx);
	    DualFloat u_l = get_vfacet(u,d_u,i,j,ny,nx);
	    DualFloat v_lt = get_hfacet(v,d_v,i,j-1,ny,nx);
	    DualFloat v_t = get_hfacet(v,d_v,i,j,ny,nx);
	    DualFloat sigma_xy_tl = get_sigma_xy_dual({u_tl,u_l,v_lt,v_t,eta_H_tl},dx_inv,i,j,ny,nx);

	    d_ru_l += sigma_xy_tl.d * dx_inv;
	    }

	    {
	    DualFloat eta_l  = eta_local[bi][bj - 1];
	    DualFloat eta_c  = eta_local[bi][bj];
	    DualFloat eta_bl = eta_local[bi + 1][bj - 1];
	    DualFloat eta_b  = eta_local[bi + 1][bj];
	    
	    DualFloat H_l    = get_cell(H,d_H,i,j-1,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
	    DualFloat H_bl   = get_cell(H,d_H,i+1,j-1,ny,nx);
	    DualFloat H_b    = get_cell(H,d_H,i+1,j,ny,nx);

	    DualFloat eta_H_bl = get_eta_H_vertex_dual({eta_l,eta_c,eta_bl,eta_b,H_l,H_c,H_bl,H_b});
	    
	    DualFloat u_l    = get_vfacet(u,d_u,i,j,ny,nx);
	    DualFloat u_bl   = get_vfacet(u,d_u,i+1,j,ny,nx);
	    DualFloat v_lb   = get_hfacet(v,d_v,i+1,j-1,ny,nx);
	    DualFloat v_b    = get_hfacet(v,d_v,i+1,j,ny,nx);
	    DualFloat sigma_xy_bl = get_sigma_xy_dual({u_l,u_bl,v_lb,v_b,eta_H_bl},dx_inv,i + 1,j,ny,nx);

	    d_ru_l -= sigma_xy_bl.d * dx_inv;
	    }
	
            {    
            DualFloat u_l    = get_vfacet(u,d_u,i,j,ny,nx);
	    DualFloat H_l    = get_cell(H,d_H,i,j-1,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    DualFloat tau_bx = get_tau_bx_dual({u_l,H_l,H_c,bed_l,bed_c,beta_l,beta_c,0.001f});
	    d_ru_l += tau_bx.d;
	    }

	    {
	    DualFloat H_l    = get_cell(H,d_H,i,j-1,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    DualFloat tau_dx = get_tau_dx_dual({H_l,H_c,bed_l,bed_c},dx_inv,i,j,ny,nx);
	    d_ru_l -= tau_dx.d;
	    }
	    jvp_u[i * (nx + 1) + j] = d_ru_l;
	
 	}
	
	if (has_v){
	    float d_rv_t = 0.0f;

	    {
	    DualFloat eta_t = eta_local[bi - 1][bj];
	    DualFloat H_t  = get_cell(H,d_H,i-1,j,ny,nx);
	    DualFloat eta_H_t = get_eta_H_cell_dual({eta_t,H_t});
	    
	    DualFloat u_tl = get_vfacet(u,d_u,i-1,j,ny,nx);
	    DualFloat u_tr = get_vfacet(u,d_u,i-1,j+1,ny,nx);
	    DualFloat v_tt = get_hfacet(v,d_v,i-1,j,ny,nx);
	    DualFloat v_t = get_hfacet(v,d_v,i,j,ny,nx);
	    DualFloat sigma_yy_t = get_sigma_yy_dual({u_tl,u_tr,v_tt,v_t,eta_H_t},dx_inv,i-1,j,ny,nx);
            d_rv_t += sigma_yy_t.d * dx_inv;
	    }
            
	    {
	    DualFloat eta_c = eta_local[bi][bj];
	    DualFloat H_c = get_cell(H,d_H,i,j,ny,nx);
	    DualFloat eta_H_c = get_eta_H_cell_dual({eta_c,H_c});

            DualFloat u_l = get_vfacet(u,d_u,i,j,ny,nx);
	    DualFloat u_r = get_vfacet(u,d_u,i,j+1,ny,nx);
	    DualFloat v_t = get_hfacet(v,d_v,i,j,ny,nx);
	    DualFloat v_b = get_hfacet(v,d_v,i+1,j,ny,nx);
            DualFloat sigma_yy_c = get_sigma_yy_dual({u_l,u_r,v_t,v_b,eta_H_c},dx_inv,i,j,ny,nx);
	    d_rv_t -= sigma_yy_c.d * dx_inv;
	    }
	    
	    {
	    DualFloat eta_tl = eta_local[bi - 1][bj - 1];
	    DualFloat eta_t  = eta_local[bi - 1][bj];
	    DualFloat eta_l  = eta_local[bi][bj - 1];
	    DualFloat eta_c  = eta_local[bi][bj];
	    
	    DualFloat H_tl   = get_cell(H,d_H,i-1,j-1,ny,nx);
	    DualFloat H_t    = get_cell(H,d_H,i-1,j,ny,nx);
	    DualFloat H_l    = get_cell(H,d_H,i,j-1,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
            
	    DualFloat eta_H_tl = get_eta_H_vertex_dual({eta_tl,eta_t,eta_l,eta_c,H_tl,H_t,H_l,H_c});
	    
	    DualFloat u_tl = get_vfacet(u,d_u,i-1,j,ny,nx);
	    DualFloat u_l = get_vfacet(u,d_u,i,j,ny,nx);
	    DualFloat v_lt = get_hfacet(v,d_v,i,j-1,ny,nx);
	    DualFloat v_t = get_hfacet(v,d_v,i,j,ny,nx);
	    
	    DualFloat sigma_xy_tl = get_sigma_xy_dual({u_tl,u_l,v_lt,v_t,eta_H_tl},dx_inv,i,j,ny,nx);

	    d_rv_t -= sigma_xy_tl.d * dx_inv;
	    }

	    {
	    DualFloat eta_t  = eta_local[bi - 1][bj];
	    DualFloat eta_tr = eta_local[bi - 1][bj + 1];
	    DualFloat eta_c  = eta_local[bi][bj];
	    DualFloat eta_r  = eta_local[bi][bj + 1];

	    DualFloat H_t  = get_cell(H,d_H,i-1,j,ny,nx);
	    DualFloat H_tr = get_cell(H,d_H,i-1,j+1,ny,nx);
	    DualFloat H_c  = get_cell(H,d_H,i,j,ny,nx);
	    DualFloat H_r  = get_cell(H,d_H,i,j+1,ny,nx);

	    DualFloat eta_H_tr = get_eta_H_vertex_dual({eta_t,eta_tr,eta_c,eta_r,H_t,H_tr,H_c,H_r});

	    DualFloat u_tr = get_vfacet(u,d_u,i-1,j+1,ny,nx);
	    DualFloat u_r  = get_vfacet(u,d_u,i,j+1,ny,nx);
	    DualFloat v_t  = get_hfacet(v,d_v,i,j,ny,nx);
	    DualFloat v_rt = get_hfacet(v,d_v,i,j+1,ny,nx);
	    DualFloat sigma_xy_tr = get_sigma_xy_dual({u_tr,u_r,v_t,v_rt,eta_H_tr},dx_inv,i,j+1,ny,nx);
	    d_rv_t += sigma_xy_tr.d * dx_inv;
	    }

	    {
	    DualFloat v_t    = get_hfacet(v,d_v,i,j,ny,nx);
	    DualFloat H_t    = get_cell(H,d_H,i-1,j,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
	    float bed_t      = get_cell(bed,i-1,j,ny,nx);
	    float bed_c      = get_cell(bed,i,j,ny,nx);
	    float beta_t     = get_cell(beta,i-1,j,ny,nx);
	    float beta_c     = get_cell(beta,i,j,ny,nx);

	    DualFloat tau_by = get_tau_by_dual({v_t,H_t,H_c,bed_t,bed_c,beta_t,beta_c,0.001f});
	    d_rv_t += tau_by.d;
	    }

	    {
	    DualFloat H_t    = get_cell(H,d_H,i-1,j,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
	    float bed_t = get_cell(bed,i-1,j,ny,nx);
	    float bed_c = get_cell(bed,i,j,ny,nx);

	    DualFloat tau_dy = get_tau_dy_dual({H_t,H_c,bed_t,bed_c},dx_inv,i,j,ny,nx);
	    d_rv_t -= tau_dy.d;
	    }

	    jvp_v[i * nx + j] = d_rv_t;

	}
    }
}


/*=========================================================
  ==================== VJP Computation ====================
  =========================================================*/

extern "C" __global__
void compute_vjp(
    float* __restrict__ vjp_u,
    float* __restrict__ vjp_v,
    float* __restrict__ vjp_H,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ lambda_u,
    const float* __restrict__ lambda_v,
    const float* __restrict__ lambda_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ mask,
    const float* __restrict__ gamma,
    float n, float eps_reg, 
    float dx, float dt,
    int ny, int nx, int stride, int halo) 
{
    const int bny = 16;
    const int bnx = 16;

    int bi = threadIdx.y;
    int bj = threadIdx.x;
    int tid = bi * blockDim.x + bj;

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    // SHARED MEMORY ACCUMULATORS
    __shared__ float s_adj_u[bny][bnx+1]; 
    __shared__ float s_adj_v[bny+1][bnx];
    __shared__ float s_adj_H[bny][bnx];

    for (int k = tid; k < 272; k += 256) {
    s_adj_u[k / 17][k % 17] = 0.0f;
    }
    // Same for V-tile (272 elements)
    for (int k = tid; k < 272; k += 256) {
    s_adj_v[k / 16][k % 16] = 0.0f;
    }
    s_adj_H[bi][bj] = 0.0f;

    __shared__ DualFloat eta_local[bny][bnx];
    
    populate_viscosity(eta_local, bi, bj, i, j, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    bool has_cell = i >= 0 && i <  ny && j >= 0 && j <  nx;
    bool has_u    = i >= 0 && i <  ny && j >= 0 && j <= nx;
    bool has_v    = i >= 0 && i <= ny && j >= 0 && j <  nx;

    if ( is_active ) {
	float dx_inv = 1.0f/dx;

	if (has_cell){
	    float H_c        = get_cell(H,i,j,ny,nx);
	    float lambda_H_c = get_masked_cell(lambda_H,mask,i,j,ny,nx);

	    // Mass matrix contribution
	    atomicAdd(&s_adj_H[bi][bj], lambda_H_c/dt);

	    float bed_c = get_cell(bed,i,j,ny,nx);
	    CellCalvingJacobian j_calve = get_cell_calving_jac({H_c,bed_c,1.0f},i, j, ny, nx);
	    atomicAdd(&s_adj_H[bi][bj] , -lambda_H_c*j_calve.d_H);

	    float H_l = get_cell(H,i,j-1,ny,nx);
	    float u_l = get_vfacet(u,i,j,ny,nx);
	    HorizontalFluxJacobian j_q_l = get_horizontal_flux_jac({u_l,H_l,H_c}, i, j, ny, nx);
	    atomicAdd(&s_adj_H[bi][bj]  , -lambda_H_c*j_q_l.d_H_r*dx_inv);
            atomicAdd(&s_adj_H[bi][bj-1], -lambda_H_c*j_q_l.d_H_l*dx_inv);
            atomicAdd(&s_adj_u[bi][bj]  , -lambda_H_c*j_q_l.d_u*dx_inv);
           
	    float H_r = get_cell(H,i,j+1,ny,nx);
	    float u_r = get_vfacet(u,i,j+1,ny,nx);
	    HorizontalFluxJacobian j_q_r = get_horizontal_flux_jac({u_r,H_c,H_r}, i, j + 1, ny, nx);
            atomicAdd(&s_adj_H[bi][bj],   lambda_H_c*j_q_r.d_H_l*dx_inv);
            atomicAdd(&s_adj_H[bi][bj+1], lambda_H_c*j_q_r.d_H_r*dx_inv);
            atomicAdd(&s_adj_u[bi][bj+1], lambda_H_c*j_q_r.d_u*dx_inv);

	    float H_t = get_cell(H,i-1,j,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    VerticalFluxJacobian j_q_t = get_vertical_flux_jac({v_t,H_t,H_c}, i, j, ny, nx);
	    atomicAdd(&s_adj_H[bi][bj],   lambda_H_c*j_q_t.d_H_b*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj], lambda_H_c*j_q_t.d_H_t*dx_inv);
	    atomicAdd(&s_adj_v[bi][bj],   lambda_H_c*j_q_t.d_v*dx_inv);

	    float H_b = get_cell(H,i+1,j,ny,nx);
	    float v_b = get_hfacet(v,i+1,j,ny,nx);
	    VerticalFluxJacobian j_q_b = get_vertical_flux_jac({v_b,H_c,H_b}, i + 1, j, ny, nx);
            atomicAdd(&s_adj_H[bi][bj],   -lambda_H_c*j_q_b.d_H_t*dx_inv);
            atomicAdd(&s_adj_H[bi+1][bj], -lambda_H_c*j_q_b.d_H_b*dx_inv);
	    atomicAdd(&s_adj_v[bi+1][bj], -lambda_H_c*j_q_b.d_v*dx_inv);

	    float masked = get_cell(mask,i,j,ny,nx);
	    float lambda_H_c_ = get_cell(lambda_H,i,j,ny,nx);
            atomicAdd(&s_adj_H[bi][bj],    lambda_H_c_*masked);
	}

	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	
	if (has_u){
	    {
	    DualFloat eta_c = eta_local[bi][bj];
	    float H_c = get_cell(H,i,j,ny,nx);
	    EtaHCellJacobian eta_H_c = get_eta_H_cell_jac({eta_c.v,H_c});

            float u_l = get_vfacet(u,i,j,ny,nx);
	    float u_r = get_vfacet(u,i,j+1,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    float v_b = get_hfacet(v,i+1,j,ny,nx);
            
	    float lambda_u_l = get_vfacet(lambda_u,i,j,ny,nx);
	    float lambda_u_r = get_vfacet(lambda_u,i,j+1,ny,nx);
	    float lambda_v_t = get_hfacet(lambda_v,i,j,ny,nx);
	    float lambda_v_b = get_hfacet(lambda_v,i+1,j,ny,nx);
            SigmaNormalJacobian j_sigma_xx_c = get_sigma_xx_jac({u_l,u_r,v_t,v_b,eta_H_c.res},dx_inv,i,j,ny,nx);

	    //sigma_xx jvp applied to lambda with d_H = 0 
	    float lambda_sigma_xx_c = j_sigma_xx_c.apply_jvp({lambda_u_l,lambda_u_r,lambda_v_t,lambda_v_b,eta_H_c.apply_jvp({eta_c.d,0.0f})});

            atomicAdd(&s_adj_u[bi][bj],lambda_sigma_xx_c * dx_inv);
            atomicAdd(&s_adj_H[bi][bj],lambda_u_l*j_sigma_xx_c.d_eta_H*eta_H_c.d_H*dx_inv);
	    }

	    {
	    DualFloat eta_l  = eta_local[bi][bj - 1];
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    EtaHCellJacobian eta_H_l = get_eta_H_cell_jac({eta_l.v,H_l});

            float u_l    = get_vfacet(u,i,j,ny,nx);
	    float u_ll   = get_vfacet(u,i,j-1,ny,nx);
	    float v_lt   = get_hfacet(v,i,j-1,ny,nx);
	    float v_lb   = get_hfacet(v,i+1,j-1,ny,nx);
            
	    float lambda_u_l    = get_vfacet(lambda_u,i,j,ny,nx);
	    float lambda_u_ll   = get_vfacet(lambda_u,i,j-1,ny,nx);
	    float lambda_v_lt   = get_hfacet(lambda_v,i,j-1,ny,nx);
	    float lambda_v_lb   = get_hfacet(lambda_v,i+1,j-1,ny,nx);
            SigmaNormalJacobian j_sigma_xx_l = get_sigma_xx_jac({u_ll,u_l,v_lt,v_lb,eta_H_l.res},dx_inv,i,j - 1,ny,nx);

	    float lambda_sigma_xx_l = j_sigma_xx_l.apply_jvp({lambda_u_ll,lambda_u_l,lambda_v_lt,lambda_v_lb,eta_H_l.apply_jvp({eta_l.d,0.0f})});

	    atomicAdd(&s_adj_u[bi][bj],  -lambda_sigma_xx_l*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj-1],-lambda_u_l*j_sigma_xx_l.d_eta_H*eta_H_l.d_H*dx_inv);
	    }

	    {
	    DualFloat eta_tl = eta_local[bi - 1][bj - 1];
	    DualFloat eta_t  = eta_local[bi - 1][bj];
	    DualFloat eta_l  = eta_local[bi][bj - 1];
	    DualFloat eta_c  = eta_local[bi][bj];

	    float H_tl   = get_cell(H,i-1,j-1,ny,nx);
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
            
	    EtaHVertexJacobian eta_H_tl = get_eta_H_vertex_jac({eta_tl.v,eta_t.v,eta_l.v,eta_c.v,H_tl,H_t,H_l,H_c});
	    
	    float u_tl = get_vfacet(u,i-1,j,ny,nx);
	    float u_l = get_vfacet(u,i,j,ny,nx);
	    float v_lt = get_hfacet(v,i,j-1,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    
	    float lambda_u_tl = get_vfacet(lambda_u,i-1,j,ny,nx);
	    float lambda_u_l = get_vfacet(lambda_u,i,j,ny,nx);
	    float lambda_v_lt = get_hfacet(lambda_v,i,j-1,ny,nx);
	    float lambda_v_t = get_hfacet(lambda_v,i,j,ny,nx);
	    
	    SigmaShearJacobian j_sigma_xy_tl = get_sigma_xy_jac({u_tl,u_l,v_lt,v_t,eta_H_tl.res},dx_inv,i,j,ny,nx);

	    float lambda_sigma_xy_tl = j_sigma_xy_tl.apply_jvp({lambda_u_tl,lambda_u_l,lambda_v_lt,lambda_v_t,eta_H_tl.apply_jvp({eta_tl.d,eta_t.d,eta_l.d,eta_c.d,0.0f,0.0f,0.0f,0.0f})});

	    atomicAdd(&s_adj_u[bi][bj],    lambda_sigma_xy_tl*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj-1],lambda_u_l * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_tl*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj],  lambda_u_l * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_tr*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj-1],  lambda_u_l * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_bl*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj],    lambda_u_l * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_br*dx_inv);
	    }

	    {
	    DualFloat eta_l  = eta_local[bi][bj - 1];
	    DualFloat eta_c  = eta_local[bi][bj];
	    DualFloat eta_bl = eta_local[bi + 1][bj - 1];
	    DualFloat eta_b  = eta_local[bi + 1][bj];
	    
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float H_bl   = get_cell(H,i+1,j-1,ny,nx);
	    float H_b    = get_cell(H,i+1,j,ny,nx);

	    EtaHVertexJacobian eta_H_bl = get_eta_H_vertex_jac({eta_l.v,eta_c.v,eta_bl.v,eta_b.v,H_l,H_c,H_bl,H_b});
	    
	    float u_l    = get_vfacet(u,i,j,ny,nx);
	    float u_bl   = get_vfacet(u,i+1,j,ny,nx);
	    float v_lb   = get_hfacet(v,i+1,j-1,ny,nx);
	    float v_b    = get_hfacet(v,i+1,j,ny,nx);

	    float lambda_u_l    = get_vfacet(lambda_u,i,j,ny,nx);
	    float lambda_u_bl   = get_vfacet(lambda_u,i+1,j,ny,nx);
	    float lambda_v_lb   = get_hfacet(lambda_v,i+1,j-1,ny,nx);
	    float lambda_v_b    = get_hfacet(lambda_v,i+1,j,ny,nx);

	    SigmaShearJacobian j_sigma_xy_bl = get_sigma_xy_jac({u_l,u_bl,v_lb,v_b,eta_H_bl.res},dx_inv,i + 1,j,ny,nx);
	    
	    float lambda_sigma_xy_bl = j_sigma_xy_bl.apply_jvp({lambda_u_l,lambda_u_bl,lambda_v_lb,lambda_v_b,eta_H_bl.apply_jvp({eta_l.d,eta_c.d,eta_bl.d,eta_b.d,0.0f,0.0f,0.0f,0.0f})});
    
	    atomicAdd(&s_adj_u[bi][bj]  ,  -lambda_sigma_xy_bl*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj-1],  -lambda_u_l * j_sigma_xy_bl.d_eta_H*eta_H_bl.d_H_tl*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj],    -lambda_u_l * j_sigma_xy_bl.d_eta_H*eta_H_bl.d_H_tr*dx_inv);
	    atomicAdd(&s_adj_H[bi+1][bj-1],-lambda_u_l * j_sigma_xy_bl.d_eta_H*eta_H_bl.d_H_bl*dx_inv);
	    atomicAdd(&s_adj_H[bi+1][bj],  -lambda_u_l * j_sigma_xy_bl.d_eta_H*eta_H_bl.d_H_br*dx_inv);
	    }

            {    
            float u_l    = get_vfacet(u,i,j,ny,nx);
            float lambda_u_l    = get_vfacet(lambda_u,i,j,ny,nx);
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    TauBxJacobian j_tau_bx = get_tau_bx_jac({u_l,H_l,H_c,bed_l,bed_c,beta_l,beta_c,0.001f});

	    atomicAdd(&s_adj_u[bi][bj],lambda_u_l * j_tau_bx.d_u);
	    }

	    {
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
            float lambda_u_l    = get_vfacet(lambda_u,i,j,ny,nx);
	    
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    TauDxJacobian j_tau_dx = get_tau_dx_jac({H_l,H_c,bed_l,bed_c},dx_inv,i,j,ny,nx);

            atomicAdd(&s_adj_H[bi][bj-1],-lambda_u_l * j_tau_dx.d_H_l);
            atomicAdd(&s_adj_H[bi][bj],  -lambda_u_l * j_tau_dx.d_H_r);
	    }
	
 	}

	if (has_v){
	    {
	    DualFloat eta_t = eta_local[bi - 1][bj];
	    float H_t  = get_cell(H,i-1,j,ny,nx);
	    EtaHCellJacobian eta_H_t = get_eta_H_cell_jac({eta_t.v,H_t});
	    
	    float u_tl = get_vfacet(u,i-1,j,ny,nx);
	    float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
	    float v_tt = get_hfacet(v,i-1,j,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    
	    float lambda_u_tl = get_vfacet(lambda_u,i-1,j,ny,nx);
	    float lambda_u_tr = get_vfacet(lambda_u,i-1,j+1,ny,nx);
	    float lambda_v_tt = get_hfacet(lambda_v,i-1,j,ny,nx);
	    float lambda_v_t  = get_hfacet(lambda_v,i,j,ny,nx);
	    SigmaNormalJacobian j_sigma_yy_t = get_sigma_yy_jac({u_tl,u_tr,v_tt,v_t,eta_H_t.res},dx_inv,i-1,j,ny,nx);

	    float lambda_sigma_yy_t = j_sigma_yy_t.apply_jvp({lambda_u_tl,lambda_u_tr,lambda_v_tt,lambda_v_t,eta_H_t.apply_jvp({eta_t.d,0.0f})});

	    atomicAdd(&s_adj_v[bi][bj],  lambda_sigma_yy_t * dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj],lambda_v_t*j_sigma_yy_t.d_eta_H*eta_H_t.d_H*dx_inv);
	    }

	    {
	    DualFloat eta_c = eta_local[bi][bj];
	    float H_c = get_cell(H,i,j,ny,nx);
	    EtaHCellJacobian eta_H_c = get_eta_H_cell_jac({eta_c.v,H_c});

            float u_l = get_vfacet(u,i,j,ny,nx);
	    float u_r = get_vfacet(u,i,j+1,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    float v_b = get_hfacet(v,i+1,j,ny,nx);
	    
	    float lambda_u_l = get_vfacet(lambda_u,i,j,ny,nx);
	    float lambda_u_r = get_vfacet(lambda_u,i,j+1,ny,nx);
	    float lambda_v_t = get_hfacet(lambda_v,i,j,ny,nx);
	    float lambda_v_b = get_hfacet(lambda_v,i+1,j,ny,nx);
            SigmaNormalJacobian j_sigma_yy_c = get_sigma_yy_jac({u_l,u_r,v_t,v_b,eta_H_c.res},dx_inv,i,j,ny,nx);
	    
	    float lambda_sigma_yy_c = j_sigma_yy_c.apply_jvp({lambda_u_l,lambda_u_r,lambda_v_t,lambda_v_b,eta_H_c.apply_jvp({eta_c.d,0.0f})});
	    atomicAdd(&s_adj_v[bi][bj],-lambda_sigma_yy_c*dx_inv);
            atomicAdd(&s_adj_H[bi][bj],-lambda_v_t*j_sigma_yy_c.d_eta_H*eta_H_c.d_H*dx_inv);
	    }
	    
	    {
	    DualFloat eta_tl = eta_local[bi - 1][bj - 1];
	    DualFloat eta_t  = eta_local[bi - 1][bj];
	    DualFloat eta_l  = eta_local[bi][bj - 1];
	    DualFloat eta_c  = eta_local[bi][bj];

	    float H_tl   = get_cell(H,i-1,j-1,ny,nx);
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
            
	    EtaHVertexJacobian eta_H_tl = get_eta_H_vertex_jac({eta_tl.v,eta_t.v,eta_l.v,eta_c.v,H_tl,H_t,H_l,H_c});
	    
	    float u_tl = get_vfacet(u,i-1,j,ny,nx);
	    float u_l = get_vfacet(u,i,j,ny,nx);
	    float v_lt = get_hfacet(v,i,j-1,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    
	    float lambda_u_tl = get_vfacet(lambda_u,i-1,j,ny,nx);
	    float lambda_u_l = get_vfacet(lambda_u,i,j,ny,nx);
	    float lambda_v_lt = get_hfacet(lambda_v,i,j-1,ny,nx);
	    float lambda_v_t = get_hfacet(lambda_v,i,j,ny,nx);
	    
	    SigmaShearJacobian j_sigma_xy_tl = get_sigma_xy_jac({u_tl,u_l,v_lt,v_t,eta_H_tl.res},dx_inv,i,j,ny,nx);

	    float lambda_sigma_xy_tl = j_sigma_xy_tl.apply_jvp({lambda_u_tl,lambda_u_l,lambda_v_lt,lambda_v_t,eta_H_tl.apply_jvp({eta_tl.d,eta_t.d,eta_l.d,eta_c.d,0.0f,0.0f,0.0f,0.0f})});

	    atomicAdd(&s_adj_v[bi][bj],    -lambda_sigma_xy_tl*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj-1],-lambda_v_t * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_tl*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj],  -lambda_v_t * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_tr*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj-1],  -lambda_v_t * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_bl*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj],    -lambda_v_t * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_br*dx_inv);
	    }

	    {
	    DualFloat eta_t  = eta_local[bi - 1][bj];
	    DualFloat eta_tr = eta_local[bi - 1][bj + 1];
	    DualFloat eta_c  = eta_local[bi][bj];
	    DualFloat eta_r = eta_local[bi][bj + 1];

	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_tr = get_cell(H,i-1,j+1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float H_r = get_cell(H,i,j+1,ny,nx);

	    EtaHVertexJacobian eta_H_tr = get_eta_H_vertex_jac({eta_t.v,eta_tr.v,eta_c.v,eta_r.v,H_t,H_tr,H_c,H_r});

	    float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
	    float u_r = get_vfacet(u,i,j+1,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    float v_rt = get_hfacet(v,i,j+1,ny,nx);
	    
	    float lambda_u_tr = get_vfacet(lambda_u,i-1,j+1,ny,nx);
	    float lambda_u_r = get_vfacet(lambda_u,i,j+1,ny,nx);
	    float lambda_v_t = get_hfacet(lambda_v,i,j,ny,nx);
	    float lambda_v_rt = get_hfacet(lambda_v,i,j+1,ny,nx);
	    SigmaShearJacobian j_sigma_xy_tr = get_sigma_xy_jac({u_tr,u_r,v_t,v_rt,eta_H_tr.res},dx_inv,i,j+1,ny,nx);

	    float lambda_sigma_xy_tr = j_sigma_xy_tr.apply_jvp({lambda_u_tr,lambda_u_r,lambda_v_t,lambda_v_rt,eta_H_tr.apply_jvp({eta_t.d,eta_tr.d,eta_c.d,eta_r.d,0.0f,0.0f,0.0f,0.0f})});

	    atomicAdd(&s_adj_v[bi][bj],    lambda_sigma_xy_tr*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj],  lambda_v_t * j_sigma_xy_tr.d_eta_H*eta_H_tr.d_H_tl*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj+1],lambda_v_t * j_sigma_xy_tr.d_eta_H*eta_H_tr.d_H_tr*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj],    lambda_v_t * j_sigma_xy_tr.d_eta_H*eta_H_tr.d_H_bl*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj+1],  lambda_v_t * j_sigma_xy_tr.d_eta_H*eta_H_tr.d_H_br*dx_inv);            
	    }

	    {
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    float lambda_v_t = get_hfacet(lambda_v,i,j,ny,nx);
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_t = get_cell(bed,i-1,j,ny,nx);
	    float bed_c = get_cell(bed,i,j,ny,nx);
	    float beta_t = get_cell(beta,i-1,j,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);

	    TauByJacobian j_tau_by = get_tau_by_jac({v_t,H_t,H_c,bed_t,bed_c,beta_t,beta_c,0.001f});
	    atomicAdd(&s_adj_v[bi][bj],lambda_v_t * j_tau_by.d_v);
	    }

	    {
            float lambda_v_t    = get_hfacet(lambda_v,i,j,ny,nx);
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_t = get_cell(bed,i-1,j,ny,nx);
	    float bed_c = get_cell(bed,i,j,ny,nx);

	    TauDyJacobian j_tau_dy = get_tau_dy_jac({H_t,H_c,bed_t,bed_c},dx_inv,i,j,ny,nx);
	    atomicAdd(&s_adj_H[bi-1][bj],-lambda_v_t * j_tau_dy.d_H_t);
	    atomicAdd(&s_adj_H[bi][bj],  -lambda_v_t * j_tau_dy.d_H_b);
	    }

	}

    }
    __syncthreads();

    // Global Base indices for thread(0,0) of this block
    int g_base_y = blockIdx.y * stride - halo;
    int g_base_x = blockIdx.x * stride - halo;
    

    // Flush U (16x17)
    for (int k = tid; k < 272; k += 256) {

        int r = k / 17;
        int c = k % 17;
        float val = s_adj_u[r][c];
        if (fabsf(val) > 0.0f) {
            int gy = g_base_y + r;
            int gx = g_base_x + c;
            if (gy >= 0 && gy < ny && gx >= 0 && gx <= nx) // Global Bounds Check
                atomicAdd(&vjp_u[gy * (nx+1) + gx], val);
        }
    }

    // Flush V (17x16)
    for (int k = tid; k < 272; k += 256) {
        int r = k / 16;
        int c = k % 16;
        float val = s_adj_v[r][c];
        if (fabsf(val) > 0.0f) {
            int gy = g_base_y + r;
            int gx = g_base_x + c;
            if (gy >= 0 && gy <= ny && gx >= 0 && gx < nx) 
                atomicAdd(&vjp_v[gy * nx + gx], val);
        }
    }

    // Flush H (16x16)
    if (tid < 256) {
        float val = s_adj_H[bi][bj];
        if (fabsf(val) > 0.0f) {
            int gy = g_base_y + bi;
            int gx = g_base_x + bj;
            if (gy >= 0 && gy < ny && gx >= 0 && gx < nx)
                atomicAdd(&vjp_H[gy * nx + gx], val);
        }
    }

}



extern "C" __global__
void vanka_smooth(
    float* __restrict__ u_out,
    float* __restrict__ v_out,
    float* __restrict__ H_out,
    float* __restrict__ mask,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ f_u,
    const float* __restrict__ f_v,
    const float* __restrict__ f_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float n, float eps_reg, 
    float dx, float dt,
    int ny, int nx, int stride, int halo,
    int color, int n_newton, float omega
    ) 
{
    const int bny = 16;
    const int bnx = 16;

    int bi = threadIdx.y;
    int bj = threadIdx.x;

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    __shared__ float eta_local[bny][bnx];
    
    if (i < 0 || i >= ny || j<0 || j >= nx) return;

    populate_viscosity(eta_local, bi, bj, i, j, u, v, B, n, eps_reg, dx, ny, nx);

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active && ((i + j) % 2 == color)) {
	float dx_inv = 1.0f/dx;

	float masked = get_cell(mask, i, j, ny, nx);

	float u_l = get_vfacet(u, i, j, ny, nx);
	float u_r = get_vfacet(u, i, j + 1, ny, nx);
	float v_t = get_hfacet(v, i, j, ny, nx);
	float v_b = get_hfacet(v, i + 1, j, ny, nx);
	float H_c = get_cell(H, i, j, ny, nx);
	float thklim = get_cell(gamma,i,j,ny,nx);
	float rnorm = 0.0f;

        for (int k = 0; k < n_newton; k++) {
	    float J[25] = {0};
	    float r[5] = {0};

	    r[0] -= get_vfacet(f_u,i,j,ny,nx);
	    r[1] -= get_vfacet(f_u,i,j+1,ny,nx);
	    r[2] -= get_hfacet(f_v,i,j,ny,nx);
	    r[3] -= get_hfacet(f_v,i+1,j,ny,nx);
	    r[4] -= get_hfacet(f_H,i,j,ny,nx);

	    // Mass Conservation Assembly
	    {
	    // Standard Mass Conservation: dH/dt + div(q) - smb = 0
	    J[24] = 1.0f / dt;
	    r[4] += H_c/dt;// H_prev/dt - smb handled by f_H - (H_c - H_prev_c) / dt - smb_c;

	    
	    float bed_c = get_cell(bed,i,j,ny,nx);
	    CellCalvingJacobian j_calve = get_cell_calving_jac({H_c,bed_c,1.0f},i, j, ny, nx);
	    J[24] -= j_calve.d_H;
	    r[4] -= j_calve.res;

	    // X-Fluxes
	    float H_l = get_cell(H,i,j-1,ny,nx);
	    HorizontalFluxJacobian j_l = get_horizontal_flux_jac({u_l, H_l, H_c}, i, j, ny, nx);
	    J[20] -= j_l.d_u   * dx_inv;
	    J[24] -= j_l.d_H_r * dx_inv;
	    r[4]  -= j_l.res   * dx_inv;

	    float H_r = get_cell(H,i,j+1,ny,nx);
	    HorizontalFluxJacobian j_r = get_horizontal_flux_jac({u_r, H_c, H_r}, i, j+1, ny, nx);
	    J[21] += j_r.d_u   * dx_inv;
	    J[24] += j_r.d_H_l * dx_inv;
	    r[4]  += j_r.res   * dx_inv;

	    // Y-Fluxes (Vertical in grid coordinates)
	    float H_t = get_cell(H,i-1,j,ny,nx);
	    VerticalFluxJacobian j_t = get_vertical_flux_jac({v_t, H_t, H_c}, i, j, ny, nx);
	    J[22] += j_t.d_v   * dx_inv;
	    J[24] += j_t.d_H_b * dx_inv;
	    r[4]  += j_t.res   * dx_inv;

	    float H_b = get_cell(H,i+1,j,ny,nx);
	    VerticalFluxJacobian j_b = get_vertical_flux_jac({v_b, H_c, H_b}, i+1, j, ny, nx);
	    J[23] -= j_b.d_v   * dx_inv;
	    J[24] -= j_b.d_H_t * dx_inv;
	    r[4]  -= j_b.res   * dx_inv;

	    if ((H_c - dt*r[4]) <= thklim) {
		// Active set constraint: Force H = thklim
		//masked = 1.0f;
		for(int k=0; k<5; ++k) J[20 + k] = 0.0f;
		J[24] = 1.0f;
		r[4] = H_c - thklim;
	    } else {
	        //masked = 0.0f;
	    
	    }
	    
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
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    TauBxJacobian tau_bx_l = get_tau_bx_jac({u_l,H_l,H_c,bed_l,bed_c,beta_l,beta_c,0.001f});
	    r[0] += tau_bx_l.res;
            J[0] += tau_bx_l.d_u;
	    J[4] += tau_bx_l.d_H_r;
	    }

	    // Basal shear stress for right momentum
            {    
	    float H_r    = get_cell(H,i,j+1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float bed_r  = get_cell(bed,i,j+1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    float beta_r = get_cell(beta,i,j+1,ny,nx);
	    TauBxJacobian tau_bx_r = get_tau_bx_jac({u_r,H_c,H_r,bed_c,bed_r,beta_c,beta_r,0.001f});
	    r[1] += tau_bx_r.res;
            J[6] += tau_bx_r.d_u;
	    J[9] += tau_bx_r.d_H_l;
	    }

	    // Basal shear stress for top momentum
            {    
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float bed_t  = get_cell(bed,i-1,j,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float beta_t = get_cell(beta,i-1,j,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    TauByJacobian tau_by_t = get_tau_by_jac({v_t,H_t,H_c,bed_t,bed_c,beta_t,beta_c,0.001f});
	    r[2]  += tau_by_t.res;
            J[12] += tau_by_t.d_v;
	    J[14] += tau_by_t.d_H_b;
	    }

	    // Basal shear stress for bottom momentum
            {    
	    float H_b    = get_cell(H,i+1,j,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float bed_b  = get_cell(bed,i+1,j,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    float beta_b = get_cell(beta,i+1,j,ny,nx);
	    TauByJacobian tau_by_b = get_tau_by_jac({v_b,H_c,H_b,bed_c,bed_b,beta_c,beta_b,0.001f});
	    r[3]  += tau_by_b.res;
            J[18] += tau_by_b.d_v;
	    J[19] += tau_by_b.d_H_t;
	    }
            
            // Driving stress for left momentum (u)
	    {
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    TauDxJacobian tau_dx_l = get_tau_dx_jac({H_l,H_c,bed_l,bed_c},dx_inv,i,j,ny,nx);
	    r[0] -= tau_dx_l.res;
	    J[4] -= tau_dx_l.d_H_r;
	    }

	    // Driving stress for right momentum (u)
	    {
	    float H_r    = get_cell(H,i,j+1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float bed_r  = get_cell(bed,i,j+1,ny,nx);
	    TauDxJacobian tau_dx_r = get_tau_dx_jac({H_c,H_r,bed_c,bed_r},dx_inv,i,j+1,ny,nx);
	    r[1] -= tau_dx_r.res;
	    J[9] -= tau_dx_r.d_H_l;
	    }

	    // Driving stress for top momentum (v)
	    {
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float bed_t  = get_cell(bed,i-1,j,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    TauDyJacobian tau_dy_t = get_tau_dy_jac({H_t,H_c,bed_t,bed_c},dx_inv,i,j,ny,nx);
	    r[2]  -= tau_dy_t.res;
	    J[14] -= tau_dy_t.d_H_b;
	    }

	    // Driving stress for bottom momentum (v)
	    {
	    float H_b    = get_cell(H,i+1,j,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float bed_b  = get_cell(bed,i+1,j,ny,nx);
	    TauDyJacobian tau_dy_b = get_tau_dy_jac({H_c,H_b,bed_c,bed_b},dx_inv,i+1,j,ny,nx);
	    r[3]  -= tau_dy_b.res;
	    J[19] -= tau_dy_b.d_H_t;
	    }

	    //float r_norm = 0.f;
	    //for (int q = 0; q < 5; q++) r_norm += r[q]*r[q];
	    //if (r_norm < 0.001f) break;

            J[0]  -= 1.0f;
            J[6]  -= 1.0f;
            J[12] -= 1.0f;
            J[18] -= 1.0f;
            J[24] += 1.0f;

	    float delta_x[5] = {0};
	    lu_5x5_solve(J,r,delta_x);


            float relaxation_factor = 0.5f;

	    u_l -= relaxation_factor*delta_x[0];
	    u_r -= relaxation_factor*delta_x[1];
	    v_t -= relaxation_factor*delta_x[2];
	    v_b -= relaxation_factor*delta_x[3];
	    H_c -= relaxation_factor*delta_x[4];
	    H_c = fmaxf(H_c,thklim);

        }
	
	float u_l_prev = get_vfacet(u, i, j, ny, nx);
	float u_r_prev = get_vfacet(u, i, j + 1, ny, nx);
	float v_t_prev = get_hfacet(v, i, j, ny, nx);
	float v_b_prev = get_hfacet(v, i + 1, j, ny, nx);
	float H_c_prev = get_cell(H, i, j, ny, nx);

	u_out[i * (nx + 1) + j]     += 0.5f*(u_l - u_l_prev);
	u_out[i * (nx + 1) + j + 1] += 0.5f*(u_r - u_r_prev);
	v_out[i * nx + j]           += 0.5f*(v_t - v_t_prev);
	v_out[(i + 1) * nx + j ]    += 0.5f*(v_b - v_b_prev);
	H_out[i * nx + j]           = (H_c - H_c_prev);
    }
}

extern "C" __global__
void vanka_smooth_adjoint(
    float* __restrict__ lambda_u_out,
    float* __restrict__ lambda_v_out,
    float* __restrict__ lambda_H_out,
    const float* __restrict__ lambda_u,
    const float* __restrict__ lambda_v,
    const float* __restrict__ lambda_H,
    const float* __restrict__ mask,
    const float* __restrict__ r_adj_u,  //J^T lambda + d(cost)/dU
    const float* __restrict__ r_adj_v,
    const float* __restrict__ r_adj_H,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float n, float eps_reg, 
    float dx, float dt,
    int ny, int nx, int stride, int halo,
    int color, float omega
    ) 
{
    const int bny = 16;
    const int bnx = 16;

    int bi = threadIdx.y;
    int bj = threadIdx.x;

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    __shared__ float eta_local[bny][bnx];
    
    if (i < 0 || i >= ny || j<0 || j >= nx) return;

    populate_viscosity(eta_local, bi, bj, i, j, u, v, B, n, eps_reg, dx, ny, nx);

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active && ((i + j) % 2 == color)) {
	float dx_inv = 1.0f/dx;

	float masked = get_cell(mask, i, j, ny, nx);

	float u_l = get_vfacet(u, i, j, ny, nx);
	float u_r = get_vfacet(u, i, j + 1, ny, nx);
	float v_t = get_hfacet(v, i, j, ny, nx);
	float v_b = get_hfacet(v, i + 1, j, ny, nx);
	float H_c = get_cell(H, i, j, ny, nx);

	float thklim = get_cell(gamma,i,j,ny,nx);
	float rnorm = 0.0f;

	float J[25] = {0};

	// Mass Conservation Assembly
	{
	// Standard Mass Conservation: dH/dt + div(q) - smb = 0
	//float H_prev_c = get_cell(H_prev, i, j, ny, nx);
	//float smb_c    = get_cell(smb, i, j, ny, nx);
	
	J[24] = 1.0f / dt;

        float bed_c = get_cell(bed,i,j,ny,nx);
	CellCalvingJacobian j_calve = get_cell_calving_jac({H_c,bed_c,1.0f},i, j, ny, nx);
	J[24] -= j_calve.d_H;

	// X-Fluxes
	float H_l = get_cell(H,i,j-1,ny,nx);
	HorizontalFluxJacobian j_l = get_horizontal_flux_jac({u_l, H_l, H_c}, i, j, ny, nx);
	J[20] -= j_l.d_u   * dx_inv;
	J[24] -= j_l.d_H_r * dx_inv;

	float H_r = get_cell(H,i,j+1,ny,nx);
	HorizontalFluxJacobian j_r = get_horizontal_flux_jac({u_r, H_c, H_r}, i, j+1, ny, nx);
	J[21] += j_r.d_u   * dx_inv;
	J[24] += j_r.d_H_l * dx_inv;

	// Y-Fluxes (Vertical in grid coordinates)
	float H_t = get_cell(H,i-1,j,ny,nx);
	VerticalFluxJacobian j_t = get_vertical_flux_jac({v_t, H_t, H_c}, i, j, ny, nx);
	J[22] += j_t.d_v   * dx_inv;
	J[24] += j_t.d_H_b * dx_inv;

	float H_b = get_cell(H,i+1,j,ny,nx);
	VerticalFluxJacobian j_b = get_vertical_flux_jac({v_b, H_c, H_b}, i+1, j, ny, nx);
	J[23] -= j_b.d_v   * dx_inv;
	J[24] -= j_b.d_H_t * dx_inv;

	if (masked > 0.5) {
	    // Active set constraint: Force H = thklim
	    for(int k=0; k<5; ++k) J[20 + k] = 0.0f;
	    J[24] = 1.0f;
	    } 
	
	}
	
	{
	float eta_c = eta_local[bi][bj];
	EtaHCellJacobian eta_H_c = get_eta_H_cell_jac({eta_c,H_c});
	
	// Compute the contribution of sigma_xx at the center to both the left and right u-residuals (since it is used by both)
	SigmaNormalJacobian sigma_xx_c = get_sigma_xx_jac({u_l,u_r,v_t,v_b,eta_H_c.res},dx_inv,i,j,ny,nx);
	
	J[0] += sigma_xx_c.d_u_l * dx_inv;
	J[1] += sigma_xx_c.d_u_r * dx_inv;
	J[2] += sigma_xx_c.d_v_t * dx_inv;
	J[3] += sigma_xx_c.d_v_b * dx_inv;
	J[4] += sigma_xx_c.d_eta_H * eta_H_c.d_H * dx_inv;
	
	J[5] -= sigma_xx_c.d_u_l * dx_inv;
	J[6] -= sigma_xx_c.d_u_r * dx_inv;
	J[7] -= sigma_xx_c.d_v_t * dx_inv;
	J[8] -= sigma_xx_c.d_v_b * dx_inv;
	J[9] -= sigma_xx_c.d_eta_H * eta_H_c.d_H * dx_inv;

	SigmaNormalJacobian sigma_yy_c = get_sigma_yy_jac({u_l,u_r,v_t,v_b,eta_H_c.res},dx_inv,i,j,ny,nx);
	J[10] -= sigma_yy_c.d_u_l * dx_inv;
	J[11] -= sigma_yy_c.d_u_r * dx_inv;
	J[12] -= sigma_yy_c.d_v_t * dx_inv;
	J[13] -= sigma_yy_c.d_v_b * dx_inv;
	J[14] -= sigma_yy_c.d_eta_H * eta_H_c.d_H * dx_inv;

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
	J[0] += sigma_xy_tl.d_u_b * dx_inv;
	J[4] += sigma_xy_tl.d_eta_H * eta_H_tl.d_H_br * dx_inv;

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
	J[6] += sigma_xy_tr.d_u_b * dx_inv;
	J[9] += sigma_xy_tr.d_eta_H * eta_H_tr.d_H_bl * dx_inv;

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
	J[0] -= sigma_xy_bl.d_u_t * dx_inv;
	J[4] -= sigma_xy_bl.d_eta_H * eta_H_bl.d_H_tr * dx_inv;

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
	J[6] -= sigma_xy_br.d_u_t * dx_inv;
	J[9] -= sigma_xy_br.d_eta_H * eta_H_br.d_H_tl * dx_inv;

	J[18] += sigma_xy_br.d_v_l * dx_inv;
	J[19] += sigma_xy_br.d_eta_H * eta_H_br.d_H_tl * dx_inv;
	}
	
	
	// Basal shear stress for left momentum
	{    
	float H_l    = get_cell(H,i,j-1,ny,nx);
	float bed_l  = get_cell(bed,i,j-1,ny,nx);
	float bed_c  = get_cell(bed,i,j,ny,nx);
	float beta_l = get_cell(beta,i,j-1,ny,nx);
	float beta_c = get_cell(beta,i,j,ny,nx);
	TauBxJacobian tau_bx_l = get_tau_bx_jac({u_l,H_l,H_c,bed_l,bed_c,beta_l,beta_c,0.001f});
	J[0] += tau_bx_l.d_u;
	J[4] += tau_bx_l.d_H_r;
	}

	// Basal shear stress for right momentum
	{    
	float H_r    = get_cell(H,i,j+1,ny,nx);
	float bed_c  = get_cell(bed,i,j,ny,nx);
	float bed_r  = get_cell(bed,i,j+1,ny,nx);
	float beta_c = get_cell(beta,i,j,ny,nx);
	float beta_r = get_cell(beta,i,j+1,ny,nx);
	TauBxJacobian tau_bx_r = get_tau_bx_jac({u_r,H_c,H_r,bed_c,bed_r,beta_c,beta_r,0.001f});
	J[6] += tau_bx_r.d_u;
	J[9] += tau_bx_r.d_H_l;
	}

	// Basal shear stress for top momentum
	{    
	float H_t    = get_cell(H,i-1,j,ny,nx);
	float bed_t  = get_cell(bed,i-1,j,ny,nx);
	float bed_c  = get_cell(bed,i,j,ny,nx);
	float beta_t = get_cell(beta,i-1,j,ny,nx);
	float beta_c = get_cell(beta,i,j,ny,nx);
	TauByJacobian tau_by_t = get_tau_by_jac({v_t,H_t,H_c,bed_t,bed_c,beta_t,beta_c,0.001f});
	J[12] += tau_by_t.d_v;
	J[14] += tau_by_t.d_H_b;
	}

	// Basal shear stress for bottom momentum
	{    
	float H_b    = get_cell(H,i+1,j,ny,nx);
	float bed_c  = get_cell(bed,i,j,ny,nx);
	float bed_b  = get_cell(bed,i+1,j,ny,nx);
	float beta_c = get_cell(beta,i,j,ny,nx);
	float beta_b = get_cell(beta,i+1,j,ny,nx);
	TauByJacobian tau_by_b = get_tau_by_jac({v_b,H_c,H_b,bed_c,bed_b,beta_c,beta_b,0.001f});
	J[18] += tau_by_b.d_v;
	J[19] += tau_by_b.d_H_t;
	}
	
	// Driving stress for left momentum (u)
	{
	float H_l    = get_cell(H,i,j-1,ny,nx);
	float bed_l  = get_cell(bed,i,j-1,ny,nx);
	float bed_c  = get_cell(bed,i,j,ny,nx);
	TauDxJacobian tau_dx_l = get_tau_dx_jac({H_l,H_c,bed_l,bed_c},dx_inv,i,j,ny,nx);
	J[4] -= tau_dx_l.d_H_r;
	}

	// Driving stress for right momentum (u)
	{
	float H_r    = get_cell(H,i,j+1,ny,nx);
	float bed_c  = get_cell(bed,i,j,ny,nx);
	float bed_r  = get_cell(bed,i,j+1,ny,nx);
	TauDxJacobian tau_dx_r = get_tau_dx_jac({H_c,H_r,bed_c,bed_r},dx_inv,i,j+1,ny,nx);
	J[9] -= tau_dx_r.d_H_l;
	}

	// Driving stress for top momentum (v)
	{
	float H_t    = get_cell(H,i-1,j,ny,nx);
	float bed_t  = get_cell(bed,i-1,j,ny,nx);
	float bed_c  = get_cell(bed,i,j,ny,nx);
	TauDyJacobian tau_dy_t = get_tau_dy_jac({H_t,H_c,bed_t,bed_c},dx_inv,i,j,ny,nx);
	J[14] -= tau_dy_t.d_H_b;
	}

	// Driving stress for bottom momentum (v)
	{
	float H_b    = get_cell(H,i+1,j,ny,nx);
	float bed_c  = get_cell(bed,i,j,ny,nx);
	float bed_b  = get_cell(bed,i+1,j,ny,nx);
	TauDyJacobian tau_dy_b = get_tau_dy_jac({H_c,H_b,bed_c,bed_b},dx_inv,i+1,j,ny,nx);
	J[19] -= tau_dy_b.d_H_t;
	}

        float J_T[25];
        #pragma unroll
        for(int r=0; r<5; ++r) {
            #pragma unroll
            for(int c=0; c<5; ++c) {
                J_T[r*5 + c] = J[c*5 + r];
            }
        }

	float rhs[5];
        rhs[0] = get_vfacet(r_adj_u, i, j, ny, nx);
        rhs[1] = get_vfacet(r_adj_u, i, j+1, ny, nx);
        rhs[2] = get_hfacet(r_adj_v, i, j, ny, nx);
        rhs[3] = get_hfacet(r_adj_v, i+1, j, ny, nx);
        rhs[4] = get_cell(r_adj_H, i, j, ny, nx);	

	float delta_lambda[5] = {0};
	lu_5x5_solve(J_T,rhs,delta_lambda);

	float l_u_l = get_vfacet(lambda_u, i, j, ny, nx);
	float l_u_r = get_vfacet(lambda_u, i, j + 1, ny, nx);
	float l_v_t = get_hfacet(lambda_v, i, j, ny, nx);
	float l_v_b = get_hfacet(lambda_v, i + 1, j, ny, nx);
	float l_H_c = get_cell(lambda_H, i, j, ny, nx);

        //if(i==10 && j==10){printf("%f,%f,%f,%f,%f\n",lambda_new[0],lambda_new[1],lambda_new[2],lambda_new[3],lambda_new[4]);}

	lambda_u_out[i * (nx + 1) + j]     = l_u_l + 0.5f*omega*delta_lambda[0];
	lambda_u_out[i * (nx + 1) + j + 1] = l_u_r + 0.5f*omega*delta_lambda[1];
	lambda_v_out[i * nx + j]           = l_v_t + 0.5f*omega*delta_lambda[2];
	lambda_v_out[(i + 1) * nx + j ]    = l_v_b + 0.5f*omega*delta_lambda[3];
	lambda_H_out[i * nx + j]           = l_H_c +      omega*delta_lambda[4];
    }
}


extern "C" __global__
void compute_grad_beta(
    float* __restrict__ grad_beta,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ lambda_u,
    const float* __restrict__ lambda_v,
    const float* __restrict__ lambda_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ mask,
    const float* __restrict__ gamma,
    float n, float eps_reg, 
    float dx, float dt,
    int ny, int nx, int stride, int halo) 
{
    const int bny = 16;
    const int bnx = 16;

    int bi = threadIdx.y;
    int bj = threadIdx.x;

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    bool has_u    = i >= 0 && i <  ny && j >= 0 && j <= nx;
    bool has_v    = i >= 0 && i <= ny && j >= 0 && j <  nx;

    if ( is_active ) {
	float dx_inv = 1.0f/dx;

	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	
	if (has_u){

            float u_l    = get_vfacet(u,i,j,ny,nx);
            float lambda_u_l    = get_vfacet(lambda_u,i,j,ny,nx);
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    TauBxJacobian j_tau_bx = get_tau_bx_jac({u_l,H_l,H_c,bed_l,bed_c,beta_l,beta_c,0.001f});

	    if (j>0     )  {atomicAdd(&grad_beta[i * nx + j - 1],lambda_u_l * j_tau_bx.d_beta_l);}
	    if (j<(nx-1))  {atomicAdd(&grad_beta[i * nx + j]    ,lambda_u_l * j_tau_bx.d_beta_r);}

 	}

	if (has_v){

	    float v_t = get_hfacet(v,i,j,ny,nx);
	    float lambda_v_t = get_hfacet(lambda_v,i,j,ny,nx);
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_t = get_cell(bed,i-1,j,ny,nx);
	    float bed_c = get_cell(bed,i,j,ny,nx);
	    float beta_t = get_cell(beta,i-1,j,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);

	    TauByJacobian j_tau_by = get_tau_by_jac({v_t,H_t,H_c,bed_t,bed_c,beta_t,beta_c,0.001f});
	    
	    if (i>0     ) {atomicAdd(&grad_beta[(i-1) * nx + j],lambda_v_t * j_tau_by.d_beta_t);}
	    if (i<(ny-1)) {atomicAdd(&grad_beta[i * nx + j]    ,lambda_v_t * j_tau_by.d_beta_b);}
	}
    }
}


extern "C" __global__
void vanka_smooth_local(
    float* __restrict__ u_out,
    float* __restrict__ v_out,
    float* __restrict__ H_out,
    float* __restrict__ mask,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ f_u,
    const float* __restrict__ f_v,
    const float* __restrict__ f_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float n, float eps_reg, 
    float dx, float dt,
    int ny, int nx, int stride, int halo,
    int color, int n_newton, float omega
    ) 
{
    const int bny = 16;
    const int bnx = 16;

    int bi = threadIdx.y;
    int bj = threadIdx.x;

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    if (i < 0 || i >= ny || j<0 || j >= nx) return;
    
    bool my_cell_bad = false;
    if( get_cell(mask,i,j,ny,nx) > 0.5f ) my_cell_bad = true;
    __shared__ bool block_is_active;
    if (bi == 0 && bj == 0) block_is_active = false;
    __syncthreads();
    if (my_cell_bad) block_is_active = true;
    __syncthreads();

    if (!block_is_active) return;
    
    __shared__ float eta_local[bny][bnx];

    populate_viscosity(eta_local, bi, bj, i, j, u, v, B, n, eps_reg, dx, ny, nx);

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active && ((i + j) % 2 == color)) {
	float dx_inv = 1.0f/dx;

	float u_l = get_vfacet(u, i, j, ny, nx);
	float u_r = get_vfacet(u, i, j + 1, ny, nx);
	float v_t = get_hfacet(v, i, j, ny, nx);
	float v_b = get_hfacet(v, i + 1, j, ny, nx);
	float H_c = get_cell(H, i, j, ny, nx);
	float thklim = get_cell(gamma,i,j,ny,nx);
	float rnorm = 0.0f;

        for (int k = 0; k < n_newton; k++) {
	    float J[25] = {0};
	    float r[5] = {0};

	    r[0] -= get_vfacet(f_u,i,j,ny,nx);
	    r[1] -= get_vfacet(f_u,i,j+1,ny,nx);
	    r[2] -= get_hfacet(f_v,i,j,ny,nx);
	    r[3] -= get_hfacet(f_v,i+1,j,ny,nx);
	    r[4] -= get_hfacet(f_H,i,j,ny,nx);

	    // Mass Conservation Assembly
	    {
	    // Standard Mass Conservation: dH/dt + div(q) - smb = 0
	    //float H_prev_c = get_cell(H_prev, i, j, ny, nx);
	    //float smb_c    = get_cell(smb, i, j, ny, nx);
	    
	    J[24] = 1.0f / dt;
	    r[4] += H_c/dt;// H_prev/dt - smb handled by f_H - (H_c - H_prev_c) / dt - smb_c;

	    
	    float bed_c = get_cell(bed,i,j,ny,nx);
	    CellCalvingJacobian j_calve = get_cell_calving_jac({H_c,bed_c,1.0f},i, j, ny, nx);
	    J[24] -= j_calve.d_H;
	    r[4] -= j_calve.res;

	    // X-Fluxes
	    float H_l = get_cell(H,i,j-1,ny,nx);
	    HorizontalFluxJacobian j_l = get_horizontal_flux_jac({u_l, H_l, H_c}, i, j, ny, nx);
	    J[20] -= j_l.d_u   * dx_inv;
	    J[24] -= j_l.d_H_r * dx_inv;
	    r[4]  -= j_l.res   * dx_inv;

	    float H_r = get_cell(H,i,j+1,ny,nx);
	    HorizontalFluxJacobian j_r = get_horizontal_flux_jac({u_r, H_c, H_r}, i, j+1, ny, nx);
	    J[21] += j_r.d_u   * dx_inv;
	    J[24] += j_r.d_H_l * dx_inv;
	    r[4]  += j_r.res   * dx_inv;

	    // Y-Fluxes (Vertical in grid coordinates)
	    float H_t = get_cell(H,i-1,j,ny,nx);
	    VerticalFluxJacobian j_t = get_vertical_flux_jac({v_t, H_t, H_c}, i, j, ny, nx);
	    J[22] += j_t.d_v   * dx_inv;
	    J[24] += j_t.d_H_b * dx_inv;
	    r[4]  += j_t.res   * dx_inv;

	    float H_b = get_cell(H,i+1,j,ny,nx);
	    VerticalFluxJacobian j_b = get_vertical_flux_jac({v_b, H_c, H_b}, i+1, j, ny, nx);
	    J[23] -= j_b.d_v   * dx_inv;
	    J[24] -= j_b.d_H_t * dx_inv;
	    r[4]  -= j_b.res   * dx_inv;

	    if ((H_c - dt*r[4]) <= thklim) {
		// Active set constraint: Force H = thklim
		//masked = 1.0f;
		for(int k=0; k<5; ++k) J[20 + k] = 0.0f;
		J[24] = 1.0f;
		r[4] = H_c - thklim;
	    } else {
	        //masked = 0.0f;
	    
	    }
	    
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
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    TauBxJacobian tau_bx_l = get_tau_bx_jac({u_l,H_l,H_c,bed_l,bed_c,beta_l,beta_c,0.001f});
	    r[0] += tau_bx_l.res;
            J[0] += tau_bx_l.d_u;
	    J[4] += tau_bx_l.d_H_r;
	    }

	    // Basal shear stress for right momentum
            {    
	    float H_r    = get_cell(H,i,j+1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float bed_r  = get_cell(bed,i,j+1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    float beta_r = get_cell(beta,i,j+1,ny,nx);
	    TauBxJacobian tau_bx_r = get_tau_bx_jac({u_r,H_c,H_r,bed_c,bed_r,beta_c,beta_r,0.001f});
	    r[1] += tau_bx_r.res;
            J[6] += tau_bx_r.d_u;
	    J[9] += tau_bx_r.d_H_l;
	    }

	    // Basal shear stress for top momentum
            {    
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float bed_t  = get_cell(bed,i-1,j,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float beta_t = get_cell(beta,i-1,j,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    TauByJacobian tau_by_t = get_tau_by_jac({v_t,H_t,H_c,bed_t,bed_c,beta_t,beta_c,0.001f});
	    r[2]  += tau_by_t.res;
            J[12] += tau_by_t.d_v;
	    J[14] += tau_by_t.d_H_b;
	    }

	    // Basal shear stress for bottom momentum
            {    
	    float H_b    = get_cell(H,i+1,j,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float bed_b  = get_cell(bed,i+1,j,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    float beta_b = get_cell(beta,i+1,j,ny,nx);
	    TauByJacobian tau_by_b = get_tau_by_jac({v_b,H_c,H_b,bed_c,bed_b,beta_c,beta_b,0.001f});
	    r[3]  += tau_by_b.res;
            J[18] += tau_by_b.d_v;
	    J[19] += tau_by_b.d_H_t;
	    }
            
            // Driving stress for left momentum (u)
	    {
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    TauDxJacobian tau_dx_l = get_tau_dx_jac({H_l,H_c,bed_l,bed_c},dx_inv,i,j,ny,nx);
	    r[0] -= tau_dx_l.res;
	    J[4] -= tau_dx_l.d_H_r;
	    }

	    // Driving stress for right momentum (u)
	    {
	    float H_r    = get_cell(H,i,j+1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float bed_r  = get_cell(bed,i,j+1,ny,nx);
	    TauDxJacobian tau_dx_r = get_tau_dx_jac({H_c,H_r,bed_c,bed_r},dx_inv,i,j+1,ny,nx);
	    r[1] -= tau_dx_r.res;
	    J[9] -= tau_dx_r.d_H_l;
	    }

	    // Driving stress for top momentum (v)
	    {
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float bed_t  = get_cell(bed,i-1,j,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    TauDyJacobian tau_dy_t = get_tau_dy_jac({H_t,H_c,bed_t,bed_c},dx_inv,i,j,ny,nx);
	    r[2]  -= tau_dy_t.res;
	    J[14] -= tau_dy_t.d_H_b;
	    }

	    // Driving stress for bottom momentum (v)
	    {
	    float H_b    = get_cell(H,i+1,j,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float bed_b  = get_cell(bed,i+1,j,ny,nx);
	    TauDyJacobian tau_dy_b = get_tau_dy_jac({H_c,H_b,bed_c,bed_b},dx_inv,i+1,j,ny,nx);
	    r[3]  -= tau_dy_b.res;
	    J[19] -= tau_dy_b.d_H_t;
	    }

	    //float r_norm = 0.f;
	    //for (int q = 0; q < 5; q++) r_norm += r[q]*r[q];
	    //if (r_norm < 0.1f) break;

            J[0]  -= 1.0f;
            J[6]  -= 1.0f;
            J[12] -= 1.0f;
            J[18] -= 1.0f;
            J[24] += 1.0f;

	    float delta_x[5] = {0};
	    lu_5x5_solve(J,r,delta_x);


            float relaxation_factor = 0.5f;

	    u_l -= relaxation_factor*delta_x[0];
	    u_r -= relaxation_factor*delta_x[1];
	    v_t -= relaxation_factor*delta_x[2];
	    v_b -= relaxation_factor*delta_x[3];
	    H_c -= relaxation_factor*delta_x[4];
	    H_c = fmaxf(H_c,thklim);

        }
	
	float u_l_prev = get_vfacet(u, i, j, ny, nx);
	float u_r_prev = get_vfacet(u, i, j + 1, ny, nx);
	float v_t_prev = get_hfacet(v, i, j, ny, nx);
	float v_b_prev = get_hfacet(v, i + 1, j, ny, nx);
	float H_c_prev = get_cell(H, i, j, ny, nx);

	u_out[i * (nx + 1) + j]     += 0.5f*(u_l - u_l_prev);
	u_out[i * (nx + 1) + j + 1] += 0.5f*(u_r - u_r_prev);
	v_out[i * nx + j]           += 0.5f*(v_t - v_t_prev);
	v_out[(i + 1) * nx + j ]    += 0.5f*(v_b - v_b_prev);
	H_out[i * nx + j]           = (H_c - H_c_prev);
    }
}

