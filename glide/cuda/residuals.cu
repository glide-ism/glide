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
    const PhysicsParams* params,
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

    populate_viscosity(eta_local, bi, bj, i, j, u, v, B, params->n, params->eps_reg, dx, ny, nx);

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
	    CellCalvingJacobian j_calve = get_cell_calving_jac({H_c,bed_c,params->calving_rate,params->gl_sigmoid_c,params->gl_derivatives},i, j, ny, nx);
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
	    TauBxJacobian tau_bx = get_tau_bx_jac({u_l,H_l,H_c,bed_l,bed_c,beta_l,beta_c,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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

	    if (j == 0 || j == nx) {
		ru_l = get_vfacet(u,i,j,ny,nx);
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

	    TauByJacobian tau_by = get_tau_by_jac({v_t,H_t,H_c,bed_t,bed_c,beta_t,beta_c,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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

	    if (i == 0 || i == ny) {
		rv_t = get_hfacet(v,i,j,ny,nx);
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
    const PhysicsParams* params,
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

    populate_viscosity(eta_local, bi, bj, i, j, u, v, d_u, d_v, B, params->n, params->eps_reg, dx, ny, nx);

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
	    DualFloat calve = get_cell_calving_dual({H_c,bed_c,params->calving_rate,params->gl_sigmoid_c,params->gl_derivatives},i, j, ny, nx);
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
	    DualFloat tau_bx = get_tau_bx_dual({u_l,H_l,H_c,bed_l,bed_c,beta_l,beta_c,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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

	    DualFloat tau_by = get_tau_by_dual({v_t,H_t,H_c,bed_t,bed_c,beta_t,beta_c,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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
    const PhysicsParams* params,
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

    populate_viscosity(eta_local, bi, bj, i, j, u, v, lambda_u, lambda_v, B, params->n, params->eps_reg, dx, ny, nx);

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
	    CellCalvingJacobian j_calve = get_cell_calving_jac({H_c,bed_c,params->calving_rate,params->gl_sigmoid_c,params->gl_derivatives},i, j, ny, nx);
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
	    TauBxJacobian j_tau_bx = get_tau_bx_jac({u_l,H_l,H_c,bed_l,bed_c,beta_l,beta_c,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});

	    atomicAdd(&s_adj_u[bi][bj],lambda_u_l * j_tau_bx.d_u);
	    atomicAdd(&s_adj_H[bi][bj-1], lambda_u_l * j_tau_bx.d_H_l);
	    atomicAdd(&s_adj_H[bi][bj],   lambda_u_l * j_tau_bx.d_H_r);
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

	    TauByJacobian j_tau_by = get_tau_by_jac({v_t,H_t,H_c,bed_t,bed_c,beta_t,beta_c,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
	    atomicAdd(&s_adj_v[bi][bj],lambda_v_t * j_tau_by.d_v);
	    atomicAdd(&s_adj_H[bi-1][bj], lambda_v_t * j_tau_by.d_H_t);
	    atomicAdd(&s_adj_H[bi][bj],   lambda_v_t * j_tau_by.d_H_b);
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
