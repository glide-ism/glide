/*=========================================================
  ========= Vanka Smoother (Frozen Fields) ================
  =========================================================

  This smoother uses precomputed eta, beta_eff, c_eff fields
  instead of computing them from the current state.
  =========================================================*/

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
    const float* __restrict__ eta,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ alpha_u,
    const float* __restrict__ alpha_v,
    const float* __restrict__ c_eff,
    const float* __restrict__ gamma,
    float dx, float dt,
    int ny, int nx, int stride, int halo,
    int n_newton
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

    eta_local[bi][bj] = get_cell(eta,i,j,ny,nx);

    __syncthreads();

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    //if ( is_active && ((i + j) % 2 == color)) {
    if ( is_active ) {
	float dx_inv = 1.0f/dx;
	float masked = 0.0f;
	float thklim = get_cell(gamma,i,j,ny,nx);

	float u_l = get_vfacet(u, i, j, ny, nx);
	float u_r = get_vfacet(u, i, j + 1, ny, nx);
	float v_t = get_hfacet(v, i, j, ny, nx);
	float v_b = get_hfacet(v, i + 1, j, ny, nx);
	float H_c = get_cell(H, i, j, ny, nx);
	
	float c_u_l = 0.0f;
	float c_u_r = 0.0f;
	float c_v_t = 0.0f;
	float c_v_b = 0.0f;
	float c_H_c = 0.0f;

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

	    float c_eff_c = get_cell(c_eff,i,j,ny,nx);
	    CellCalvingFrozenJacobian j_calve = get_cell_calving_frozen_jac({H_c,c_eff_c},i, j, ny, nx);
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
	    float alpha_l = get_vfacet(alpha_u,i,j,ny,nx);
	    TauBxFrozenJacobian tau_bx_l = get_tau_bx_frozen_jac({u_l,alpha_l});
	    r[0] += tau_bx_l.res;
            J[0] += tau_bx_l.d_u;
	    }

	    // Basal shear stress for right momentum
            {
	    float alpha_r = get_vfacet(alpha_u,i,j+1,ny,nx);
	    TauBxFrozenJacobian tau_bx_r = get_tau_bx_frozen_jac({u_r,alpha_r});
	    r[1] += tau_bx_r.res;
            J[6] += tau_bx_r.d_u;
	    }

	    // Basal shear stress for top momentum
            {
	    float alpha_t = get_hfacet(alpha_v,i,j,ny,nx);
	    TauByFrozenJacobian tau_by_t = get_tau_by_frozen_jac({v_t,alpha_t});
	    r[2]  += tau_by_t.res;
            J[12] += tau_by_t.d_v;
	    }

	    // Basal shear stress for bottom momentum
            {
	    float alpha_b = get_hfacet(alpha_v,i+1,j,ny,nx);
	    TauByFrozenJacobian tau_by_b = get_tau_by_frozen_jac({v_b,alpha_b});
	    r[3]  += tau_by_b.res;
            J[18] += tau_by_b.d_v;
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

	    
            J[0]  -= 1.0f;
            J[6]  -= 1.0f;
            J[12] -= 1.0f;
            J[18] -= 1.0f;
            J[24] += 1.0f;
            

	    if (j == 0) {
	    	for(int k=0; k<5; ++k) J[0 + k] = 0.0f;
		J[0] = 1.0f;
		r[0] = u_l;
	    }

	    if (j == (nx - 1)) {
	    	for(int k=0; k<5; ++k) J[5 + k] = 0.0f;
		J[6] = 1.0f;
		r[1] = u_r;
	    }

	    if (i == 0) {
	    	for(int k=0; k<5; ++k) J[10 + k] = 0.0f;
		J[12] = 1.0f;
		r[2] = v_t;
	    }

	    if (i == (ny-1)) {
	    	for(int k=0; k<5; ++k) J[15 + k] = 0.0f;
		J[18] = 1.0f;
		r[3] = v_b;
	    }

	    if ((H_c - dt*r[4]) <= thklim) {
		// Active set constraint: Force H = thklim
		masked = 1.0f;
		for(int k=0; k<5; ++k) J[20 + k] = 0.0f;
		J[24] = 1.0f;
		//J[4] = 0.0f;
		//J[9] = 0.0f;
		//J[14] = 0.0f;
		//J[19] = 0.0f;
		r[4] = H_c - thklim;
	    } else {
	        masked = 0.0f;
	    }

	    float delta_x[5] = {0};
	    lu_5x5_solve(J,r,delta_x);

            float relaxation_factor = 0.25f;

	    float y_u_l = -relaxation_factor*delta_x[0] - c_u_l;
	    float t_u_l = u_l + y_u_l;
	    c_u_l = (t_u_l - u_l) - y_u_l;
	    u_l = t_u_l;
	    
	    float y_u_r = -relaxation_factor*delta_x[1] - c_u_r;
	    float t_u_r = u_r + y_u_r;
	    c_u_r = (t_u_r - u_r) - y_u_r;
	    u_r = t_u_r;

	    float y_v_t = -relaxation_factor*delta_x[2] - c_v_t;
	    float t_v_t = v_t + y_v_t;
	    c_v_t = (t_v_t - v_t) - y_v_t;
	    v_t = t_v_t;
	    
	    float y_v_b = -relaxation_factor*delta_x[3] - c_v_b;
	    float t_v_b = v_b + y_v_b;
	    c_v_b = (t_v_b - v_b) - y_v_b;
	    v_b = t_v_b;

	    float y_H_c = -relaxation_factor*delta_x[4] - c_H_c;
	    float t_H_c = H_c + y_H_c;
	    c_H_c = (t_H_c - H_c) - y_H_c;
	    H_c = t_H_c;

	    //u_l -= relaxation_factor*delta_x[0];
	    //u_r -= relaxation_factor*delta_x[1];
	    //v_t -= relaxation_factor*delta_x[2];
	    //v_b -= relaxation_factor*delta_x[3];
	    //H_c -= relaxation_factor*delta_x[4];
	    H_c = fmaxf(H_c,thklim);

        }
	
	float u_l_prev = get_vfacet(u, i, j, ny, nx);
	float u_r_prev = get_vfacet(u, i, j + 1, ny, nx);
	float v_t_prev = get_hfacet(v, i, j, ny, nx);
	float v_b_prev = get_hfacet(v, i + 1, j, ny, nx);
	float H_c_prev = get_cell(H, i, j, ny, nx);

	atomicAdd(&u_out[i * (nx + 1) + j],       0.5f*(u_l - u_l_prev));
	atomicAdd(&u_out[i * (nx + 1) + j + 1],   0.5f*(u_r - u_r_prev));
	atomicAdd(&v_out[i * nx + j],             0.5f*(v_t - v_t_prev));
	atomicAdd(&v_out[(i + 1) * nx + j ],      0.5f*(v_b - v_b_prev));
	H_out[i * nx + j] =                             (H_c - H_c_prev);
	mask[i * nx + j] =                                        masked;
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
    const float* __restrict__ eta,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ alpha_u,
    const float* __restrict__ alpha_v,
    const float* __restrict__ c_eff,
    const float* __restrict__ gamma,
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

    eta_local[bi][bj] = get_cell(eta,i,j,ny,nx);

    __syncthreads();

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

	float J[25] = {0};

	// Mass Conservation Assembly
	{
	// Standard Mass Conservation: dH/dt + div(q) - smb = 0
	//float H_prev_c = get_cell(H_prev, i, j, ny, nx);
	//float smb_c    = get_cell(smb, i, j, ny, nx);

	J[24] = 1.0f / dt;

	float c_eff_c = get_cell(c_eff,i,j,ny,nx);
	CellCalvingFrozenJacobian j_calve = get_cell_calving_frozen_jac({H_c,c_eff_c},i, j, ny, nx);
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
	float alpha_l = get_vfacet(alpha_u,i,j,ny,nx);
	TauBxFrozenJacobian tau_bx_l = get_tau_bx_frozen_jac({u_l,alpha_l});
	J[0] += tau_bx_l.d_u;
	}

	// Basal shear stress for right momentum
	{
	float alpha_r = get_vfacet(alpha_u,i,j+1,ny,nx);
	TauBxFrozenJacobian tau_bx_r = get_tau_bx_frozen_jac({u_r,alpha_r});
	J[6] += tau_bx_r.d_u;
	}

	// Basal shear stress for top momentum
	{
	float alpha_t = get_hfacet(alpha_v,i,j,ny,nx);
	TauByFrozenJacobian tau_by_t = get_tau_by_frozen_jac({v_t,alpha_t});
	J[12] += tau_by_t.d_v;
	}

	// Basal shear stress for bottom momentum
	{
	float alpha_b = get_hfacet(alpha_v,i+1,j,ny,nx);
	TauByFrozenJacobian tau_by_b = get_tau_by_frozen_jac({v_b,alpha_b});
	J[18] += tau_by_b.d_v;
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
	if (j == 0) {
	    for(int k=0; k<5; ++k) J_T[0 + k] = 0.0f;
	    J_T[0] = 1.0f;
	}

	if (j == (nx - 1)) {
	    for(int k=0; k<5; ++k) J_T[5 + k] = 0.0f;
	    J_T[6] = 1.0f;
	}

	if (i == 0) {
	    for(int k=0; k<5; ++k) J_T[10 + k] = 0.0f;
	    J_T[12] = 1.0f;
	}

	if (i == (ny-1)) {
	    for(int k=0; k<5; ++k) J_T[15 + k] = 0.0f;
	    J_T[18] = 1.0f;
	}

	if (masked > 0.5) {
	    // Active set constraint: Force H = thklim
	    for(int k=0; k<5; ++k) J_T[20 + k] = 0.0f;
	    J_T[24] = 1.0f;
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

	lambda_u_out[i * (nx + 1) + j]     = l_u_l + 0.5f*omega*delta_lambda[0];
	lambda_u_out[i * (nx + 1) + j + 1] = l_u_r + 0.5f*omega*delta_lambda[1];
	lambda_v_out[i * nx + j]           = l_v_t + 0.5f*omega*delta_lambda[2];
	lambda_v_out[(i + 1) * nx + j ]    = l_v_b + 0.5f*omega*delta_lambda[3];
	lambda_H_out[i * nx + j]           = l_H_c +      omega*delta_lambda[4];
    }
}

extern "C" __global__
void vanka_matrix_dump(
    float* __restrict__ J_out,
    float* __restrict__ R_out,
    float* __restrict__ Delta_out,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ f_u,
    const float* __restrict__ f_v,
    const float* __restrict__ f_H,
    const float* __restrict__ eta,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ alpha_u,
    const float* __restrict__ alpha_v,
    const float* __restrict__ c_eff,
    const float* __restrict__ gamma,
    float dx, float dt,
    int ny, int nx, int stride, int halo
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

    eta_local[bi][bj] = get_cell(eta,i,j,ny,nx);

    __syncthreads();

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    //if ( is_active && ((i + j) % 2 == color)) {
    if ( is_active ) {
	float dx_inv = 1.0f/dx;
	
	float u_l = get_vfacet(u, i, j, ny, nx);
	float u_r = get_vfacet(u, i, j + 1, ny, nx);
	float v_t = get_hfacet(v, i, j, ny, nx);
	float v_b = get_hfacet(v, i + 1, j, ny, nx);
	float H_c = get_cell(H, i, j, ny, nx);
	float thklim = get_cell(gamma,i,j,ny,nx);

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


	float c_eff_c = get_cell(c_eff,i,j,ny,nx);
	CellCalvingFrozenJacobian j_calve = get_cell_calving_frozen_jac({H_c,c_eff_c},i, j, ny, nx);
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
	float alpha_l = get_vfacet(alpha_u,i,j,ny,nx);
	TauBxFrozenJacobian tau_bx_l = get_tau_bx_frozen_jac({u_l,alpha_l});
	r[0] += tau_bx_l.res;
	J[0] += tau_bx_l.d_u;
	}

	// Basal shear stress for right momentum
	{
	float alpha_r = get_vfacet(alpha_u,i,j+1,ny,nx);
	TauBxFrozenJacobian tau_bx_r = get_tau_bx_frozen_jac({u_r,alpha_r});
	r[1] += tau_bx_r.res;
	J[6] += tau_bx_r.d_u;
	}

	// Basal shear stress for top momentum
	{
	float alpha_t = get_hfacet(alpha_v,i,j,ny,nx);
	TauByFrozenJacobian tau_by_t = get_tau_by_frozen_jac({v_t,alpha_t});
	r[2]  += tau_by_t.res;
	J[12] += tau_by_t.d_v;
	}

	// Basal shear stress for bottom momentum
	{
	float alpha_b = get_hfacet(alpha_v,i+1,j,ny,nx);
	TauByFrozenJacobian tau_by_b = get_tau_by_frozen_jac({v_b,alpha_b});
	r[3]  += tau_by_b.res;
	J[18] += tau_by_b.d_v;
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

	if (j == 0) {
	    for(int k=0; k<5; ++k) J[0 + k] = 0.0f;
	    J[0] = 1.0f;
	    r[0] = u_l;
	}

	if (j == (nx - 1)) {
	    for(int k=0; k<5; ++k) J[5 + k] = 0.0f;
	    J[6] = 1.0f;
	    r[1] = u_r;
	}

	if (i == 0) {
	    for(int k=0; k<5; ++k) J[10 + k] = 0.0f;
	    J[12] = 1.0f;
	    r[2] = v_t;
	}

	if (i == (ny-1)) {
	    for(int k=0; k<5; ++k) J[15 + k] = 0.0f;
	    J[18] = 1.0f;
	    r[3] = v_b;
	}

	if ((H_c - dt*r[4]) <= thklim) {
	    // Active set constraint: Force H = thklim
	    for(int k=0; k<5; ++k) J[20 + k] = 0.0f;
	    J[24] = 1.0f;
	    r[4] = H_c - thklim;
	} 

    for(int k=0;k<25;k++){
	J_out[25*(i * nx + j) + k] = J[k];
    }
	R_out[5*(i*nx+j) + 0] = r[0];
	R_out[5*(i*nx+j) + 1] = r[1];
	R_out[5*(i*nx+j) + 2] = r[2];
	R_out[5*(i*nx+j) + 3] = r[3];
	R_out[5*(i*nx+j) + 4] = r[4];

	float delta_x[5] = {0};
	lu_5x5_solve(J,r,delta_x);

	Delta_out[5*(i*nx+j) + 0] = delta_x[0];
	Delta_out[5*(i*nx+j) + 1] = delta_x[1];
	Delta_out[5*(i*nx+j) + 2] = delta_x[2];
	Delta_out[5*(i*nx+j) + 3] = delta_x[3];
	Delta_out[5*(i*nx+j) + 4] = delta_x[4];



    }
    
}


