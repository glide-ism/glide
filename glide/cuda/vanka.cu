

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
    const PhysicsParams* params,
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

    populate_viscosity(eta_local, bi, bj, i, j, u, v, B, params->n, params->eps_reg, dx, ny, nx);

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    //if ( is_active && ((i + j) % 2 == color)) {
    if ( is_active ) {
	float dx_inv = 1.0f/dx;

	float masked = get_cell(mask, i, j, ny, nx);

	float u_l = get_vfacet(u, i, j, ny, nx);
	float u_r = get_vfacet(u, i, j + 1, ny, nx);
	float v_t = get_hfacet(v, i, j, ny, nx);
	float v_b = get_hfacet(v, i + 1, j, ny, nx);
	float H_c = get_cell(H, i, j, ny, nx);
	float thklim = get_cell(gamma,i,j,ny,nx);

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
	    CellCalvingJacobian j_calve = get_cell_calving_jac({H_c,bed_c,params->calving_rate,params->gl_sigmoid_c,params->gl_derivatives},i, j, ny, nx);
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
	    TauBxJacobian tau_bx_l = get_tau_bx_jac({u_l,H_l,H_c,bed_l,bed_c,beta_l,beta_c,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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
	    TauBxJacobian tau_bx_r = get_tau_bx_jac({u_r,H_c,H_r,bed_c,bed_r,beta_c,beta_r,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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
	    TauByJacobian tau_by_t = get_tau_by_jac({v_t,H_t,H_c,bed_t,bed_c,beta_t,beta_c,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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
	    TauByJacobian tau_by_b = get_tau_by_jac({v_b,H_c,H_b,bed_c,bed_b,beta_c,beta_b,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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

            J[0]  -= 1.0f;
            J[6]  -= 1.0f;
            J[12] -= 1.0f;
            J[18] -= 1.0f;
            J[24] += 1.0f;

	    float delta_x[5] = {0};
	    lu_5x5_solve(J,r,delta_x);


            float relaxation_factor = 0.25f;

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

	atomicAdd(&u_out[i * (nx + 1) + j],       0.5f*(u_l - u_l_prev));
	atomicAdd(&u_out[i * (nx + 1) + j + 1],   0.5f*(u_r - u_r_prev));
	atomicAdd(&v_out[i * nx + j],             0.5f*(v_t - v_t_prev));
	atomicAdd(&v_out[(i + 1) * nx + j ],      0.5f*(v_b - v_b_prev));
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
    const PhysicsParams* params,
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

    populate_viscosity(eta_local, bi, bj, i, j, u, v, B, params->n, params->eps_reg, dx, ny, nx);

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
	CellCalvingJacobian j_calve = get_cell_calving_jac({H_c,bed_c,params->calving_rate,params->gl_sigmoid_c,params->gl_derivatives},i, j, ny, nx);
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
	TauBxJacobian tau_bx_l = get_tau_bx_jac({u_l,H_l,H_c,bed_l,bed_c,beta_l,beta_c,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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
	TauBxJacobian tau_bx_r = get_tau_bx_jac({u_r,H_c,H_r,bed_c,bed_r,beta_c,beta_r,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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
	TauByJacobian tau_by_t = get_tau_by_jac({v_t,H_t,H_c,bed_t,bed_c,beta_t,beta_c,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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
	TauByJacobian tau_by_b = get_tau_by_jac({v_b,H_c,H_b,bed_c,bed_b,beta_c,beta_b,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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
    const PhysicsParams* params,
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

    // Check bounds but don't return yet - all threads must reach __syncthreads()
    bool in_bounds = (i >= 0 && i < ny && j >= 0 && j < nx);

    bool my_cell_bad = false;
    if (in_bounds && get_cell(mask,i,j,ny,nx) > 0.5f) my_cell_bad = true;

    __shared__ bool block_is_active;
    if (bi == 0 && bj == 0) block_is_active = false;
    __syncthreads();
    if (my_cell_bad) block_is_active = true;
    __syncthreads();

    // Now safe to return after all threads have passed the barriers
    if (!block_is_active || !in_bounds) return;

    __shared__ float eta_local[bny][bnx];

    populate_viscosity(eta_local, bi, bj, i, j, u, v, B, params->n, params->eps_reg, dx, ny, nx);

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
	    CellCalvingJacobian j_calve = get_cell_calving_jac({H_c,bed_c,params->calving_rate,params->gl_sigmoid_c,params->gl_derivatives},i, j, ny, nx);
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
	    TauBxJacobian tau_bx_l = get_tau_bx_jac({u_l,H_l,H_c,bed_l,bed_c,beta_l,beta_c,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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
	    TauBxJacobian tau_bx_r = get_tau_bx_jac({u_r,H_c,H_r,bed_c,bed_r,beta_c,beta_r,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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
	    TauByJacobian tau_by_t = get_tau_by_jac({v_t,H_t,H_c,bed_t,bed_c,beta_t,beta_c,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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
	    TauByJacobian tau_by_b = get_tau_by_jac({v_b,H_c,H_b,bed_c,bed_b,beta_c,beta_b,params->water_drag,params->gl_sigmoid_c,params->gl_derivatives});
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

            J[0]  -= 1.0f;
            J[6]  -= 1.0f;
            J[12] -= 1.0f;
            J[18] -= 1.0f;
            J[24] += 1.0f;

	    float delta_x[5] = {0};
	    lu_5x5_solve(J,r,delta_x);


            float relaxation_factor = 0.25f;

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

	atomicAdd(&u_out[i * (nx + 1) + j],       0.5f*(u_l - u_l_prev));
	atomicAdd(&u_out[i * (nx + 1) + j + 1],   0.5f*(u_r - u_r_prev));
	atomicAdd(&v_out[i * nx + j],             0.5f*(v_t - v_t_prev));
	atomicAdd(&v_out[(i + 1) * nx + j ],      0.5f*(v_b - v_b_prev));
	H_out[i * nx + j]           = (H_c - H_c_prev);
    }
}


