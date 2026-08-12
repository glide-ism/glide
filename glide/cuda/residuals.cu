/*=========================================================
  ================= Residual Computation ==================
  =========================================================*/

extern "C" __global__
void compute_residual(
    float* __restrict__ r_u,
    float* __restrict__ r_v,
    float* __restrict__ r_ud,
    float* __restrict__ r_vd,
    float* __restrict__ r_H,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ ud,
    const float* __restrict__ vd,
    const float* __restrict__ H,
    const float* __restrict__ phi,
    const float* __restrict__ xi,
    const float* __restrict__ mask,
    const float* __restrict__ f_u,
    const float* __restrict__ f_v,
    const float* __restrict__ f_ud,
    const float* __restrict__ f_vd,
    const float* __restrict__ f_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    bool use_forcing, bool use_mask, bool ssa,
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

    if (i > ny || j > nx) return;

    populate_viscosity(eta_local, bi, bj, i, j, u, v, ud, vd, H, B, n, eps_reg, H_reg, dx, ny, nx);

    __syncthreads();

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active ) {
	float dx_inv = 1.0f/dx;
	bool has_cell = i >= 0 && i <  ny && j >= 0 && j <  nx;
	bool has_u    = i >= 0 && i <  ny && j >= 0 && j <= nx;
	bool has_v    = i >= 0 && i <= ny && j >= 0 && j <  nx;

	if (has_cell){

	    float H_c        = get_cell(H,i,j,ny,nx);
	    float phi_c = get_cell(phi,i,j,ny,nx);

	    float rH = H_c/dt;
            if (use_forcing) rH -= get_cell(f_H,i,j,ny,nx);

	    float H_l = get_cell(H,i,j-1,ny,nx);
	    float u_l = get_vfacet(u,i,j,ny,nx);
	    HorizontalFluxJacobian j_l = get_horizontal_flux_jac({u_l,H_l,H_c}, i, j, ny, nx);
	    rH -= j_l.res*dx_inv;
	    
	    float phi_l = get_cell(phi,i,j-1,ny,nx);
	    FacetCalvingJacobian j_calve_l = get_facet_calving_jac({H_c,H_l,phi_c,phi_l,calving_rate,flotation_reg_calving},i,j,ny,nx);
	    rH += j_calve_l.res*dx_inv;

	    float H_r = get_cell(H,i,j+1,ny,nx);
	    float u_r = get_vfacet(u,i,j+1,ny,nx);
	    HorizontalFluxJacobian j_r = get_horizontal_flux_jac({u_r,H_c,H_r}, i, j + 1, ny, nx);
            rH += j_r.res*dx_inv;
	    
	    float phi_r = get_cell(phi,i,j+1,ny,nx);
	    FacetCalvingJacobian j_calve_r = get_facet_calving_jac({H_c,H_r,phi_c,phi_r,calving_rate,flotation_reg_calving},i,j+1,ny,nx);
	    rH += j_calve_r.res*dx_inv;

	    float H_t = get_cell(H,i-1,j,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    VerticalFluxJacobian j_t = get_vertical_flux_jac({v_t,H_t,H_c}, i, j, ny, nx);
	    rH += j_t.res*dx_inv;

	    float phi_t = get_cell(phi,i-1,j,ny,nx);
	    FacetCalvingJacobian j_calve_t = get_facet_calving_jac({H_c,H_t,phi_c,phi_t,calving_rate,flotation_reg_calving},i,j,ny,nx);
	    rH += j_calve_t.res*dx_inv;


	    float H_b = get_cell(H,i+1,j,ny,nx);
	    float v_b = get_hfacet(v,i+1,j,ny,nx);
	    VerticalFluxJacobian j_b = get_vertical_flux_jac({v_b,H_c,H_b}, i + 1, j, ny, nx);
            rH -= j_b.res*dx_inv;


	    float phi_b = get_cell(phi,i+1,j,ny,nx);
	    FacetCalvingJacobian j_calve_b = get_facet_calving_jac({H_c,H_b,phi_c,phi_b,calving_rate,flotation_reg_calving},i+1,j,ny,nx);
	    rH += j_calve_b.res*dx_inv;

	    float masked = use_mask ? get_cell(mask,i,j,ny,nx) : 0.0f;
	    float thklim = get_cell(gamma,i,j,ny,nx);
            r_H[i * nx + j] = (1.0f - masked) * rH + masked * (H_c - thklim);
	}

	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	
	if (has_u){

            float K_1 = 1.0f / (2.0f * n + 3.0f);
            float S_1 = (n + 2.0f) * (n + 2.0f) / (2.0f * n + 1.0f);
            float c_1 = __powf(1.0f / (n + 2.0f),2.0f * n / (n + 1.0f)) * (2.0f * n + 1.0f);

	    float ru_l = 0.0f;
            float rud_l = 0.0f;
	    if (use_forcing) {
                ru_l -= get_vfacet(f_u,i,j,ny,nx);
                rud_l -= get_vfacet(f_ud,i,j,ny,nx);
            }

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

            float ud_l = get_vfacet(ud,i,j,ny,nx);
	    float ud_r = get_vfacet(ud,i,j+1,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);
	    float vd_b = get_hfacet(vd,i+1,j,ny,nx);
            SigmaNormalJacobian sigmad_xx_c = get_sigma_xx_jac({ud_l,ud_r,vd_t,vd_b,eta_H_c.res},dx_inv,i,j,ny,nx);
            
	    rud_l += K_1 * sigmad_xx_c.res * dx_inv;
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

            float ud_l    = get_vfacet(ud,i,j,ny,nx);
	    float ud_ll   = get_vfacet(ud,i,j-1,ny,nx);
	    float vd_lt   = get_hfacet(vd,i,j-1,ny,nx);
	    float vd_lb   = get_hfacet(vd,i+1,j-1,ny,nx);
            SigmaNormalJacobian sigmad_xx_l = get_sigma_xx_jac({ud_ll,ud_l,vd_lt,vd_lb,eta_H_l.res},dx_inv,i,j - 1,ny,nx);

	    rud_l -= K_1 * sigmad_xx_l.res * dx_inv;

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

	    float ud_tl = get_vfacet(ud,i-1,j,ny,nx);
	    float ud_l = get_vfacet(ud,i,j,ny,nx);
	    float vd_lt = get_hfacet(vd,i,j-1,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);
	    
	    SigmaShearJacobian sigmad_xy_tl = get_sigma_xy_jac({ud_tl,ud_l,vd_lt,vd_t,eta_H_tl.res},dx_inv,i,j,ny,nx);

	    rud_l += K_1 * sigmad_xy_tl.res * dx_inv;


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

	    float ud_l    = get_vfacet(ud,i,j,ny,nx);
	    float ud_bl   = get_vfacet(ud,i+1,j,ny,nx);
	    float vd_lb   = get_hfacet(vd,i+1,j-1,ny,nx);
	    float vd_b    = get_hfacet(vd,i+1,j,ny,nx);
	    SigmaShearJacobian sigmad_xy_bl = get_sigma_xy_jac({ud_l,ud_bl,vd_lb,vd_b,eta_H_bl.res},dx_inv,i + 1,j,ny,nx);
    
	    rud_l -= K_1 * sigmad_xy_bl.res * dx_inv;

	    }

            {
            float eta_l = eta_local[bi][bj-1];
            float eta_c = eta_local[bi][bj];
            float H_l = get_cell(H,i,j-1,ny,nx);
            float H_c = get_cell(H,i,j,ny,nx);
            float ud_l = get_vfacet(ud,i,j,ny,nx);
            SigmaVertXZJacobian sigmad_xz_l = get_sigma_xz_jac({ud_l,eta_l,eta_c,H_l,H_c},c_1,S_1,H_reg,i,j,ny,nx);
            
            rud_l += sigmad_xz_l.res;
            }
	
            {    
            float u_l    = get_vfacet(u,i,j,ny,nx);
            float ud_l   = get_vfacet(ud,i,j,ny,nx);
            float ub_l   = u_l - ud_l;

            float u_ll   = get_vfacet(u,i,j-1,ny,nx);
            float ud_ll  = get_vfacet(ud,i,j-1,ny,nx);
            float ub_ll  = u_ll - ud_ll; 

            float u_r    = get_vfacet(u,i,j+1,ny,nx);
            float ud_r   = get_vfacet(ud,i,j+1,ny,nx);
            float ub_r   = u_r - ud_r;
 
            float v_tl   = get_hfacet(v,i,j-1,ny,nx);
            float vd_tl  = get_hfacet(vd,i,j-1,ny,nx);
	    float vb_tl  = v_tl - vd_tl;

            float v_tr   = get_hfacet(v,i,j,ny,nx);
            float vd_tr  = get_hfacet(vd,i,j,ny,nx);
            float vb_tr  = v_tr - vd_tr;
	    
            float v_bl   = get_hfacet(v,i+1,j-1,ny,nx);
            float vd_bl  = get_hfacet(vd,i+1,j-1,ny,nx);
            float vb_bl  = v_bl - vd_bl;

	    float v_br   = get_hfacet(v,i+1,j,ny,nx);
	    float vd_br  = get_hfacet(vd,i+1,j,ny,nx);
            float vb_br  = v_br - vd_br;

	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float phi_l = get_cell(phi,i,j-1,ny,nx);
	    float phi_c = get_cell(phi,i,j,ny,nx);
	    float xi_l = get_cell(xi,i,j-1,ny,nx);
	    float xi_c = get_cell(xi,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    TauBxJacobian tau_bx = get_tau_bx_jac({ub_l,ub_ll,ub_r,vb_tl,vb_tr,vb_bl,vb_br,H_l,H_c,xi_l,xi_c,beta_l,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
	    ru_l += tau_bx.res;
            rud_l -= tau_bx.res;
	    }

	    {
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float phi_l = get_cell(phi,i,j-1,ny,nx);
	    float phi_c = get_cell(phi,i,j,ny,nx);
	    TauDxJacobian tau_dx = get_tau_dx_jac({H_l,H_c,bed_l,bed_c,phi_l,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);
	    ru_l -= tau_dx.res;
	    }

	    if (j == 0 || j == nx) {
		ru_l = get_vfacet(u,i,j,ny,nx);
	    }
	    // SSA mode pins the deformational component everywhere:
	    // every ud row is an identity row (see common.cu convention)
	    if (ssa || j == 0 || j == nx) {
                rud_l = get_vfacet(ud,i,j,ny,nx);
	    }
	    r_u[i * (nx + 1) + j] = ru_l;
            r_ud[i * (nx + 1) + j] = rud_l;
	}

	if (has_v){
            float K_1 = 1.0f / (2.0f * n + 3.0f);
            float S_1 = (n + 2.0f) * (n + 2.0f) / (2.0f * n + 1.0f);
            float c_1 = __powf(1.0f / (n + 2.0f),2.0f * n / (n + 1.0f)) * (2.0f * n + 1.0f);

	    float rv_t = 0.0f;
            float rvd_t = 0.0f;
	    if (use_forcing) {
                rv_t -= get_hfacet(f_v,i,j,ny,nx);
                rvd_t -= get_hfacet(f_vd,i,j,ny,nx);
            }

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

	    float ud_tl = get_vfacet(ud,i-1,j,ny,nx);
	    float ud_tr = get_vfacet(ud,i-1,j+1,ny,nx);
	    float vd_tt = get_hfacet(vd,i-1,j,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);
	    SigmaNormalJacobian sigmad_yy_t = get_sigma_yy_jac({ud_tl,ud_tr,vd_tt,vd_t,eta_H_t.res},dx_inv,i-1,j,ny,nx);
            rvd_t += K_1 * sigmad_yy_t.res * dx_inv;
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

            float ud_l = get_vfacet(ud,i,j,ny,nx);
	    float ud_r = get_vfacet(ud,i,j+1,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);
	    float vd_b = get_hfacet(vd,i+1,j,ny,nx);
            SigmaNormalJacobian sigmad_yy_c = get_sigma_yy_jac({ud_l,ud_r,vd_t,vd_b,eta_H_c.res},dx_inv,i,j,ny,nx);
	    rvd_t -= K_1 * sigmad_yy_c.res * dx_inv;
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

	    float ud_tl = get_vfacet(ud,i-1,j,ny,nx);
	    float ud_l = get_vfacet(ud,i,j,ny,nx);
	    float vd_lt = get_hfacet(vd,i,j-1,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);

	    SigmaShearJacobian sigmad_xy_tl = get_sigma_xy_jac({ud_tl,ud_l,vd_lt,vd_t,eta_H_tl.res},dx_inv,i,j,ny,nx);
	    
            rvd_t -= K_1 * sigmad_xy_tl.res * dx_inv;

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

	    float ud_tr = get_vfacet(ud,i-1,j+1,ny,nx);
	    float ud_r = get_vfacet(ud,i,j+1,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);
	    float vd_rt = get_hfacet(vd,i,j+1,ny,nx);
	    SigmaShearJacobian sigmad_xy_tr = get_sigma_xy_jac({ud_tr,ud_r,vd_t,vd_rt,eta_H_tr.res},dx_inv,i,j+1,ny,nx);
	    rvd_t += K_1 * sigmad_xy_tr.res * dx_inv;

	    }
 
            
             {
            float eta_t = eta_local[bi-1][bj];
            float eta_b = eta_local[bi][bj];
            float H_t = get_cell(H,i-1,j,ny,nx);
            float H_b = get_cell(H,i,j,ny,nx);
            float vd_t = get_hfacet(vd,i,j,ny,nx);
            SigmaVertYZJacobian sigmad_yz_l = get_sigma_yz_jac({vd_t,eta_t,eta_b,H_t,H_b},c_1,S_1,H_reg,i,j,ny,nx);
            
            rvd_t += sigmad_yz_l.res;
            }
            

	    {
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);
            float vb_t = v_t - vd_t;

	    float v_tt = get_hfacet(v,i-1,j,ny,nx);
	    float vd_tt = get_hfacet(vd,i-1,j,ny,nx);
            float vb_tt = v_tt - vd_tt;

	    float v_b = get_hfacet(v,i+1,j,ny,nx);
	    float vd_b = get_hfacet(vd,i+1,j,ny,nx);
            float vb_b = v_b - vd_b;

            float u_tl = get_vfacet(u,i-1,j,ny,nx);
            float ud_tl = get_vfacet(ud,i-1,j,ny,nx);
            float ub_tl = u_tl - ud_tl;

            float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
            float ud_tr = get_vfacet(ud,i-1,j+1,ny,nx);
            float ub_tr = u_tr - ud_tr;
 
            float u_bl = get_vfacet(u,i,j,ny,nx);
            float ud_bl = get_vfacet(ud,i,j,ny,nx);
            float ub_bl = u_bl - ud_bl;

            float u_br = get_vfacet(u,i,j+1,ny,nx);
            float ud_br = get_vfacet(ud,i,j+1,ny,nx);
            float ub_br = u_br - ud_br;

	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float phi_t = get_cell(phi,i-1,j,ny,nx);
	    float phi_c = get_cell(phi,i,j,ny,nx);
	    float xi_t = get_cell(xi,i-1,j,ny,nx);
	    float xi_c = get_cell(xi,i,j,ny,nx);
	    float beta_t = get_cell(beta,i-1,j,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);

	    TauByJacobian tau_by = get_tau_by_jac({vb_t,vb_tt,vb_b,ub_tl,ub_tr,ub_bl,ub_br,H_t,H_c,xi_t,xi_c,beta_t,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
	    rv_t += tau_by.res;
            rvd_t -= tau_by.res;
	    }

	    {
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_t = get_cell(bed,i-1,j,ny,nx);
	    float bed_c = get_cell(bed,i,j,ny,nx);
	    float phi_t = get_cell(phi,i-1,j,ny,nx);
	    float phi_c = get_cell(phi,i,j,ny,nx);

	    TauDyJacobian tau_dy = get_tau_dy_jac({H_t,H_c,bed_t,bed_c,phi_t,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);
	    rv_t -= tau_dy.res;
	    }

	    if (i == 0 || i == ny) {
		rv_t = get_hfacet(v,i,j,ny,nx);
	    }
	    if (ssa || i == 0 || i == ny) {
                rvd_t = get_hfacet(vd,i,j,ny,nx);
	    }

	    r_v[i * nx + j] = rv_t;
            r_vd[i * nx + j] = rvd_t;
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
    float* __restrict__ jvp_ud,
    float* __restrict__ jvp_vd,
    float* __restrict__ jvp_H,
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
    const float* __restrict__ phi,
    const float* __restrict__ xi,
    const float* __restrict__ mask,
    const float* __restrict__ f_u,
    const float* __restrict__ f_v,
    const float* __restrict__ f_ud,
    const float* __restrict__ f_vd,
    const float* __restrict__ f_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    bool use_mask, bool ssa,
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

    __shared__ DualFloat eta_local[bny][bnx];

    if (i > ny || j > nx) return;

    populate_viscosity(eta_local, bi, bj, i, j, u, v, ud, vd, H, d_u, d_v, d_ud, d_vd, d_H, B, n, eps_reg, H_reg, dx, ny, nx);

    __syncthreads();
    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active ) {
	float dx_inv = 1.0f/dx;
	bool has_cell = i >= 0 && i <  ny && j >= 0 && j <  nx;
	bool has_u    = i >= 0 && i <  ny && j >= 0 && j <= nx;
	bool has_v    = i >= 0 && i <= ny && j >= 0 && j <  nx;

	if (has_cell){
	    DualFloat H_c      = get_cell(H,d_H,i,j,ny,nx);
	    float phi_c        = get_cell(phi,i,j,ny,nx);

	    float d_rH = H_c.d/dt;

	    float bed_c = get_cell(bed,i,j,ny,nx);

	    DualFloat H_l = get_cell(H,d_H,i,j-1,ny,nx);
	    DualFloat u_l = get_vfacet(u,d_u,i,j,ny,nx);
	    DualFloat q_l = get_horizontal_flux_dual({u_l,H_l,H_c}, i, j, ny, nx);
	    d_rH -= q_l.d*dx_inv;

	    float phi_l = get_cell(phi,i,j-1,ny,nx);
	    DualFloat q_calve_l = get_facet_calving_dual({H_c,H_l,phi_c,phi_l,calving_rate,flotation_reg_calving},i,j,ny,nx);
	    d_rH += q_calve_l.d*dx_inv;

	    DualFloat H_r = get_cell(H,d_H,i,j+1,ny,nx);
	    DualFloat u_r = get_vfacet(u,d_u,i,j+1,ny,nx);
	    DualFloat q_r = get_horizontal_flux_dual({u_r,H_c,H_r}, i, j + 1, ny, nx);
            d_rH += q_r.d*dx_inv;

	    float phi_r = get_cell(phi,i,j+1,ny,nx);
	    DualFloat q_calve_r = get_facet_calving_dual({H_c,H_r,phi_c,phi_r,calving_rate,flotation_reg_calving},i,j+1,ny,nx);
	    d_rH += q_calve_r.d*dx_inv;

	    DualFloat H_t = get_cell(H,d_H,i-1,j,ny,nx);
	    DualFloat v_t = get_hfacet(v,d_v,i,j,ny,nx);
	    DualFloat q_t = get_vertical_flux_dual({v_t,H_t,H_c}, i, j, ny, nx);
	    d_rH += q_t.d*dx_inv;

	    float phi_t = get_cell(phi,i-1,j,ny,nx);
	    DualFloat q_calve_t = get_facet_calving_dual({H_c,H_t,phi_c,phi_t,calving_rate,flotation_reg_calving},i,j,ny,nx);
	    d_rH += q_calve_t.d*dx_inv;

	    DualFloat H_b = get_cell(H,d_H,i+1,j,ny,nx);
	    DualFloat v_b = get_hfacet(v,d_v,i+1,j,ny,nx);
	    DualFloat q_b = get_vertical_flux_dual({v_b,H_c,H_b}, i + 1, j, ny, nx);
            d_rH -= q_b.d*dx_inv;

	    float phi_b = get_cell(phi,i+1,j,ny,nx);
	    DualFloat q_calve_b = get_facet_calving_dual({H_c,H_b,phi_c,phi_b,calving_rate,flotation_reg_calving},i+1,j,ny,nx);
	    d_rH += q_calve_b.d*dx_inv;

	    // Identity row on the active set, mirroring compute_residual
	    float masked = use_mask ? get_cell(mask,i,j,ny,nx) : 0.0f;
            jvp_H[i * nx + j] = (1.0f - masked) * d_rH + masked * get_cell(d_H,i,j,ny,nx);

	}

	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	if (has_u){
            float K_1 = 1.0f / (2.0f * n + 3.0f);
            float S_1 = (n + 2.0f) * (n + 2.0f) / (2.0f * n + 1.0f);
            float c_1 = __powf(1.0f / (n + 2.0f),2.0f * n / (n + 1.0f)) * (2.0f * n + 1.0f);

	    float d_ru_l = 0.0f;
	    float d_rud_l = 0.0f;

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

            DualFloat ud_l = get_vfacet(ud,d_ud,i,j,ny,nx);
	    DualFloat ud_r = get_vfacet(ud,d_ud,i,j+1,ny,nx);
	    DualFloat vd_t = get_hfacet(vd,d_vd,i,j,ny,nx);
	    DualFloat vd_b = get_hfacet(vd,d_vd,i+1,j,ny,nx);
	    DualFloat sigmad_xx_c = get_sigma_xx_dual({ud_l,ud_r,vd_t,vd_b,eta_H_c},dx_inv,i,j,ny,nx);

	    d_rud_l += K_1 * sigmad_xx_c.d*dx_inv;
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

            DualFloat ud_l    = get_vfacet(ud,d_ud,i,j,ny,nx);
	    DualFloat ud_ll   = get_vfacet(ud,d_ud,i,j-1,ny,nx);
	    DualFloat vd_lt   = get_hfacet(vd,d_vd,i,j-1,ny,nx);
	    DualFloat vd_lb   = get_hfacet(vd,d_vd,i+1,j-1,ny,nx);
            DualFloat sigmad_xx_l = get_sigma_xx_dual({ud_ll,ud_l,vd_lt,vd_lb,eta_H_l},dx_inv,i,j-1,ny,nx);

	    d_rud_l -= K_1 * sigmad_xx_l.d * dx_inv;
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

	    DualFloat ud_tl = get_vfacet(ud,d_ud,i-1,j,ny,nx);
	    DualFloat ud_l = get_vfacet(ud,d_ud,i,j,ny,nx);
	    DualFloat vd_lt = get_hfacet(vd,d_vd,i,j-1,ny,nx);
	    DualFloat vd_t = get_hfacet(vd,d_vd,i,j,ny,nx);
	    DualFloat sigmad_xy_tl = get_sigma_xy_dual({ud_tl,ud_l,vd_lt,vd_t,eta_H_tl},dx_inv,i,j,ny,nx);

	    d_rud_l += K_1 * sigmad_xy_tl.d * dx_inv;
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

	    DualFloat ud_l    = get_vfacet(ud,d_ud,i,j,ny,nx);
	    DualFloat ud_bl   = get_vfacet(ud,d_ud,i+1,j,ny,nx);
	    DualFloat vd_lb   = get_hfacet(vd,d_vd,i+1,j-1,ny,nx);
	    DualFloat vd_b    = get_hfacet(vd,d_vd,i+1,j,ny,nx);
	    DualFloat sigmad_xy_bl = get_sigma_xy_dual({ud_l,ud_bl,vd_lb,vd_b,eta_H_bl},dx_inv,i + 1,j,ny,nx);

	    d_rud_l -= K_1 * sigmad_xy_bl.d * dx_inv;
	    }

            {
            DualFloat eta_l = eta_local[bi][bj-1];
            DualFloat eta_c = eta_local[bi][bj];
            DualFloat H_l = get_cell(H,d_H,i,j-1,ny,nx);
            DualFloat H_c = get_cell(H,d_H,i,j,ny,nx);
            DualFloat ud_l = get_vfacet(ud,d_ud,i,j,ny,nx);
            DualFloat sigmad_xz_l = get_sigma_xz_dual({ud_l,eta_l,eta_c,H_l,H_c},c_1,S_1,H_reg,i,j,ny,nx);

            d_rud_l += sigmad_xz_l.d;
            }

            {
            DualFloat u_l    = get_vfacet(u,d_u,i,j,ny,nx);
            DualFloat u_ll    = get_vfacet(u,d_u,i,j-1,ny,nx);
            DualFloat u_r    = get_vfacet(u,d_u,i,j+1,ny,nx);
            DualFloat v_tl   = get_hfacet(v,d_v,i,j-1,ny,nx);
	    DualFloat v_tr   = get_hfacet(v,d_v,i,j,ny,nx);
	    DualFloat v_bl   = get_hfacet(v,d_v,i+1,j-1,ny,nx);
	    DualFloat v_br   = get_hfacet(v,d_v,i+1,j,ny,nx);

            DualFloat ub_l  = u_l  - get_vfacet(ud,d_ud,i,j,ny,nx);
            DualFloat ub_ll = u_ll - get_vfacet(ud,d_ud,i,j-1,ny,nx);
            DualFloat ub_r  = u_r  - get_vfacet(ud,d_ud,i,j+1,ny,nx);
            DualFloat vb_tl = v_tl - get_hfacet(vd,d_vd,i,j-1,ny,nx);
            DualFloat vb_tr = v_tr - get_hfacet(vd,d_vd,i,j,ny,nx);
            DualFloat vb_bl = v_bl - get_hfacet(vd,d_vd,i+1,j-1,ny,nx);
            DualFloat vb_br = v_br - get_hfacet(vd,d_vd,i+1,j,ny,nx);

	    DualFloat H_l    = get_cell(H,d_H,i,j-1,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
	    float phi_l  = get_cell(phi,i,j-1,ny,nx);
	    float phi_c  = get_cell(phi,i,j,ny,nx);
	    float xi_l  = get_cell(xi,i,j-1,ny,nx);
	    float xi_c  = get_cell(xi,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    DualFloat tau_bx = get_tau_bx_dual({ub_l,ub_ll,ub_r,vb_tl,vb_tr,vb_bl,vb_br,H_l,H_c,xi_l,xi_c,beta_l,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
	    d_ru_l += tau_bx.d;
	    d_rud_l -= tau_bx.d;
	    }

	    {
	    DualFloat H_l    = get_cell(H,d_H,i,j-1,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float phi_l = get_cell(phi,i,j-1,ny,nx);
	    float phi_c = get_cell(phi,i,j,ny,nx);
	    DualFloat tau_dx = get_tau_dx_dual({H_l,H_c,bed_l,bed_c,phi_l,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);
	    d_ru_l -= tau_dx.d;
	    }

	    // Identity rows at Dirichlet facets (and all ud rows in SSA
	    // mode), mirroring compute_residual
	    if (j <= 0 || j >= nx) {
		d_ru_l = get_vfacet(d_u,i,j,ny,nx);
	    }
	    if (ssa || j <= 0 || j >= nx) {
		d_rud_l = get_vfacet(d_ud,i,j,ny,nx);
	    }
	    jvp_u[i * (nx + 1) + j] = d_ru_l;
	    jvp_ud[i * (nx + 1) + j] = d_rud_l;

 	}

	if (has_v){
            float K_1 = 1.0f / (2.0f * n + 3.0f);
            float S_1 = (n + 2.0f) * (n + 2.0f) / (2.0f * n + 1.0f);
            float c_1 = __powf(1.0f / (n + 2.0f),2.0f * n / (n + 1.0f)) * (2.0f * n + 1.0f);

	    float d_rv_t = 0.0f;
	    float d_rvd_t = 0.0f;

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

	    DualFloat ud_tl = get_vfacet(ud,d_ud,i-1,j,ny,nx);
	    DualFloat ud_tr = get_vfacet(ud,d_ud,i-1,j+1,ny,nx);
	    DualFloat vd_tt = get_hfacet(vd,d_vd,i-1,j,ny,nx);
	    DualFloat vd_t = get_hfacet(vd,d_vd,i,j,ny,nx);
	    DualFloat sigmad_yy_t = get_sigma_yy_dual({ud_tl,ud_tr,vd_tt,vd_t,eta_H_t},dx_inv,i-1,j,ny,nx);
            d_rvd_t += K_1 * sigmad_yy_t.d * dx_inv;
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

            DualFloat ud_l = get_vfacet(ud,d_ud,i,j,ny,nx);
	    DualFloat ud_r = get_vfacet(ud,d_ud,i,j+1,ny,nx);
	    DualFloat vd_t = get_hfacet(vd,d_vd,i,j,ny,nx);
	    DualFloat vd_b = get_hfacet(vd,d_vd,i+1,j,ny,nx);
            DualFloat sigmad_yy_c = get_sigma_yy_dual({ud_l,ud_r,vd_t,vd_b,eta_H_c},dx_inv,i,j,ny,nx);
	    d_rvd_t -= K_1 * sigmad_yy_c.d * dx_inv;
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

	    DualFloat ud_tl = get_vfacet(ud,d_ud,i-1,j,ny,nx);
	    DualFloat ud_l = get_vfacet(ud,d_ud,i,j,ny,nx);
	    DualFloat vd_lt = get_hfacet(vd,d_vd,i,j-1,ny,nx);
	    DualFloat vd_t = get_hfacet(vd,d_vd,i,j,ny,nx);

	    DualFloat sigmad_xy_tl = get_sigma_xy_dual({ud_tl,ud_l,vd_lt,vd_t,eta_H_tl},dx_inv,i,j,ny,nx);

	    d_rvd_t -= K_1 * sigmad_xy_tl.d * dx_inv;
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

	    DualFloat ud_tr = get_vfacet(ud,d_ud,i-1,j+1,ny,nx);
	    DualFloat ud_r  = get_vfacet(ud,d_ud,i,j+1,ny,nx);
	    DualFloat vd_t  = get_hfacet(vd,d_vd,i,j,ny,nx);
	    DualFloat vd_rt = get_hfacet(vd,d_vd,i,j+1,ny,nx);
	    DualFloat sigmad_xy_tr = get_sigma_xy_dual({ud_tr,ud_r,vd_t,vd_rt,eta_H_tr},dx_inv,i,j+1,ny,nx);
	    d_rvd_t += K_1 * sigmad_xy_tr.d * dx_inv;
	    }

            {
            DualFloat eta_t = eta_local[bi-1][bj];
            DualFloat eta_b = eta_local[bi][bj];
            DualFloat H_t = get_cell(H,d_H,i-1,j,ny,nx);
            DualFloat H_b = get_cell(H,d_H,i,j,ny,nx);
            DualFloat vd_t = get_hfacet(vd,d_vd,i,j,ny,nx);
            DualFloat sigmad_yz_t = get_sigma_yz_dual({vd_t,eta_t,eta_b,H_t,H_b},c_1,S_1,H_reg,i,j,ny,nx);

            d_rvd_t += sigmad_yz_t.d;
            }

	    {
	    DualFloat v_t    = get_hfacet(v,d_v,i,j,ny,nx);
	    DualFloat v_tt    = get_hfacet(v,d_v,i-1,j,ny,nx);
	    DualFloat v_b    = get_hfacet(v,d_v,i+1,j,ny,nx);

            DualFloat u_tl = get_vfacet(u,d_u,i-1,j,ny,nx);
            DualFloat u_tr = get_vfacet(u,d_u,i-1,j+1,ny,nx);
            DualFloat u_bl = get_vfacet(u,d_u,i,j,ny,nx);
            DualFloat u_br = get_vfacet(u,d_u,i,j+1,ny,nx);

            DualFloat vb_t  = v_t  - get_hfacet(vd,d_vd,i,j,ny,nx);
            DualFloat vb_tt = v_tt - get_hfacet(vd,d_vd,i-1,j,ny,nx);
            DualFloat vb_b  = v_b  - get_hfacet(vd,d_vd,i+1,j,ny,nx);
            DualFloat ub_tl = u_tl - get_vfacet(ud,d_ud,i-1,j,ny,nx);
            DualFloat ub_tr = u_tr - get_vfacet(ud,d_ud,i-1,j+1,ny,nx);
            DualFloat ub_bl = u_bl - get_vfacet(ud,d_ud,i,j,ny,nx);
            DualFloat ub_br = u_br - get_vfacet(ud,d_ud,i,j+1,ny,nx);

	    DualFloat H_t    = get_cell(H,d_H,i-1,j,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
	    float phi_t      = get_cell(phi,i-1,j,ny,nx);
	    float phi_c      = get_cell(phi,i,j,ny,nx);
	    float xi_t      = get_cell(xi,i-1,j,ny,nx);
	    float xi_c      = get_cell(xi,i,j,ny,nx);
	    float beta_t     = get_cell(beta,i-1,j,ny,nx);
	    float beta_c     = get_cell(beta,i,j,ny,nx);

	    DualFloat tau_by = get_tau_by_dual({vb_t,vb_tt,vb_b,ub_tl,ub_tr,ub_bl,ub_br,H_t,H_c,xi_t,xi_c,beta_t,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
	    d_rv_t += tau_by.d;
	    d_rvd_t -= tau_by.d;
	    }

	    {
	    DualFloat H_t    = get_cell(H,d_H,i-1,j,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
	    float bed_t = get_cell(bed,i-1,j,ny,nx);
	    float bed_c = get_cell(bed,i,j,ny,nx);
	    float phi_t      = get_cell(phi,i-1,j,ny,nx);
	    float phi_c      = get_cell(phi,i,j,ny,nx);

	    DualFloat tau_dy = get_tau_dy_dual({H_t,H_c,bed_t,bed_c,phi_t,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);
	    d_rv_t -= tau_dy.d;
	    }

	    // Identity rows at Dirichlet facets (and all vd rows in SSA
	    // mode), mirroring compute_residual
	    if (i <= 0 || i >= ny) {
		d_rv_t = get_hfacet(d_v,i,j,ny,nx);
	    }
	    if (ssa || i <= 0 || i >= ny) {
		d_rvd_t = get_hfacet(d_vd,i,j,ny,nx);
	    }
	    jvp_v[i * nx + j] = d_rv_t;
	    jvp_vd[i * nx + j] = d_rvd_t;

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
    float* __restrict__ vjp_ud,
    float* __restrict__ vjp_vd,
    float* __restrict__ vjp_H,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ ud,
    const float* __restrict__ vd,
    const float* __restrict__ H,
    const float* __restrict__ lambda_u,
    const float* __restrict__ lambda_v,
    const float* __restrict__ lambda_ud,
    const float* __restrict__ lambda_vd,
    const float* __restrict__ lambda_H,
    const float* __restrict__ phi,
    const float* __restrict__ xi,
    const float* __restrict__ mask,
    const float* __restrict__ f_u,
    const float* __restrict__ f_v,
    const float* __restrict__ f_ud,
    const float* __restrict__ f_vd,
    const float* __restrict__ f_H,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    bool use_forcing, bool use_mask,
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
    int tid = bi * blockDim.x + bj;

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    // SHARED MEMORY ACCUMULATORS
    __shared__ float s_adj_u[bny][bnx+1];
    __shared__ float s_adj_v[bny+1][bnx];
    __shared__ float s_adj_ud[bny][bnx+1];
    __shared__ float s_adj_vd[bny+1][bnx];
    __shared__ float s_adj_H[bny][bnx];
    // Accumulates sum over rows of lambda_row * dR_row/d(eta_cell); folded
    // into s_adj_H through d(eta)/d(H) after the main pass.
    __shared__ float s_adj_eta[bny][bnx];

    for (int k = tid; k < 272; k += 256) {
    s_adj_u[k / 17][k % 17] = 0.0f;
    s_adj_ud[k / 17][k % 17] = 0.0f;
    }
    // Same for V-tile (272 elements)
    for (int k = tid; k < 272; k += 256) {
    s_adj_v[k / 16][k % 16] = 0.0f;
    s_adj_vd[k / 16][k % 16] = 0.0f;
    }
    s_adj_H[bi][bj] = 0.0f;
    s_adj_eta[bi][bj] = 0.0f;

    __shared__ DualFloat eta_local[bny][bnx];
    __shared__ float deta_dH_local[bny][bnx];

    populate_viscosity_vjp(eta_local, deta_dH_local, bi, bj, i, j,
        u, v, ud, vd, H, lambda_u, lambda_v, lambda_ud, lambda_vd,
        B, n, eps_reg, H_reg, dx, ny, nx);

    __syncthreads();
    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    bool has_cell = i >= 0 && i <  ny && j >= 0 && j <  nx;
    bool has_u    = i >= 0 && i <  ny && j >= 0 && j <= nx;
    bool has_v    = i >= 0 && i <= ny && j >= 0 && j <  nx;

    if ( is_active ) {
	float dx_inv = 1.0f/dx;

	if (has_cell){
	    float H_c        = get_cell(H,i,j,ny,nx);
	    // Row projection of lambda_H (active set) is applied by the
	    // Python wrapper before launch; the kernel is the pure physics
	    // transpose (see the constraint convention in common.cu)
	    float lambda_H_c = get_cell(lambda_H,i,j,ny,nx);

	    // Mass matrix contribution
	    atomicAdd(&s_adj_H[bi][bj], lambda_H_c/dt);

	    float phi_c = get_cell(phi,i,j,ny,nx);

	    float H_l = get_cell(H,i,j-1,ny,nx);
	    float u_l = get_vfacet(u,i,j,ny,nx);
	    HorizontalFluxJacobian j_q_l = get_horizontal_flux_jac({u_l,H_l,H_c}, i, j, ny, nx);
	    atomicAdd(&s_adj_H[bi][bj]  , -lambda_H_c*j_q_l.d_H_r*dx_inv);
            atomicAdd(&s_adj_H[bi][bj-1], -lambda_H_c*j_q_l.d_H_l*dx_inv);
            atomicAdd(&s_adj_u[bi][bj]  , -lambda_H_c*j_q_l.d_u*dx_inv);

	    float phi_l = get_cell(phi,i,j-1,ny,nx);
	    FacetCalvingJacobian j_calve_l = get_facet_calving_jac({H_c,H_l,phi_c,phi_l,calving_rate,flotation_reg_calving},i,j,ny,nx);
	    atomicAdd(&s_adj_H[bi][bj],   lambda_H_c*j_calve_l.d_H_this*dx_inv);
           
	    float H_r = get_cell(H,i,j+1,ny,nx);
	    float u_r = get_vfacet(u,i,j+1,ny,nx);
	    HorizontalFluxJacobian j_q_r = get_horizontal_flux_jac({u_r,H_c,H_r}, i, j + 1, ny, nx);
            atomicAdd(&s_adj_H[bi][bj],   lambda_H_c*j_q_r.d_H_l*dx_inv);
            atomicAdd(&s_adj_H[bi][bj+1], lambda_H_c*j_q_r.d_H_r*dx_inv);
            atomicAdd(&s_adj_u[bi][bj+1], lambda_H_c*j_q_r.d_u*dx_inv);

	    float phi_r = get_cell(phi,i,j+1,ny,nx);
	    FacetCalvingJacobian j_calve_r = get_facet_calving_jac({H_c,H_r,phi_c,phi_r,calving_rate,flotation_reg_calving},i,j+1,ny,nx);
	    atomicAdd(&s_adj_H[bi][bj], lambda_H_c*j_calve_r.d_H_this*dx_inv);

	    float H_t = get_cell(H,i-1,j,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    VerticalFluxJacobian j_q_t = get_vertical_flux_jac({v_t,H_t,H_c}, i, j, ny, nx);
	    atomicAdd(&s_adj_H[bi][bj],   lambda_H_c*j_q_t.d_H_b*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj], lambda_H_c*j_q_t.d_H_t*dx_inv);
	    atomicAdd(&s_adj_v[bi][bj],   lambda_H_c*j_q_t.d_v*dx_inv);

	    float phi_t = get_cell(phi,i-1,j,ny,nx);
	    FacetCalvingJacobian j_calve_t = get_facet_calving_jac({H_c,H_t,phi_c,phi_t,calving_rate,flotation_reg_calving},i,j,ny,nx);
	    atomicAdd(&s_adj_H[bi][bj], lambda_H_c*j_calve_t.d_H_this*dx_inv);


	    float H_b = get_cell(H,i+1,j,ny,nx);
	    float v_b = get_hfacet(v,i+1,j,ny,nx);
	    VerticalFluxJacobian j_q_b = get_vertical_flux_jac({v_b,H_c,H_b}, i + 1, j, ny, nx);
            atomicAdd(&s_adj_H[bi][bj],   -lambda_H_c*j_q_b.d_H_t*dx_inv);
            atomicAdd(&s_adj_H[bi+1][bj], -lambda_H_c*j_q_b.d_H_b*dx_inv);
	    atomicAdd(&s_adj_v[bi+1][bj], -lambda_H_c*j_q_b.d_v*dx_inv);

	    float phi_b = get_cell(phi,i+1,j,ny,nx);
	    FacetCalvingJacobian j_calve_b = get_facet_calving_jac({H_c,H_b,phi_c,phi_b,calving_rate,flotation_reg_calving},i+1,j,ny,nx);
	    atomicAdd(&s_adj_H[bi][bj], lambda_H_c*j_calve_b.d_H_this*dx_inv);

	    //float masked = use_mask ? get_cell(mask,i,j,ny,nx) : 0.0f;
	    //float lambda_H_c_ = get_cell(lambda_H,i,j,ny,nx);
            //atomicAdd(&s_adj_H[bi][bj], (1.0f - masked) * lambda_H_c_);
	    if (use_forcing) atomicAdd(&s_adj_H[bi][bj], -get_cell(f_H,i,j,ny,nx));
	}

	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	
	if (has_u){
            float K_1 = 1.0f / (2.0f * n + 3.0f);
            float S_1 = (n + 2.0f) * (n + 2.0f) / (2.0f * n + 1.0f);
            float c_1 = __powf(1.0f / (n + 2.0f),2.0f * n / (n + 1.0f)) * (2.0f * n + 1.0f);

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
            atomicAdd(&s_adj_eta[bi][bj],lambda_u_l*j_sigma_xx_c.d_eta_H*eta_H_c.d_eta*dx_inv);

            float ud_l = get_vfacet(ud,i,j,ny,nx);
	    float ud_r = get_vfacet(ud,i,j+1,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);
	    float vd_b = get_hfacet(vd,i+1,j,ny,nx);

	    float lambda_ud_l = get_vfacet(lambda_ud,i,j,ny,nx);
	    float lambda_ud_r = get_vfacet(lambda_ud,i,j+1,ny,nx);
	    float lambda_vd_t = get_hfacet(lambda_vd,i,j,ny,nx);
	    float lambda_vd_b = get_hfacet(lambda_vd,i+1,j,ny,nx);
            SigmaNormalJacobian j_sigmad_xx_c = get_sigma_xx_jac({ud_l,ud_r,vd_t,vd_b,eta_H_c.res},dx_inv,i,j,ny,nx);

	    float lambda_sigmad_xx_c = j_sigmad_xx_c.apply_jvp({lambda_ud_l,lambda_ud_r,lambda_vd_t,lambda_vd_b,eta_H_c.apply_jvp({eta_c.d,0.0f})});

            atomicAdd(&s_adj_ud[bi][bj],K_1*lambda_sigmad_xx_c * dx_inv);
            atomicAdd(&s_adj_H[bi][bj],K_1*lambda_ud_l*j_sigmad_xx_c.d_eta_H*eta_H_c.d_H*dx_inv);
            atomicAdd(&s_adj_eta[bi][bj],K_1*lambda_ud_l*j_sigmad_xx_c.d_eta_H*eta_H_c.d_eta*dx_inv);
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
	    atomicAdd(&s_adj_eta[bi][bj-1],-lambda_u_l*j_sigma_xx_l.d_eta_H*eta_H_l.d_eta*dx_inv);

            float ud_l    = get_vfacet(ud,i,j,ny,nx);
	    float ud_ll   = get_vfacet(ud,i,j-1,ny,nx);
	    float vd_lt   = get_hfacet(vd,i,j-1,ny,nx);
	    float vd_lb   = get_hfacet(vd,i+1,j-1,ny,nx);

	    float lambda_ud_l    = get_vfacet(lambda_ud,i,j,ny,nx);
	    float lambda_ud_ll   = get_vfacet(lambda_ud,i,j-1,ny,nx);
	    float lambda_vd_lt   = get_hfacet(lambda_vd,i,j-1,ny,nx);
	    float lambda_vd_lb   = get_hfacet(lambda_vd,i+1,j-1,ny,nx);
            SigmaNormalJacobian j_sigmad_xx_l = get_sigma_xx_jac({ud_ll,ud_l,vd_lt,vd_lb,eta_H_l.res},dx_inv,i,j - 1,ny,nx);

	    float lambda_sigmad_xx_l = j_sigmad_xx_l.apply_jvp({lambda_ud_ll,lambda_ud_l,lambda_vd_lt,lambda_vd_lb,eta_H_l.apply_jvp({eta_l.d,0.0f})});

	    atomicAdd(&s_adj_ud[bi][bj],  -K_1*lambda_sigmad_xx_l*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj-1],-K_1*lambda_ud_l*j_sigmad_xx_l.d_eta_H*eta_H_l.d_H*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj-1],-K_1*lambda_ud_l*j_sigmad_xx_l.d_eta_H*eta_H_l.d_eta*dx_inv);
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
	    atomicAdd(&s_adj_eta[bi-1][bj-1],lambda_u_l * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_eta_tl*dx_inv);
	    atomicAdd(&s_adj_eta[bi-1][bj],  lambda_u_l * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_eta_tr*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj-1],  lambda_u_l * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_eta_bl*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj],    lambda_u_l * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_eta_br*dx_inv);

	    float ud_tl = get_vfacet(ud,i-1,j,ny,nx);
	    float ud_l = get_vfacet(ud,i,j,ny,nx);
	    float vd_lt = get_hfacet(vd,i,j-1,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);

	    float lambda_ud_tl = get_vfacet(lambda_ud,i-1,j,ny,nx);
	    float lambda_ud_l = get_vfacet(lambda_ud,i,j,ny,nx);
	    float lambda_vd_lt = get_hfacet(lambda_vd,i,j-1,ny,nx);
	    float lambda_vd_t = get_hfacet(lambda_vd,i,j,ny,nx);

	    SigmaShearJacobian j_sigmad_xy_tl = get_sigma_xy_jac({ud_tl,ud_l,vd_lt,vd_t,eta_H_tl.res},dx_inv,i,j,ny,nx);

	    float lambda_sigmad_xy_tl = j_sigmad_xy_tl.apply_jvp({lambda_ud_tl,lambda_ud_l,lambda_vd_lt,lambda_vd_t,eta_H_tl.apply_jvp({eta_tl.d,eta_t.d,eta_l.d,eta_c.d,0.0f,0.0f,0.0f,0.0f})});

	    atomicAdd(&s_adj_ud[bi][bj],   K_1*lambda_sigmad_xy_tl*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj-1],K_1*lambda_ud_l * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_H_tl*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj],  K_1*lambda_ud_l * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_H_tr*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj-1],  K_1*lambda_ud_l * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_H_bl*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj],    K_1*lambda_ud_l * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_H_br*dx_inv);
	    atomicAdd(&s_adj_eta[bi-1][bj-1],K_1*lambda_ud_l * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_eta_tl*dx_inv);
	    atomicAdd(&s_adj_eta[bi-1][bj],  K_1*lambda_ud_l * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_eta_tr*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj-1],  K_1*lambda_ud_l * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_eta_bl*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj],    K_1*lambda_ud_l * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_eta_br*dx_inv);
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
	    atomicAdd(&s_adj_eta[bi][bj-1],  -lambda_u_l * j_sigma_xy_bl.d_eta_H*eta_H_bl.d_eta_tl*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj],    -lambda_u_l * j_sigma_xy_bl.d_eta_H*eta_H_bl.d_eta_tr*dx_inv);
	    atomicAdd(&s_adj_eta[bi+1][bj-1],-lambda_u_l * j_sigma_xy_bl.d_eta_H*eta_H_bl.d_eta_bl*dx_inv);
	    atomicAdd(&s_adj_eta[bi+1][bj],  -lambda_u_l * j_sigma_xy_bl.d_eta_H*eta_H_bl.d_eta_br*dx_inv);

	    float ud_l    = get_vfacet(ud,i,j,ny,nx);
	    float ud_bl   = get_vfacet(ud,i+1,j,ny,nx);
	    float vd_lb   = get_hfacet(vd,i+1,j-1,ny,nx);
	    float vd_b    = get_hfacet(vd,i+1,j,ny,nx);

	    float lambda_ud_l    = get_vfacet(lambda_ud,i,j,ny,nx);
	    float lambda_ud_bl   = get_vfacet(lambda_ud,i+1,j,ny,nx);
	    float lambda_vd_lb   = get_hfacet(lambda_vd,i+1,j-1,ny,nx);
	    float lambda_vd_b    = get_hfacet(lambda_vd,i+1,j,ny,nx);

	    SigmaShearJacobian j_sigmad_xy_bl = get_sigma_xy_jac({ud_l,ud_bl,vd_lb,vd_b,eta_H_bl.res},dx_inv,i + 1,j,ny,nx);

	    float lambda_sigmad_xy_bl = j_sigmad_xy_bl.apply_jvp({lambda_ud_l,lambda_ud_bl,lambda_vd_lb,lambda_vd_b,eta_H_bl.apply_jvp({eta_l.d,eta_c.d,eta_bl.d,eta_b.d,0.0f,0.0f,0.0f,0.0f})});

	    atomicAdd(&s_adj_ud[bi][bj] ,  -K_1*lambda_sigmad_xy_bl*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj-1],  -K_1*lambda_ud_l * j_sigmad_xy_bl.d_eta_H*eta_H_bl.d_H_tl*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj],    -K_1*lambda_ud_l * j_sigmad_xy_bl.d_eta_H*eta_H_bl.d_H_tr*dx_inv);
	    atomicAdd(&s_adj_H[bi+1][bj-1],-K_1*lambda_ud_l * j_sigmad_xy_bl.d_eta_H*eta_H_bl.d_H_bl*dx_inv);
	    atomicAdd(&s_adj_H[bi+1][bj],  -K_1*lambda_ud_l * j_sigmad_xy_bl.d_eta_H*eta_H_bl.d_H_br*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj-1],  -K_1*lambda_ud_l * j_sigmad_xy_bl.d_eta_H*eta_H_bl.d_eta_tl*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj],    -K_1*lambda_ud_l * j_sigmad_xy_bl.d_eta_H*eta_H_bl.d_eta_tr*dx_inv);
	    atomicAdd(&s_adj_eta[bi+1][bj-1],-K_1*lambda_ud_l * j_sigmad_xy_bl.d_eta_H*eta_H_bl.d_eta_bl*dx_inv);
	    atomicAdd(&s_adj_eta[bi+1][bj],  -K_1*lambda_ud_l * j_sigmad_xy_bl.d_eta_H*eta_H_bl.d_eta_br*dx_inv);
	    }

	    // Vertical shear row (r_ud): symmetric part via lambda-direction JVP,
	    // plus explicit H-column scatters (direct 1/(H^2+H_reg^2) leg; the
	    // eta(H) leg goes through s_adj_eta)
	    {
	    DualFloat eta_l = eta_local[bi][bj-1];
	    DualFloat eta_c = eta_local[bi][bj];
	    float H_l = get_cell(H,i,j-1,ny,nx);
	    float H_c = get_cell(H,i,j,ny,nx);
	    float ud_l = get_vfacet(ud,i,j,ny,nx);
	    float lambda_ud_l = get_vfacet(lambda_ud,i,j,ny,nx);

	    SigmaVertXZJacobian j_sigmad_xz_l = get_sigma_xz_jac({ud_l,eta_l.v,eta_c.v,H_l,H_c},c_1,S_1,H_reg,i,j,ny,nx);

	    float lambda_sigmad_xz_l = j_sigmad_xz_l.apply_jvp({lambda_ud_l,eta_l.d,eta_c.d,0.0f,0.0f});

	    atomicAdd(&s_adj_ud[bi][bj], lambda_sigmad_xz_l);
	    atomicAdd(&s_adj_H[bi][bj-1], lambda_ud_l * j_sigmad_xz_l.d_H_l);
	    atomicAdd(&s_adj_H[bi][bj],   lambda_ud_l * j_sigmad_xz_l.d_H_r);
	    atomicAdd(&s_adj_eta[bi][bj-1], lambda_ud_l * j_sigmad_xz_l.d_eta_l);
	    atomicAdd(&s_adj_eta[bi][bj],   lambda_ud_l * j_sigmad_xz_l.d_eta_r);
	    }
            
            {
            // Drag rows: R_u = +tau_bx(u_b), R_ud = -tau_bx(u_b), u_b = u - ud.
            // Jacobian block structure (1,-1)(1,-1)^T (x) J_drag: combine both
            // rows with lam_eff = lambda_u - lambda_ud, scatter + into u/v
            // columns and - into ud/vd columns.
            float ub_l  = get_vfacet(u,i,j,ny,nx)     - get_vfacet(ud,i,j,ny,nx);
            float ub_ll = get_vfacet(u,i,j-1,ny,nx)   - get_vfacet(ud,i,j-1,ny,nx);
            float ub_r  = get_vfacet(u,i,j+1,ny,nx)   - get_vfacet(ud,i,j+1,ny,nx);
            float vb_tl = get_hfacet(v,i,j-1,ny,nx)   - get_hfacet(vd,i,j-1,ny,nx);
	    float vb_tr = get_hfacet(v,i,j,ny,nx)     - get_hfacet(vd,i,j,ny,nx);
	    float vb_bl = get_hfacet(v,i+1,j-1,ny,nx) - get_hfacet(vd,i+1,j-1,ny,nx);
	    float vb_br = get_hfacet(v,i+1,j,ny,nx)   - get_hfacet(vd,i+1,j,ny,nx);

	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float xi_l  = get_cell(xi,i,j-1,ny,nx);
	    float xi_c  = get_cell(xi,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    TauBxJacobian j_tau_bx = get_tau_bx_jac({ub_l,ub_ll,ub_r,vb_tl,vb_tr,vb_bl,vb_br,H_l,H_c,xi_l,xi_c,beta_l,beta_c,m,u_reg,water_drag,flotation_reg_sliding});


	    float lambda_u_l = get_vfacet(lambda_u,i,j,ny,nx);
	    float lambda_ud_l = get_vfacet(lambda_ud,i,j,ny,nx);
	    float lam_eff = lambda_u_l - lambda_ud_l;

	    atomicAdd(&s_adj_u[bi][bj],     lam_eff * j_tau_bx.d_u_c);
	    atomicAdd(&s_adj_u[bi][bj-1],   lam_eff * j_tau_bx.d_u_l);
	    atomicAdd(&s_adj_u[bi][bj+1],   lam_eff * j_tau_bx.d_u_r);
	    atomicAdd(&s_adj_v[bi][bj-1],   lam_eff * j_tau_bx.d_v_tl);
	    atomicAdd(&s_adj_v[bi][bj],     lam_eff * j_tau_bx.d_v_tr);
	    atomicAdd(&s_adj_v[bi+1][bj-1], lam_eff * j_tau_bx.d_v_bl);
	    atomicAdd(&s_adj_v[bi+1][bj],   lam_eff * j_tau_bx.d_v_br);

	    atomicAdd(&s_adj_ud[bi][bj],     -lam_eff * j_tau_bx.d_u_c);
	    atomicAdd(&s_adj_ud[bi][bj-1],   -lam_eff * j_tau_bx.d_u_l);
	    atomicAdd(&s_adj_ud[bi][bj+1],   -lam_eff * j_tau_bx.d_u_r);
	    atomicAdd(&s_adj_vd[bi][bj-1],   -lam_eff * j_tau_bx.d_v_tl);
	    atomicAdd(&s_adj_vd[bi][bj],     -lam_eff * j_tau_bx.d_v_tr);
	    atomicAdd(&s_adj_vd[bi+1][bj-1], -lam_eff * j_tau_bx.d_v_bl);
	    atomicAdd(&s_adj_vd[bi+1][bj],   -lam_eff * j_tau_bx.d_v_br);

	    atomicAdd(&s_adj_H[bi][bj-1],   lam_eff * j_tau_bx.d_H_l);
	    atomicAdd(&s_adj_H[bi][bj],     lam_eff * j_tau_bx.d_H_r);

	    }


	    {
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
            float lambda_u_l    = get_vfacet(lambda_u,i,j,ny,nx);

	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float phi_l  = get_cell(phi,i,j-1,ny,nx);
	    float phi_c  = get_cell(phi,i,j,ny,nx);
	    TauDxJacobian j_tau_dx = get_tau_dx_jac({H_l,H_c,bed_l,bed_c,phi_l,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);

            atomicAdd(&s_adj_H[bi][bj-1],-lambda_u_l * j_tau_dx.d_H_l);
            atomicAdd(&s_adj_H[bi][bj],  -lambda_u_l * j_tau_dx.d_H_r);
	    }

	    if (use_forcing) atomicAdd(&s_adj_u[bi][bj], -get_vfacet(f_u,i,j,ny,nx));
	    if (use_forcing) atomicAdd(&s_adj_ud[bi][bj], -get_vfacet(f_ud,i,j,ny,nx));

 	}

	if (has_v){
            float K_1 = 1.0f / (2.0f * n + 3.0f);
            float S_1 = (n + 2.0f) * (n + 2.0f) / (2.0f * n + 1.0f);
            float c_1 = __powf(1.0f / (n + 2.0f),2.0f * n / (n + 1.0f)) * (2.0f * n + 1.0f);

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
	    atomicAdd(&s_adj_eta[bi-1][bj],lambda_v_t*j_sigma_yy_t.d_eta_H*eta_H_t.d_eta*dx_inv);

	    float ud_tl = get_vfacet(ud,i-1,j,ny,nx);
	    float ud_tr = get_vfacet(ud,i-1,j+1,ny,nx);
	    float vd_tt = get_hfacet(vd,i-1,j,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);

	    float lambda_ud_tl = get_vfacet(lambda_ud,i-1,j,ny,nx);
	    float lambda_ud_tr = get_vfacet(lambda_ud,i-1,j+1,ny,nx);
	    float lambda_vd_tt = get_hfacet(lambda_vd,i-1,j,ny,nx);
	    float lambda_vd_t  = get_hfacet(lambda_vd,i,j,ny,nx);
	    SigmaNormalJacobian j_sigmad_yy_t = get_sigma_yy_jac({ud_tl,ud_tr,vd_tt,vd_t,eta_H_t.res},dx_inv,i-1,j,ny,nx);

	    float lambda_sigmad_yy_t = j_sigmad_yy_t.apply_jvp({lambda_ud_tl,lambda_ud_tr,lambda_vd_tt,lambda_vd_t,eta_H_t.apply_jvp({eta_t.d,0.0f})});

	    atomicAdd(&s_adj_vd[bi][bj],  K_1*lambda_sigmad_yy_t * dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj],K_1*lambda_vd_t*j_sigmad_yy_t.d_eta_H*eta_H_t.d_H*dx_inv);
	    atomicAdd(&s_adj_eta[bi-1][bj],K_1*lambda_vd_t*j_sigmad_yy_t.d_eta_H*eta_H_t.d_eta*dx_inv);
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
            atomicAdd(&s_adj_eta[bi][bj],-lambda_v_t*j_sigma_yy_c.d_eta_H*eta_H_c.d_eta*dx_inv);

            float ud_l = get_vfacet(ud,i,j,ny,nx);
	    float ud_r = get_vfacet(ud,i,j+1,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);
	    float vd_b = get_hfacet(vd,i+1,j,ny,nx);

	    float lambda_ud_l = get_vfacet(lambda_ud,i,j,ny,nx);
	    float lambda_ud_r = get_vfacet(lambda_ud,i,j+1,ny,nx);
	    float lambda_vd_t = get_hfacet(lambda_vd,i,j,ny,nx);
	    float lambda_vd_b = get_hfacet(lambda_vd,i+1,j,ny,nx);
            SigmaNormalJacobian j_sigmad_yy_c = get_sigma_yy_jac({ud_l,ud_r,vd_t,vd_b,eta_H_c.res},dx_inv,i,j,ny,nx);

	    float lambda_sigmad_yy_c = j_sigmad_yy_c.apply_jvp({lambda_ud_l,lambda_ud_r,lambda_vd_t,lambda_vd_b,eta_H_c.apply_jvp({eta_c.d,0.0f})});
	    atomicAdd(&s_adj_vd[bi][bj],-K_1*lambda_sigmad_yy_c*dx_inv);
            atomicAdd(&s_adj_H[bi][bj],-K_1*lambda_vd_t*j_sigmad_yy_c.d_eta_H*eta_H_c.d_H*dx_inv);
            atomicAdd(&s_adj_eta[bi][bj],-K_1*lambda_vd_t*j_sigmad_yy_c.d_eta_H*eta_H_c.d_eta*dx_inv);
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
	    atomicAdd(&s_adj_eta[bi-1][bj-1],-lambda_v_t * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_eta_tl*dx_inv);
	    atomicAdd(&s_adj_eta[bi-1][bj],  -lambda_v_t * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_eta_tr*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj-1],  -lambda_v_t * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_eta_bl*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj],    -lambda_v_t * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_eta_br*dx_inv);

	    float ud_tl = get_vfacet(ud,i-1,j,ny,nx);
	    float ud_l = get_vfacet(ud,i,j,ny,nx);
	    float vd_lt = get_hfacet(vd,i,j-1,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);

	    float lambda_ud_tl = get_vfacet(lambda_ud,i-1,j,ny,nx);
	    float lambda_ud_l = get_vfacet(lambda_ud,i,j,ny,nx);
	    float lambda_vd_lt = get_hfacet(lambda_vd,i,j-1,ny,nx);
	    float lambda_vd_t = get_hfacet(lambda_vd,i,j,ny,nx);

	    SigmaShearJacobian j_sigmad_xy_tl = get_sigma_xy_jac({ud_tl,ud_l,vd_lt,vd_t,eta_H_tl.res},dx_inv,i,j,ny,nx);

	    float lambda_sigmad_xy_tl = j_sigmad_xy_tl.apply_jvp({lambda_ud_tl,lambda_ud_l,lambda_vd_lt,lambda_vd_t,eta_H_tl.apply_jvp({eta_tl.d,eta_t.d,eta_l.d,eta_c.d,0.0f,0.0f,0.0f,0.0f})});

	    atomicAdd(&s_adj_vd[bi][bj],   -K_1*lambda_sigmad_xy_tl*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj-1],-K_1*lambda_vd_t * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_H_tl*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj],  -K_1*lambda_vd_t * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_H_tr*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj-1],  -K_1*lambda_vd_t * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_H_bl*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj],    -K_1*lambda_vd_t * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_H_br*dx_inv);
	    atomicAdd(&s_adj_eta[bi-1][bj-1],-K_1*lambda_vd_t * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_eta_tl*dx_inv);
	    atomicAdd(&s_adj_eta[bi-1][bj],  -K_1*lambda_vd_t * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_eta_tr*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj-1],  -K_1*lambda_vd_t * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_eta_bl*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj],    -K_1*lambda_vd_t * j_sigmad_xy_tl.d_eta_H*eta_H_tl.d_eta_br*dx_inv);
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
	    atomicAdd(&s_adj_eta[bi-1][bj],  lambda_v_t * j_sigma_xy_tr.d_eta_H*eta_H_tr.d_eta_tl*dx_inv);
	    atomicAdd(&s_adj_eta[bi-1][bj+1],lambda_v_t * j_sigma_xy_tr.d_eta_H*eta_H_tr.d_eta_tr*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj],    lambda_v_t * j_sigma_xy_tr.d_eta_H*eta_H_tr.d_eta_bl*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj+1],  lambda_v_t * j_sigma_xy_tr.d_eta_H*eta_H_tr.d_eta_br*dx_inv);

	    float ud_tr = get_vfacet(ud,i-1,j+1,ny,nx);
	    float ud_r = get_vfacet(ud,i,j+1,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);
	    float vd_rt = get_hfacet(vd,i,j+1,ny,nx);

	    float lambda_ud_tr = get_vfacet(lambda_ud,i-1,j+1,ny,nx);
	    float lambda_ud_r = get_vfacet(lambda_ud,i,j+1,ny,nx);
	    float lambda_vd_t = get_hfacet(lambda_vd,i,j,ny,nx);
	    float lambda_vd_rt = get_hfacet(lambda_vd,i,j+1,ny,nx);
	    SigmaShearJacobian j_sigmad_xy_tr = get_sigma_xy_jac({ud_tr,ud_r,vd_t,vd_rt,eta_H_tr.res},dx_inv,i,j+1,ny,nx);

	    float lambda_sigmad_xy_tr = j_sigmad_xy_tr.apply_jvp({lambda_ud_tr,lambda_ud_r,lambda_vd_t,lambda_vd_rt,eta_H_tr.apply_jvp({eta_t.d,eta_tr.d,eta_c.d,eta_r.d,0.0f,0.0f,0.0f,0.0f})});

	    atomicAdd(&s_adj_vd[bi][bj],   K_1*lambda_sigmad_xy_tr*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj],  K_1*lambda_vd_t * j_sigmad_xy_tr.d_eta_H*eta_H_tr.d_H_tl*dx_inv);
	    atomicAdd(&s_adj_H[bi-1][bj+1],K_1*lambda_vd_t * j_sigmad_xy_tr.d_eta_H*eta_H_tr.d_H_tr*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj],    K_1*lambda_vd_t * j_sigmad_xy_tr.d_eta_H*eta_H_tr.d_H_bl*dx_inv);
	    atomicAdd(&s_adj_H[bi][bj+1],  K_1*lambda_vd_t * j_sigmad_xy_tr.d_eta_H*eta_H_tr.d_H_br*dx_inv);
	    atomicAdd(&s_adj_eta[bi-1][bj],  K_1*lambda_vd_t * j_sigmad_xy_tr.d_eta_H*eta_H_tr.d_eta_tl*dx_inv);
	    atomicAdd(&s_adj_eta[bi-1][bj+1],K_1*lambda_vd_t * j_sigmad_xy_tr.d_eta_H*eta_H_tr.d_eta_tr*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj],    K_1*lambda_vd_t * j_sigmad_xy_tr.d_eta_H*eta_H_tr.d_eta_bl*dx_inv);
	    atomicAdd(&s_adj_eta[bi][bj+1],  K_1*lambda_vd_t * j_sigmad_xy_tr.d_eta_H*eta_H_tr.d_eta_br*dx_inv);
	    }

	    // Vertical shear row (r_vd)
	    {
	    DualFloat eta_t = eta_local[bi-1][bj];
	    DualFloat eta_b = eta_local[bi][bj];
	    float H_t = get_cell(H,i-1,j,ny,nx);
	    float H_b = get_cell(H,i,j,ny,nx);
	    float vd_t = get_hfacet(vd,i,j,ny,nx);
	    float lambda_vd_t = get_hfacet(lambda_vd,i,j,ny,nx);

	    SigmaVertYZJacobian j_sigmad_yz_t = get_sigma_yz_jac({vd_t,eta_t.v,eta_b.v,H_t,H_b},c_1,S_1,H_reg,i,j,ny,nx);

	    float lambda_sigmad_yz_t = j_sigmad_yz_t.apply_jvp({lambda_vd_t,eta_t.d,eta_b.d,0.0f,0.0f});

	    atomicAdd(&s_adj_vd[bi][bj], lambda_sigmad_yz_t);
	    atomicAdd(&s_adj_H[bi-1][bj], lambda_vd_t * j_sigmad_yz_t.d_H_t);
	    atomicAdd(&s_adj_H[bi][bj],   lambda_vd_t * j_sigmad_yz_t.d_H_b);
	    atomicAdd(&s_adj_eta[bi-1][bj], lambda_vd_t * j_sigmad_yz_t.d_eta_t);
	    atomicAdd(&s_adj_eta[bi][bj],   lambda_vd_t * j_sigmad_yz_t.d_eta_b);
	    }
             
	    
	    {
            // Drag rows: R_v = +tau_by(v_b), R_vd = -tau_by(v_b); combined
            // scatter with lam_eff = lambda_v - lambda_vd (cf. tau_bx above)
	    float vb_t  = get_hfacet(v,i,j,ny,nx)     - get_hfacet(vd,i,j,ny,nx);
	    float vb_tt = get_hfacet(v,i-1,j,ny,nx)   - get_hfacet(vd,i-1,j,ny,nx);
	    float vb_b  = get_hfacet(v,i+1,j,ny,nx)   - get_hfacet(vd,i+1,j,ny,nx);
            float ub_tl = get_vfacet(u,i-1,j,ny,nx)   - get_vfacet(ud,i-1,j,ny,nx);
            float ub_tr = get_vfacet(u,i-1,j+1,ny,nx) - get_vfacet(ud,i-1,j+1,ny,nx);
            float ub_bl = get_vfacet(u,i,j,ny,nx)     - get_vfacet(ud,i,j,ny,nx);
            float ub_br = get_vfacet(u,i,j+1,ny,nx)   - get_vfacet(ud,i,j+1,ny,nx);

	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float xi_t  = get_cell(xi,i-1,j,ny,nx);
	    float xi_c  = get_cell(xi,i,j,ny,nx);
	    float beta_t = get_cell(beta,i-1,j,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);

	    TauByJacobian j_tau_by = get_tau_by_jac({vb_t,vb_tt,vb_b,ub_tl,ub_tr,ub_bl,ub_br,H_t,H_c,xi_t,xi_c,beta_t,beta_c,m,u_reg,water_drag,flotation_reg_sliding});

	    float lambda_v_t = get_hfacet(lambda_v,i,j,ny,nx);
	    float lambda_vd_t = get_hfacet(lambda_vd,i,j,ny,nx);
	    float lam_eff = lambda_v_t - lambda_vd_t;

	    atomicAdd(&s_adj_v[bi][bj],     lam_eff * j_tau_by.d_v_c);
	    atomicAdd(&s_adj_v[bi-1][bj],   lam_eff * j_tau_by.d_v_t);
	    atomicAdd(&s_adj_v[bi+1][bj],   lam_eff * j_tau_by.d_v_b);
            atomicAdd(&s_adj_u[bi-1][bj],   lam_eff * j_tau_by.d_u_tl);
            atomicAdd(&s_adj_u[bi-1][bj+1], lam_eff * j_tau_by.d_u_tr);
            atomicAdd(&s_adj_u[bi][bj],     lam_eff * j_tau_by.d_u_bl);
            atomicAdd(&s_adj_u[bi][bj+1],   lam_eff * j_tau_by.d_u_br);

	    atomicAdd(&s_adj_vd[bi][bj],     -lam_eff * j_tau_by.d_v_c);
	    atomicAdd(&s_adj_vd[bi-1][bj],   -lam_eff * j_tau_by.d_v_t);
	    atomicAdd(&s_adj_vd[bi+1][bj],   -lam_eff * j_tau_by.d_v_b);
            atomicAdd(&s_adj_ud[bi-1][bj],   -lam_eff * j_tau_by.d_u_tl);
            atomicAdd(&s_adj_ud[bi-1][bj+1], -lam_eff * j_tau_by.d_u_tr);
            atomicAdd(&s_adj_ud[bi][bj],     -lam_eff * j_tau_by.d_u_bl);
            atomicAdd(&s_adj_ud[bi][bj+1],   -lam_eff * j_tau_by.d_u_br);

	    atomicAdd(&s_adj_H[bi-1][bj],  lam_eff * j_tau_by.d_H_t);
	    atomicAdd(&s_adj_H[bi][bj],    lam_eff * j_tau_by.d_H_b);

	    }


	    {
            float lambda_v_t    = get_hfacet(lambda_v,i,j,ny,nx);
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_t = get_cell(bed,i-1,j,ny,nx);
	    float bed_c = get_cell(bed,i,j,ny,nx);
	    float phi_t  = get_cell(phi,i-1,j,ny,nx);
	    float phi_c  = get_cell(phi,i,j,ny,nx);

	    TauDyJacobian j_tau_dy = get_tau_dy_jac({H_t,H_c,bed_t,bed_c,phi_t,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);
	    atomicAdd(&s_adj_H[bi-1][bj],-lambda_v_t * j_tau_dy.d_H_t);
	    atomicAdd(&s_adj_H[bi][bj],  -lambda_v_t * j_tau_dy.d_H_b);
	    }

	    if (use_forcing) atomicAdd(&s_adj_v[bi][bj], -get_hfacet(f_v,i,j,ny,nx));
	    if (use_forcing) atomicAdd(&s_adj_vd[bi][bj], -get_hfacet(f_vd,i,j,ny,nx));
	}

    }
    __syncthreads();

    // Fold the accumulated eta-column adjoint through d(eta)/d(H): each
    // thread owns its own cell of both tiles, so no atomics are needed.
    s_adj_H[bi][bj] += s_adj_eta[bi][bj] * deta_dH_local[bi][bj];

    __syncthreads();

    // Global Base indices for thread(0,0) of this block
    int g_base_y = blockIdx.y * stride - halo;
    int g_base_x = blockIdx.x * stride - halo;


    // Flushes are pure bounds checks: constrained COLUMNS are genuine
    // entries of J^T and are kept (see constraint convention in common.cu)

    // Flush U (16x17)
    for (int k = tid; k < 272; k += 256) {

        int r = k / 17;
        int c = k % 17;
        float val = s_adj_u[r][c];
        if (fabsf(val) > 0.0f) {
            int gy = g_base_y + r;
            int gx = g_base_x + c;
            if (gy >= 0 && gy < ny && gx >= 0 && gx <= nx)
                atomicAdd(&vjp_u[gy * (nx+1) + gx], val);
        }
    }

    // Flush UD (16x17)
    for (int k = tid; k < 272; k += 256) {

        int r = k / 17;
        int c = k % 17;
        float val = s_adj_ud[r][c];
        if (fabsf(val) > 0.0f) {
            int gy = g_base_y + r;
            int gx = g_base_x + c;
            if (gy >= 0 && gy < ny && gx >= 0 && gx <= nx)
                atomicAdd(&vjp_ud[gy * (nx+1) + gx], val);
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

    // Flush VD (17x16)
    for (int k = tid; k < 272; k += 256) {
        int r = k / 16;
        int c = k % 16;
        float val = s_adj_vd[r][c];
        if (fabsf(val) > 0.0f) {
            int gy = g_base_y + r;
            int gx = g_base_x + c;
            if (gy >= 0 && gy <= ny && gx >= 0 && gx < nx)
                atomicAdd(&vjp_vd[gy * nx + gx], val);
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
