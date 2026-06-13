// Auto-generated from cuda/residuals.cu by _gen_residuals.py.
// Flat one-thread-per-(i,j) port; eta_local tile -> eta_at/eta_dual_at;
// compute_vjp shared-tile scatter -> direct global scatter_u/v/H.
// Relies on common/viscosity/stress/flux .metal (concatenated first).


inline void scatter_u(device atomic_uint* vjp_u, int gy, int gx, float val, int ny, int nx) {
    if (gy >= 0 && gy < ny && gx > 0 && gx < nx)
        atomic_add_float(&vjp_u[gy * (nx + 1) + gx], val);
}
inline void scatter_v(device atomic_uint* vjp_v, int gy, int gx, float val, int ny, int nx) {
    if (gy > 0 && gy < ny && gx >= 0 && gx < nx)
        atomic_add_float(&vjp_v[gy * nx + gx], val);
}
inline void scatter_H(device atomic_uint* vjp_H, device const float* mask,
                      int gy, int gx, float val, int ny, int nx) {
    if (gy >= 0 && gy < ny && gx >= 0 && gx < nx) {
        if (get_cell(mask, gy, gx, ny, nx) > 0.0f) val = 0.0f;
        atomic_add_float(&vjp_H[gy * nx + gx], val);
    }
}

kernel void compute_residual(
    device float* r_u [[buffer(0)]],
    device float* r_v [[buffer(1)]],
    device float* r_H [[buffer(2)]],
    device const float* u [[buffer(3)]],
    device const float* v [[buffer(4)]],
    device const float* H [[buffer(5)]],
    device const float* phi [[buffer(6)]],
    device const float* mask [[buffer(7)]],
    device const float* f_u [[buffer(8)]],
    device const float* f_v [[buffer(9)]],
    device const float* f_H [[buffer(10)]],
    device const float* bed [[buffer(11)]],
    device const float* B [[buffer(12)]],
    device const float* beta [[buffer(13)]],
    device const float* gamma [[buffer(14)]],
    device const float* params [[buffer(15)]],
    uint tid [[thread_position_in_grid]]
)
{
    bool use_forcing = params[0] != 0.0f;
    bool use_mask = params[1] != 0.0f;
    float n = params[2];
    float eps_reg = params[3];
    float flotation_reg_driving = params[4];
    float m = params[5];
    float u_reg = params[6];
    float water_drag = params[7];
    float flotation_reg_sliding = params[8];
    float calving_rate = params[9];
    float flotation_reg_calving = params[10];
    float dx = params[11];
    float dt = params[12];
    int ny = (int)params[13];
    int nx = (int)params[14];

    int i = (int)tid / (nx + 1);
    int j = (int)tid % (nx + 1);
    if (i > ny || j > nx) return;
    {

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

	    float ru_l = 0.0f;
	    if (use_forcing) ru_l -= get_vfacet(f_u,i,j,ny,nx);

	    {
	    float eta_c = eta_at(i, j, u, v, B, n, eps_reg, dx, ny, nx);
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
	    float eta_l  = eta_at(i, j - 1, u, v, B, n, eps_reg, dx, ny, nx);
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
	    float eta_tl = eta_at(i - 1, j - 1, u, v, B, n, eps_reg, dx, ny, nx);
	    float eta_t  = eta_at(i - 1, j, u, v, B, n, eps_reg, dx, ny, nx);
	    float eta_l  = eta_at(i, j - 1, u, v, B, n, eps_reg, dx, ny, nx);
	    float eta_c  = eta_at(i, j, u, v, B, n, eps_reg, dx, ny, nx);
	    
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
	    float eta_l  = eta_at(i, j - 1, u, v, B, n, eps_reg, dx, ny, nx);
	    float eta_c  = eta_at(i, j, u, v, B, n, eps_reg, dx, ny, nx);
	    float eta_bl = eta_at(i + 1, j - 1, u, v, B, n, eps_reg, dx, ny, nx);
	    float eta_b  = eta_at(i + 1, j, u, v, B, n, eps_reg, dx, ny, nx);
	    
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
            float v_tl   = get_hfacet(v,i,j-1,ny,nx);
	    float v_tr   = get_hfacet(v,i,j,ny,nx);
	    float v_bl   = get_hfacet(v,i+1,j-1,ny,nx);
	    float v_br   = get_hfacet(v,i+1,j,ny,nx);

	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float phi_l = get_cell(phi,i,j-1,ny,nx);
	    float phi_c = get_cell(phi,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    TauBxJacobian tau_bx = get_tau_bx_jac({u_l,v_tl,v_tr,v_bl,v_br,H_l,H_c,phi_l,phi_c,beta_l,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
	    ru_l += tau_bx.res;
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
	    r_u[i * (nx + 1) + j] = ru_l;
	}

	if (has_v){

	    float rv_t = 0.0f;
	    if (use_forcing) rv_t -= get_hfacet(f_v,i,j,ny,nx);

	    {
	    float eta_t = eta_at(i - 1, j, u, v, B, n, eps_reg, dx, ny, nx);
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
	    float eta_c = eta_at(i, j, u, v, B, n, eps_reg, dx, ny, nx);
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
	    float eta_tl = eta_at(i - 1, j - 1, u, v, B, n, eps_reg, dx, ny, nx);
	    float eta_t  = eta_at(i - 1, j, u, v, B, n, eps_reg, dx, ny, nx);
	    float eta_l  = eta_at(i, j - 1, u, v, B, n, eps_reg, dx, ny, nx);
	    float eta_c  = eta_at(i, j, u, v, B, n, eps_reg, dx, ny, nx);

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
	    float eta_t  = eta_at(i - 1, j, u, v, B, n, eps_reg, dx, ny, nx);
	    float eta_tr = eta_at(i - 1, j + 1, u, v, B, n, eps_reg, dx, ny, nx);
	    float eta_c  = eta_at(i, j, u, v, B, n, eps_reg, dx, ny, nx);
	    float eta_r = eta_at(i, j + 1, u, v, B, n, eps_reg, dx, ny, nx);

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
            float u_tl = get_vfacet(u,i-1,j,ny,nx);
            float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
            float u_bl = get_vfacet(u,i,j,ny,nx);
            float u_br = get_vfacet(u,i,j+1,ny,nx);

	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float phi_t = get_cell(phi,i-1,j,ny,nx);
	    float phi_c = get_cell(phi,i,j,ny,nx);
	    float beta_t = get_cell(beta,i-1,j,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);

	    TauByJacobian tau_by = get_tau_by_jac({v_t,u_tl,u_tr,u_bl,u_br,H_t,H_c,phi_t,phi_c,beta_t,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
	    rv_t += tau_by.res;
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

	    r_v[i * nx + j] = rv_t;
	}
    
    }
}

kernel void compute_jvp(
    device float* jvp_u [[buffer(0)]],
    device float* jvp_v [[buffer(1)]],
    device float* jvp_H [[buffer(2)]],
    device const float* u [[buffer(3)]],
    device const float* v [[buffer(4)]],
    device const float* H [[buffer(5)]],
    device const float* d_u [[buffer(6)]],
    device const float* d_v [[buffer(7)]],
    device const float* d_H [[buffer(8)]],
    device const float* phi [[buffer(9)]],
    device const float* mask [[buffer(10)]],
    device const float* f_u [[buffer(11)]],
    device const float* f_v [[buffer(12)]],
    device const float* f_H [[buffer(13)]],
    device const float* bed [[buffer(14)]],
    device const float* B [[buffer(15)]],
    device const float* beta [[buffer(16)]],
    device const float* gamma [[buffer(17)]],
    device const float* params [[buffer(18)]],
    uint tid [[thread_position_in_grid]]
)
{
    bool use_mask = params[0] != 0.0f;
    float n = params[1];
    float eps_reg = params[2];
    float flotation_reg_driving = params[3];
    float m = params[4];
    float u_reg = params[5];
    float water_drag = params[6];
    float flotation_reg_sliding = params[7];
    float calving_rate = params[8];
    float flotation_reg_calving = params[9];
    float dx = params[10];
    float dt = params[11];
    int ny = (int)params[12];
    int nx = (int)params[13];

    int i = (int)tid / (nx + 1);
    int j = (int)tid % (nx + 1);
    if (i > ny || j > nx) return;
    {

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

	    float masked = use_mask ? get_cell(mask,i,j,ny,nx) : 0.0f;
            jvp_H[i * nx + j] = (1.0f - masked) * d_rH;

	}

	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	if (has_u){
	    float d_ru_l = 0.0f;

	    {
	    DualFloat eta_c = eta_dual_at(i, j, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
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
	    DualFloat eta_l  = eta_dual_at(i, j - 1, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
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
	    DualFloat eta_tl = eta_dual_at(i - 1, j - 1, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_t  = eta_dual_at(i - 1, j, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_l  = eta_dual_at(i, j - 1, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_c  = eta_dual_at(i, j, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    
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
	    DualFloat eta_l  = eta_dual_at(i, j - 1, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_c  = eta_dual_at(i, j, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_bl = eta_dual_at(i + 1, j - 1, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_b  = eta_dual_at(i + 1, j, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    
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
            DualFloat v_tl   = get_hfacet(v,d_v,i,j-1,ny,nx);
	    DualFloat v_tr   = get_hfacet(v,d_v,i,j,ny,nx);
	    DualFloat v_bl   = get_hfacet(v,d_v,i+1,j-1,ny,nx);
	    DualFloat v_br   = get_hfacet(v,d_v,i+1,j,ny,nx);

	    DualFloat H_l    = get_cell(H,d_H,i,j-1,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
	    float phi_l  = get_cell(phi,i,j-1,ny,nx);
	    float phi_c  = get_cell(phi,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    DualFloat tau_bx = get_tau_bx_dual({u_l,v_tl,v_tr,v_bl,v_br,H_l,H_c,phi_l,phi_c,beta_l,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
	    d_ru_l += tau_bx.d;
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

	    if (j <= 0 || j >= nx) {d_ru_l = 0.0f;}
	    jvp_u[i * (nx + 1) + j] = d_ru_l;
	
 	}
	
	if (has_v){
	    float d_rv_t = 0.0f;

	    {
	    DualFloat eta_t = eta_dual_at(i - 1, j, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
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
	    DualFloat eta_c = eta_dual_at(i, j, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
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
	    DualFloat eta_tl = eta_dual_at(i - 1, j - 1, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_t  = eta_dual_at(i - 1, j, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_l  = eta_dual_at(i, j - 1, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_c  = eta_dual_at(i, j, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    
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
	    DualFloat eta_t  = eta_dual_at(i - 1, j, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_tr = eta_dual_at(i - 1, j + 1, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_c  = eta_dual_at(i, j, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_r  = eta_dual_at(i, j + 1, u, v, d_u, d_v, B, n, eps_reg, dx, ny, nx);

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

            DualFloat u_tl = get_vfacet(u,d_u,i-1,j,ny,nx);
            DualFloat u_tr = get_vfacet(u,d_u,i-1,j+1,ny,nx);
            DualFloat u_bl = get_vfacet(u,d_u,i,j,ny,nx);
            DualFloat u_br = get_vfacet(u,d_u,i,j+1,ny,nx);

	    DualFloat H_t    = get_cell(H,d_H,i-1,j,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
	    float phi_t      = get_cell(phi,i-1,j,ny,nx);
	    float phi_c      = get_cell(phi,i,j,ny,nx);
	    float beta_t     = get_cell(beta,i-1,j,ny,nx);
	    float beta_c     = get_cell(beta,i,j,ny,nx);

	    DualFloat tau_by = get_tau_by_dual({v_t,u_tl,u_tr,u_bl,u_br,H_t,H_c,phi_t,phi_c,beta_t,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
	    d_rv_t += tau_by.d;
	    }

	    {
	    DualFloat H_t    = get_cell(H,d_H,i-1,j,ny,nx);
	    DualFloat H_c    = get_cell(H,d_H,i,j,ny,nx);
	    float bed_t = get_cell(bed,i-1,j,ny,nx);
	    float bed_c = get_cell(bed,i,j,ny,nx);
	    float phi_t      = get_cell(phi,i-1,j,ny,nx);
	    float phi_c      = get_cell(phi,i,j,ny,nx);

	    DualFloat tau_dy = get_tau_dy_dual({H_t,H_c,bed_t,bed_c,phi_t,phi_c,flotation_reg_sliding},dx_inv,i,j,ny,nx);
	    d_rv_t -= tau_dy.d;
	    }

	    if (i <= 0 || i >= ny) { d_rv_t = 0.0f;}
	    jvp_v[i * nx + j] = d_rv_t;

	}
    
    }
}

kernel void compute_vjp(
    device atomic_uint* vjp_u [[buffer(0)]],
    device atomic_uint* vjp_v [[buffer(1)]],
    device atomic_uint* vjp_H [[buffer(2)]],
    device const float* u [[buffer(3)]],
    device const float* v [[buffer(4)]],
    device const float* H [[buffer(5)]],
    device const float* lambda_u [[buffer(6)]],
    device const float* lambda_v [[buffer(7)]],
    device const float* lambda_H [[buffer(8)]],
    device const float* phi [[buffer(9)]],
    device const float* mask [[buffer(10)]],
    device const float* f_u [[buffer(11)]],
    device const float* f_v [[buffer(12)]],
    device const float* f_H [[buffer(13)]],
    device const float* bed [[buffer(14)]],
    device const float* B [[buffer(15)]],
    device const float* beta [[buffer(16)]],
    device const float* gamma [[buffer(17)]],
    device const float* params [[buffer(18)]],
    uint tid [[thread_position_in_grid]]
)
{
    bool use_forcing = params[0] != 0.0f;
    bool use_mask = params[1] != 0.0f;
    float n = params[2];
    float eps_reg = params[3];
    float flotation_reg_driving = params[4];
    float m = params[5];
    float u_reg = params[6];
    float water_drag = params[7];
    float flotation_reg_sliding = params[8];
    float calving_rate = params[9];
    float flotation_reg_calving = params[10];
    float dx = params[11];
    float dt = params[12];
    int ny = (int)params[13];
    int nx = (int)params[14];

    int i = (int)tid / (nx + 1);
    int j = (int)tid % (nx + 1);
    if (i > ny || j > nx) return;
    {
    bool has_cell = i >= 0 && i <  ny && j >= 0 && j <  nx;
    bool has_u    = i >= 0 && i <  ny && j >= 0 && j <= nx;
    bool has_v    = i >= 0 && i <= ny && j >= 0 && j <  nx;

	float dx_inv = 1.0f/dx;

	if (has_cell){
	    float H_c        = get_cell(H,i,j,ny,nx);
	    float lambda_H_c = get_masked_cell(lambda_H,mask,i,j,ny,nx);

	    // Mass matrix contribution
	    scatter_H(vjp_H, mask, i, j, lambda_H_c/dt, ny, nx);

	    float phi_c = get_cell(phi,i,j,ny,nx);

	    float H_l = get_cell(H,i,j-1,ny,nx);
	    float u_l = get_vfacet(u,i,j,ny,nx);
	    HorizontalFluxJacobian j_q_l = get_horizontal_flux_jac({u_l,H_l,H_c}, i, j, ny, nx);
	    scatter_H(vjp_H, mask, i, j, -lambda_H_c*j_q_l.d_H_r*dx_inv, ny, nx);
            scatter_H(vjp_H, mask, i, j - 1, -lambda_H_c*j_q_l.d_H_l*dx_inv, ny, nx);
            scatter_u(vjp_u, i, j, -lambda_H_c*j_q_l.d_u*dx_inv, ny, nx);

	    float phi_l = get_cell(phi,i,j-1,ny,nx);
	    FacetCalvingJacobian j_calve_l = get_facet_calving_jac({H_c,H_l,phi_c,phi_l,calving_rate,flotation_reg_calving},i,j,ny,nx);
	    scatter_H(vjp_H, mask, i, j, lambda_H_c*j_calve_l.d_H_this*dx_inv, ny, nx);
           
	    float H_r = get_cell(H,i,j+1,ny,nx);
	    float u_r = get_vfacet(u,i,j+1,ny,nx);
	    HorizontalFluxJacobian j_q_r = get_horizontal_flux_jac({u_r,H_c,H_r}, i, j + 1, ny, nx);
            scatter_H(vjp_H, mask, i, j, lambda_H_c*j_q_r.d_H_l*dx_inv, ny, nx);
            scatter_H(vjp_H, mask, i, j + 1, lambda_H_c*j_q_r.d_H_r*dx_inv, ny, nx);
            scatter_u(vjp_u, i, j + 1, lambda_H_c*j_q_r.d_u*dx_inv, ny, nx);

	    float phi_r = get_cell(phi,i,j+1,ny,nx);
	    FacetCalvingJacobian j_calve_r = get_facet_calving_jac({H_c,H_r,phi_c,phi_r,calving_rate,flotation_reg_calving},i,j+1,ny,nx);
	    scatter_H(vjp_H, mask, i, j, lambda_H_c*j_calve_r.d_H_this*dx_inv, ny, nx);

	    float H_t = get_cell(H,i-1,j,ny,nx);
	    float v_t = get_hfacet(v,i,j,ny,nx);
	    VerticalFluxJacobian j_q_t = get_vertical_flux_jac({v_t,H_t,H_c}, i, j, ny, nx);
	    scatter_H(vjp_H, mask, i, j, lambda_H_c*j_q_t.d_H_b*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i - 1, j, lambda_H_c*j_q_t.d_H_t*dx_inv, ny, nx);
	    scatter_v(vjp_v, i, j, lambda_H_c*j_q_t.d_v*dx_inv, ny, nx);

	    float phi_t = get_cell(phi,i-1,j,ny,nx);
	    FacetCalvingJacobian j_calve_t = get_facet_calving_jac({H_c,H_t,phi_c,phi_t,calving_rate,flotation_reg_calving},i,j,ny,nx);
	    scatter_H(vjp_H, mask, i, j, lambda_H_c*j_calve_t.d_H_this*dx_inv, ny, nx);


	    float H_b = get_cell(H,i+1,j,ny,nx);
	    float v_b = get_hfacet(v,i+1,j,ny,nx);
	    VerticalFluxJacobian j_q_b = get_vertical_flux_jac({v_b,H_c,H_b}, i + 1, j, ny, nx);
            scatter_H(vjp_H, mask, i, j, -lambda_H_c*j_q_b.d_H_t*dx_inv, ny, nx);
            scatter_H(vjp_H, mask, i + 1, j, -lambda_H_c*j_q_b.d_H_b*dx_inv, ny, nx);
	    scatter_v(vjp_v, i + 1, j, -lambda_H_c*j_q_b.d_v*dx_inv, ny, nx);

	    float phi_b = get_cell(phi,i+1,j,ny,nx);
	    FacetCalvingJacobian j_calve_b = get_facet_calving_jac({H_c,H_b,phi_c,phi_b,calving_rate,flotation_reg_calving},i+1,j,ny,nx);
	    scatter_H(vjp_H, mask, i, j, lambda_H_c*j_calve_b.d_H_this*dx_inv, ny, nx);

	    //float masked = use_mask ? get_cell(mask,i,j,ny,nx) : 0.0f;
	    //float lambda_H_c_ = get_cell(lambda_H,i,j,ny,nx);
            //scatter_H(vjp_H, mask, i, j, (1.0f - masked) * lambda_H_c_, ny, nx);
	    if (use_forcing) scatter_H(vjp_H, mask, i, j, -get_cell(f_H,i,j,ny,nx), ny, nx);
	}

	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	
	if (has_u){
	    {
	    DualFloat eta_c = eta_dual_at(i, j, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
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

            scatter_u(vjp_u, i, j, lambda_sigma_xx_c * dx_inv, ny, nx);
            scatter_H(vjp_H, mask, i, j, lambda_u_l*j_sigma_xx_c.d_eta_H*eta_H_c.d_H*dx_inv, ny, nx);
	    }

	    {
	    DualFloat eta_l  = eta_dual_at(i, j - 1, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
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

	    scatter_u(vjp_u, i, j, -lambda_sigma_xx_l*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i, j - 1, -lambda_u_l*j_sigma_xx_l.d_eta_H*eta_H_l.d_H*dx_inv, ny, nx);
	    }

	    {
	    DualFloat eta_tl = eta_dual_at(i - 1, j - 1, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_t  = eta_dual_at(i - 1, j, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_l  = eta_dual_at(i, j - 1, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_c  = eta_dual_at(i, j, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);

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

	    scatter_u(vjp_u, i, j, lambda_sigma_xy_tl*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i - 1, j - 1, lambda_u_l * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_tl*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i - 1, j, lambda_u_l * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_tr*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i, j - 1, lambda_u_l * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_bl*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i, j, lambda_u_l * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_br*dx_inv, ny, nx);
	    }

	    {
	    DualFloat eta_l  = eta_dual_at(i, j - 1, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_c  = eta_dual_at(i, j, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_bl = eta_dual_at(i + 1, j - 1, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_b  = eta_dual_at(i + 1, j, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
	    
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
    
	    scatter_u(vjp_u, i, j, -lambda_sigma_xy_bl*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i, j - 1, -lambda_u_l * j_sigma_xy_bl.d_eta_H*eta_H_bl.d_H_tl*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i, j, -lambda_u_l * j_sigma_xy_bl.d_eta_H*eta_H_bl.d_H_tr*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i + 1, j - 1, -lambda_u_l * j_sigma_xy_bl.d_eta_H*eta_H_bl.d_H_bl*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i + 1, j, -lambda_u_l * j_sigma_xy_bl.d_eta_H*eta_H_bl.d_H_br*dx_inv, ny, nx);
	    }
            
            {    
            float u_l    = get_vfacet(u,i,j,ny,nx);
            float v_tl   = get_hfacet(v,i,j-1,ny,nx);
	    float v_tr   = get_hfacet(v,i,j,ny,nx);
	    float v_bl   = get_hfacet(v,i+1,j-1,ny,nx);
	    float v_br   = get_hfacet(v,i+1,j,ny,nx);

	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float phi_l  = get_cell(phi,i,j-1,ny,nx);
	    float phi_c  = get_cell(phi,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    TauBxJacobian j_tau_bx = get_tau_bx_jac({u_l,v_tl,v_tr,v_bl,v_br,H_l,H_c,phi_l,phi_c,beta_l,beta_c,m,u_reg,water_drag,flotation_reg_sliding});


	    float lambda_u_l = get_vfacet(lambda_u,i,j,ny,nx);
	    scatter_u(vjp_u, i, j, lambda_u_l * j_tau_bx.d_u, ny, nx);
	    scatter_v(vjp_v, i, j - 1, lambda_u_l * j_tau_bx.d_v_tl, ny, nx);
	    scatter_v(vjp_v, i, j, lambda_u_l * j_tau_bx.d_v_tr, ny, nx);
	    scatter_v(vjp_v, i + 1, j - 1, lambda_u_l * j_tau_bx.d_v_bl, ny, nx);
	    scatter_v(vjp_v, i + 1, j, lambda_u_l * j_tau_bx.d_v_br, ny, nx);
	    scatter_H(vjp_H, mask, i, j - 1, lambda_u_l * j_tau_bx.d_H_l, ny, nx);
	    scatter_H(vjp_H, mask, i, j, lambda_u_l * j_tau_bx.d_H_r, ny, nx);

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

            scatter_H(vjp_H, mask, i, j - 1, -lambda_u_l * j_tau_dx.d_H_l, ny, nx);
            scatter_H(vjp_H, mask, i, j, -lambda_u_l * j_tau_dx.d_H_r, ny, nx);
	    }

	    if (use_forcing) scatter_u(vjp_u, i, j, -get_vfacet(f_u,i,j,ny,nx), ny, nx);
	
 	}

	if (has_v){
	    {
	    DualFloat eta_t = eta_dual_at(i - 1, j, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
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

	    scatter_v(vjp_v, i, j, lambda_sigma_yy_t * dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i - 1, j, lambda_v_t*j_sigma_yy_t.d_eta_H*eta_H_t.d_H*dx_inv, ny, nx);
	    }

	    {
	    DualFloat eta_c = eta_dual_at(i, j, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
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
	    scatter_v(vjp_v, i, j, -lambda_sigma_yy_c*dx_inv, ny, nx);
            scatter_H(vjp_H, mask, i, j, -lambda_v_t*j_sigma_yy_c.d_eta_H*eta_H_c.d_H*dx_inv, ny, nx);
	    }
	    
	    {
	    DualFloat eta_tl = eta_dual_at(i - 1, j - 1, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_t  = eta_dual_at(i - 1, j, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_l  = eta_dual_at(i, j - 1, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_c  = eta_dual_at(i, j, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);

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

	    scatter_v(vjp_v, i, j, -lambda_sigma_xy_tl*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i - 1, j - 1, -lambda_v_t * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_tl*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i - 1, j, -lambda_v_t * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_tr*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i, j - 1, -lambda_v_t * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_bl*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i, j, -lambda_v_t * j_sigma_xy_tl.d_eta_H*eta_H_tl.d_H_br*dx_inv, ny, nx);
	    }

	    {
	    DualFloat eta_t  = eta_dual_at(i - 1, j, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_tr = eta_dual_at(i - 1, j + 1, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_c  = eta_dual_at(i, j, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);
	    DualFloat eta_r = eta_dual_at(i, j + 1, u, v, lambda_u, lambda_v, B, n, eps_reg, dx, ny, nx);

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

	    scatter_v(vjp_v, i, j, lambda_sigma_xy_tr*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i - 1, j, lambda_v_t * j_sigma_xy_tr.d_eta_H*eta_H_tr.d_H_tl*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i - 1, j + 1, lambda_v_t * j_sigma_xy_tr.d_eta_H*eta_H_tr.d_H_tr*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i, j, lambda_v_t * j_sigma_xy_tr.d_eta_H*eta_H_tr.d_H_bl*dx_inv, ny, nx);
	    scatter_H(vjp_H, mask, i, j + 1, lambda_v_t * j_sigma_xy_tr.d_eta_H*eta_H_tr.d_H_br*dx_inv, ny, nx);            
	    }
             
	    
	    {
	    float v_t  = get_hfacet(v,i,j,ny,nx);
            float u_tl = get_vfacet(u,i-1,j,ny,nx);
            float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
            float u_bl = get_vfacet(u,i,j,ny,nx);
            float u_br = get_vfacet(u,i,j+1,ny,nx);

	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float phi_t  = get_cell(phi,i-1,j,ny,nx);
	    float phi_c  = get_cell(phi,i,j,ny,nx);
	    float beta_t = get_cell(beta,i-1,j,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);

	    TauByJacobian j_tau_by = get_tau_by_jac({v_t,u_tl,u_tr,u_bl,u_br,H_t,H_c,phi_t,phi_c,beta_t,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
	    
	    float lambda_v_t = get_hfacet(lambda_v,i,j,ny,nx);
	    
	    scatter_v(vjp_v, i, j, lambda_v_t * j_tau_by.d_v, ny, nx);
            scatter_u(vjp_u, i - 1, j, lambda_v_t * j_tau_by.d_u_tl, ny, nx);
            scatter_u(vjp_u, i - 1, j + 1, lambda_v_t * j_tau_by.d_u_tr, ny, nx);
            scatter_u(vjp_u, i, j, lambda_v_t * j_tau_by.d_u_bl, ny, nx);
            scatter_u(vjp_u, i, j + 1, lambda_v_t * j_tau_by.d_u_br, ny, nx);
	    scatter_H(vjp_H, mask, i - 1, j, lambda_v_t * j_tau_by.d_H_t, ny, nx);
	    scatter_H(vjp_H, mask, i, j, lambda_v_t * j_tau_by.d_H_b, ny, nx);

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
	    scatter_H(vjp_H, mask, i - 1, j, -lambda_v_t * j_tau_dy.d_H_t, ny, nx);
	    scatter_H(vjp_H, mask, i, j, -lambda_v_t * j_tau_dy.d_H_b, ny, nx);
	    }
	    
	    if (use_forcing) scatter_v(vjp_v, i, j, -get_hfacet(f_v,i,j,ny,nx), ny, nx);
	}

    
    }
}
