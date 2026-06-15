// Auto-generated from cuda/grad.cu by the metal port.
// Flat scatter of adjoint gradients (grad_beta/grad_bed atomic_uint).
// Relies on common/stress .metal (concatenated first).

kernel void compute_gradient_beta(
    device atomic_uint* grad_beta [[buffer(0)]],
    device const float* u [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* H [[buffer(3)]],
    device const float* lambda_u [[buffer(4)]],
    device const float* lambda_v [[buffer(5)]],
    device const float* lambda_H [[buffer(6)]],
    device const float* phi [[buffer(7)]],
    device const float* mask [[buffer(8)]],
    device const float* bed [[buffer(9)]],
    device const float* B [[buffer(10)]],
    device const float* beta [[buffer(11)]],
    device const float* gamma [[buffer(12)]],
    device const float* params [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
)
{
    float n = params[0];
    float eps_reg = params[1];
    float flotation_reg_driving = params[2];
    float m = params[3];
    float u_reg = params[4];
    float water_drag = params[5];
    float flotation_reg_sliding = params[6];
    float calving_rate = params[7];
    float flotation_reg_calving = params[8];
    float dx = params[9];
    float dt = params[10];
    int ny = (int)params[11];
    int nx = (int)params[12];
    int stride = (int)params[13];
    int halo = (int)params[14];

    int i = (int)tid / (nx + 1);
    int j = (int)tid % (nx + 1);
    if (i > ny || j > nx) return;
    float dx_inv = 1.0f / dx;
    bool has_u = i >= 0 && i <  ny && j >= 0 && j <= nx;
    bool has_v = i >= 0 && i <= ny && j >= 0 && j <  nx;
    {


	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	
	if (has_u){

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

	    if (j>0     )  {atomic_add_float(&grad_beta[i * nx + j - 1],lambda_u_l * j_tau_bx.d_beta_l);}
	    if (j<(nx-1))  {atomic_add_float(&grad_beta[i * nx + j]    ,lambda_u_l * j_tau_bx.d_beta_r);}
 	}

	if (has_v){

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
	    
	    if (i>0     ) {atomic_add_float(&grad_beta[(i-1) * nx + j],lambda_v_t * j_tau_by.d_beta_t);}
	    if (i<(ny-1)) {atomic_add_float(&grad_beta[i * nx + j]    ,lambda_v_t * j_tau_by.d_beta_b);}
	}
    
    }
}

kernel void compute_gradient_bed(
    device atomic_uint* grad_bed [[buffer(0)]],
    device const float* u [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* H [[buffer(3)]],
    device const float* lambda_u [[buffer(4)]],
    device const float* lambda_v [[buffer(5)]],
    device const float* lambda_H [[buffer(6)]],
    device const float* phi [[buffer(7)]],
    device const float* mask [[buffer(8)]],
    device const float* bed [[buffer(9)]],
    device const float* B [[buffer(10)]],
    device const float* beta [[buffer(11)]],
    device const float* gamma [[buffer(12)]],
    device const float* params [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
)
{
    float n = params[0];
    float eps_reg = params[1];
    float flotation_reg_driving = params[2];
    float m = params[3];
    float u_reg = params[4];
    float water_drag = params[5];
    float flotation_reg_sliding = params[6];
    float calving_rate = params[7];
    float flotation_reg_calving = params[8];
    float dx = params[9];
    float dt = params[10];
    int ny = (int)params[11];
    int nx = (int)params[12];
    int stride = (int)params[13];
    int halo = (int)params[14];

    int i = (int)tid / (nx + 1);
    int j = (int)tid % (nx + 1);
    if (i > ny || j > nx) return;
    float dx_inv = 1.0f / dx;
    bool has_u = i >= 0 && i <  ny && j >= 0 && j <= nx;
    bool has_v = i >= 0 && i <= ny && j >= 0 && j <  nx;
    {


	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	
	if (has_u){
	    {
	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    
	    float bed_l  = get_cell(bed,i,j-1,ny,nx);
	    float bed_c  = get_cell(bed,i,j,ny,nx);
	    float phi_l  = get_cell(phi,i,j-1,ny,nx);
	    float phi_c  = get_cell(phi,i,j,ny,nx);
	    TauDxJacobian j_tau_dx = get_tau_dx_jac({H_l,H_c,bed_l,bed_c,phi_l,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);

            float lambda_u_l    = get_vfacet(lambda_u,i,j,ny,nx);
	    
	    if (j>0     )  {atomic_add_float(&grad_bed[i * nx + j - 1],-lambda_u_l * j_tau_dx.d_bed_l);}
	    if (j<(nx-1))  {atomic_add_float(&grad_bed[i * nx + j]    ,-lambda_u_l * j_tau_dx.d_bed_r);}
	    }
 	}

	if (has_v){
	    {
	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    float bed_t = get_cell(bed,i-1,j,ny,nx);
	    float bed_c = get_cell(bed,i,j,ny,nx);
	    float phi_t  = get_cell(phi,i-1,j,ny,nx);
	    float phi_c  = get_cell(phi,i,j,ny,nx);

	    TauDyJacobian j_tau_dy = get_tau_dy_jac({H_t,H_c,bed_t,bed_c,phi_t,phi_c,flotation_reg_driving},dx_inv,i,j,ny,nx);
            
	    float lambda_v_t    = get_hfacet(lambda_v,i,j,ny,nx);
	    
	    if (i>0     ) {atomic_add_float(&grad_bed[(i-1) * nx + j],-lambda_v_t * j_tau_dy.d_bed_t);}
	    if (i<(ny-1)) {atomic_add_float(&grad_bed[i * nx + j]    ,-lambda_v_t * j_tau_dy.d_bed_b);}
	    }	    
	}
    
    }
}
