extern "C" __global__
void compute_gradient_beta(
    float* __restrict__ grad_beta,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ lambda_u,
    const float* __restrict__ lambda_v,
    const float* __restrict__ lambda_H,
    const float* __restrict__ phi,
    const float* __restrict__ xi,
    const float* __restrict__ mask,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float n, float eps_reg, float flotation_reg_driving,
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

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    bool has_u    = i >= 0 && i <  ny && j >= 0 && j <= nx;
    bool has_v    = i >= 0 && i <= ny && j >= 0 && j <  nx;


    if ( is_active ) {

	// Residual for the u-momentum equation on the left side of the cell
	// the right side residual is handled by the next cell to the right!
	
	if (has_u){

            float u_l    = get_vfacet(u,i,j,ny,nx);
            float u_ll    = get_vfacet(u,i,j-1,ny,nx);
            float u_r    = get_vfacet(u,i,j+1,ny,nx);
            float v_tl   = get_hfacet(v,i,j-1,ny,nx);
	    float v_tr   = get_hfacet(v,i,j,ny,nx);
	    float v_bl   = get_hfacet(v,i+1,j-1,ny,nx);
	    float v_br   = get_hfacet(v,i+1,j,ny,nx);

	    float H_l    = get_cell(H,i,j-1,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    //float phi_l  = get_cell(phi,i,j-1,ny,nx);
	    //float phi_c  = get_cell(phi,i,j,ny,nx);
	    float xi_l  = get_cell(xi,i,j-1,ny,nx);
	    float xi_c  = get_cell(xi,i,j,ny,nx);
	    float beta_l = get_cell(beta,i,j-1,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);
	    TauBxJacobian j_tau_bx = get_tau_bx_jac({u_l,u_ll,u_r,v_tl,v_tr,v_bl,v_br,H_l,H_c,xi_l,xi_c,beta_l,beta_c,m,u_reg,water_drag,flotation_reg_sliding});

	    float lambda_u_l = get_vfacet(lambda_u,i,j,ny,nx);

	    if (j>0     )  {atomicAdd(&grad_beta[i * nx + j - 1],lambda_u_l * j_tau_bx.d_beta_l);}
	    if (j<(nx-1))  {atomicAdd(&grad_beta[i * nx + j]    ,lambda_u_l * j_tau_bx.d_beta_r);}
 	}

	if (has_v){

	    float v_t  = get_hfacet(v,i,j,ny,nx);
	    float v_tt  = get_hfacet(v,i-1,j,ny,nx);
	    float v_b  = get_hfacet(v,i+1,j,ny,nx);
            float u_tl = get_vfacet(u,i-1,j,ny,nx);
            float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
            float u_bl = get_vfacet(u,i,j,ny,nx);
            float u_br = get_vfacet(u,i,j+1,ny,nx);

	    float H_t    = get_cell(H,i-1,j,ny,nx);
	    float H_c    = get_cell(H,i,j,ny,nx);
	    //float phi_t  = get_cell(phi,i-1,j,ny,nx);
	    //float phi_c  = get_cell(phi,i,j,ny,nx);
	    float xi_t  = get_cell(xi,i-1,j,ny,nx);
	    float xi_c  = get_cell(xi,i,j,ny,nx);
	    float beta_t = get_cell(beta,i-1,j,ny,nx);
	    float beta_c = get_cell(beta,i,j,ny,nx);

	    TauByJacobian j_tau_by = get_tau_by_jac({v_t,v_tt,v_b,u_tl,u_tr,u_bl,u_br,H_t,H_c,xi_t,xi_c,beta_t,beta_c,m,u_reg,water_drag,flotation_reg_sliding});
	    
	    float lambda_v_t = get_hfacet(lambda_v,i,j,ny,nx);
	    
	    if (i>0     ) {atomicAdd(&grad_beta[(i-1) * nx + j],lambda_v_t * j_tau_by.d_beta_t);}
	    if (i<(ny-1)) {atomicAdd(&grad_beta[i * nx + j]    ,lambda_v_t * j_tau_by.d_beta_b);}
	}
    }
}

extern "C" __global__
void compute_gradient_bed(
    float* __restrict__ grad_bed,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ lambda_u,
    const float* __restrict__ lambda_v,
    const float* __restrict__ lambda_H,
    const float* __restrict__ phi,
    const float* __restrict__ xi,
    const float* __restrict__ mask,
    const float* __restrict__ bed,
    const float* __restrict__ B,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float n, float eps_reg, float flotation_reg_driving,
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

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    bool has_u    = i >= 0 && i <  ny && j >= 0 && j <= nx;
    bool has_v    = i >= 0 && i <= ny && j >= 0 && j <  nx;
    
    float dx_inv = 1.0f / dx;

    if ( is_active ) {

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
	    
	    if (j>0     )  {atomicAdd(&grad_bed[i * nx + j - 1],-lambda_u_l * j_tau_dx.d_bed_l);}
	    if (j<(nx-1))  {atomicAdd(&grad_bed[i * nx + j]    ,-lambda_u_l * j_tau_dx.d_bed_r);}
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
	    
	    if (i>0     ) {atomicAdd(&grad_bed[(i-1) * nx + j],-lambda_v_t * j_tau_dy.d_bed_t);}
	    if (i<(ny-1)) {atomicAdd(&grad_bed[i * nx + j]    ,-lambda_v_t * j_tau_dy.d_bed_b);}
	    }	    
	}
    }
}
