extern "C" __global__
void compute_grad_alpha(
    float* __restrict__ grad_alpha_u,
    float* __restrict__ grad_alpha_v,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ H,
    const float* __restrict__ lambda_u,
    const float* __restrict__ lambda_v,
    const float* __restrict__ lambda_H,
    const float* __restrict__ alpha_u,
    const float* __restrict__ alpha_v,
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

	    float alpha_u_l = get_vfacet(alpha_u,i,j,ny,nx);
	    float alpha_v_tl = get_hfacet(alpha_v,i,j-1,ny,nx);
	    float alpha_v_tr = get_hfacet(alpha_v,i,j,ny,nx);
	    float alpha_v_bl = get_hfacet(alpha_v,i+1,j-1,ny,nx);
	    float alpha_v_br = get_hfacet(alpha_v,i+1,j,ny,nx);

	    AlphaBarUJacobian alpha_bar_u_l = get_alpha_bar_u_jac({alpha_u_l,alpha_v_tl,alpha_v_tr,alpha_v_bl,alpha_v_br});

            float u_l    = get_vfacet(u,i,j,ny,nx);
            float lambda_u_l = get_vfacet(lambda_u,i,j,ny,nx);
	    TauBxFrozenJacobian tau_bx = get_tau_bx_frozen_jac({u_l,alpha_bar_u_l.res});

	    atomicAdd(&grad_alpha_u[i * (nx + 1) + j], lambda_u_l * tau_bx.d_alpha * alpha_bar_u_l.d_alpha_u);
	    atomicAdd(&grad_alpha_v[i * nx + max(j-1,0)], lambda_u_l * tau_bx.d_alpha * alpha_bar_u_l.d_alpha_v_tl);
	    atomicAdd(&grad_alpha_v[i * nx + min(j,nx-1)], lambda_u_l * tau_bx.d_alpha * alpha_bar_u_l.d_alpha_v_tr);
	    atomicAdd(&grad_alpha_v[(i + 1) * nx + max(j-1,0)], lambda_u_l * tau_bx.d_alpha * alpha_bar_u_l.d_alpha_v_bl); 
	    atomicAdd(&grad_alpha_v[(i + 1) * nx + min(j,nx-1)], lambda_u_l * tau_bx.d_alpha * alpha_bar_u_l.d_alpha_v_br); 

 	}

	if (has_v){

	    float alpha_v_t = get_hfacet(alpha_v,i,j,ny,nx);
	    float alpha_u_tl = get_vfacet(alpha_u,i-1,j,ny,nx);
	    float alpha_u_tr = get_vfacet(alpha_u,i-1,j+1,ny,nx);
	    float alpha_u_bl = get_vfacet(alpha_u,i,j,ny,nx);
	    float alpha_u_br = get_vfacet(alpha_u,i,j+1,ny,nx);

	    AlphaBarVJacobian alpha_bar_v_t = get_alpha_bar_v_jac({alpha_v_t,alpha_u_tl,alpha_u_tr,alpha_u_bl,alpha_u_br});

	    float v_t = get_hfacet(v,i,j,ny,nx);
	    float lambda_v_t = get_hfacet(lambda_v,i,j,ny,nx);

	    TauByFrozenJacobian tau_by = get_tau_by_frozen_jac({v_t,alpha_bar_v_t.res});	    
	    
	    atomicAdd(&grad_alpha_v[i * nx + j], lambda_v_t * tau_by.d_alpha * alpha_bar_v_t.d_alpha_v);
	    atomicAdd(&grad_alpha_u[max(i - 1,0) * (nx + 1) + j], lambda_v_t * tau_by.d_alpha * alpha_bar_v_t.d_alpha_u_tl);
	    atomicAdd(&grad_alpha_u[max(i - 1,0) * (nx + 1) + j + 1], lambda_v_t * tau_by.d_alpha * alpha_bar_v_t.d_alpha_u_tr);
	    atomicAdd(&grad_alpha_u[min(i,ny-1) * (nx + 1) + j], lambda_v_t * tau_by.d_alpha * alpha_bar_v_t.d_alpha_u_bl);
	    atomicAdd(&grad_alpha_u[min(i,ny-1) * (nx + 1) + j + 1], lambda_v_t * tau_by.d_alpha * alpha_bar_v_t.d_alpha_u_br);
	}
    }
}

extern "C" __global__ void compute_grad_beta(
    float* __restrict__ grad_beta,     // Output: (ny, nx + 1)
    const float* __restrict__ grad_alpha_u,
    const float* __restrict__ grad_alpha_v,
    const float* __restrict__ u,      // u - velocity (ny, nx + 1)
    const float* __restrict__ v,      // v - velocity: (ny + 1, nx)
    const float* __restrict__ grounded,      // Thickness: (ny, nx)
    float m,
    float eps_sliding,
    int ny, int nx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < ny * (nx+1)) {
	int i = idx / (nx + 1);
	int j = idx % (nx + 1);

	float grounded_l = get_cell(grounded,i,j-1,ny,nx);
        float grounded_r = get_cell(grounded,i,j,ny,nx);

        float u_c = get_vfacet(u,i,j,ny,nx);
        float v_tl = get_hfacet(v,i,j-1,ny,nx);
        float v_tr = get_hfacet(v,i,j,ny,nx);
        float v_bl = get_hfacet(v,i+1,j-1,ny,nx);
        float v_br = get_hfacet(v,i+1,j,ny,nx);
        
	float unorm_sq = u_c * u_c + 0.25f*(v_tl * v_tl + v_tr * v_tr + v_bl * v_bl + v_br * v_br);
        float g_alpha_u = get_vfacet(grad_alpha_u,i,j,ny,nx);

        atomicAdd(&grad_beta[i * nx + max(j - 1,  0)], 0.5f * g_alpha_u * __powf(unorm_sq + eps_sliding,(m - 1.0f)/2.0f) * grounded_l);	
        atomicAdd(&grad_beta[i * nx + min(j, nx - 1)], 0.5f * g_alpha_u * __powf(unorm_sq + eps_sliding,(m - 1.0f)/2.0f) * grounded_r);	
    } else if ( idx >= ny * (nx + 1) && idx < ny * (nx + 1) + (ny + 1) * nx ) {
        int idx_v = idx - ( ny * (nx + 1));
	int i = idx_v / nx;
	int j = idx_v % nx;

        float grounded_t = get_cell(grounded,i-1,j,ny,nx);
        float grounded_b = get_cell(grounded,i,j,ny,nx);

        float v_c = get_hfacet(v,i,j,ny,nx);
	float u_tl = get_vfacet(u,i-1,j,ny,nx);
	float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
	float u_bl = get_vfacet(u,i,j,ny,nx);
	float u_br = get_vfacet(u,i,j+1,ny,nx);

	float unorm_sq = v_c * v_c + 0.25f*(u_tl * u_tl + u_tr * u_tr + u_bl * u_bl + u_br * u_br);
        float g_alpha_v = get_hfacet(grad_alpha_v,i,j,ny,nx);

        atomicAdd(&grad_beta[max(i - 1,  0) * nx + j], 0.5f * g_alpha_v * __powf(unorm_sq + eps_sliding,(m - 1.0f)/2.0f) * grounded_t);	
        atomicAdd(&grad_beta[min(i, ny - 1) * nx + j], 0.5f * g_alpha_v * __powf(unorm_sq + eps_sliding,(m - 1.0f)/2.0f) * grounded_b);	
    }
}


