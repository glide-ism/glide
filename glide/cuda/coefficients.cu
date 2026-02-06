/*==============================================================
  FROZEN FIELDS: Precomputed coefficients for Picard linearization

  These fields are computed once at the finest grid level before
  a V-cycle and then restricted to coarser grids, rather than
  being recomputed at each level. This ensures operator consistency
  across the multigrid hierarchy.
  ==============================================================*/

/*--------------------------------------------------------------
  Compute cell-centered viscosity field eta

  eta = 0.5 * B * eps_II^((1-n)/(2n))

  where eps_II is the second invariant of the strain rate tensor.
  --------------------------------------------------------------*/
extern "C" __global__ void compute_eta(
    float* __restrict__ eta,          // Output: (ny, nx)
    const float* __restrict__ u,      // u-velocity: (ny, nx+1)
    const float* __restrict__ v,      // v-velocity: (ny+1, nx)
    const float* __restrict__ B,      // Rate factor: (ny, nx)
    float n,                          // Glen's exponent
    float eps_reg,                    // Strain rate regularization
    float dx,
    int ny, int nx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ny * nx) return;

    int i = idx / nx;  // row (y direction)
    int j = idx % nx;  // column (x direction)

    float dx_inv = 1.0f / dx;
    float glen_exp = (1.0f - n) / (2.0f * n);

    // Get velocities around cell (i,j)
    float u_l = get_vfacet(u,i,j,ny,nx);
    float u_r = get_vfacet(u,i,j+1,ny,nx);
    float v_t = get_hfacet(v,i,j,ny,nx);
    float v_b = get_hfacet(v,i+1,j,ny,nx);

    // Normal strain rates
    float dudx = (u_r - u_l) * dx_inv;
    float dvdy = (v_t - v_b) * dx_inv;

    // Shear strain rate at 4 corners, then average
    // Top-left corner (i-1/2, j-1/2)
    float tl_mask = (i > 0 && j > 0) ? 1.0f : 0.0f;
    float u_tl = (i > 0) ? u[(i - 1) * (nx + 1) + j] : u_l;
    float v_lt = (j > 0) ? v[i * nx + j - 1] : v_t;
    float eps_xy_tl = 0.5f * ((u_tl - u_l) * dx_inv + (v_t - v_lt) * dx_inv) * tl_mask;

    // Top-right corner (i-1/2, j+1/2)
    float tr_mask = (i > 0 && j < (nx - 1)) ? 1.0f : 0.0f;
    float u_tr = (i > 0) ? u[(i - 1) * (nx + 1) + j + 1] : u_r;
    float v_rt = (j < (nx - 1)) ? v[i * nx + j + 1] : v_t;
    float eps_xy_tr = 0.5f * ((u_tr - u_r) * dx_inv + (v_rt - v_t) * dx_inv) * tr_mask;

    // Bottom-left corner (i+1/2, j-1/2)
    float bl_mask = (i < (ny - 1) && j > 0) ? 1.0f : 0.0f;
    float u_bl = (i < (ny - 1)) ? u[(i + 1) * (nx + 1) + j] : u_l;
    float v_lb = (j > 0) ? v[(i + 1) * nx + j - 1] : v_b;
    float eps_xy_bl = 0.5f * ((u_l - u_bl) * dx_inv + (v_b - v_lb) * dx_inv) * bl_mask;

    // Bottom-right corner (i+1/2, j+1/2)
    float br_mask = (i < (ny - 1) && j < (nx - 1)) ? 1.0f : 0.0f;
    float u_br = (i < (ny - 1)) ? u[(i + 1) * (nx + 1) + j + 1] : u_r;
    float v_rb = (j < (nx - 1)) ? v[(i + 1) * nx + j + 1] : v_b;
    float eps_xy_br = 0.5f * ((u_r - u_br) * dx_inv + (v_rb - v_b) * dx_inv) * br_mask;

    // Average of squared shear strains
    float eps_xy2_bar = 0.25f * (eps_xy_tl * eps_xy_tl + eps_xy_tr * eps_xy_tr +
                                  eps_xy_bl * eps_xy_bl + eps_xy_br * eps_xy_br);

    // Second invariant of strain rate
    float eps_II = dudx * dudx + dvdy * dvdy + dudx * dvdy + eps_xy2_bar + eps_reg;

    float B_ = get_cell(B,i,j,ny,nx);

    // Viscosity: eta = 0.5 * B * eps_II^((1-n)/(2n))
    eta[idx] = 0.5f * B_ * __powf(eps_II, glen_exp);
}

extern "C" __global__ void compute_eta_with_dual(
    float* __restrict__ eta,          // Output: (ny, nx)
    float* __restrict__ d_eta,
    const float* __restrict__ u,      // u-velocity: (ny, nx+1)
    const float* __restrict__ v,      // v-velocity: (ny+1, nx)
    const float* __restrict__ d_u,      // u-velocity: (ny, nx+1)
    const float* __restrict__ d_v,      // v-velocity: (ny+1, nx)
    const float* __restrict__ B,      // Rate factor: (ny, nx)
    float n,                          // Glen's exponent
    float eps_reg,                    // Strain rate regularization
    float dx,
    int ny, int nx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ny * nx) return;

    int i = idx / nx;  // row (y direction)
    int j = idx % nx;  // column (x direction)

    float dx_inv = 1.0f / dx;
    float glen_exp = (1.0f - n) / (2.0f * n);

    // Get velocities around cell (i,j)
    DualFloat u_l = get_vfacet(u,d_u,i,j,ny,nx);
    DualFloat u_r = get_vfacet(u,d_u,i,j+1,ny,nx);
    DualFloat v_t = get_hfacet(v,d_v,i,j,ny,nx);
    DualFloat v_b = get_hfacet(v,d_v,i+1,j,ny,nx);

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

    DualFloat eta_ = 0.5f*get_cell(B,i,j,ny,nx)*__powf(eps_II_c,glen_exp);

    // Viscosity: eta = 0.5 * B * eps_II^((1-n)/(2n))
    eta[idx] = eta_.v;
    d_eta[idx] = eta_.d;
}

/*
struct BetaEffStencil {
    float bed;
    float H;
    float beta;
    float sigmoid_c;
};

struct BetaEffJacobian {
    float res;
    float d_bed;
    float d_H;
    float d_beta;

    __device__ __forceinline__
    float apply_jvp(const BetaEffStencil& dot) const {
	return d_bed * dot.bed
	       + d_H * dot.H
	       + d_beta * dot.beta;
    }
};

__device__ __forceinline__
BetaEffJacobian get_beta_eff_jac(BetaEffStencil s){
    BetaEffJacobian jac;
    
    float z = fminf(fmaxf(s.sigmoid_c * (s.bed + 0.917f*s.H). -20.0f, 20.0f));
    float grounded = 1.0f / (1.0f + __expf(-z));
    j.res = grounded * s.beta + (1.0f - grounded

*/	


extern "C" __global__ void compute_alpha(
    float* __restrict__ alpha_u,     // Output: (ny, nx + 1)
    float* __restrict__ alpha_v,     // Output: (ny + 1, nx)
    const float* __restrict__ u,      // u - velocity (ny, nx + 1)
    const float* __restrict__ v,      // v - velocity: (ny + 1, nx)
    const float* __restrict__ beta,
    const float* __restrict__ grounded,      // Thickness: (ny, nx)
    float m,
    float eps_sliding,
    float water_drag,
    int ny, int nx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < ny * (nx+1)) {
	int i = idx / (nx + 1);
	int j = idx % (nx + 1);

	float grounded_l = get_cell(grounded,i,j-1,ny,nx);
	float beta_l = get_cell(beta,i,j-1,ny,nx);
	float beta_eff_l = grounded_l * beta_l + (1.0f - grounded_l)*water_drag;

        float grounded_r = get_cell(grounded,i,j,ny,nx);
	float beta_r = get_cell(beta,i,j,ny,nx);
	float beta_eff_r = grounded_r * beta_r + (1.0f - grounded_r)*water_drag;

        float beta_eff = 0.5f*(beta_eff_l + beta_eff_r);
        float u_c = get_vfacet(u,i,j,ny,nx);
        float v_tl = get_hfacet(v,i,j-1,ny,nx);
        float v_tr = get_hfacet(v,i,j,ny,nx);
        float v_bl = get_hfacet(v,i+1,j-1,ny,nx);
        float v_br = get_hfacet(v,i+1,j,ny,nx);
        
	float unorm_sq = u_c * u_c + 0.25f*(v_tl * v_tl + v_tr * v_tr + v_bl * v_bl + v_br * v_br);
        
        alpha_u[i * (nx + 1) + j] = beta_eff * __powf(unorm_sq + eps_sliding,(m - 1.0f)/2.0f);	
    } else if ( idx >= ny * (nx + 1) && idx < ny * (nx + 1) + (ny + 1) * nx ) {
        int idx_v = idx - ( ny * (nx + 1));
	int i = idx_v / nx;
	int j = idx_v % nx;

        float grounded_t = get_cell(grounded,i-1,j,ny,nx);
	float beta_t = get_cell(beta,i-1,j,ny,nx);
	float beta_eff_t = grounded_t * beta_t + (1.0f - grounded_t)*water_drag;

        float grounded_b = get_cell(grounded,i,j,ny,nx);
	float beta_b = get_cell(beta,i,j,ny,nx);
	float beta_eff_b = grounded_b * beta_b + (1.0f - grounded_b)*water_drag;

	float beta_eff = 0.5f*(beta_eff_t + beta_eff_b);
        float v_c = get_hfacet(v,i,j,ny,nx);
	float u_tl = get_vfacet(u,i-1,j,ny,nx);
	float u_tr = get_vfacet(u,i-1,j+1,ny,nx);
	float u_bl = get_vfacet(u,i,j,ny,nx);
	float u_br = get_vfacet(u,i,j+1,ny,nx);

	float unorm_sq = v_c * v_c + 0.25f*(u_tl * u_tl + u_tr * u_tr + u_bl * u_bl + u_br * u_br);

	alpha_v[i * nx + j] = beta_eff * __powf(unorm_sq + eps_sliding,(m - 1.0f)/2.0f);
    }
}

extern "C" __global__ void compute_alpha_with_dual(
    float* __restrict__ alpha_u,     // Output: (ny, nx + 1)
    float* __restrict__ alpha_v,     // Output: (ny + 1, nx)
    float* __restrict__ d_alpha_u,     // Output: (ny, nx + 1)
    float* __restrict__ d_alpha_v,     // Output: (ny + 1, nx)
    const float* __restrict__ u,      // u - velocity (ny, nx + 1)
    const float* __restrict__ v,      // v - velocity: (ny + 1, nx)
    const float* __restrict__ d_u,      // u - velocity (ny, nx + 1)
    const float* __restrict__ d_v,      // v - velocity: (ny + 1, nx)
    const float* __restrict__ beta,
    const float* __restrict__ grounded,   // Basal traction: (ny, nx)
    float m,
    float eps_sliding,
    float water_drag,
    int ny, int nx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < ny * (nx+1)) {
	int i = idx / (nx + 1);
	int j = idx % (nx + 1);

	float grounded_l = get_cell(grounded,i,j-1,ny,nx);
	float beta_l = get_cell(beta,i,j-1,ny,nx);
	float beta_eff_l = grounded_l * beta_l + (1.0f - grounded_l)*water_drag;

        float grounded_r = get_cell(grounded,i,j,ny,nx);
	float beta_r = get_cell(beta,i,j,ny,nx);
	float beta_eff_r = grounded_r * beta_r + (1.0f - grounded_r)*water_drag;

        float beta_eff = 0.5f*(beta_eff_l + beta_eff_r);
        DualFloat u_c = get_vfacet(u,d_u,i,j,ny,nx);
        DualFloat v_tl = get_hfacet(v,d_v,i,j-1,ny,nx);
        DualFloat v_tr = get_hfacet(v,d_v,i,j,ny,nx);
        DualFloat v_bl = get_hfacet(v,d_v,i+1,j-1,ny,nx);
        DualFloat v_br = get_hfacet(v,d_v,i+1,j,ny,nx);
        
	DualFloat unorm_sq = u_c * u_c + 0.25f*(v_tl * v_tl + v_tr * v_tr + v_bl * v_bl + v_br * v_br);
        DualFloat alpha_ = beta_eff * __powf(unorm_sq + eps_sliding,(m - 1.0f)/2.0f);	

        alpha_u[i * (nx + 1) + j] = alpha_.v;
        d_alpha_u[i * (nx + 1) + j] = alpha_.d;
    } else if ( idx >= ny * (nx + 1) && idx < ny * (nx + 1) + (ny + 1) * nx ) {
        int idx_v = idx - ( ny * (nx + 1));
	int i = idx_v / nx;
	int j = idx_v % nx;

        float grounded_t = get_cell(grounded,i-1,j,ny,nx);
	float beta_t = get_cell(beta,i-1,j,ny,nx);
	float beta_eff_t = grounded_t * beta_t + (1.0f - grounded_t)*water_drag;

        float grounded_b = get_cell(grounded,i,j,ny,nx);
	float beta_b = get_cell(beta,i,j,ny,nx);
	float beta_eff_b = grounded_b * beta_b + (1.0f - grounded_b)*water_drag;

	float beta_eff = 0.5f*(beta_eff_t + beta_eff_b);

        DualFloat v_c = get_hfacet(v,d_v,i,j,ny,nx);
	DualFloat u_tl = get_vfacet(u,d_u,i-1,j,ny,nx);
	DualFloat u_tr = get_vfacet(u,d_u,i-1,j+1,ny,nx);
	DualFloat u_bl = get_vfacet(u,d_u,i,j,ny,nx);
	DualFloat u_br = get_vfacet(u,d_u,i,j+1,ny,nx);

	DualFloat unorm_sq = v_c * v_c + 0.25f*(u_tl * u_tl + u_tr * u_tr + u_bl * u_bl + u_br * u_br);

	DualFloat alpha_ = beta_eff * __powf(unorm_sq + eps_sliding,(m - 1.0f)/2.0f);

	alpha_v[i * nx + j] = alpha_.v;
	d_alpha_v[i * nx + j] = alpha_.d;
    }
}

/*--------------------------------------------------------------
  Compute cell-centered effective calving rate c_eff

  c_eff = (1 - grounded) * calving_rate

  This represents mass loss rate for floating ice.
  --------------------------------------------------------------*/
extern "C" __global__ void compute_c_eff(
    float* __restrict__ c_eff,        // Output: (ny, nx)
    const float* __restrict__ grounded,      // Thickness: (ny, nx)
    float calving_rate,
    int ny, int nx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ny * nx) return;

    c_eff[idx] = (1.0f - grounded[idx]) * calving_rate;
}

extern "C" __global__ void compute_grounded(
    float* __restrict__ grounded,        // Output: (ny, nx)
    const float* __restrict__ H,      // Thickness: (ny, nx)
    const float* __restrict__ bed,    // Bed topography: (ny, nx)
    float sigmoid_c,
    float relaxation,
    int ny, int nx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ny * nx) return;

    float z = bed[idx] + 0.917f * H[idx];

    // Sigmoid for smooth grounding line transition
    float scaled_z = fminf(fmaxf(sigmoid_c * z, -20.0f), 20.0f);
    grounded[idx] = relaxation * grounded[idx] + (1.0f - relaxation) / (1.0f + __expf(-scaled_z));
}

