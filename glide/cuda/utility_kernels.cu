extern "C" __global__
void restrict_u(
    const float* u_fine,      // (ny_fine, nx_fine+1)
    float* u_coarse,          // (ny_coarse, nx_coarse+1)
    const int ny_coarse,
    const int nx_coarse)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny_coarse * (nx_coarse + 1);
    
    if (idx < total) {
        int J = idx / (nx_coarse + 1);
        int I = idx % (nx_coarse + 1);
        
        // Average tangentially (in y), no averaging in x
        u_coarse[idx] = 0.5f * (u_fine[2*J * (2*nx_coarse + 1) + 2*I] + 
                               u_fine[(2*J + 1) * (2*nx_coarse + 1) + 2*I]);
    }
}

extern "C" __global__
void prolongate_u(
    const float* u_coarse,    // (ny_coarse, nx_coarse+1)
    float* u_fine,            // (ny_fine, nx_fine+1)
    const int ny_fine,
    const int nx_fine)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny_fine * (nx_fine + 1);
    
    if (idx < total) {
        int j = idx / (nx_fine + 1);
        int i = idx % (nx_fine + 1);
        
        int J = j / 2;  // Coarse cell
        int I = i / 2;  // Coarse face position
        
        if (i % 2 == 0) {
            // Fine face aligns with coarse face
            // Constant in tangential (y) direction
            u_fine[idx] = u_coarse[J * (nx_fine/2 + 1) + I];
        } else {
            // Fine face between coarse faces
            // Linear interpolation in normal (x) direction
            if (I < nx_fine/2) {
                u_fine[idx] = 0.5f * (u_coarse[J * (nx_fine/2 + 1) + I] + 
                                     u_coarse[J * (nx_fine/2 + 1) + I + 1]);
            } else {
                // Boundary case
                u_fine[idx] = u_coarse[J * (nx_fine/2 + 1) + I];
            }
        }
    }
}

extern "C" __global__
void restrict_cell_centered(
    const float* h_fine,      // (ny_fine, nx_fine)
    float* h_coarse,          // (ny_coarse, nx_coarse)
    const int ny_coarse,
    const int nx_coarse)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < ny_coarse * nx_coarse) {
        int J = idx / nx_coarse;
        int I = idx % nx_coarse;
        
        // Full-weighting (average 4 fine cells)
        h_coarse[idx] = 0.25f * (h_fine[2*J * (2*nx_coarse) + 2*I] +
                                h_fine[2*J * (2*nx_coarse) + 2*I + 1] +
                                h_fine[(2*J + 1) * (2*nx_coarse) + 2*I] +
                                h_fine[(2*J + 1) * (2*nx_coarse) + 2*I + 1]);
    }
}

extern "C" __global__
void restrict_max_pool(
    const float* h_fine,      // (ny_fine, nx_fine)
    float* h_coarse,          // (ny_coarse, nx_coarse)
    const int ny_coarse,
    const int nx_coarse)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < ny_coarse * nx_coarse) {
        int J = idx / nx_coarse;
        int I = idx % nx_coarse;
        
        // Full-weighting (average 4 fine cells)
        float h_tl = h_fine[2*J * (2*nx_coarse) + 2*I];
        float h_tr = h_fine[2*J * (2*nx_coarse) + 2*I + 1];
        float h_bl =  h_fine[(2*J + 1) * (2*nx_coarse) + 2*I];
        float h_br =  h_fine[(2*J + 1) * (2*nx_coarse) + 2*I + 1];

	float max_h_t = fmaxf(h_tl,h_tr);
	float max_h_b = fmaxf(h_bl,h_br);

        h_coarse[idx] = fmaxf(max_h_t,max_h_b);
    }
}


extern "C" __global__
void prolongate_cell_centered(
    const float* h_coarse,    // (ny_coarse, nx_coarse)
    float* h_fine,            // (ny_fine, nx_fine)
    const int ny_fine,
    const int nx_fine)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < ny_fine * nx_fine) {
        int j = idx / nx_fine;
        int i = idx % nx_fine;
        
        int J = j / 2;
        int I = i / 2;
        
        // Injection (as you suggested)
        h_fine[idx] = h_coarse[J * (nx_fine/2) + I];
    }
}

extern "C" __global__
void restrict_v(
    const float* v_fine,      // (ny_fine+1, nx_fine)
    float* v_coarse,          // (ny_coarse+1, nx_coarse)
    const int ny_coarse,
    const int nx_coarse)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (ny_coarse + 1) * nx_coarse;
    
    if (idx < total) {
        int J = idx / nx_coarse;
        int I = idx % nx_coarse;
        
        // Average tangentially (in x), no averaging in y (faces align!)
        v_coarse[idx] = 0.5f * (v_fine[2*J * (2*nx_coarse) + 2*I] + 
                               v_fine[2*J * (2*nx_coarse) + 2*I + 1]);
    }
}

extern "C" __global__
void prolongate_v(
    const float* v_coarse,    // (ny_coarse+1, nx_coarse)
    float* v_fine,            // (ny_fine+1, nx_fine)
    const int ny_fine,
    const int nx_fine)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (ny_fine + 1) * nx_fine;
    
    if (idx < total) {
        int j = idx / nx_fine;
        int i = idx % nx_fine;
        
        int J = j / 2;  // Coarse face row
        int I = i / 2;  // Coarse cell column
        
        if (j % 2 == 0) {
            // Fine face aligns with coarse face in y
            // Constant in tangential (x) direction
            v_fine[idx] = v_coarse[J * (nx_fine/2) + I];
        } else {
            // Fine face between coarse faces in y
            // Linear interpolation in normal (y) direction
            if (J < ny_fine/2) {
                v_fine[idx] = 0.5f * (v_coarse[J * (nx_fine/2) + I] + 
                                     v_coarse[(J + 1) * (nx_fine/2) + I]);
            } else {
                // Boundary case at bottom
                v_fine[idx] = v_coarse[J * (nx_fine/2) + I];
            }
        }
    }
}

extern "C" __global__
void prolongate_u_smooth(
    const float* u_coarse,    // (ny_coarse, nx_coarse+1)
    float* u_fine,            // (ny_fine, nx_fine+1)
    const int ny_fine,
    const int nx_fine)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny_fine * (nx_fine + 1);
    
    if (idx >= total) return;
    
    int j = idx / (nx_fine + 1);  // Fine grid row
    int i = idx % (nx_fine + 1);  // Fine grid col (face index)
    
    int ny_coarse = ny_fine / 2;
    int nx_coarse = nx_fine / 2;
    
    // u is on vertical faces: u[j,i] at position (j+0.5, i) in fine cell units
    // Coarse u[J,I] at position (2J+1, 2I) in fine cell units
    //
    // Map fine to coarse:
    //   Y: 2*J_float + 1 = j + 0.5  =>  J_float = (j - 0.5) / 2
    //   X: 2*I_float = i  =>  I_float = i / 2
    
    float J_float = (j - 0.5f) * 0.5f;
    float I_float = i * 0.5f;
    
    // Clamp to valid interpolation region
    J_float = fmaxf(0.0f, fminf(J_float, (float)(ny_coarse - 1)));
    I_float = fmaxf(0.0f, fminf(I_float, (float)nx_coarse));  // nx_coarse+1 faces, last index is nx_coarse
    
    // Integer indices for surrounding coarse values
    int J_lo = (int)J_float;
    int I_lo = (int)I_float;
    int J_hi = min(J_lo + 1, ny_coarse - 1);
    int I_hi = min(I_lo + 1, nx_coarse);
    
    // Interpolation weights
    float t_y = J_float - J_lo;
    float t_x = I_float - I_lo;
    
    // Load coarse values (stride is nx_coarse+1)
    int stride = nx_coarse + 1;
    float v00 = u_coarse[J_lo * stride + I_lo];
    float v01 = u_coarse[J_lo * stride + I_hi];
    float v10 = u_coarse[J_hi * stride + I_lo];
    float v11 = u_coarse[J_hi * stride + I_hi];
    
    // Bilinear interpolation
    u_fine[idx] = (1.0f - t_y) * ((1.0f - t_x) * v00 + t_x * v01)
                + t_y         * ((1.0f - t_x) * v10 + t_x * v11);
}


extern "C" __global__
void prolongate_v_smooth(
    const float* v_coarse,    // (ny_coarse+1, nx_coarse)
    float* v_fine,            // (ny_fine+1, nx_fine)
    const int ny_fine,
    const int nx_fine)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (ny_fine + 1) * nx_fine;
    
    if (idx >= total) return;
    
    int j = idx / nx_fine;  // Fine grid row (face index)
    int i = idx % nx_fine;  // Fine grid col
    
    int ny_coarse = ny_fine / 2;
    int nx_coarse = nx_fine / 2;
    
    // v is on horizontal faces: v[j,i] at position (j, i+0.5) in fine cell units
    // Coarse v[J,I] at position (2J, 2I+1) in fine cell units
    //
    // Map fine to coarse:
    //   Y: 2*J_float = j  =>  J_float = j / 2
    //   X: 2*I_float + 1 = i + 0.5  =>  I_float = (i - 0.5) / 2
    
    float J_float = j * 0.5f;
    float I_float = (i - 0.5f) * 0.5f;
    
    // Clamp to valid interpolation region
    J_float = fmaxf(0.0f, fminf(J_float, (float)ny_coarse));  // ny_coarse+1 faces, last index is ny_coarse
    I_float = fmaxf(0.0f, fminf(I_float, (float)(nx_coarse - 1)));
    
    // Integer indices for surrounding coarse values
    int J_lo = (int)J_float;
    int I_lo = (int)I_float;
    int J_hi = min(J_lo + 1, ny_coarse);
    int I_hi = min(I_lo + 1, nx_coarse - 1);
    
    // Interpolation weights
    float t_y = J_float - J_lo;
    float t_x = I_float - I_lo;
    
    // Load coarse values (stride is nx_coarse)
    int stride = nx_coarse;
    float v00 = v_coarse[J_lo * stride + I_lo];
    float v01 = v_coarse[J_lo * stride + I_hi];
    float v10 = v_coarse[J_hi * stride + I_lo];
    float v11 = v_coarse[J_hi * stride + I_hi];
    
    // Bilinear interpolation
    v_fine[idx] = (1.0f - t_y) * ((1.0f - t_x) * v00 + t_x * v01)
                + t_y         * ((1.0f - t_x) * v10 + t_x * v11);
}

extern "C" __global__
void prolongate_H_smooth(
    const float* H_coarse,
    float* H_fine,
    const int ny_fine,
    const int nx_fine)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny_fine * nx_fine;
    
    if (idx >= total) return;
    
    int j = idx / nx_fine;  // Fine grid row
    int i = idx % nx_fine;  // Fine grid col
    
    int ny_coarse = ny_fine / 2;
    int nx_coarse = nx_fine / 2;
    
    // Fine cell (j,i) center is at (j+0.5, i+0.5) in fine cell units
    // Coarse cell (J,I) center is at (2J+1, 2I+1) in fine cell units
    // Map fine position to coarse grid coordinates:
    //   J_float = (j + 0.5 - 1) / 2 = (j - 0.5) / 2
    float J_float = (j - 0.5f) * 0.5f;
    float I_float = (i - 0.5f) * 0.5f;
    
    // Clamp to valid interpolation region [0, n_coarse-1]
    J_float = fmaxf(0.0f, fminf(J_float, (float)(ny_coarse - 1)));
    I_float = fmaxf(0.0f, fminf(I_float, (float)(nx_coarse - 1)));
    
    // Integer indices for the 4 surrounding coarse cells
    int J_lo = (int)J_float;
    int I_lo = (int)I_float;
    int J_hi = min(J_lo + 1, ny_coarse - 1);
    int I_hi = min(I_lo + 1, nx_coarse - 1);
    
    // Fractional position within the coarse cell
    float t_y = J_float - J_lo;
    float t_x = I_float - I_lo;
    
    // Load the 4 coarse values
    float v00 = H_coarse[J_lo * nx_coarse + I_lo];
    float v01 = H_coarse[J_lo * nx_coarse + I_hi];
    float v10 = H_coarse[J_hi * nx_coarse + I_lo];
    float v11 = H_coarse[J_hi * nx_coarse + I_hi];
    
    // Bilinear interpolation
    H_fine[idx] = (1.0f - t_y) * ((1.0f - t_x) * v00 + t_x * v01)
                + t_y         * ((1.0f - t_x) * v10 + t_x * v11);
}



extern "C" __global__
void monolithic_matvec_B(
    // Current state (being multiplied)
    const float* __restrict__ u,           // velocity u (ny, nx+1)
    const float* __restrict__ v,           // velocity v (ny+1, nx)
    const float* __restrict__ H,           // thickness (ny, nx)
    // Lagged values for linearization
    const float* __restrict__ u_lagged,    // lagged u for 
    const float* __restrict__ v_lagged,    // lagged v for 
    const float* __restrict__ H_lagged,    // lagged thickness 
    // SSA parameters
    const float* __restrict__ eta_H_cells, // viscosity at cells
    const float* __restrict__ eta_H_verts, // viscosity at vertices
    const float* __restrict__ beta_u,      // basal drag u
    const float* __restrict__ beta_v,      // basal drag v
    const float* __restrict__ B,
    // Outputs
    float* __restrict__ out_u,             // momentum equation u
    float* __restrict__ out_v,             // momentum equation v
    float* __restrict__ out_H,             // mass equation
    // Parameters
    const float dx,
    const float dt,
    const int ny,
    const int nx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nu = ny * (nx + 1);
    int nv = (ny + 1) * nx;
    int nh = ny * nx;
    int total_work = nu + nv + nh;
    
    if (idx < total_work) {
        if (idx < nu) {
            // ======== U-momentum equation: A*u + B^T*H ========
            int row = idx / (nx + 1);
            int col = idx % (nx + 1);
            float result = 0.0f;
            
            // [Insert existing SSA terms from ssa_matvec_fused here]
            // ... normal stress, shear stress, basal drag ...
            // ---- Normal stress divergence: d/dx[2*eta*H*(2*du/dx + dv/dy)] ----
	    
            if (col == 0) {
                // Left boundary - only stress from right cell
                float dudx = u[row * (nx + 1) + 1] - u[row * (nx + 1) + 0];
                float dvdy = v[row * nx + 0] - v[(row + 1) * nx + 0];
                float stress = 2.0f * eta_H_cells[row * nx + 0] * (2.0f * dudx + dvdy);
                result += stress;  // No factor of 2!
            }
            else if (col == nx) {
                // Right boundary - only stress from left cell
                float dudx = u[row * (nx + 1) + nx] - u[row * (nx + 1) + (nx - 1)];
                float dvdy = v[row * nx + (nx - 1)] - v[(row + 1) * nx + (nx - 1)];
                float stress = 2.0f * eta_H_cells[row * nx + (nx - 1)] * (2.0f * dudx + dvdy);
                result += -stress;  // No factor of 2!
            }
            else {
                // Interior
                // Left cell
                float dudx_left = u[row * (nx + 1) + col] - u[row * (nx + 1) + (col - 1)];
                float dvdy_left = v[row * nx + (col - 1)] - v[(row + 1) * nx + (col - 1)];
                float stress_left = 2.0f * eta_H_cells[row * nx + (col - 1)] * 
                                    (2.0f * dudx_left + dvdy_left);
                
                // Right cell
                float dudx_right = u[row * (nx + 1) + (col + 1)] - u[row * (nx + 1) + col];
                float dvdy_right = v[row * nx + col] - v[(row + 1) * nx + col];
                float stress_right = 2.0f * eta_H_cells[row * nx + col] * 
                                     (2.0f * dudx_right + dvdy_right);
                
                result += stress_right - stress_left;
            }
            
            // ---- Shear stress divergence: d/dy[eta*H*(du/dy + dv/dx)] ----
            // Top vertex (row, col) with y-up convention
            float shear_top = 0.0f;
            if (row == 0) {
                // Top boundary: shear = 0
                shear_top = 0.0f;
            } else {
                float dudy_top = u[(row - 1) * (nx + 1) + col] - u[row * (nx + 1) + col];
                float dvdx_top = 0.0f;
                if (col > 0 && col < nx) {
                    dvdx_top = v[row * nx + col] - v[row * nx + (col - 1)];
                }
                shear_top = eta_H_verts[row * (nx + 1) + col] * (dudy_top + dvdx_top);
            }
            
            // Bottom vertex (row+1, col) with y-up convention
            float shear_bottom = 0.0f;
            if (row == ny - 1) {
                // Bottom boundary: shear = 0
                shear_bottom = 0.0f;
            } else {
                float dudy_bottom = u[row * (nx + 1) + col] - u[(row + 1) * (nx + 1) + col];
                float dvdx_bottom = 0.0f;
                if (col > 0 && col < nx) {
                    dvdx_bottom = v[(row + 1) * nx + col] - v[(row + 1) * nx + (col - 1)];
                }
                shear_bottom = eta_H_verts[(row + 1) * (nx + 1) + col] * 
                              (dudy_bottom + dvdx_bottom);
            }
            
            result += shear_top - shear_bottom;
            
	    result /= (dx*dx);
            // ---- Basal drag ----
            result -= beta_u[idx] * u[idx];
	    
            
            // Add B^T contribution: H_lagged_at_face * dH/dx (no rho*g)
            if (col > 0 && col < nx) {
                // Interior: average H_lagged to face, centered gradient of current H
                float H_face = 0.5f * (H_lagged[row * nx + col] + 
                                       H_lagged[row * nx + (col-1)]);
                float dHdx = (H[row * nx + col] - H[row * nx + (col-1)]) / dx;
                result -= H_face * dHdx;
            }

	    if (col > 0 && col < nx) {
		float H_bar = 0.5f * (H[row * nx + col] +
				      H[row * nx + (col - 1)]);
		float dBdx = (B[row * nx + col] - B[row * nx + (col-1) ]) / dx;
		result -= H_bar * dBdx;
	    }
            // Boundaries: zero gradient BC means no contribution
            
            out_u[idx] = result;
        }
        else if (idx < nu + nv) {
            // ======== V-momentum equation: A*v + B^T*H ========
            int v_idx = idx - nu;
            int row = v_idx / nx;
            int col = v_idx % nx;
            float result = 0.0f;
            
            // ---- Normal stress divergence: d/dy[2*eta*H*(du/dx + 2*dv/dy)] ----
            // Using y-up convention
            if (row == 0) {
                // Top boundary (highest y) - only stress from cell below
                float dudx = u[0 * (nx + 1) + (col + 1)] - u[0 * (nx + 1) + col];
                float dvdy = v[0 * nx + col] - v[1 * nx + col];  // y-up: current - below
                float stress = 2.0f * eta_H_cells[0 * nx + col] * (dudx + 2.0f * dvdy);
                result += -stress;  // No factor of 2!
            }
            else if (row == ny) {
                // Bottom boundary (lowest y) - only stress from cell above
                float dudx = u[(ny - 1) * (nx + 1) + (col + 1)] - 
                             u[(ny - 1) * (nx + 1) + col];
                float dvdy = v[(ny - 1) * nx + col] - v[ny * nx + col];  // y-up: above - current
                float stress = 2.0f * eta_H_cells[(ny - 1) * nx + col] * 
                               (dudx + 2.0f * dvdy);
                result += stress;  // No factor of 2!
            }
            else {
                // Interior
                // Cell above (row-1) in y-up convention
                float dudx_above = u[(row - 1) * (nx + 1) + (col + 1)] - 
                                   u[(row - 1) * (nx + 1) + col];
                float dvdy_above = v[(row - 1) * nx + col] - v[row * nx + col];  // y-up
                float stress_above = 2.0f * eta_H_cells[(row - 1) * nx + col] * 
                                     (dudx_above + 2.0f * dvdy_above);
                
                // Cell below (row) in y-up convention
                float dudx_below = u[row * (nx + 1) + (col + 1)] - 
                                   u[row * (nx + 1) + col];
                float dvdy_below = v[row * nx + col] - v[(row + 1) * nx + col];  // y-up
                float stress_below = 2.0f * eta_H_cells[row * nx + col] * 
                                     (dudx_below + 2.0f * dvdy_below);
                
                // Divergence with y-up convention
                result += stress_above - stress_below;
            }
            
            // ---- Shear stress divergence: d/dx[eta*H*(du/dy + dv/dx)] ----
            // Left vertex (row, col)
            float shear_left = 0.0f;
            if (col == 0) {
                // Left boundary: shear = 0
                shear_left = 0.0f;
            } else {
                float dvdx_left =  v[row * nx + col] - v[row * nx + (col - 1)];
                float dudy_left = 0.0f;
                if (row > 0 && row < ny) {
                    dudy_left = u[(row - 1) * (nx + 1) + col] - u[row * (nx + 1) + col];
                }
                shear_left = eta_H_verts[row * (nx + 1) + col] * (dudy_left + dvdx_left);
            }
            
            // Right vertex (row, col+1)
            float shear_right = 0.0f;
            if (col == nx - 1) {
                // Right boundary: shear = 0
                shear_right = 0.0f;
            } else {
                float dvdx_right = v[row * nx + (col + 1)] - v[row * nx + col];
                float dudy_right = 0.0f;
                if (row > 0 && row < ny) {
                    dudy_right = u[(row - 1) * (nx + 1) + (col + 1)] - 
                                u[row * (nx + 1) + (col + 1)];
                }
                shear_right = eta_H_verts[row * (nx + 1) + (col + 1)] * 
                             (dudy_right + dvdx_right);
            }
            
            result += shear_right - shear_left;
	    result /= (dx*dx);
            
            // ---- Basal drag ----
            result -= beta_v[v_idx] * v[v_idx];
             
            // Add B^T contribution: H_lagged_at_face * dH/dy
            if (row > 0 && row < ny) {
                // Interior: average H_lagged to horizontal face
                float H_face = 0.5f * (H_lagged[row * nx + col] + 
                                       H_lagged[(row-1) * nx + col]);
                // Centered gradient with y-up convention
                float dHdy = (H[(row-1) * nx + col] - H[row * nx + col]) / dx;
                result -= H_face * dHdy;
            }

	    if (row > 0 && row < ny) {
                // Interior: average H_lagged to horizontal face
                float H_bar = 0.5f * (H[row * nx + col] + 
                                       H[(row-1) * nx + col]);
                // Centered gradient with y-up convention
                float dBdy = (B[(row-1) * nx + col] - B[row * nx + col]) / dx;
                result -= H_bar * dBdy;
            }

            // Boundaries: zero gradient BC
            
            out_v[v_idx] = result;
        }
        else {
            // ======== Mass equation: B*[u,v] + (I/dt + K)*H ========
            int h_idx = idx - nu - nv;
            int row = h_idx / nx;
            int col = h_idx % nx;
            float result = 0.0f;
            
            // ---- B term: div(H_lagged * [u,v]) ----
            // Left/right faces (vertical)
            float flux_x_left = 0.0f, flux_x_right = 0.0f;
            
            if (col > 0) {
                float H_left = 0.5f * (H_lagged[row * nx + col] + 
                                       H_lagged[row * nx + (col-1)]);
                flux_x_left = H_left * u[row * (nx+1) + col];
            } else {
                // Left boundary: zero flux BC
                flux_x_left = 0.0f;
            }
            
            if (col < nx-1) {
                float H_right = 0.5f * (H_lagged[row * nx + col] + 
                                        H_lagged[row * nx + (col+1)]);
                flux_x_right = H_right * u[row * (nx+1) + (col+1)];
            } else {
                // Right boundary: zero flux BC
                flux_x_right = 0.0f;
            }
            
            // Top/bottom faces (horizontal) with y-up
            float flux_y_top = 0.0f, flux_y_bottom = 0.0f;
            
            if (row > 0) {
                float H_top = 0.5f * (H_lagged[row * nx + col] + 
                                     H_lagged[(row-1) * nx + col]);
                flux_y_top = H_top * v[row * nx + col];
            } else {
                // Top boundary: zero flux BC
                flux_y_top = 0.0f;
            }
            
            if (row < ny-1) {
                float H_bottom = 0.5f * (H_lagged[row * nx + col] + 
                                        H_lagged[(row+1) * nx + col]);
                flux_y_bottom = H_bottom * v[(row+1) * nx + col];
            } else {
                // Bottom boundary: zero flux BC
                flux_y_bottom = 0.0f;
            }
            
            // Divergence (centered)
            result += (flux_x_right - flux_x_left) / dx + 
                     (flux_y_top - flux_y_bottom) / dx;
            
            // ---- Mass matrix term: H/dt ----
            result += H[h_idx] / dt;
            
            // Streamline diffusion for stability
            
            // X-direction diffusion
            float diff_flux_x_left = 0.0f, diff_flux_x_right = 0.0f;
            
            if (col > 0) {
                float u_mag_left = fabs(u_lagged[row * (nx+1) + col]);
                // Gradient of H at left face (current H)
                float dHdx_left = (H[row * nx + col] - H[row * nx + (col-1)]) / dx;
                diff_flux_x_left = (dx * u_mag_left / 2.0f) * dHdx_left;
            }
            
            if (col < nx-1) {
                float u_mag_right = fabs(u_lagged[row * (nx+1) + (col+1)]);
                // Gradient of H at right face (current H)
                float dHdx_right = (H[row * nx + (col+1)] - H[row * nx + col]) / dx;
                diff_flux_x_right = (dx * u_mag_right / 2.0f) * dHdx_right;
            }
            
            // Y-direction diffusion
            float diff_flux_y_top = 0.0f, diff_flux_y_bottom = 0.0f;
            
            if (row > 0) {
                float v_mag_top = fabs(v_lagged[row * nx + col]);
                // Gradient of H at top face (current H) with y-up
                float dHdy_top = (H[(row-1) * nx + col] - H[row * nx + col]) / dx;
                diff_flux_y_top = (dx * v_mag_top / 2.0f) * dHdy_top;
            }
            
            if (row < ny-1) {
                float v_mag_bottom = fabs(v_lagged[(row+1) * nx + col]);
                // Gradient of H at bottom face with y-up
                float dHdy_bottom = (H[row * nx + col] - H[(row+1) * nx + col]) / dx;
                diff_flux_y_bottom = (dx * v_mag_bottom / 2.0f) * dHdy_bottom;
            }
            
            // Divergence of diffusive flux (negative for diffusion)
            result -= 2.0f*(diff_flux_x_right - diff_flux_x_left) / dx;
            result -= 2.0f*(diff_flux_y_top - diff_flux_y_bottom) / dx;
            
            out_H[h_idx] = result;
        }
    }
}

