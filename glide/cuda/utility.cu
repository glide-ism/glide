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

