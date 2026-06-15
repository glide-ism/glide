// Auto-translated from cuda/transfer.cu by the metal port.
// Grid-transfer kernels: flat linear-index, 2 buffers + (ny,nx) params.
// Relies on common.metal (concatenated first).

kernel void restrict_vfacet(
    device const float* f_fine [[buffer(0)]],
    device float* f_coarse [[buffer(1)]],
    device const float* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int ny_coarse = (int)params[0];
    int nx_coarse = (int)params[1];

    int idx = (int)tid;
    int total = ny_coarse * (nx_coarse + 1);
    
    if (idx < total) {
        int J = idx / (nx_coarse + 1);
        int I = idx % (nx_coarse + 1);
        
        
        f_coarse[idx] = 0.5f * (f_fine[2*J * (2*nx_coarse + 1) + 2*I] + 
                                f_fine[(2*J + 1) * (2*nx_coarse + 1) + 2*I]);
    }
}


kernel void restrict_hfacet(
    device const float* f_fine [[buffer(0)]],
    device float* f_coarse [[buffer(1)]],
    device const float* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int ny_coarse = (int)params[0];
    int nx_coarse = (int)params[1];

    int idx = (int)tid;
    int total = (ny_coarse + 1) * nx_coarse;
    
    if (idx < total) {
        int J = idx / nx_coarse;
        int I = idx % nx_coarse;
        
        
        f_coarse[idx] = 0.5f * (f_fine[2*J * (2*nx_coarse) + 2*I] + 
                                f_fine[2*J * (2*nx_coarse) + 2*I + 1]);
    }
}


kernel void restrict_cell_avg(
    device const float* f_fine [[buffer(0)]],
    device float* f_coarse [[buffer(1)]],
    device const float* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int ny_coarse = (int)params[0];
    int nx_coarse = (int)params[1];

    int idx = (int)tid;
    
    if (idx < ny_coarse * nx_coarse) {
        int J = idx / nx_coarse;
        int I = idx % nx_coarse;
        
        
        f_coarse[idx] = 0.25f * (f_fine[2*J * (2*nx_coarse) + 2*I] +
                                 f_fine[2*J * (2*nx_coarse) + 2*I + 1] +
                                 f_fine[(2*J + 1) * (2*nx_coarse) + 2*I] +
                                 f_fine[(2*J + 1) * (2*nx_coarse) + 2*I + 1]);
    }
}


kernel void restrict_cell_max(
    device const float* f_fine [[buffer(0)]],
    device float* f_coarse [[buffer(1)]],
    device const float* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int ny_coarse = (int)params[0];
    int nx_coarse = (int)params[1];

    int idx = (int)tid;
    
    if (idx < ny_coarse * nx_coarse) {
        int J = idx / nx_coarse;
        int I = idx % nx_coarse;
        
        
        float f_tl =  f_fine[2*J * (2*nx_coarse) + 2*I];
        float f_tr =  f_fine[2*J * (2*nx_coarse) + 2*I + 1];
        float f_bl =  f_fine[(2*J + 1) * (2*nx_coarse) + 2*I];
        float f_br =  f_fine[(2*J + 1) * (2*nx_coarse) + 2*I + 1];

	float max_f_t = fmax(f_tl,f_tr);
	float max_f_b = fmax(f_bl,f_br);

        f_coarse[idx] = fmax(max_f_t,max_f_b);
    }
}


kernel void restrict_cell_min(
    device const float* f_fine [[buffer(0)]],
    device float* f_coarse [[buffer(1)]],
    device const float* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int ny_coarse = (int)params[0];
    int nx_coarse = (int)params[1];

    int idx = (int)tid;
    
    if (idx < ny_coarse * nx_coarse) {
        int J = idx / nx_coarse;
        int I = idx % nx_coarse;
        
        
        float f_tl = f_fine[2*J * (2*nx_coarse) + 2*I];
        float f_tr = f_fine[2*J * (2*nx_coarse) + 2*I + 1];
        float f_bl = f_fine[(2*J + 1) * (2*nx_coarse) + 2*I];
        float f_br = f_fine[(2*J + 1) * (2*nx_coarse) + 2*I + 1];

	float min_f_t = fmin(f_tl,f_tr);
	float min_f_b = fmin(f_bl,f_br);

        f_coarse[idx] = fmin(min_f_t,min_f_b);
    }
}


kernel void restrict_cell_var(
    device const float* f_fine [[buffer(0)]],
    device float* v_coarse [[buffer(1)]],
    device const float* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int ny_coarse = (int)params[0];
    int nx_coarse = (int)params[1];

    int idx = (int)tid;
    
    if (idx < ny_coarse * nx_coarse) {
        int J = idx / nx_coarse;
        int I = idx % nx_coarse;
        
        float mean = 0.25f * (f_fine[2*J * (2*nx_coarse) + 2*I] +
                                 f_fine[2*J * (2*nx_coarse) + 2*I + 1] +
                                 f_fine[(2*J + 1) * (2*nx_coarse) + 2*I] +
                                 f_fine[(2*J + 1) * (2*nx_coarse) + 2*I + 1]);

	float d1 = f_fine[2*J * (2*nx_coarse) + 2*I] - mean;
        float d2 = f_fine[2*J * (2*nx_coarse) + 2*I + 1] - mean;
        float d3 = f_fine[(2*J + 1) * (2*nx_coarse) + 2*I] - mean; 
	float d4 = f_fine[(2*J + 1) * (2*nx_coarse) + 2*I + 1] - mean;

        v_coarse[idx] = 0.25f * (d1*d1 + d2*d2 + d3*d3 + d4*d4);
    }
}


kernel void prolongate_vfacet_injection(
    device const float* u_coarse [[buffer(0)]],
    device float* u_fine [[buffer(1)]],
    device const float* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int ny_fine = (int)params[0];
    int nx_fine = (int)params[1];

    int idx = (int)tid;
    int total = ny_fine * (nx_fine + 1);
    
    if (idx < total) {
        int j = idx / (nx_fine + 1);
        int i = idx % (nx_fine + 1);
        
        int J = j / 2;  
        int I = i / 2;  
        
        if (i % 2 == 0) {
            
            
            u_fine[idx] = u_coarse[J * (nx_fine/2 + 1) + I];
        } else {
            
            
            if (I < nx_fine/2) {
                u_fine[idx] = 0.5f * (u_coarse[J * (nx_fine/2 + 1) + I] + 
                                     u_coarse[J * (nx_fine/2 + 1) + I + 1]);
            } else {
                
                u_fine[idx] = u_coarse[J * (nx_fine/2 + 1) + I];
            }
        }
    }
}


kernel void prolongate_vfacet_bilinear(
    device const float* u_coarse [[buffer(0)]],
    device float* u_fine [[buffer(1)]],
    device const float* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int ny_fine = (int)params[0];
    int nx_fine = (int)params[1];

    int idx = (int)tid;
    int total = ny_fine * (nx_fine + 1);
    
    if (idx >= total) return;
    
    int j = idx / (nx_fine + 1);  
    int i = idx % (nx_fine + 1);  
    
    int ny_coarse = ny_fine / 2;
    int nx_coarse = nx_fine / 2;
    
    
    
    
    
    
    
    
    float J_float = (j - 0.5f) * 0.5f;
    float I_float = i * 0.5f;
    
    
    J_float = fmax(0.0f, fmin(J_float, (float)(ny_coarse - 1)));
    I_float = fmax(0.0f, fmin(I_float, (float)nx_coarse));  
    
    
    int J_lo = (int)J_float;
    int I_lo = (int)I_float;
    int J_hi = min(J_lo + 1, ny_coarse - 1);
    int I_hi = min(I_lo + 1, nx_coarse);
    
    
    float t_y = J_float - J_lo;
    float t_x = I_float - I_lo;
    
    
    int stride = nx_coarse + 1;
    float v00 = u_coarse[J_lo * stride + I_lo];
    float v01 = u_coarse[J_lo * stride + I_hi];
    float v10 = u_coarse[J_hi * stride + I_lo];
    float v11 = u_coarse[J_hi * stride + I_hi];
    
    
    u_fine[idx] = (1.0f - t_y) * ((1.0f - t_x) * v00 + t_x * v01)
                + t_y         * ((1.0f - t_x) * v10 + t_x * v11);
}



kernel void prolongate_hfacet_injection(
    device const float* v_coarse [[buffer(0)]],
    device float* v_fine [[buffer(1)]],
    device const float* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int ny_fine = (int)params[0];
    int nx_fine = (int)params[1];

    int idx = (int)tid;
    int total = (ny_fine + 1) * nx_fine;
    
    if (idx < total) {
        int j = idx / nx_fine;
        int i = idx % nx_fine;
        
        int J = j / 2;  
        int I = i / 2;  
        
        if (j % 2 == 0) {
            
            
            v_fine[idx] = v_coarse[J * (nx_fine/2) + I];
        } else {
            
            
            if (J < ny_fine/2) {
                v_fine[idx] = 0.5f * (v_coarse[J * (nx_fine/2) + I] + 
                                     v_coarse[(J + 1) * (nx_fine/2) + I]);
            } else {
                
                v_fine[idx] = v_coarse[J * (nx_fine/2) + I];
            }
        }
    }
}





kernel void prolongate_hfacet_bilinear(
    device const float* v_coarse [[buffer(0)]],
    device float* v_fine [[buffer(1)]],
    device const float* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int ny_fine = (int)params[0];
    int nx_fine = (int)params[1];

    int idx = (int)tid;
    int total = (ny_fine + 1) * nx_fine;
    
    if (idx >= total) return;
    
    int j = idx / nx_fine;  
    int i = idx % nx_fine;  
    
    int ny_coarse = ny_fine / 2;
    int nx_coarse = nx_fine / 2;
    
    
    
    
    
    
    
    
    float J_float = j * 0.5f;
    float I_float = (i - 0.5f) * 0.5f;
    
    
    J_float = fmax(0.0f, fmin(J_float, (float)ny_coarse));  
    I_float = fmax(0.0f, fmin(I_float, (float)(nx_coarse - 1)));
    
    
    int J_lo = (int)J_float;
    int I_lo = (int)I_float;
    int J_hi = min(J_lo + 1, ny_coarse);
    int I_hi = min(I_lo + 1, nx_coarse - 1);
    
    
    float t_y = J_float - J_lo;
    float t_x = I_float - I_lo;
    
    
    int stride = nx_coarse;
    float v00 = v_coarse[J_lo * stride + I_lo];
    float v01 = v_coarse[J_lo * stride + I_hi];
    float v10 = v_coarse[J_hi * stride + I_lo];
    float v11 = v_coarse[J_hi * stride + I_hi];
    
    
    v_fine[idx] = (1.0f - t_y) * ((1.0f - t_x) * v00 + t_x * v01)
                + t_y         * ((1.0f - t_x) * v10 + t_x * v11);
}




kernel void prolongate_cell_injection(
    device const float* h_coarse [[buffer(0)]],
    device float* h_fine [[buffer(1)]],
    device const float* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int ny_fine = (int)params[0];
    int nx_fine = (int)params[1];

    int idx = (int)tid;
    
    if (idx < ny_fine * nx_fine) {
        int j = idx / nx_fine;
        int i = idx % nx_fine;
        
        int J = j / 2;
        int I = i / 2;
        
        h_fine[idx] = h_coarse[J * (nx_fine/2) + I];
    }
}




kernel void prolongate_cell_bilinear(
    device const float* H_coarse [[buffer(0)]],
    device float* H_fine [[buffer(1)]],
    device const float* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int ny_fine = (int)params[0];
    int nx_fine = (int)params[1];

    int idx = (int)tid;
    int total = ny_fine * nx_fine;
    
    if (idx >= total) return;
    
    int j = idx / nx_fine;  
    int i = idx % nx_fine;  
    
    int ny_coarse = ny_fine / 2;
    int nx_coarse = nx_fine / 2;
    
    
    
    
    
    float J_float = (j - 0.5f) * 0.5f;
    float I_float = (i - 0.5f) * 0.5f;
    
    
    J_float = fmax(0.0f, fmin(J_float, (float)(ny_coarse - 1)));
    I_float = fmax(0.0f, fmin(I_float, (float)(nx_coarse - 1)));
    
    
    int J_lo = (int)J_float;
    int I_lo = (int)I_float;
    int J_hi = min(J_lo + 1, ny_coarse - 1);
    int I_hi = min(I_lo + 1, nx_coarse - 1);
    
    
    float t_y = J_float - J_lo;
    float t_x = I_float - I_lo;
    
    
    float v00 = H_coarse[J_lo * nx_coarse + I_lo];
    float v01 = H_coarse[J_lo * nx_coarse + I_hi];
    float v10 = H_coarse[J_hi * nx_coarse + I_lo];
    float v11 = H_coarse[J_hi * nx_coarse + I_hi];
    
    
    H_fine[idx] = (1.0f - t_y) * ((1.0f - t_x) * v00 + t_x * v01)
                + t_y         * ((1.0f - t_x) * v10 + t_x * v11);
}

