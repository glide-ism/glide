// =====================================================================
// COMMON UTILITIES: DualFloat, array access helpers, LU solvers
// =====================================================================
struct DualFloat {
    float v; // Primal value
    float d; // Derivative/Perturbation component

    // Addition: (u + v, du + dv)
    __device__ __forceinline__ friend DualFloat operator+(DualFloat a, DualFloat b) {
        return {a.v + b.v, a.d + b.d};
    }

    // Subtraction: (u - v, du - dv)
    __device__ __forceinline__ friend DualFloat operator-(DualFloat a, DualFloat b) {
        return {a.v - b.v, a.d - b.d};
    }

    // Multiplication: (u * v, u * dv + v * du)
    __device__ __forceinline__ friend DualFloat operator*(DualFloat a, DualFloat b) {
        return {a.v * b.v, __fmaf_rn(a.v, b.d, a.d * b.v)};
    }

    // Multiplication by Scalar: (u * s, du * s)
    __device__ __forceinline__ friend DualFloat operator*(DualFloat a, float s) {
        return {a.v * s, a.d * s};
    }

    __device__ __forceinline__ friend DualFloat operator*(float s, DualFloat a) {
        return {a.v * s, a.d * s};
    }

    // Addition with Scalar: (u + s, du)
    __device__ __forceinline__ friend DualFloat operator+(DualFloat a, float s) {
	return {a.v + s, a.d};
    }

    // Commutative version: (s + u, du)
    __device__ __forceinline__ friend DualFloat operator+(float s, DualFloat a) {
	return {s + a.v, a.d};
    }

    // Subtraction with Scalar
    __device__ __forceinline__ friend DualFloat operator-(DualFloat a, float s) {
	return {a.v - s, a.d};
    }

    // Division by Scalar: (u / s, du / s)
    __device__ __forceinline__ friend DualFloat operator/(DualFloat a, float s) {
	float inv_s = 1.0f / s; // Compiler will likely use RCP
	return {a.v * inv_s, a.d * inv_s};
    }

};


__device__ __forceinline__ DualFloat __powf(DualFloat u, float p) {
    // High-performance hardware intrinsic pow
    float val = __powf(u.v, p);

    // d/dx(u^p) = p * u^(p-1) * du
    // If u.v is zero, derivative is technically singular; eps_reg handles this.
    float deriv = p * __powf(u.v, p - 1.0f) * u.d;

    return {val, deriv};
}

__device__ __forceinline__ float sigmoid(const float z, const float c) {
   float scaled_z = fminf(fmaxf(c*z,-10.0f),10.0f);
   return 1.0f/(1.0f + __expf(-scaled_z));
}

__device__ __forceinline__ float get_vfacet(const float* __restrict__ u, int i, int j, int ny, int nx) {
    //if (i < 0 || i >= ny || j < 0 || j > nx) return 0.0f;
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx),0);
    return u[i * (nx + 1) + j];
}

__device__ __forceinline__ DualFloat get_vfacet(const float* __restrict__ u, const float* __restrict__ du, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx),0);
    int idx = i * (nx + 1) + j;
    return {u[idx],du[idx]};
}

__device__ __forceinline__ float get_hfacet(const float* __restrict__ v, int i, int j, int ny, int nx) {
    //if (i < 0 || i > ny || j < 0 || j >= nx) return 0.0f;
    i = max(min(i,ny),0);
    j = max(min(j,nx - 1),0);
    return v[i * nx + j];
}

__device__ __forceinline__ DualFloat get_hfacet(const float* __restrict__ v, const float* __restrict__ dv, int i, int j, int ny, int nx) {
    i = max(min(i,ny),0);
    j = max(min(j,nx - 1),0);
    int idx = i * nx + j;
    return {v[idx],dv[idx]};
}

__device__ __forceinline__ float get_cell(const float* __restrict__ arr, int i, int j, int ny, int nx) {
    //if (i < 0 || i >= ny || j < 0 || j >= nx) return 0.0f;
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx - 1),0);
    return arr[i * nx + j];
}

__device__ __forceinline__ DualFloat get_cell(const float* __restrict__ arr, const float* __restrict__ darr, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx - 1),0);
    int idx = i * nx + j;
    return {arr[idx],darr[idx]};
}

__device__ __forceinline__ float get_masked_cell(const float* __restrict__ arr, const float* __restrict__ mask, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx - 1),0);
    int idx = i * nx + j;
    return arr[idx]*(1.0f - mask[idx]);
}



// ============================================================
// LU Solve for 5x5 Systems (Vanka smoother)
// ============================================================
__device__ void lu_5x5_solve(
    const float* A,  // 25 entries: full 5x5 row-major
    const float* b,  // 5 entries
    float* x)        // 5 entries (output)
{
    float LU[5][5];

    #pragma unroll
    for (int i = 0; i < 5; i++) {
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            LU[i][j] = A[i * 5 + j];
        }
    }

    // LU factorization (Doolittle, no pivoting)
    #pragma unroll
    for (int k = 0; k < 5; k++) {
        float inv_diag = 1.0f / LU[k][k];
        #pragma unroll
        for (int i = k + 1; i < 5; i++) {
            LU[i][k] *= inv_diag;
            #pragma unroll
            for (int j = k + 1; j < 5; j++) {
                LU[i][j] -= LU[i][k] * LU[k][j];
            }
        }
    }

    // Forward solve: L*y = b
    float y[5];
    y[0] = b[0];
    y[1] = b[1] - LU[1][0]*y[0];
    y[2] = b[2] - LU[2][0]*y[0] - LU[2][1]*y[1];
    y[3] = b[3] - LU[3][0]*y[0] - LU[3][1]*y[1] - LU[3][2]*y[2];
    y[4] = b[4] - LU[4][0]*y[0] - LU[4][1]*y[1] - LU[4][2]*y[2] - LU[4][3]*y[3];

    // Backward solve: U*x = y
    x[4] = y[4] / LU[4][4];
    x[3] = (y[3] - LU[3][4]*x[4]) / LU[3][3];
    x[2] = (y[2] - LU[2][3]*x[3] - LU[2][4]*x[4]) / LU[2][2];
    x[1] = (y[1] - LU[1][2]*x[2] - LU[1][3]*x[3] - LU[1][4]*x[4]) / LU[1][1];
    x[0] = (y[0] - LU[0][1]*x[1] - LU[0][2]*x[2] - LU[0][3]*x[3] - LU[0][4]*x[4]) / LU[0][0];
}


// =====================================================================
// LU SOLVE (5x5) WITH PIVOTING
// =====================================================================

__device__ void lu_5x5_solve_pivot(float A[25], const float b[5], float x[5])
{
    float rhs[5] = {b[0], b[1], b[2], b[3], b[4]};
    int piv[5] = {0, 1, 2, 3, 4};

    for (int k = 0; k < 4; k++) {
        float max_val = fabsf(A[piv[k] * 5 + k]);
        int max_idx = k;
        for (int ii = k + 1; ii < 5; ii++) {
            float val = fabsf(A[piv[ii] * 5 + k]);
            if (val > max_val) {
                max_val = val;
                max_idx = ii;
            }
        }

        int tmp = piv[k];
        piv[k] = piv[max_idx];
        piv[max_idx] = tmp;

        float pivot = A[piv[k] * 5 + k];
        if (fabsf(pivot) < 1e-14f) continue;

        for (int ii = k + 1; ii < 5; ii++) {
            float factor = A[piv[ii] * 5 + k] / pivot;
            A[piv[ii] * 5 + k] = factor;
            for (int jj = k + 1; jj < 5; jj++) {
                A[piv[ii] * 5 + jj] -= factor * A[piv[k] * 5 + jj];
            }
        }
    }

    float y[5];
    for (int ii = 0; ii < 5; ii++) {
        y[ii] = rhs[piv[ii]];
        for (int jj = 0; jj < ii; jj++) {
            y[ii] -= A[piv[ii] * 5 + jj] * y[jj];
        }
    }

    for (int ii = 4; ii >= 0; ii--) {
        x[ii] = y[ii];
        for (int jj = ii + 1; jj < 5; jj++) {
            x[ii] -= A[piv[ii] * 5 + jj] * x[jj];
        }
        float diag = A[piv[ii] * 5 + ii];
        if (fabsf(diag) > 1e-14f) {
            x[ii] /= diag;
        } else {
            x[ii] = 0.0f;
        }
    }
}

// =====================================================================
// Physics Parameters Struct
// Passed from Python via cupy structured array with matching dtype
// =====================================================================
struct PhysicsParams {
    float n;            // Glen's flow law exponent
    float eps_reg;      // Strain rate regularization
    float water_drag;   // Drag coefficient for floating ice
    float calving_rate; // Calving rate for mass loss at margins
};
