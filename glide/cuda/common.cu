// =====================================================================
// COMMON UTILITIES: DualFloat, array access helpers, LU solvers
// =====================================================================

// Compile-time stress scheme switch. GLIDE_MOLHO=1 (default) compiles the
// full two-field model; GLIDE_MOLHO=0 (passed as -DGLIDE_MOLHO=0 when
// grid.stress_scheme == 'ssa') compiles out all deformational physics and
// shrinks the Vanka patch solves to the 5 live dofs. The runtime `ssa`
// kernel flag remains the semantic source of truth (see the constraint
// convention below); the SSA build is its specialization and must produce
// the same results as a MOLHO build running with ssa=true.
#ifndef GLIDE_MOLHO
#define GLIDE_MOLHO 1
#endif
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
   float scaled_z = fminf(fmaxf(c*z,-20.0f),20.0f);
   return 1.0f/(1.0f + __expf(-scaled_z));
}

// Derivative of sigmoid w.r.t. z: d(sigmoid)/dz = c * sigmoid * (1 - sigmoid)
__device__ __forceinline__ float sigmoid_deriv(const float z, const float c) {
   float s = sigmoid(z, c);
   return c * s * (1.0f - s);
}


//__device__ __forceinline__ float get_grounded(const float H, const float bed, const float sigmoid_c) 
//{
//   float z = bed + 0.917f*H;
//   return sigmoid(z, sigmoid_c);
//}

//__device__ __forceinline__ float get_grounded(const float H, const float bed, const float sigmoid_c) 
//{
//   float depth = fmaxf(-bed,0.0f);
//   float z = 0.917f*H - depth;
//   return fmaxf( fminf(1.0f + sigmoid_c*z,0.99f),0.01f);
//}

__device__ __forceinline__ float get_grounded(const float H, const float depth, const float sigmoid_c, const float sigmoid_k) 
{
   float z = 0.917f*H - depth + sigmoid_k/sigmoid_c;
   return sigmoid(z,sigmoid_c);
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

/* =====================================================================
   CONSTRAINT CONVENTION (single source of truth)

   Constrained dofs are Dirichlet velocity facets (u/ud at j in {0,nx},
   v/vd at i in {0,ny}) and active-set thickness cells (mask = 1).
   In SSA mode (the ssa kernel flag / grid.stress_scheme == 'ssa'), EVERY
   ud/vd facet is additionally constrained to zero, which reduces the
   MOLHO momentum balance exactly to the SSA; all of the machinery below
   applies to those dofs unchanged.
   compute_residual defines the convention: constrained dofs have IDENTITY
   residual rows, R_c = x_c - x_bc (r_u = u, r_H = H - thklim), while all
   other rows retain their genuine stencil dependence on constrained dofs.
   Everything else follows verbatim:

   - compute_jvp is the exact derivative: constrained rows return the
     direction component; nothing else is masked.
   - compute_vjp is the exact transpose: the kernel computes the pure
     physics transpose, and the constrained-row structure (project
     multipliers off constrained rows, add the identity part lambda_c) is
     applied ONCE in the Python wrapper (operators.py, _launch_vjp).
   - Parameter gradient kernels project out constrained-row multipliers
     explicitly (dR_c/dp = 0), so they are correct for any lambda.
   - The Vanka patch solves are PRECONDITIONERS and deliberately deviate
     from the true Jacobian at constrained dofs: they use symmetric
     row+column elimination (unit diagonal). Identity-row-only patches
     are exact but unstable when a cell ENTERS the active set - the
     momentum rows then extrapolate the full H -> thklim collapse
     linearly through their d/dH columns, which blows up velocities at
     thin margins. Fixed points are unaffected: the smoother rhs is
     always the exact residual (never zeroed), so the forward smoother
     converges to R = 0 and the adjoint smoother converges lambda_c to
     its true multiplier equation.

   Under this convention lambda at constrained dofs is the constraint
   multiplier (not zero); no consumer may assume it vanishes.
   ===================================================================== */


