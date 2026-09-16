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
   return 1.0f/(1.0f + expf(-scaled_z));
}

// Derivative of sigmoid w.r.t. z: d(sigmoid)/dz = c * sigmoid * (1 - sigmoid)
__device__ __forceinline__ float sigmoid_deriv(const float z, const float c) {
   float s = sigmoid(z, c);
   return c * s * (1.0f - s);
}


/*==================================================
  ============ Flotation (single source) ===========
  ==================================================*/
// Ice/water density ratio. Every grounded/floating quantity in the model
// derives from the flotation excess below, so this is the only place the
// ratio is typed.
#define RHO_I_OVER_RHO_W 0.917f

// Thickness floor for ratios in H: the constraint machinery can leave H at
// tiny negative values in the ocean.
#define FLOTATION_H_MIN 1e-3f

// Flotation excess z = rho_i/rho_w * H - depth, in metres of water column.
// depth is the signed head deficit (water level minus bed): positive under
// water, negative on dry land. z > 0 grounded, z < 0 floating, z = 0 at
// flotation; the effective pressure is N = rho_w g z.
__device__ __forceinline__ float flotation_excess(const float H, const float depth)
{
   return RHO_I_OVER_RHO_W*H - depth;
}

// Grounded flag phi = sigmoid(c z): the swish-form blend weight for the
// driving stress. Unbiased: phi = 1/2 exactly at flotation.
__device__ __forceinline__ float get_grounded(const float H, const float depth, const float sigmoid_c)
{
   return sigmoid(flotation_excess(H, depth), sigmoid_c);
}

// Flotation fraction xi = N / (rho_i g H) = z / (rho_i/rho_w H), clipped to
// [0,1]. Used as beta * xi^p in the sliding law. The floored H is used in z
// as well, so a bed at the water line with vanishing thickness reads as
// grounded (xi = 1) rather than as an H/H_MIN fraction.
__device__ __forceinline__ float get_flotation_fraction(const float H, const float depth)
{
   float Hf = fmaxf(H, FLOTATION_H_MIN);
   float z  = flotation_excess(Hf, depth);
   return fminf(fmaxf(z / (RHO_I_OVER_RHO_W*Hf), 0.0f), 1.0f);
}

// Calving flag: a single MONOTONE criterion psi = sigmoid(c r (H - H_calve))
// with the critical thickness blended between the grounded (height-above-
// buoyancy) and shelf (minimum thickness) laws by the gap between ice base
// and bed:
//
//   H_g     = (depth / r + h0) / (1 - q)          grounded: HAB < q H + h0
//   H_s     = min(H_c, H_g)                        shelf: H < H_c, capped by
//                                                  the grounded threshold so
//                                                  H_c acts only on tongues in
//                                                  water deeper than H_c / r
//   gap     = max(depth - r H, 0)                  ice base above the bed
//   G       = r |H_g - H_s|
//   w       = max(1 - gap / G, 0)                  w(0) = 1, w -> 0 detached
//   H_calve = H_s + w (H_g - H_s)
//
// The linear ramp with scale G is the least-grounded-like blend for which
// F = H - H_calve is non-decreasing in H (dH_calve/dH <= 1): thinning never
// reduces calving, which is what lets the implicit solve converge (a phi-
// blended pair of criteria protects a tongue root as it thins and stalls).
// Regimes for H_g > H_s: grounded ice F = H - H_g; floating ice down to
// H_c - h0 has F = -h0 (the margin's sign decides, no H dependence); thinner
// floating ice F = H - H_c. So a positive margin removes floating ice, a
// negative one lets a tongue exist above H_c. H_c = inf: F = H - H_g.
// Without the cap, a shallow margin whose flotation thickness is below H_c
// would lose any ice that goes afloat at once ("floating ice thinner than
// H_c calves" is not a shelf criterion in water shallower than H_c / r),
// and the grounded ice behind it thins and ungrounds in a retreat cascade —
// most of the model's sensitivity to H_c came from there (2026-09-17).
// With the cap the grounded law governs shallow margins and H_c only sets
// the minimum thickness of tongues in deep water. The sink on a cell is
// (1 - psi) H / tau.
__device__ __forceinline__ float calving_F(const float H, const float depth, const float q, const float h0, const float H_c)
{
   float Hg = (depth / RHO_I_OVER_RHO_W + h0) / (1.0f - q);
   float Hs = fminf(H_c, Hg);
   float gap = fmaxf(depth - RHO_I_OVER_RHO_W * H, 0.0f);
   float G = fmaxf(RHO_I_OVER_RHO_W * (Hg - Hs), 1e-3f);
   float w = fmaxf(1.0f - gap / G, 0.0f);      // H_c = inf: Hs = Hg, G -> 0, H_calve = Hg
   float Hcalve = Hs + w * (Hg - Hs);
   return H - Hcalve;
}

__device__ __forceinline__ float get_calving_flag(const float H, const float depth, const float sigmoid_c, const float q, const float h0, const float H_c)
{
   return sigmoid(RHO_I_OVER_RHO_W * calving_F(H, depth, q, h0, H_c), sigmoid_c);
}

// d psi / dH and d psi / d bed (depth = -bed), by central differences of
// the piecewise-linear F, for the calving sink's Jacobian.
__device__ __forceinline__ void get_calving_flag_derivs(const float H, const float depth, const float sigmoid_c, const float q, const float h0, const float H_c, float& dpsi_dH, float& dpsi_dbed)
{
   const float d = 0.25f;
   float F = calving_F(H, depth, q, h0, H_c);
   float psi = sigmoid(RHO_I_OVER_RHO_W * F, sigmoid_c);
   float dpsi_dF = sigmoid_c * RHO_I_OVER_RHO_W * psi * (1.0f - psi);
   float dF_dH = (calving_F(H + d, depth, q, h0, H_c) - calving_F(H - d, depth, q, h0, H_c)) / (2.0f * d);
   float dF_ddepth = (calving_F(H, depth + d, q, h0, H_c) - calving_F(H, depth - d, q, h0, H_c)) / (2.0f * d);
   dpsi_dH = dpsi_dF * dF_dH;
   dpsi_dbed = -dpsi_dF * dF_ddepth;
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


