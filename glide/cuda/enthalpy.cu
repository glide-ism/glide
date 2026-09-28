/* =========================================================
   ================== Enthalpy Kernels =======================
   =========================================================

   Solves the conservative enthalpy equation in terrain-following
   (sigma) coordinates:

   rho_i [d(HE)/dt + d(HuE)/dx + d(HvE)/dy + d(E*omega)/dsigma]
       = (1/H) d/dsigma(K dE/dsigma) + H*phi - H*rho_w L Dw(omega)

   where omega = H*sigma_dot is the scaled vertical velocity.

   The conservative form tracks depth-integrated enthalpy HE,
   ensuring exact energy conservation under grid deformation.
   See notes/derivation.md for the derivation and notes/discretization.md
   for the discretization details.

   Discretization:
   - Horizontal: Lax-Friedrichs finite volume on MAC grid
   - Vertical: conservative finite volume on uniform sigma nodes
   - Column-wise Newton/Thomas solve as smoother (Vanka-style)

   Architecture:
   Each physical term is decomposed into a Stencil/Jacobian pair
   following the same pattern as the momentum balance (flux.cu,
   stress.cu). The get_*_jac() function is the single source of
   truth for both the residual and its derivatives. Both the
   residual kernel and the column smoother call the same functions.
   ========================================================= */

// Physical constants (RHO_I, RHO_W, C_I, K_I, L_HEAT, T_REF, T_MELT,
// BETA_CC, GRAVITY, K_TEMP_FACTOR, K_COLD) are injected as #define
// directives by EnthalpyOperators.__init__ from the Python-side values
// in glide/enthalpy.py — single source of truth.

// Maximum number of vertical sigma levels
#define MAX_NZ 64

// Drainage smoothing sharpness (same as K(E) sigmoid)
#define DRAIN_SHARPNESS 0.01f

// Nonlinear drainage "valve": basal frictional heating at fast temperate outlets
// can inject enthalpy faster than a constant-rate drainage can remove it, letting
// the water content run away (>>100%).  Above DRAIN_OMEGA_CAP the effective drainage
// rate ramps up by DRAIN_HIGH_FACTOR (a smooth softplus hinge of sharpness
// DRAIN_HINGE_SHARP), pinning omega near the cap -- the standard Aschwanden/Greve
// nonlinear drainage.  Below the cap it reduces to the usual linear drain_rate*omega.
#define DRAIN_OMEGA_CAP   0.01f    // basal water-content ceiling (~1%, Aschwanden 2012)
#define DRAIN_HIGH_FACTOR 2000.0f  // effective-rate multiplier above the cap
// Softplus sharpness (1/omega) of the cap transition.  Must be sharp enough that the
// hinge is negligible for omega << cap even after the DRAIN_HIGH_FACTOR amplification
// (otherwise cold ice, omega~0, gets a spurious sink): the sub-cap leak is
// HIGH_FACTOR*softplus(-SHARP*cap)/SHARP ~ 4e-7 here, vs the 0.01 cap.
#define DRAIN_HINGE_SHARP 1500.0f

// Term toggle bitmask flags
#define TERM_HORIZ_ADV    (1 << 0)
#define TERM_SIGMA_DOT    (1 << 1)
#define TERM_DRAINAGE     (1 << 2)


// ---- Helper: enthalpy at pressure melting point (scaled) ----
__device__ __forceinline__
float get_E_pmp(float sigma, float H) {
    // Returns E_pmp / E_SCALE (dimensionless, O(1)).
    float depth = (1.0f - sigma) * H;
    float T_pmp = T_MELT - BETA_CC * RHO_I * GRAVITY * depth;
    return C_I * (T_pmp - T_REF) / E_SCALE;
}

// ---- Helper: diffusivity K(E) with smooth transition ----
// Arguments are in scaled units (E/E_SCALE). The sigmoid
// reconstructs physical-scale differences internally.
__device__ __forceinline__
float get_K(float E, float E_pmp) {
    float z = (E - E_pmp) * E_SCALE * 0.01f;
    z = fminf(fmaxf(z, -20.0f), 20.0f);
    float s = 1.0f / (1.0f + __expf(-z));
    return K_COLD * (1.0f - s + K_TEMP_FACTOR * s);
}

// ---- Helper: water content from enthalpy (smoothed) ----
// Arguments are in scaled units. Reconstructs physical E for
// the softplus, returns physical water content (dimensionless).
__device__ __forceinline__
float get_omega(float E, float E_pmp) {
    float x = (E - E_pmp) * E_SCALE;  // physical difference
    float ax = DRAIN_SHARPNESS * x;
    ax = fminf(fmaxf(ax, -20.0f), 20.0f);
    float sp = (ax > 10.0f) ? x : __logf(1.0f + __expf(ax)) / DRAIN_SHARPNESS;
    return sp / L_HEAT;
}

// ---- Helper: d(omega)/dE_phys — sigmoid ----
// Arguments are in scaled units. Returns the derivative with
// respect to *physical* E (not scaled), i.e. d(omega)/d(E_phys).
// The drainage Jacobian d(R_scaled)/d(E_scaled) = d(R_phys)/d(E_phys)
// because the E_SCALE factors cancel (see derivation in enthalpy.py).
__device__ __forceinline__
float get_domega_dE(float E, float E_pmp) {
    float z = DRAIN_SHARPNESS * (E - E_pmp) * E_SCALE;
    z = fminf(fmaxf(z, -20.0f), 20.0f);
    float s = 1.0f / (1.0f + __expf(-z));
    return s / L_HEAT;
}

// ---- Helper: drainage function Dw(omega) ----
// Linear drain_rate*omega below DRAIN_OMEGA_CAP, plus a steep softplus valve above
// it so the water content cannot run away when frictional heating is large.
__device__ __forceinline__
float get_drainage(float omega, float drain_rate) {
    float x = DRAIN_HINGE_SHARP * (omega - DRAIN_OMEGA_CAP);
    float hinge = (x > 10.0f) ? (omega - DRAIN_OMEGA_CAP)
                              : __logf(1.0f + __expf(x)) / DRAIN_HINGE_SHARP;
    return drain_rate * (omega + DRAIN_HIGH_FACTOR * hinge);
}

// ---- Helper: d(Dw)/d(omega) for the drainage Jacobian ----
__device__ __forceinline__
float get_ddrainage_domega(float omega, float drain_rate) {
    float z = DRAIN_HINGE_SHARP * (omega - DRAIN_OMEGA_CAP);
    z = fminf(fmaxf(z, -20.0f), 20.0f);
    float s = 1.0f / (1.0f + __expf(-z));               // sigmoid = d(hinge)/d(omega)
    return drain_rate * (1.0f + DRAIN_HIGH_FACTOR * s);
}

// ---- Helper: 3D cell-centered access for E (ny, nx, nz layout) ----
__device__ __forceinline__
float get_E(const float* __restrict__ E, int i, int j, int k,
            int ny, int nx, int nz) {
    i = max(min(i, ny - 1), 0);
    j = max(min(j, nx - 1), 0);
    k = max(min(k, nz - 1), 0);
    return E[(i * nx + j) * nz + k];
}

// ---- Helper: 3D facet access for u (nz, ny, nx+1 layout) ----
__device__ __forceinline__
float get_u3d(const float* __restrict__ u, int k, int i, int j,
              int nz, int ny, int nx) {
    k = max(min(k, nz - 1), 0);
    i = max(min(i, ny - 1), 0);
    j = max(min(j, nx), 0);
    return u[(k * ny + i) * (nx + 1) + j];
}

// ---- Helper: 3D facet access for v (nz, ny+1, nx layout) ----
__device__ __forceinline__
float get_v3d(const float* __restrict__ v, int k, int i, int j,
              int nz, int ny, int nx) {
    k = max(min(k, nz - 1), 0);
    i = max(min(i, ny), 0);
    j = max(min(j, nx - 1), 0);
    return v[(k * (ny + 1) + i) * nx + j];
}


/* =========================================================
   ========== Jacobian Structs for Enthalpy Terms ===========
   =========================================================

   Each physical term is decomposed into:
     1. Stencil  — local inputs needed to evaluate the term
     2. Jacobian — residual + partial derivatives w.r.t. column unknowns
     3. get_*_jac() — single function computing both res and d_E_*

   This mirrors the momentum balance pattern in flux.cu / stress.cu
   (HorizontalFluxStencil/Jacobian, SigmaNormalStencil/Jacobian, etc.)
   ========================================================= */


/* ---------------------------------------------------------
   Conservative Time Term (implicit Euler):
     rho_i * (H * E_k - H_prev * E_prev_k) / dt
   The Jacobian carries derivatives w.r.t. all four state inputs:
   the current solver unknown E_k, the cached previous state E_prev_k
   (frozen during the smoother but live for adjoints), and both
   thicknesses H and H_prev (also frozen during the smoother).
   --------------------------------------------------------- */
struct TimeStencil {
    float E_k;
    float E_prev_k;
    float H;
    float H_prev;
    float dt;
};

struct TimeStencilDual {
    DualFloat E_k;
    DualFloat E_prev_k;
    DualFloat H;
    DualFloat H_prev;
    float dt;

    __device__ __forceinline__
    TimeStencil get_primals() const {
        return {E_k.v, E_prev_k.v, H.v, H_prev.v, dt};
    }

    __device__ __forceinline__
    TimeStencil get_diffs() const {
        return {E_k.d, E_prev_k.d, H.d, H_prev.d, 0.0f};
    }
};

struct TimeJacobian {
    float res;
    float d_E_k;
    float d_E_prev_k;
    float d_H;
    float d_H_prev;

    __device__ __forceinline__
    float apply_jvp(const TimeStencil& dot) const {
        return d_E_k * dot.E_k + d_E_prev_k * dot.E_prev_k
             + d_H * dot.H + d_H_prev * dot.H_prev;
    }
};

__device__ __forceinline__
TimeJacobian get_time_jac(TimeStencil s) {
    TimeJacobian jac = {0};
    float inv_dt = 1.0f / s.dt;
    jac.res        =  RHO_I * (s.H * s.E_k - s.H_prev * s.E_prev_k) * inv_dt;
    jac.d_E_k      =  RHO_I * s.H * inv_dt;
    jac.d_E_prev_k = -RHO_I * s.H_prev * inv_dt;
    jac.d_H        =  RHO_I * s.E_k * inv_dt;
    jac.d_H_prev   = -RHO_I * s.E_prev_k * inv_dt;
    return jac;
}

__device__ __forceinline__
DualFloat get_time_dual(TimeStencilDual s) {
    TimeJacobian jac = get_time_jac(s.get_primals());
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}


/* ---------------------------------------------------------
   Vertical Diffusion (conservative form):
     -(1/H) d/dsigma(K dE/dsigma)
   at an interior node k with neighbors k-1, k+1.
   Returns tridiagonal contributions (d_E_km1, d_E_k, d_E_kp1).
   --------------------------------------------------------- */
struct ColumnDiffusionStencil {
    float E_km1, E_k, E_kp1;
    float E_pmp_km1, E_pmp_k, E_pmp_kp1;
    float dsig;
    float h_inv;
};

struct ColumnDiffusionStencilDual {
    DualFloat E_km1, E_k, E_kp1;
    float E_pmp_km1, E_pmp_k, E_pmp_kp1;
    float dsig;
    float h_inv;

    __device__ __forceinline__
    ColumnDiffusionStencil get_primals() const {
        return {E_km1.v, E_k.v, E_kp1.v,
                E_pmp_km1, E_pmp_k, E_pmp_kp1,
                dsig, h_inv};
    }

    __device__ __forceinline__
    ColumnDiffusionStencil get_diffs() const {
        return {E_km1.d, E_k.d, E_kp1.d,
                0.0f, 0.0f, 0.0f,
                0.0f, 0.0f};
    }
};

struct ColumnDiffusionJacobian {
    float res;
    float d_E_km1, d_E_k, d_E_kp1;

    __device__ __forceinline__
    float apply_jvp(const ColumnDiffusionStencil& dot) const {
        return d_E_km1 * dot.E_km1 + d_E_k * dot.E_k + d_E_kp1 * dot.E_kp1;
    }
};

__device__ __forceinline__
ColumnDiffusionJacobian get_column_diffusion_jac(ColumnDiffusionStencil s) {
    ColumnDiffusionJacobian jac = {0};

    float K_upper = get_K(0.5f * (s.E_k + s.E_kp1),
                          0.5f * (s.E_pmp_k + s.E_pmp_kp1));
    float K_lower = get_K(0.5f * (s.E_k + s.E_km1),
                          0.5f * (s.E_pmp_k + s.E_pmp_km1));

    float dsig2_inv = 1.0f / (s.dsig * s.dsig);

    float dE_upper = s.E_kp1 - s.E_k;
    float dE_lower = s.E_k - s.E_km1;

    // Residual: -(1/H) * [K_upper*(E_kp1-E_k) - K_lower*(E_k-E_km1)] / dsig^2
    jac.res = -s.h_inv * dsig2_inv * (K_upper * dE_upper - K_lower * dE_lower);

    // Jacobian: freeze K at current state 
    float diff_lower = s.h_inv * dsig2_inv * K_lower;
    float diff_upper = s.h_inv * dsig2_inv * K_upper;

    jac.d_E_km1 = -diff_lower;
    jac.d_E_kp1 = -diff_upper;
    jac.d_E_k   =  diff_lower + diff_upper;

    return jac;
}

__device__ __forceinline__
DualFloat get_column_diffusion_dual(ColumnDiffusionStencilDual s) {
    ColumnDiffusionJacobian jac = get_column_diffusion_jac(s.get_primals());
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}


/* ---------------------------------------------------------
   Bed Diffusion (conservative form, basal flux BC):
     -(1/H) K_{1/2} (E_1 - E_0)/dsig - (Q_geo + Q_fh)
   The Q_geo+Q_fh term no longer has the 1/H factor because
   the original source -(Q_geo+Q_fh)/H gets multiplied by H
   in the conservative form, cancelling the denominator.
   Returns contributions for k=0 only (d_E_k = d_E_0, d_E_kp1 = d_E_1).
   --------------------------------------------------------- */
struct BedDiffusionStencil {
    float E_k, E_kp1;
    float E_pmp_k, E_pmp_kp1;
    float dsig;
    float h_inv;
    float Q_geo, Q_fh;
};

struct BedDiffusionStencilDual {
    DualFloat E_k, E_kp1;
    float E_pmp_k, E_pmp_kp1;
    float dsig;
    float h_inv;
    float Q_geo, Q_fh;

    __device__ __forceinline__
    BedDiffusionStencil get_primals() const {
        return {E_k.v, E_kp1.v, E_pmp_k, E_pmp_kp1,
                dsig, h_inv, Q_geo, Q_fh};
    }

    __device__ __forceinline__
    BedDiffusionStencil get_diffs() const {
        return {E_k.d, E_kp1.d, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f};
    }
};

struct BedDiffusionJacobian {
    float res;
    float d_E_k, d_E_kp1;

    __device__ __forceinline__
    float apply_jvp(const BedDiffusionStencil& dot) const {
        return d_E_k * dot.E_k + d_E_kp1 * dot.E_kp1;
    }
};

__device__ __forceinline__
BedDiffusionJacobian get_bed_diffusion_jac(BedDiffusionStencil s) {
    BedDiffusionJacobian jac = {0};

    float K_half = get_K(0.5f * (s.E_k + s.E_kp1),
                         0.5f * (s.E_pmp_k + s.E_pmp_kp1));

    // Half-cell control volume width at the bed boundary.
    float dsig_half = 0.5f * s.dsig;

    // Residual:
    //   -(1/H) * K_{1/2} * (E_1-E_0)/dsig / dsig_half - (Q_geo+Q_fh) / dsig_half
    jac.res = (-s.h_inv * K_half * (s.E_kp1 - s.E_k) / s.dsig
               - (s.Q_geo + s.Q_fh)) / dsig_half;

    // Jacobian: freeze K at current state
    float coeff = s.h_inv * K_half / (s.dsig * dsig_half);
    jac.d_E_k   =  coeff;
    jac.d_E_kp1 = -coeff;

    return jac;
}

__device__ __forceinline__
DualFloat get_bed_diffusion_dual(BedDiffusionStencilDual s) {
    BedDiffusionJacobian jac = get_bed_diffusion_jac(s.get_primals());
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}


/* ---------------------------------------------------------
   Sigma (Vertical) Advection:
     rho_i * d(E * omega) / dsigma
   where omega = H * sigma_dot is the scaled vertical velocity.

   The conservative flux at interface k+1/2 is:
     F_{k+1/2} = E_upwind * omega_{k+1/2}
   where omega_{k+1/2} = 0.5*(omega_k + omega_{k+1}).
   Returns tridiagonal contributions.
   --------------------------------------------------------- */
struct SigmaAdvectionStencil {
    float E_km1, E_k, E_kp1;
    float omega_km1, omega_k, omega_kp1;  // omega = H*sigma_dot at nodes
    float dsig;
};

struct SigmaAdvectionStencilDual {
    DualFloat E_km1, E_k, E_kp1;
    float omega_km1, omega_k, omega_kp1;
    float dsig;

    __device__ __forceinline__
    SigmaAdvectionStencil get_primals() const {
        return {E_km1.v, E_k.v, E_kp1.v,
                omega_km1, omega_k, omega_kp1, dsig};
    }

    __device__ __forceinline__
    SigmaAdvectionStencil get_diffs() const {
        return {E_km1.d, E_k.d, E_kp1.d,
                0.0f, 0.0f, 0.0f, 0.0f};
    }
};

struct SigmaAdvectionJacobian {
    float res;
    float d_E_km1, d_E_k, d_E_kp1;

    __device__ __forceinline__
    float apply_jvp(const SigmaAdvectionStencil& dot) const {
        return d_E_km1 * dot.E_km1 + d_E_k * dot.E_k + d_E_kp1 * dot.E_kp1;
    }
};

__device__ __forceinline__
SigmaAdvectionJacobian get_sigma_advection_jac(SigmaAdvectionStencil s) {
    SigmaAdvectionJacobian jac = {0};

    float dsig_inv = 1.0f / s.dsig;

    // Interface omega values (averaged to half-nodes)
    float omega_upper = 0.5f * (s.omega_k + s.omega_kp1);
    float omega_lower = 0.5f * (s.omega_km1 + s.omega_k);

    // Upwind fluxes: F = E_upwind * omega_half
    float omega_up_pos = fmaxf(omega_upper, 0.0f);
    float omega_up_neg = fminf(omega_upper, 0.0f);
    float F_upper = omega_up_pos * s.E_k + omega_up_neg * s.E_kp1;

    float omega_lo_pos = fmaxf(omega_lower, 0.0f);
    float omega_lo_neg = fminf(omega_lower, 0.0f);
    float F_lower = omega_lo_pos * s.E_km1 + omega_lo_neg * s.E_k;

    // Residual: rho_i * (F_upper - F_lower) / dsig
    jac.res = RHO_I * (F_upper - F_lower) * dsig_inv;

    // Jacobian w.r.t. column unknowns
    jac.d_E_km1 = -RHO_I * omega_lo_pos * dsig_inv;
    jac.d_E_kp1 =  RHO_I * omega_up_neg * dsig_inv;
    jac.d_E_k   =  RHO_I * (omega_up_pos - omega_lo_neg) * dsig_inv;

    return jac;
}

__device__ __forceinline__
DualFloat get_sigma_advection_dual(SigmaAdvectionStencilDual s) {
    SigmaAdvectionJacobian jac = get_sigma_advection_jac(s.get_primals());
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}


/* ---------------------------------------------------------
   Bed Sigma Advection (k=0):
     Uses half-cell. The bed boundary flux is F_bed = omega_bed * E_bed.
     With no basal melt, omega_bed ≈ 0 so F_bed ≈ 0.
     The upper interface flux uses upwinding on omega_{1/2}.
   --------------------------------------------------------- */
struct BedSigmaAdvectionStencil {
    float E_k, E_kp1;
    float omega_k, omega_kp1;  // omega at bed (k=0) and first interior (k=1)
    float dsig;
};

struct BedSigmaAdvectionStencilDual {
    DualFloat E_k, E_kp1;
    float omega_k, omega_kp1;
    float dsig;

    __device__ __forceinline__
    BedSigmaAdvectionStencil get_primals() const {
        return {E_k.v, E_kp1.v, omega_k, omega_kp1, dsig};
    }

    __device__ __forceinline__
    BedSigmaAdvectionStencil get_diffs() const {
        return {E_k.d, E_kp1.d, 0.0f, 0.0f, 0.0f};
    }
};

struct BedSigmaAdvectionJacobian {
    float res;
    float d_E_k, d_E_kp1;

    __device__ __forceinline__
    float apply_jvp(const BedSigmaAdvectionStencil& dot) const {
        return d_E_k * dot.E_k + d_E_kp1 * dot.E_kp1;
    }
};

__device__ __forceinline__
BedSigmaAdvectionJacobian get_bed_sigma_advection_jac(BedSigmaAdvectionStencil s) {
    BedSigmaAdvectionJacobian jac = {0};

    float dsig_half = 0.5f * s.dsig;
    float dsig_half_inv = 1.0f / dsig_half;

    // Upper interface flux (upwind on omega_{1/2})
    float omega_half = 0.5f * (s.omega_k + s.omega_kp1);
    float omega_pos = fmaxf(omega_half, 0.0f);
    float omega_neg = fminf(omega_half, 0.0f);
    float F_upper = omega_pos * s.E_k + omega_neg * s.E_kp1;

    // Bed boundary flux: F_bed = omega_bed * E_bed
    // omega_bed = omega[k=0] which includes basal melt (typically ~0)
    float F_bed = s.omega_k * s.E_k;

    // Residual: rho_i * (F_upper - F_bed) / dsig_half
    jac.res = RHO_I * (F_upper - F_bed) * dsig_half_inv;

    // Jacobian
    jac.d_E_k   = RHO_I * (omega_pos - s.omega_k) * dsig_half_inv;
    jac.d_E_kp1 = RHO_I * omega_neg * dsig_half_inv;

    return jac;
}

__device__ __forceinline__
DualFloat get_bed_sigma_advection_dual(BedSigmaAdvectionStencilDual s) {
    BedSigmaAdvectionJacobian jac = get_bed_sigma_advection_jac(s.get_primals());
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}


/* ---------------------------------------------------------
   Horizontal Enthalpy Flux (per facet, Lax-Friedrichs):
     F = 0.5 * u * (E_here + E_neighbor) - 0.5 * alpha * (E_neighbor - E_here)
   where alpha = sqrt(u^2 + LF_C) is the smoothed wave speed.

   Mirrors the mass flux in flux.cu (get_horizontal_flux_jac).
   With LF_C = 0: recovers exact upwind (non-differentiable at u=0).
   With LF_C > 0: smooth Jacobian with baseline diffusion sqrt(LF_C).

   The d_E_here derivative is always nonzero (no discontinuous
   upwind switch), giving a more robust diagonal contribution
   in the column smoother.
   --------------------------------------------------------- */

// Default Lax-Friedrichs regularization constant (m^2/s^2).
// Used as fallback if not passed as a kernel parameter.
#define LF_C_DEFAULT 1e-12f

struct HorizEnthalpyFluxStencil {
    float u;          // facet velocity (positive = left-to-right)
    float E_l;        // enthalpy on the left side of the face
    float E_r;        // enthalpy on the right side of the face
};

struct HorizEnthalpyFluxStencilDual {
    float u;
    DualFloat E_l;
    DualFloat E_r;

    __device__ __forceinline__
    HorizEnthalpyFluxStencil get_primals() const {
        return {u, E_l.v, E_r.v};
    }

    __device__ __forceinline__
    HorizEnthalpyFluxStencil get_diffs() const {
        return {0.0f, E_l.d, E_r.d};
    }
};

struct HorizEnthalpyFluxJacobian {
    float res;       // flux value
    float d_E_l;     // derivative w.r.t. left cell E
    float d_E_r;     // derivative w.r.t. right cell E

    __device__ __forceinline__
    float apply_jvp(const HorizEnthalpyFluxStencil& dot) const {
        return d_E_l * dot.E_l + d_E_r * dot.E_r;
    }
};

__device__ __forceinline__
HorizEnthalpyFluxJacobian get_horiz_enthalpy_flux_jac(HorizEnthalpyFluxStencil s,
                                                       float lf_c) {
    HorizEnthalpyFluxJacobian jac = {0};

    // Lax-Friedrichs flux: central + dissipation
    // Mirrors get_horizontal_flux_jac in flux.cu.
    // alpha = |u| * sqrt(1 + lf_c) regularizes the upwind switch.
    // Unlike sqrt(u^2 + lf_c), this produces zero dissipation when
    // u = 0, preventing spurious cross-diffusion on quiescent faces.
    // For nonzero u, it adds a fractional dissipation boost of
    // sqrt(1 + lf_c) - 1 ≈ lf_c/2 relative to pure upwind.
    float alpha = fabsf(s.u) * sqrtf(1.0f + lf_c);

    // F = 0.5 * u * (E_l + E_r) - 0.5 * alpha * (E_r - E_l)
    jac.res = 0.5f * s.u * (s.E_l + s.E_r)
            - 0.5f * alpha * (s.E_r - s.E_l);

    jac.d_E_l = 0.5f * (s.u + alpha);
    jac.d_E_r = 0.5f * (s.u - alpha);

    return jac;
}

__device__ __forceinline__
DualFloat get_horiz_enthalpy_flux_dual(HorizEnthalpyFluxStencilDual s,
                                       float lf_c = LF_C_DEFAULT) {
    HorizEnthalpyFluxJacobian jac = get_horiz_enthalpy_flux_jac(s.get_primals(), lf_c);
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}


/* ---------------------------------------------------------
   Drainage Source Term:
     rho_w * L * drain_rate * omega(E, E_pmp)
   Positive = removes enthalpy from temperate ice.
   --------------------------------------------------------- */
struct DrainageStencil {
    float E_k;
    float E_pmp_k;
    float drain_rate;
    float H;           // ice thickness (conservative form scales by H)
};

struct DrainageStencilDual {
    DualFloat E_k;
    float E_pmp_k;
    float drain_rate;
    float H;

    __device__ __forceinline__
    DrainageStencil get_primals() const {
        return {E_k.v, E_pmp_k, drain_rate, H};
    }

    __device__ __forceinline__
    DrainageStencil get_diffs() const {
        return {E_k.d, 0.0f, 0.0f, 0.0f};
    }
};

struct DrainageJacobian {
    float res;
    float d_E_k;

    __device__ __forceinline__
    float apply_jvp(const DrainageStencil& dot) const {
        return d_E_k * dot.E_k;
    }
};

__device__ __forceinline__
DrainageJacobian get_drainage_jac(DrainageStencil s) {
    DrainageJacobian jac = {0};

    float omega = get_omega(s.E_k, s.E_pmp_k);
    // Drainage is nonlinear in E and produces a physical-scale residual.
    // Divide by E_SCALE to match the other (E-linear) residual terms.
    // The Jacobian d(R_scaled)/d(E_scaled) = d(R_phys)/d(E_phys)
    // because the E_SCALE factors cancel in the chain rule.  The chain rule now
    // carries the nonlinear valve: d(Dw)/d(E) = d(Dw)/d(omega) * d(omega)/d(E).
    jac.res = s.H * RHO_W * L_HEAT * get_drainage(omega, s.drain_rate) / E_SCALE;
    jac.d_E_k = s.H * RHO_W * L_HEAT
              * get_ddrainage_domega(omega, s.drain_rate)
              * get_domega_dE(s.E_k, s.E_pmp_k);

    return jac;
}

__device__ __forceinline__
DualFloat get_drainage_dual(DrainageStencilDual s) {
    DrainageJacobian jac = get_drainage_jac(s.get_primals());
    return {jac.res, jac.apply_jvp(s.get_diffs())};
}


/* ---------------------------------------------------------
   Horizontal advection assembly for one layer (conservative form).

   Computes rho_i * div_h(H*u*E) in flux divergence form.
   The mass flux H*u at each face is computed from the face
   velocity and face-averaged thickness.
   --------------------------------------------------------- */
struct HorizAdvectionResult {
    float res;       // rho_i * (u dE/dx + v dE/dy)  [advective form]
    float d_E_here;  // diagonal Jacobian contribution
};

__device__ __forceinline__
HorizAdvectionResult get_horiz_advection(
    const float* __restrict__ E,
    const float* __restrict__ u3d,
    const float* __restrict__ v3d,
    const float* __restrict__ H,
    float E_here,
    int i, int j, int k,
    int ny, int nx, int nz,
    float dx_inv,
    float lf_c)
{
    HorizAdvectionResult result = {0};

    // Cell thicknesses for mass flux computation
    float H_here = H[i * nx + j];

    // x-direction: mass flux Hu at each face using Lax-Friedrichs on H,
    // matching the momentum solver (flux.cu: get_horizontal_flux_jac).
    // This ensures div(Hu) here equals the momentum solver's div(Hu),
    // so uniform E produces zero residual when combined with omega.
    float u_left  = get_u3d(u3d, k, i, j,   nz, ny, nx);
    float u_right = get_u3d(u3d, k, i, j+1, nz, ny, nx);
    float E_xm = get_E(E, i, j-1, k, ny, nx, nz);
    float E_xp = get_E(E, i, j+1, k, ny, nx, nz);

    HorizEnthalpyFluxJacobian fl = {0};
    HorizEnthalpyFluxJacobian fr = {0};

    if (j > 0) {
        float H_xm = get_cell(H, i, j-1, ny, nx);
        float H_avg = 0.5f * (H_xm + H_here);
        float Hu_left = H_avg * u_left - 0.5f * sqrtf(u_left * u_left + MASS_FLUX_REG) * (H_here - H_xm);
        fl = get_horiz_enthalpy_flux_jac({Hu_left, E_xm, E_here}, lf_c);
    } else {
        fl = get_horiz_enthalpy_flux_jac({0.0f, E_here, E_here}, lf_c);
    }
    if (j < nx - 1) {
        float H_xp = get_cell(H, i, j+1, ny, nx);
        float H_avg = 0.5f * (H_here + H_xp);
        float Hu_right = H_avg * u_right - 0.5f * sqrtf(u_right * u_right + MASS_FLUX_REG) * (H_xp - H_here);
        fr = get_horiz_enthalpy_flux_jac({Hu_right, E_here, E_xp}, lf_c);
    } else {
        fr = get_horiz_enthalpy_flux_jac({0.0f, E_here, E_here}, lf_c);
    }

    // y-direction faces.
    // The momentum solver's v-convention has v > 0 = decreasing i
    // (flux.cu: get_vertical_flux_jac, residual assembles j_t - j_b).
    // The mass flux Hv > 0 means flow toward smaller i (from E_r=E[i]
    // toward E_l=E[i-1] at the top face). The LF flux upwinds E_l when
    // u > 0. To get correct upwinding, we swap E_l and E_r so that
    // E_l is the upstream cell: {Hv, E[i], E[i-1]} at the top face.
    // This preserves the mass flux sign (for uniform-E conservation)
    // while fixing the upwind direction.
    float v_top    = get_v3d(v3d, k, i,   j, nz, ny, nx);
    float v_bottom = get_v3d(v3d, k, i+1, j, nz, ny, nx);
    float E_ym = get_E(E, i-1, j, k, ny, nx, nz);
    float E_yp = get_E(E, i+1, j, k, ny, nx, nz);

    HorizEnthalpyFluxJacobian ft = {0};
    HorizEnthalpyFluxJacobian fb = {0};

    if (i > 0) {
        float H_ym = get_cell(H, i-1, j, ny, nx);
        float H_avg = 0.5f * (H_ym + H_here);
        float Hv_top = H_avg * v_top - 0.5f * sqrtf(v_top * v_top + MASS_FLUX_REG) * (H_ym - H_here);
        ft = get_horiz_enthalpy_flux_jac({Hv_top, E_here, E_ym}, lf_c);
    } else {
        ft = get_horiz_enthalpy_flux_jac({0.0f, E_here, E_here}, lf_c);
    }
    if (i < ny - 1) {
        float H_yp = get_cell(H, i+1, j, ny, nx);
        float H_avg = 0.5f * (H_here + H_yp);
        float Hv_bottom = H_avg * v_bottom - 0.5f * sqrtf(v_bottom * v_bottom + MASS_FLUX_REG) * (H_here - H_yp);
        fb = get_horiz_enthalpy_flux_jac({Hv_bottom, E_yp, E_here}, lf_c);
    } else {
        fb = get_horiz_enthalpy_flux_jac({0.0f, E_here, E_here}, lf_c);
    }

    // Conservative flux divergence: rho_i * div(H*u*E)
    // y-convention: (ft - fb) matches the momentum solver's (j_t - j_b).
    result.res = RHO_I * ((fr.res - fl.res) + (ft.res - fb.res)) * dx_inv;
    result.d_E_here = RHO_I * ((fr.d_E_l - fl.d_E_r)
                              + (ft.d_E_l - fb.d_E_r)) * dx_inv;

    return result;
}


/* =========================================================
   Compute the conservative enthalpy residual at all (i, j, k).

   The operator F(E) contains all E-dependent terms:
     F(E) = rho_i*H*E/dt + rho_i*div_h(HuE)
            + rho_i*d(E*omega)/dsig - (1/H)*d/dsig(K*dE/dsig)
            + H*drainage(E)

   The forcing f_E contains all E-independent terms:
     f_E = rho_i*H_prev*E_prev/dt + H*phi_strain
           + (Q_geo+Q_fh)/dsig_half  [at k=0 only]

   When use_forcing=1: out = F(E) - f_E  (full residual)
   When use_forcing=0: out = F(E)        (operator only, for FAS)

   The forcing f_E is precomputed by set_rhs() on the Python side.
   ========================================================= */
extern "C" __global__
void enthalpy_compute_residual(
    float* __restrict__ out,            // (ny, nx, nz) output: r_E or F_E
    const float* __restrict__ E,        // (ny, nx, nz) current enthalpy
    const float* __restrict__ E_prev,   // (ny, nx, nz) previous-step enthalpy
    const float* __restrict__ f_E,      // (ny, nx, nz) precomputed forcing

    const float* __restrict__ u3d,      // (nz, ny, nx+1) x-velocity per layer
    const float* __restrict__ v3d,      // (nz, ny+1, nx) y-velocity per layer
    const float* __restrict__ omega,    // (ny, nx, nz) omega = H*sigma_dot
    const float* __restrict__ H,        // (ny, nx) ice thickness
    const float* __restrict__ H_prev,   // (ny, nx) previous-step thickness
    const float* __restrict__ E_surface,// (ny, nx) surface enthalpy BC
    float dx, float dt, float drain_rate, float h_thin, float lf_c,
    int term_flags, int use_forcing,
    int ny, int nx, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_columns = ny * nx;
    if (idx >= n_columns) return;

    int i = idx / nx;
    int j = idx % nx;

    float h = H[i * nx + j];

    // Thin-ice bypass: not a solver unknown, zero residual.
    if (h < h_thin) {
        for (int k = 0; k < nz; k++) {
            out[(i * nx + j) * nz + k] = 0.0f;
        }
        return;
    }

    float h_inv = 1.0f / h;
    float dx_inv = 1.0f / dx;
    float dsig = 1.0f / (float)(nz - 1);

    for (int k = 0; k < nz; k++) {
        int ijk = (i * nx + j) * nz + k;

        if (k == nz - 1) {
            out[ijk] = 0.0f;
            continue;
        }

        float E_k = E[ijk];
        float E_prev_k = E_prev[ijk];
        float h_prev = H_prev[i * nx + j];
        float sigma_k = k * dsig;
        float E_pmp_k = get_E_pmp(sigma_k, h);

        // --- Conservative time term: rho_i * (H*E - H_prev*E_prev) / dt ---
        TimeJacobian time_jac = get_time_jac({E_k, E_prev_k, h, h_prev, dt});
        float r = time_jac.res;

        // --- Horizontal advection: rho_i * div(H*u*E) ---
        if (term_flags & TERM_HORIZ_ADV) {
            HorizAdvectionResult h_adv = get_horiz_advection(
                E, u3d, v3d, H, E_k, i, j, k, ny, nx, nz, dx_inv, lf_c);
            r += h_adv.res;
        }

        // --- Vertical terms ---
        if (k == 0) {
            float E_pmp_bed = get_E_pmp(0.0f, h);
            float E_kp1 = get_E(E, i, j, 1, ny, nx, nz);

            if (term_flags & TERM_SIGMA_DOT) {
                float omega_k = omega[ijk];
                float omega_kp1 = omega[(i*nx+j)*nz + 1];
                BedSigmaAdvectionJacobian adv_jac = get_bed_sigma_advection_jac(
                    {E_k, E_kp1, omega_k, omega_kp1, dsig});
                r += adv_jac.res;
            }

            // Bed diffusion: operator part only (Q_geo/Q_fh are in f_E)
            BedDiffusionJacobian diff_jac = get_bed_diffusion_jac(
                {E_k, E_kp1, E_pmp_bed, get_E_pmp(dsig, h),
                 dsig, h_inv, 0.0f, 0.0f});
            r += diff_jac.res;

            // phi_strain is in f_E — not subtracted here

            if (term_flags & TERM_DRAINAGE) {
                DrainageJacobian drain_jac = get_drainage_jac(
                    {E_k, E_pmp_bed, drain_rate, h});
                r += drain_jac.res;
            }
        } else {
            float E_km1 = get_E(E, i, j, k-1, ny, nx, nz);
            float E_kp1 = (k < nz-1) ? get_E(E, i, j, k+1, ny, nx, nz)
                                      : E_surface[i*nx+j];

            float E_pmp_km1 = get_E_pmp(max(k-1, 0) * dsig, h);
            float E_pmp_kp1 = get_E_pmp(min(k+1, nz-1) * dsig, h);

            if (term_flags & TERM_SIGMA_DOT) {
                float omega_km1 = omega[(i*nx+j)*nz + k-1];
                float omega_k   = omega[ijk];
                float omega_kp1 = omega[(i*nx+j)*nz + min(k+1, nz-1)];
                SigmaAdvectionJacobian adv_jac = get_sigma_advection_jac(
                    {E_km1, E_k, E_kp1, omega_km1, omega_k, omega_kp1, dsig});
                r += adv_jac.res;
            }

            ColumnDiffusionJacobian diff_jac = get_column_diffusion_jac(
                {E_km1, E_k, E_kp1, E_pmp_km1, E_pmp_k, E_pmp_kp1,
                 dsig, h_inv});
            r += diff_jac.res;

            // phi_strain is in f_E — not subtracted here

            if (term_flags & TERM_DRAINAGE) {
                DrainageJacobian drain_jac = get_drainage_jac(
                    {E_k, E_pmp_k, drain_rate, h});
                r += drain_jac.res;
            }
        }

        out[ijk] = use_forcing ? (r - f_E[ijk]) : r;
    }
}


/* =========================================================
   Column-wise Newton smoother (Vanka-style).

   For each column (i,j), performs n_newton Newton steps:
   1. Evaluate the FULL PDE residual at every node using
      E_local (which starts as a copy of the global E)
      for vertical neighbors, and the global E for horizontal
      neighbors (frozen, as in the Vanka pattern).
   2. Extract the column-local tridiagonal Jacobian (vertical
      coupling + horizontal advection diagonal)
   3. Solve J_col * dE = -R via the Thomas algorithm
   4. Update E_local += relaxation * dE

   The horizontal advection residual is re-evaluated each
   Newton step using the current E_local value for the center
   cell and the global E for horizontal neighbors. This means
   the residual at the center cell is always the true PDE
   residual (no linearization of horizontal terms), while
   neighbor values remain frozen across Newton iterations.

   This mirrors the Vanka smoother for the momentum balance:
   local Newton iteration with full PDE residual, Jacobian
   restricted to the local block.
   ========================================================= */
extern "C" __global__
void enthalpy_column_smooth(
    float* __restrict__ delta_E,        // (ny, nx, nz) correction output
    const float* __restrict__ E,        // (ny, nx, nz) current enthalpy
    const float* __restrict__ E_prev,   // (ny, nx, nz) previous-step enthalpy
    const float* __restrict__ f_E,      // (ny, nx, nz) precomputed forcing

    const float* __restrict__ u3d,      // (nz, ny, nx+1) x-velocity
    const float* __restrict__ v3d,      // (nz, ny+1, nx) y-velocity
    const float* __restrict__ omega,    // (ny, nx, nz) omega = H*sigma_dot
    const float* __restrict__ H,        // (ny, nx) thickness
    const float* __restrict__ H_prev,   // (ny, nx) previous-step thickness
    const float* __restrict__ E_surface,// (ny, nx) surface BC
    float dx, float dt, float drain_rate, float h_thin, float lf_c,
    int term_flags,
    int ny, int nx, int nz,
    int n_newton, float relaxation)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_columns = ny * nx;
    if (idx >= n_columns) return;

    int i = idx / nx;
    int j = idx % nx;

    float h = H[i * nx + j];

    // Thin-ice bypass: drive column toward E_surface, capped at E_pmp
    if (h < h_thin) {
        float E_s = E_surface[i * nx + j];
        float dsig_thin = 1.0f / (float)(nz - 1);
        for (int k = 0; k < nz; k++) {
            float E_pmp_k = get_E_pmp(k * dsig_thin, h);
            float E_target = fminf(E_s, E_pmp_k);
            delta_E[(i * nx + j) * nz + k] = E_target - E[(i * nx + j) * nz + k];
        }
        return;
    }

    float h_inv = 1.0f / h;
    float h_prev = H_prev[i * nx + j];
    float dx_inv = 1.0f / dx;
    float dsig = 1.0f / (float)(nz - 1);
    float E_s = E_surface[i * nx + j];

    float E_local[MAX_NZ];
    for (int k = 0; k < nz; k++) {
        E_local[k] = E[(i * nx + j) * nz + k];
    }

    // --- Newton iteration ---
    for (int newton = 0; newton < n_newton; newton++) {

        float a[MAX_NZ], b[MAX_NZ], c_arr[MAX_NZ], rhs[MAX_NZ];

        for (int k = 0; k < nz; k++) {
            int ijk = (i * nx + j) * nz + k;

            if (k == nz - 1) {
                a[k] = 0.0f;
                b[k] = 1.0f;
                c_arr[k] = 0.0f;
                rhs[k] = -(E_local[k] - E_s);
                continue;
            }

            float E_k = E_local[k];
            float E_prev_k = E_prev[ijk];
            float sigma_k = k * dsig;
            float E_pmp_k = get_E_pmp(sigma_k, h);

            // --- Conservative time term + external forcing ---
            TimeJacobian time_jac = get_time_jac({E_k, E_prev_k, h, h_prev, dt});
            float r = time_jac.res - f_E[ijk];

            // --- Horizontal advection (conservative, Vanka-style) ---
            float horiz_diag = 0.0f;
            if (term_flags & TERM_HORIZ_ADV) {
                HorizAdvectionResult h_adv = get_horiz_advection(
                    E, u3d, v3d, H, E_k, i, j, k, ny, nx, nz, dx_inv, lf_c);
                r += h_adv.res;
                horiz_diag = h_adv.d_E_here;
            }

            // --- Vertical terms (from E_local) + Jacobian ---
            if (k == 0) {
                float E_pmp_bed = get_E_pmp(0.0f, h);
                float E_kp1 = E_local[1];

                BedSigmaAdvectionJacobian adv_jac = {0};
                if (term_flags & TERM_SIGMA_DOT) {
                    float omega_k   = omega[ijk];
                    float omega_kp1 = omega[(i*nx+j)*nz + 1];
                    adv_jac = get_bed_sigma_advection_jac(
                        {E_k, E_kp1, omega_k, omega_kp1, dsig});
                    r += adv_jac.res;
                }

                BedDiffusionJacobian diff_jac = get_bed_diffusion_jac(
                    {E_k, E_kp1, E_pmp_bed, get_E_pmp(dsig, h),
                     dsig, h_inv, 0.0f, 0.0f});
                r += diff_jac.res;

                DrainageJacobian drain_jac = {0};
                if (term_flags & TERM_DRAINAGE) {
                    drain_jac = get_drainage_jac(
                        {E_k, E_pmp_bed, drain_rate, h});
                    r += drain_jac.res;
                }

                a[0] = 0.0f;
                b[0] = time_jac.d_E_k
                     + diff_jac.d_E_k + adv_jac.d_E_k
                     + drain_jac.d_E_k + horiz_diag;
                c_arr[0] = diff_jac.d_E_kp1 + adv_jac.d_E_kp1;

            } else {
                float E_km1 = E_local[k-1];
                float E_kp1 = (k < nz-1) ? E_local[k+1] : E_s;

                float E_pmp_km1 = get_E_pmp(max(k-1, 0) * dsig, h);
                float E_pmp_kp1 = get_E_pmp(min(k+1, nz-1) * dsig, h);

                SigmaAdvectionJacobian adv_jac = {0};
                if (term_flags & TERM_SIGMA_DOT) {
                    float omega_km1 = omega[(i*nx+j)*nz + k-1];
                    float omega_k   = omega[ijk];
                    float omega_kp1 = omega[(i*nx+j)*nz + min(k+1, nz-1)];
                    adv_jac = get_sigma_advection_jac(
                        {E_km1, E_k, E_kp1, omega_km1, omega_k, omega_kp1, dsig});
                    r += adv_jac.res;
                }

                ColumnDiffusionJacobian diff_jac = get_column_diffusion_jac(
                    {E_km1, E_k, E_kp1, E_pmp_km1, E_pmp_k, E_pmp_kp1,
                     dsig, h_inv});
                r += diff_jac.res;

                DrainageJacobian drain_jac = {0};
                if (term_flags & TERM_DRAINAGE) {
                    drain_jac = get_drainage_jac(
                        {E_k, E_pmp_k, drain_rate, h});
                    r += drain_jac.res;
                }

                a[k] = diff_jac.d_E_km1 + adv_jac.d_E_km1;
                b[k] = time_jac.d_E_k
                     + diff_jac.d_E_k + adv_jac.d_E_k
                     + drain_jac.d_E_k + horiz_diag;
                c_arr[k] = diff_jac.d_E_kp1 + adv_jac.d_E_kp1;
            }

            rhs[k] = -r;
        }

        // --- Thomas algorithm (forward elimination) ---
        for (int k = 1; k < nz; k++) {
            if (fabsf(b[k-1]) < 1e-30f) continue;
            float w = a[k] / b[k-1];
            b[k]   -= w * c_arr[k-1];
            rhs[k] -= w * rhs[k-1];
        }

        // --- Back substitution ---
        float dE[MAX_NZ];
        dE[nz-1] = rhs[nz-1] / b[nz-1];
        for (int k = nz - 2; k >= 0; k--) {
            dE[k] = (rhs[k] - c_arr[k] * dE[k+1]) / b[k];
        }

        // --- Apply correction with relaxation ---
        for (int k = 0; k < nz; k++) {
            E_local[k] += relaxation * dE[k];
        }
    }

    // Write out the total correction
    for (int k = 0; k < nz; k++) {
        delta_E[(i * nx + j) * nz + k] = E_local[k] - E[(i * nx + j) * nz + k];
    }
}


/* =========================================================
   Layer-wise pointwise Jacobi smoother.

   One thread per node (i, j, k). Computes the full PDE
   residual and the diagonal Jacobian entry, then outputs
   the correction delta_E = -R / J_diag.

   This is NOT an exact solve — it's a single relaxation
   step. Its purpose is to reduce horizontal advection
   error cheaply between column sweeps (which resolve
   vertical coupling exactly but propagate horizontal
   information only one cell per sweep).
   ========================================================= */
extern "C" __global__
void enthalpy_layer_smooth(
    float* __restrict__ delta_E,        // (ny, nx, nz) correction output
    const float* __restrict__ E,        // (ny, nx, nz) current enthalpy
    const float* __restrict__ E_prev,   // (ny, nx, nz) previous-step enthalpy
    const float* __restrict__ f_E,      // (ny, nx, nz) precomputed forcing

    const float* __restrict__ u3d,      // (nz, ny, nx+1) x-velocity
    const float* __restrict__ v3d,      // (nz, ny+1, nx) y-velocity
    const float* __restrict__ omega,    // (ny, nx, nz) omega = H*sigma_dot
    const float* __restrict__ H,        // (ny, nx) thickness
    const float* __restrict__ H_prev,   // (ny, nx) previous-step thickness
    const float* __restrict__ E_surface,// (ny, nx) surface BC
    float dx, float dt, float drain_rate, float h_thin, float lf_c,
    int term_flags,
    int ny, int nx, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny * nx * nz;
    if (idx >= total) return;

    int ij = idx / nz;
    int k  = idx % nz;
    int i  = ij / nx;
    int j  = ij % nx;
    int ijk = idx;

    // Surface Dirichlet: no correction
    if (k == nz - 1) {
        delta_E[ijk] = 0.0f;
        return;
    }

    float h = H[i * nx + j];

    // Thin-ice bypass
    if (h < h_thin) {
        delta_E[ijk] = 0.0f;
        return;
    }

    float h_inv = 1.0f / h;
    float dx_inv = 1.0f / dx;
    float dsig = 1.0f / (float)(nz - 1);

    float E_k = E[ijk];
    float E_prev_k = E_prev[ijk];
    float h_prev = H_prev[i * nx + j];
    float sigma_k = k * dsig;
    float E_pmp_k = get_E_pmp(sigma_k, h);

    // --- Conservative time term + external forcing ---
    TimeJacobian time_jac = get_time_jac({E_k, E_prev_k, h, h_prev, dt});
    float r = time_jac.res - f_E[ijk];
    float diag = time_jac.d_E_k;

    // Horizontal advection
    float horiz_diag = 0.0f;
    if (term_flags & TERM_HORIZ_ADV) {
        HorizAdvectionResult h_adv = get_horiz_advection(
            E, u3d, v3d, H, E_k, i, j, k, ny, nx, nz, dx_inv, lf_c);
        r += h_adv.res;
        horiz_diag = h_adv.d_E_here;
    }
    diag += horiz_diag;

    // Vertical terms
    if (k == 0) {
        float E_pmp_bed = get_E_pmp(0.0f, h);
        float E_kp1 = get_E(E, i, j, 1, ny, nx, nz);

        BedSigmaAdvectionJacobian adv_jac = {0};
        if (term_flags & TERM_SIGMA_DOT) {
            float omega_k = omega[ijk];
            float omega_kp1 = omega[(i*nx+j)*nz + 1];
            adv_jac = get_bed_sigma_advection_jac(
                {E_k, E_kp1, omega_k, omega_kp1, dsig});
            r += adv_jac.res;
        }

        BedDiffusionJacobian diff_jac = get_bed_diffusion_jac(
            {E_k, E_kp1, E_pmp_bed, get_E_pmp(dsig, h),
             dsig, h_inv, 0.0f, 0.0f});
        r += diff_jac.res;

        DrainageJacobian drain_jac = {0};
        if (term_flags & TERM_DRAINAGE) {
            drain_jac = get_drainage_jac({E_k, E_pmp_bed, drain_rate, h});
            r += drain_jac.res;
        }

        diag += diff_jac.d_E_k + adv_jac.d_E_k + drain_jac.d_E_k;

    } else {
        float E_km1 = get_E(E, i, j, k-1, ny, nx, nz);
        float E_kp1 = (k < nz-1) ? get_E(E, i, j, k+1, ny, nx, nz)
                                  : E_surface[i*nx+j];

        float E_pmp_km1 = get_E_pmp(max(k-1, 0) * dsig, h);
        float E_pmp_kp1 = get_E_pmp(min(k+1, nz-1) * dsig, h);

        SigmaAdvectionJacobian adv_jac = {0};
        if (term_flags & TERM_SIGMA_DOT) {
            float omega_km1 = omega[(i*nx+j)*nz + k-1];
            float omega_k   = omega[ijk];
            float omega_kp1 = omega[(i*nx+j)*nz + min(k+1, nz-1)];
            adv_jac = get_sigma_advection_jac(
                {E_km1, E_k, E_kp1, omega_km1, omega_k, omega_kp1, dsig});
            r += adv_jac.res;
        }

        ColumnDiffusionJacobian diff_jac = get_column_diffusion_jac(
            {E_km1, E_k, E_kp1, E_pmp_km1, E_pmp_k, E_pmp_kp1,
             dsig, h_inv});
        r += diff_jac.res;

        DrainageJacobian drain_jac = {0};
        if (term_flags & TERM_DRAINAGE) {
            drain_jac = get_drainage_jac({E_k, E_pmp_k, drain_rate, h});
            r += drain_jac.res;
        }

        diag += diff_jac.d_E_k + adv_jac.d_E_k + drain_jac.d_E_k;
    }

    // Pointwise Jacobi correction
    delta_E[ijk] = (fabsf(diag) > 1e-30f) ? (-r / diag) : 0.0f;
}


/* =========================================================
   Compute omega = H * sigma_dot from the sigma-space
   continuity equation by vertical integration.

   The continuity equation in sigma coordinates is:
     dH/dt + d(Hu)/dx + d(Hv)/dy + d(omega)/dsigma = 0

   Rearranging: d(omega)/dsigma = -(dH/dt + div(Hu))

   We integrate upward from the bed BC (omega_b = -bmb):
     omega(k) = omega(k-1) - dsig * (dH/dt + div_Hu(k-1/2))

   The mass flux stencil H_face = 0.5*(H_l+H_r)*u_face matches
   the horizontal enthalpy flux, ensuring consistency with the
   conservative enthalpy equation.

   At the surface, omega automatically equals -SMB by construction.
   ========================================================= */
extern "C" __global__
void compute_omega(
    float* __restrict__ omega,      // (ny, nx, nz) output
    const float* __restrict__ u3d,  // (nz, ny, nx+1) layer x-velocity
    const float* __restrict__ v3d,  // (nz, ny+1, nx) layer y-velocity
    const float* __restrict__ H,    // (ny, nx) ice thickness
    const float* __restrict__ dh_dt_in, // (ny, nx) actual dH/dt (m/s)
    const float* __restrict__ bmb,  // (ny, nx) basal mass balance (m/s, typically 0)
    float dx,
    int ny, int nx, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_columns = ny * nx;
    if (idx >= n_columns) return;

    int i = idx / nx;
    int j = idx % nx;

    float h_here = H[i * nx + j];
    if (h_here < 1e-3f) {
        for (int k = 0; k < nz; k++) omega[(i*nx+j)*nz + k] = 0.0f;
        return;
    }

    float dx_inv = 1.0f / dx;
    float dsig = 1.0f / (float)(nz - 1);

    // Neighbor thicknesses for mass flux computation
    float H_xm = get_cell(H, i, j-1, ny, nx);
    float H_xp = get_cell(H, i, j+1, ny, nx);
    float H_ym = get_cell(H, i-1, j, ny, nx);
    float H_yp = get_cell(H, i+1, j, ny, nx);

    // 1. Compute layer-wise div(Hu) using the same LF mass flux as the
    //    momentum solver (flux.cu). The velocity is passed in m/yr (the
    //    momentum solver's native units) so that the float32 evaluation
    //    of sqrt(u^2 + 10) matches the momentum solver exactly, avoiding
    //    scale-dependent rounding differences that would corrupt omega
    //    through catastrophic cancellation in dH/dt + div(Hu).
    float div_Hu[MAX_NZ];

    for (int k = 0; k < nz; k++) {
        float u_l = get_u3d(u3d, k, i, j,   nz, ny, nx);
        float u_r = get_u3d(u3d, k, i, j+1, nz, ny, nx);
        float v_t = get_v3d(v3d, k, i,   j, nz, ny, nx);
        float v_b = get_v3d(v3d, k, i+1, j, nz, ny, nx);

        // x-direction LF mass flux (velocity in m/yr, reg constant in (m/yr)^2)
        float flux_r = 0.0f, flux_l = 0.0f;
        if (j < nx - 1) {
            float H_avg_r = 0.5f * (h_here + H_xp);
            float u_mag_r = sqrtf(u_r * u_r + MASS_FLUX_REG_YR);
            flux_r = H_avg_r * u_r - 0.5f * u_mag_r * (H_xp - h_here);
        }
        if (j > 0) {
            float H_avg_l = 0.5f * (H_xm + h_here);
            float u_mag_l = sqrtf(u_l * u_l + MASS_FLUX_REG_YR);
            flux_l = H_avg_l * u_l - 0.5f * u_mag_l * (h_here - H_xm);
        }

        // y-direction LF mass flux.
        // Convention matches flux.cu: dissipation is (H_top - H_bottom)
        // where top = smaller i, bottom = larger i.
        float flux_b = 0.0f, flux_t = 0.0f;
        if (i < ny - 1) {
            float H_avg_b = 0.5f * (h_here + H_yp);
            float v_mag_b = sqrtf(v_b * v_b + MASS_FLUX_REG_YR);
            flux_b = H_avg_b * v_b - 0.5f * v_mag_b * (h_here - H_yp);
        }
        if (i > 0) {
            float H_avg_t = 0.5f * (H_ym + h_here);
            float v_mag_t = sqrtf(v_t * v_t + MASS_FLUX_REG_YR);
            flux_t = H_avg_t * v_t - 0.5f * v_mag_t * (H_ym - h_here);
        }

        // Divergence convention: matches the momentum solver's continuity
        // residual which uses (j_r - j_l + j_t - j_b)/dx.
        div_Hu[k] = (flux_r - flux_l + flux_t - flux_b) * dx_inv;
    }

    // 2. Use the actual dH/dt passed from the caller.
    //    This is (H_new - H_prev)/dt from the momentum step, ensuring
    //    exact consistency: the omega field cancels the time + advection
    //    residual for uniform enthalpy.
    float dh_dt = dh_dt_in[i * nx + j];
    float bmb_val = bmb[i * nx + j];

    // 3. Integrate upward from bed BC: omega(0) = -bmb
    float current_omega = -bmb_val;
    omega[(i*nx+j)*nz + 0] = current_omega;

    for (int k = 1; k < nz; k++) {
        // Average divergence at half-node (k-1/2)
        float div_half = 0.5f * (div_Hu[k-1] + div_Hu[k]);
        current_omega = current_omega - dsig * (dh_dt + div_half);
        omega[(i*nx+j)*nz + k] = current_omega;
    }
}