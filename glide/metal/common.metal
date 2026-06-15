// =====================================================================
// COMMON UTILITIES (Metal port of common.cu)
//   DualFloat forward-mode autodiff, clamped array accessors, helpers.
//
// Port notes vs the CUDA original:
//   * Device-callable free functions must qualify pointer params with the
//     `device` address space.
//   * DualFloat operators are free `inline` functions (members are public).
//   * `__fmaf_rn`->fma, `__powf`->pow/`dpow`, `__expf`->exp, `fminf/fmaxf`->
//     fmin/fmax.  The DualFloat `__powf` overload is renamed `dpow` to avoid
//     colliding with the builtin `pow`.
//   * `atomic_add_float` emulates float atomic-add via a CAS loop on
//     atomic_uint (native float atomics are unavailable in this toolchain).
//     The output buffer is a plain float32 array bound as atomic_uint.
// =====================================================================
#include <metal_stdlib>
using namespace metal;

struct DualFloat {
    float v; // Primal value
    float d; // Derivative/Perturbation component
};

// Addition: (u + v, du + dv)
inline DualFloat operator+(DualFloat a, DualFloat b) {
    return {a.v + b.v, a.d + b.d};
}

// Subtraction: (u - v, du - dv)
inline DualFloat operator-(DualFloat a, DualFloat b) {
    return {a.v - b.v, a.d - b.d};
}

// Multiplication: (u * v, u * dv + v * du)
inline DualFloat operator*(DualFloat a, DualFloat b) {
    return {a.v * b.v, fma(a.v, b.d, a.d * b.v)};
}

// Multiplication by Scalar: (u * s, du * s)
inline DualFloat operator*(DualFloat a, float s) {
    return {a.v * s, a.d * s};
}

inline DualFloat operator*(float s, DualFloat a) {
    return {a.v * s, a.d * s};
}

// Addition with Scalar: (u + s, du)
inline DualFloat operator+(DualFloat a, float s) {
    return {a.v + s, a.d};
}

// Commutative version: (s + u, du)
inline DualFloat operator+(float s, DualFloat a) {
    return {s + a.v, a.d};
}

// Subtraction with Scalar
inline DualFloat operator-(DualFloat a, float s) {
    return {a.v - s, a.d};
}

// Division by Scalar: (u / s, du / s)
inline DualFloat operator/(DualFloat a, float s) {
    float inv_s = 1.0f / s;
    return {a.v * inv_s, a.d * inv_s};
}

inline DualFloat dpow(DualFloat u, float p) {
    float val = pow(u.v, p);

    // d/dx(u^p) = p * u^(p-1) * du
    float deriv = p * pow(u.v, p - 1.0f) * u.d;

    return {val, deriv};
}

inline float sigmoid(const float z, const float c) {
   float scaled_z = fmin(fmax(c*z,-20.0f),20.0f);
   return 1.0f/(1.0f + exp(-scaled_z));
}

// Derivative of sigmoid w.r.t. z: d(sigmoid)/dz = c * sigmoid * (1 - sigmoid)
inline float sigmoid_deriv(const float z, const float c) {
   float s = sigmoid(z, c);
   return c * s * (1.0f - s);
}

inline float get_grounded(const float H, const float depth, const float sigmoid_c, const float sigmoid_k)
{
   float z = 0.917f*H - depth + sigmoid_k/sigmoid_c;
   return sigmoid(z,sigmoid_c);
}

inline float get_vfacet(device const float* u, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx),0);
    return u[i * (nx + 1) + j];
}

inline DualFloat get_vfacet(device const float* u, device const float* du, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx),0);
    int idx = i * (nx + 1) + j;
    return {u[idx],du[idx]};
}

inline float get_hfacet(device const float* v, int i, int j, int ny, int nx) {
    i = max(min(i,ny),0);
    j = max(min(j,nx - 1),0);
    return v[i * nx + j];
}

inline DualFloat get_hfacet(device const float* v, device const float* dv, int i, int j, int ny, int nx) {
    i = max(min(i,ny),0);
    j = max(min(j,nx - 1),0);
    int idx = i * nx + j;
    return {v[idx],dv[idx]};
}

inline float get_cell(device const float* arr, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx - 1),0);
    return arr[i * nx + j];
}

inline DualFloat get_cell(device const float* arr, device const float* darr, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx - 1),0);
    int idx = i * nx + j;
    return {arr[idx],darr[idx]};
}

inline float get_masked_cell(device const float* arr, device const float* mask, int i, int j, int ny, int nx) {
    i = max(min(i,ny - 1),0);
    j = max(min(j,nx - 1),0);
    int idx = i * nx + j;
    return arr[idx]*(1.0f - mask[idx]);
}

// Float atomic-add via compare-and-swap on the underlying uint bits.
// `addr` aliases a float32 buffer element bound as atomic_uint.
inline void atomic_add_float(device atomic_uint* addr, float val) {
    uint old = atomic_load_explicit(addr, memory_order_relaxed);
    uint des;
    do {
        des = as_type<uint>(as_type<float>(old) + val);
    } while (!atomic_compare_exchange_weak_explicit(
                 addr, &old, des, memory_order_relaxed, memory_order_relaxed));
}
