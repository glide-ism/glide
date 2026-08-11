struct SigmaVertXZStencil {
    float u_c;
    float eta_l, eta_r;
    float H_l, H_r;
};

struct SigmaVertXZStencilDual {
    DualFloat u_c;
    DualFloat eta_l, eta_r;
    DualFloat H_l, H_r;

    __device__ __forceinline__
    SigmaVertXZStencil get_primals() const {
        return {u_c.v, eta_l.v,eta_r.v,H_l.v,H_r.v};
    }
    
    __device__ __forceinline__
    SigmaVertXZStencil get_diffs() const {
        return {u_c.d,eta_l.d,eta_r.d,H_l.d,H_r.d};
    }
};

struct SigmaVertXZJacobian {
    float res;
    float d_u_c;
    float d_eta_l, d_eta_r;
    float d_H_l, d_H_r;
    
    __device__ __forceinline__
    float apply_jvp(const SigmaVertXZStencil& dot) const {
        return d_u_c * dot.u_c +
	       d_eta_l * dot.eta_l +
	       d_eta_r * dot.eta_r +
	       d_H_l * dot.H_l +
	       d_H_r * dot.H_r;
    }
};

__device__ __forceinline__
SigmaVertXZJacobian get_sigma_xz_jac(
    SigmaVertXZStencil s,
    float c_1, float S_1, float H_reg,
    int i, int j,  // Defined on cells - the i,j for the cell
    int ny, int nx) {

    SigmaVertXZJacobian jac = {0};

    // 2*K_2 = 2*(c_1*S_1/4): the leading 2 matches the J = 2*eta*H*E2
    // convention used by the membrane stresses (cf. get_sigma_xx_jac).
    // eta/H is regularized to eta*H/(H^2 + H_reg^2), consistent with the
    // shear invariant's 1/(H^2 + H_reg^2) in populate_viscosity.
    float factr = -1.0f * c_1 * S_1 / 2.0f;
    float den_l = s.H_l * s.H_l + H_reg * H_reg;
    float den_r = s.H_r * s.H_r + H_reg * H_reg;
    float shear = s.eta_l * s.H_l / den_l + s.eta_r * s.H_r / den_r;

    jac.res = factr * shear * s.u_c;
    jac.d_u_c = factr * shear;
    jac.d_eta_l = factr * s.H_l / den_l * s.u_c;
    jac.d_eta_r = factr * s.H_r / den_r * s.u_c;
    jac.d_H_l = factr * s.eta_l * (H_reg * H_reg - s.H_l * s.H_l) / (den_l * den_l) * s.u_c;
    jac.d_H_r = factr * s.eta_r * (H_reg * H_reg - s.H_r * s.H_r) / (den_r * den_r) * s.u_c;

    return jac;
}

struct SigmaVertYZStencil {
    float v_c;
    float eta_t, eta_b;
    float H_t, H_b;
};

struct SigmaVertYZStencilDual {
    DualFloat v_c;
    DualFloat eta_t, eta_b;
    DualFloat H_t, H_b;

    __device__ __forceinline__
    SigmaVertYZStencil get_primals() const {
        return {v_c.v, eta_t.v,eta_b.v,H_t.v,H_b.v};
    }
    
    __device__ __forceinline__
    SigmaVertYZStencil get_diffs() const {
        return {v_c.d,eta_t.d,eta_b.d,H_t.d,H_b.d};
    }
};

struct SigmaVertYZJacobian {
    float res;
    float d_v_c;
    float d_eta_t, d_eta_b;
    float d_H_t, d_H_b;
    
    __device__ __forceinline__
    float apply_jvp(const SigmaVertYZStencil& dot) const {
        return d_v_c * dot.v_c +
	       d_eta_t * dot.eta_t +
	       d_eta_b * dot.eta_b +
	       d_H_t * dot.H_t +
	       d_H_b * dot.H_b;
    }
};

__device__ __forceinline__
SigmaVertYZJacobian get_sigma_yz_jac(
    SigmaVertYZStencil s,
    float c_1, float S_1, float H_reg,
    int i, int j,  // Defined on cells - the i,j for the cell
    int ny, int nx) {

    SigmaVertYZJacobian jac = {0};

    // 2*K_2, matching the J = 2*eta*H*E2 convention (cf. get_sigma_xx_jac),
    // with eta/H regularized as in get_sigma_xz_jac
    float factr = -1.0f * c_1 * S_1 / 2.0f;
    float den_t = s.H_t * s.H_t + H_reg * H_reg;
    float den_b = s.H_b * s.H_b + H_reg * H_reg;
    float shear = s.eta_t * s.H_t / den_t + s.eta_b * s.H_b / den_b;

    jac.res = factr * shear * s.v_c;
    jac.d_v_c = factr * shear;
    jac.d_eta_t = factr * s.H_t / den_t * s.v_c;
    jac.d_eta_b = factr * s.H_b / den_b * s.v_c;
    jac.d_H_t = factr * s.eta_t * (H_reg * H_reg - s.H_t * s.H_t) / (den_t * den_t) * s.v_c;
    jac.d_H_b = factr * s.eta_b * (H_reg * H_reg - s.H_b * s.H_b) / (den_b * den_b) * s.v_c;

    return jac;
}

__device__ __forceinline__
DualFloat get_sigma_xz_dual(
    SigmaVertXZStencilDual s,
    float c_1, float S_1, float H_reg,
    int i, int j,
    int ny, int nx) {
    SigmaVertXZJacobian jac = get_sigma_xz_jac(s.get_primals(),c_1,S_1,H_reg,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

__device__ __forceinline__
DualFloat get_sigma_yz_dual(
    SigmaVertYZStencilDual s,
    float c_1, float S_1, float H_reg,
    int i, int j,
    int ny, int nx) {
    SigmaVertYZJacobian jac = get_sigma_yz_jac(s.get_primals(),c_1,S_1,H_reg,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

// Newton correction for the diagonal of the vertical shear block: the
// shear-softening part of d(eta)/d(u_d), which is local to the facet's two
// adjacent cells (the shear invariant depends only on the cell's own
// facet values). In the shear-dominated limit this scales the frozen-eta
// diagonal by 1/n, curing the systematic Picard overshoot. Membrane
// contributions to d(eta)/d(u) remain deliberately ignored.
// E2 is recovered from eta = B/2 * E2^((1-n)/(2n)).
__device__ __forceinline__
float get_sigma_vert_dvisc(
    float u_c,
    float eta_1, float eta_2,
    float H_1, float H_2,
    float B_1, float B_2,
    float c_1, float S_1,
    float n, float H_reg) {

    float factr = -1.0f * c_1 * S_1 / 2.0f;
    float K_2 = c_1 * S_1 / 4.0f;
    float glen_exp = (1.0f - n) / (2.0f * n);
    float pw = 2.0f * n / (1.0f - n);

    float den_1 = H_1 * H_1 + H_reg * H_reg;
    float den_2 = H_2 * H_2 + H_reg * H_reg;

    float e2_1 = __powf(2.0f * eta_1 / B_1, pw);
    float e2_2 = __powf(2.0f * eta_2 / B_2, pw);

    return factr * glen_exp * K_2 * u_c * u_c *
           (eta_1 * H_1 / (den_1 * den_1 * e2_1) +
            eta_2 * H_2 / (den_2 * den_2 * e2_2));
}

/*=======================================================
  ================== Normal Stress ======================
 ========================================================*/
// Stencil items that require differentiation
struct SigmaNormalStencil {
    float u_l, u_r, v_t, v_b;
    float eta_H;
};

struct SigmaNormalStencilDual {
    DualFloat u_l, u_r, v_t, v_b;
    DualFloat eta_H;

    __device__ __forceinline__
    SigmaNormalStencil get_primals() const {
        return {u_l.v,u_r.v,v_t.v,v_b.v,eta_H.v};
    }

    __device__ __forceinline__
    SigmaNormalStencil get_diffs() const {
        return {u_l.d,u_r.d,v_t.d,v_b.d,eta_H.d};
    }
};

// Return type for sigma_xx,
// containing residual and jacobian row
struct SigmaNormalJacobian {
    float res;
    float d_u_l, d_u_r, d_v_t, d_v_b;
    float d_eta_H;

    __device__ __forceinline__
    float apply_jvp(const SigmaNormalStencil& dot) const {
        return d_u_l * dot.u_l +
	       d_u_r * dot.u_r +
	       d_v_t * dot.v_t +
	       d_v_b * dot.v_b +
	       d_eta_H * dot.eta_H;
    }

};

__device__ __forceinline__
SigmaNormalJacobian get_sigma_xx_jac(
    SigmaNormalStencil s,
    float dx_inv,
    int i, int j,  // Defined on cells - the i,j for the cell
    int ny, int nx) {

    SigmaNormalJacobian jac= {0};

    if (j < 0 || j >= nx) {
	return jac;
    }

    float eps_xx = (2.0f*(s.u_r - s.u_l)*dx_inv + (s.v_t - s.v_b)*dx_inv);
    float jac_prefactor = 2.0f * s.eta_H * dx_inv;

    jac.res = 2.0f * s.eta_H * eps_xx;
    jac.d_u_l = -2.0f * jac_prefactor;
    jac.d_u_r =  2.0f * jac_prefactor;
    jac.d_v_t =  jac_prefactor;
    jac.d_v_b = -jac_prefactor;
    jac.d_eta_H = 2.0f * eps_xx;
    return jac;
}

__device__ __forceinline__
DualFloat get_sigma_xx_dual(
    SigmaNormalStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    SigmaNormalJacobian jac = get_sigma_xx_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

__device__ __forceinline__
SigmaNormalJacobian get_sigma_yy_jac(
    SigmaNormalStencil s,
    float dx_inv,
    int i, int j,  // Defined on cells - the i,j for the cell
    int ny, int nx) {

    SigmaNormalJacobian jac= {0};

    // No normal stress on out-of-domain cells
    if (i < 0 || i >= ny) {
	return jac;
    }

    float eps_yy = ((s.u_r - s.u_l)*dx_inv + 2.0f*(s.v_t - s.v_b)*dx_inv);
    float jac_prefactor = 2.0f * s.eta_H * dx_inv;

    jac.res = 2.0f * s.eta_H * eps_yy;
    jac.d_u_l = -jac_prefactor;
    jac.d_u_r =  jac_prefactor;
    jac.d_v_t =  2.0f*jac_prefactor;
    jac.d_v_b = -2.0f*jac_prefactor;
    jac.d_eta_H = 2.0f * eps_yy;
    return jac;
}

__device__ __forceinline__
DualFloat get_sigma_yy_dual(
    SigmaNormalStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    SigmaNormalJacobian jac = get_sigma_yy_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*======================================================
  ==================== Shear Stress ====================
  ======================================================*/

// Stencil items that require differentiation
struct SigmaShearStencil {
    float u_t, u_b, v_l, v_r;
    float eta_H;
};

struct SigmaShearStencilDual {
    DualFloat u_t, u_b, v_l, v_r;
    DualFloat eta_H;

    __device__ __forceinline__
    SigmaShearStencil get_primals() const {
        return {u_t.v,u_b.v,v_l.v,v_r.v,eta_H.v};
    }

    __device__ __forceinline__
    SigmaShearStencil get_diffs() const {
        return {u_t.d,u_b.d,v_l.d,v_r.d,eta_H.d};
    }

};

// Return type for sigma_xx,
// containing residual and jacobian row
struct SigmaShearJacobian {
    float res;
    float d_u_t, d_u_b, d_v_l, d_v_r;
    float d_eta_H;

    __device__ __forceinline__
    float apply_jvp(const SigmaShearStencil& dot) const {
        return d_u_t * dot.u_t +
	       d_u_b * dot.u_b +
	       d_v_l * dot.v_l +
	       d_v_r * dot.v_r +
	       d_eta_H * dot.eta_H;
    }

};


__device__ __forceinline__
SigmaShearJacobian get_sigma_xy_jac(
    SigmaShearStencil s,
    float dx_inv,
    int i, int j, // defined on vertices, the i,j for the vertex
    int ny, int nx) {

    SigmaShearJacobian jac = {0};
    // No shear on boundary vertices
    if (i <= 0 || i >= ny || j <= 0 || j >= nx) {
        return jac;
    }

    float eps_xy = 0.5f*((s.u_t - s.u_b)*dx_inv + (s.v_r - s.v_l)*dx_inv);
    float jac_prefactor = s.eta_H * dx_inv;

    jac.res = 2.0f * s.eta_H * eps_xy;
    jac.d_u_t = jac_prefactor;
    jac.d_u_b = -jac_prefactor;
    jac.d_v_l = -jac_prefactor;
    jac.d_v_r = jac_prefactor;
    jac.d_eta_H = 2.0f * eps_xy;
    return jac;

}

__device__ __forceinline__
DualFloat get_sigma_xy_dual(
    SigmaShearStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    SigmaShearJacobian jac = get_sigma_xy_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*=========================================================
  ================== Basal Shear Stress v2 ===================
  =========================================================*/

struct TauBxStencil {
    float u_c, u_l, u_r;
    float v_tl, v_tr, v_bl, v_br;
    float H_l, H_r;
    float xi_l, xi_r;
    float beta_l, beta_r;
    float m;
    float u_reg;
    float water_drag;
    float flotation_reg_sliding;
};

struct TauBxStencilDual {
    DualFloat u_c, u_l, u_r;
    DualFloat v_tl, v_tr, v_bl, v_br;
    DualFloat H_l, H_r;
    float xi_l, xi_r;
    float beta_l, beta_r;
    float m;
    float u_reg;
    float water_drag;
    float flotation_reg_sliding;

    __device__ __forceinline__
    TauBxStencil get_primals() const {
        return {u_c.v,u_l.v,u_r.v,v_tl.v,v_tr.v,v_bl.v,v_br.v,H_l.v,H_r.v,xi_l,xi_r,beta_l,beta_r,m,u_reg,water_drag,flotation_reg_sliding};
    }

    __device__ __forceinline__
    TauBxStencil get_diffs() const {
        return {u_c.d,u_l.d,u_r.d,v_tl.d,v_tr.d,v_bl.d,v_br.d,H_l.d,H_r.d,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    }

};

struct TauBxJacobian {
    float res;
    float d_u_c, d_u_l, d_u_r;
    float d_v_tl,d_v_tr,d_v_bl,d_v_br;
    float d_H_l, d_H_r;
    float d_beta_l, d_beta_r;

    __device__ __forceinline__
    float apply_jvp(const TauBxStencil& dot) const {
        return d_u_c * dot.u_c +
               d_u_l * dot.u_l +
               d_u_r * dot.u_r +
	       d_v_tl * dot.v_tl +
	       d_v_tr * dot.v_tr +
	       d_v_bl * dot.v_bl +
	       d_v_br * dot.v_br +
	       d_H_l * dot.H_l +
	       d_H_r * dot.H_r;
    }
};

__device__ __forceinline__
TauBxJacobian get_tau_bx_jac(
   TauBxStencil s )
{
    TauBxJacobian jac = {0};

    float xi_l = powf(s.xi_l,1.0f);
    float xi_r = powf(s.xi_r,1.0f);

    float beta_eff_l = s.beta_l * xi_l;
    float beta_eff_r = s.beta_r * xi_r;

    float unorm_sq_l = 0.5f * (s.u_l * s.u_l + s.u_c * s.u_c + s.v_tl * s.v_tl + s.v_bl * s.v_bl);
    float unorm_sq_r = 0.5f * (s.u_c * s.u_c + s.u_r * s.u_r + s.v_tr * s.v_tr + s.v_br * s.v_br);

    float unorm_sq_pow_l = __powf(unorm_sq_l + s.u_reg, (s.m - 1.0f)/2.0f);
    float unorm_sq_pow_r = __powf(unorm_sq_r + s.u_reg, (s.m - 1.0f)/2.0f);

    float unorm_sq_deriv_l = (s.m - 1.0f)/2.0f * __powf(unorm_sq_l + s.u_reg, (s.m - 1.0f)/2.0f - 1.0f);
    float unorm_sq_deriv_r = (s.m - 1.0f)/2.0f * __powf(unorm_sq_r + s.u_reg, (s.m - 1.0f)/2.0f - 1.0f);
    
    float coeff = 0.5f * (beta_eff_l * unorm_sq_pow_l + beta_eff_r * unorm_sq_pow_r) + s.water_drag;

    jac.res = - coeff * s.u_c;
    jac.d_u_c = -0.5f * (beta_eff_l * unorm_sq_deriv_l + beta_eff_r * unorm_sq_deriv_r) * s.u_c * s.u_c - coeff;
    jac.d_u_l = -0.5f * beta_eff_l * unorm_sq_deriv_l * s.u_c * s.u_l;
    jac.d_u_r = -0.5f * beta_eff_r * unorm_sq_deriv_r * s.u_c * s.u_r;
    jac.d_v_tl = -0.5f * beta_eff_l * unorm_sq_deriv_l * s.u_c * s.v_tl;
    jac.d_v_bl = -0.5f * beta_eff_l * unorm_sq_deriv_l * s.u_c * s.v_bl;
    jac.d_v_tr = -0.5f * beta_eff_r * unorm_sq_deriv_r * s.u_c * s.v_tr;
    jac.d_v_br = -0.5f * beta_eff_r * unorm_sq_deriv_r * s.u_c * s.v_br;
    jac.d_beta_l = -0.5f * xi_l * unorm_sq_pow_l * s.u_c;
    jac.d_beta_r = -0.5f * xi_r * unorm_sq_pow_r * s.u_c;
    return jac;
}

__device__ __forceinline__
DualFloat get_tau_bx_dual(TauBxStencilDual s) {
    TauBxJacobian jac = get_tau_bx_jac(s.get_primals());
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

struct TauByStencil {
    float v_c, v_t, v_b;
    float u_tl, u_tr, u_bl, u_br;
    float H_t, H_b;
    float xi_t, xi_b;
    float beta_t, beta_b;
    float m;
    float u_reg;
    float water_drag;
    float flotation_reg_sliding;
};

struct TauByStencilDual {
    DualFloat v_c, v_t, v_b;
    DualFloat u_tl, u_tr, u_bl, u_br;
    DualFloat H_t, H_b;
    float xi_t, xi_b;
    float beta_t, beta_b;
    float m;
    float u_reg;
    float water_drag;
    float flotation_reg_sliding;

    __device__ __forceinline__
    TauByStencil get_primals() const {
        return {v_c.v, v_t.v, v_b.v ,u_tl.v,u_tr.v,u_bl.v,u_br.v,H_t.v,H_b.v,xi_t,xi_b,beta_t,beta_b,m,u_reg,water_drag,flotation_reg_sliding};
    }

    __device__ __forceinline__
    TauByStencil get_diffs() const {
        return {v_c.d, v_t.d, v_b.d, u_tl.d,u_tr.d,u_bl.d,u_br.d,H_t.d,H_t.d,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    }

};

struct TauByJacobian {
    float res;
    float d_v_c, d_v_t, d_v_b;
    float d_u_tl,d_u_tr,d_u_bl,d_u_br;
    float d_H_t, d_H_b;
    float d_beta_t, d_beta_b;

    __device__ __forceinline__
    float apply_jvp(const TauByStencil& dot) const {
        return d_v_c * dot.v_c +
               d_v_t * dot.v_t +
               d_v_b * dot.v_b +
	       d_u_tl * dot.u_tl +
	       d_u_tr * dot.u_tr +
	       d_u_bl * dot.u_bl +
	       d_u_br * dot.u_br +
	       d_H_t * dot.H_t +
	       d_H_b * dot.H_b;
    }
};

__device__ __forceinline__
TauByJacobian get_tau_by_jac(
   TauByStencil s )
{
    TauByJacobian jac = {0};

    float xi_t = powf(s.xi_t,1.0f);
    float xi_b = powf(s.xi_b,1.0f);

    float beta_eff_t = s.beta_t * xi_t;
    float beta_eff_b = s.beta_b * xi_b;
    
    float unorm_sq_t = 0.5f * (s.v_t * s.v_t + s.v_c * s.v_c + s.u_tl * s.u_tl + s.u_tr * s.u_tr); 
    float unorm_sq_b = 0.5f * (s.v_c * s.v_c + s.v_b * s.v_b + s.u_bl * s.u_bl + s.u_br * s.u_br); 

    float unorm_sq_pow_t = __powf(unorm_sq_t + s.u_reg,(s.m - 1.0f)/2.0f);
    float unorm_sq_pow_b = __powf(unorm_sq_b + s.u_reg,(s.m - 1.0f)/2.0f);

    float unorm_sq_deriv_t = (s.m - 1.0f)/2.0f * __powf(unorm_sq_t + s.u_reg,(s.m - 1.0f)/2.0f - 1.0f);
    float unorm_sq_deriv_b = (s.m - 1.0f)/2.0f * __powf(unorm_sq_b + s.u_reg,(s.m - 1.0f)/2.0f - 1.0f);

    float coeff = 0.5f * (beta_eff_t * unorm_sq_pow_t + beta_eff_b * unorm_sq_pow_b) + s.water_drag;

    jac.res = - coeff * s.v_c;
    jac.d_v_c = -0.5f * (beta_eff_t * unorm_sq_deriv_t + beta_eff_b * unorm_sq_deriv_b) * s.v_c * s.v_c - coeff;
    jac.d_v_t = -0.5f * beta_eff_t * unorm_sq_deriv_t * s.v_c * s.v_t;
    jac.d_v_b = -0.5f * beta_eff_b * unorm_sq_deriv_b * s.v_c * s.v_b;
    jac.d_u_tl = -0.5f * beta_eff_t * unorm_sq_deriv_t * s.v_c * s.u_tl;
    jac.d_u_tr = -0.5f * beta_eff_t * unorm_sq_deriv_t * s.v_c * s.u_tr;
    jac.d_u_bl = -0.5f * beta_eff_b * unorm_sq_deriv_b * s.v_c * s.u_bl;
    jac.d_u_br = -0.5f * beta_eff_b * unorm_sq_deriv_b * s.v_c * s.u_br;
    jac.d_beta_t = -0.5f * xi_t * unorm_sq_pow_t * s.v_c;
    jac.d_beta_b = -0.5f * xi_b * unorm_sq_pow_b * s.v_c;
    return jac;
}

__device__ __forceinline__
DualFloat get_tau_by_dual(TauByStencilDual s) {
    TauByJacobian jac = get_tau_by_jac(s.get_primals());
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*=========================================================
  ==================== Driving Stress =====================
  =========================================================*/

struct TauDxStencil {
    float H_l, H_r;
    float bed_l, bed_r;
    float phi_l, phi_r;
    float sigmoid_c;
};

struct TauDxStencilDual {
    DualFloat H_l, H_r;
    float bed_l, bed_r;
    float phi_l, phi_r;
    float sigmoid_c;

    __device__ __forceinline__
    TauDxStencil get_primals() const {
        return {H_l.v,H_r.v,bed_l,bed_r,phi_l,phi_r,sigmoid_c};
    }

    __device__ __forceinline__
    TauDxStencil get_diffs() const {
        return {H_l.d,H_r.d,0.0f,0.0f,0.0f,0.0f,0.0f};
    }

};

struct TauDxJacobian {
    float res;
    float d_H_l, d_H_r;
    float d_bed_l, d_bed_r;

    __device__ __forceinline__
    float apply_jvp(const TauDxStencil& dot) const {
        return d_H_l * dot.H_l +
	       d_H_r * dot.H_r;
    }

};

__device__ __forceinline__
TauDxJacobian get_tau_dx_jac(
    TauDxStencil s,
    float dx_inv,
    int i, int j,  // Defined on facets
    int ny, int nx) {

    TauDxJacobian jac = {0};

    // No driving stress on boundaries
    if (j <= 0 || j >= nx) {
        return jac;
    }

    float H_avg = 0.5f*(s.H_l + s.H_r);
    //float grounded_l = sigmoid(0.917f*s.H_l + s.bed_l,s.sigmoid_c);
    //float grounded_r = sigmoid(0.917f*s.H_r + s.bed_r,s.sigmoid_c);
    float grounded_l = s.phi_l;//sigmoid(s.phi_l,s.sigmoid_c);
    float grounded_r = s.phi_r;//sigmoid(s.phi_r,s.sigmoid_c);

    float base_l = grounded_l * s.bed_l - (1.0f - grounded_l)*0.917f*s.H_l;
    float base_r = grounded_r * s.bed_r - (1.0f - grounded_r)*0.917f*s.H_r;

    float dbase_dH_l = -(1.0f - grounded_l)*0.917f;
    float dbase_dH_r = -(1.0f - grounded_r)*0.917f;

    float S_l = base_l + s.H_l;
    float S_r = base_r + s.H_r;

    jac.res = H_avg * (S_r - S_l) * dx_inv;

    jac.d_H_l = 0.5f*(S_r - S_l)*dx_inv - H_avg*(1.0f + dbase_dH_l)*dx_inv;
    jac.d_H_r = 0.5f*(S_r - S_l)*dx_inv + H_avg*(1.0f + dbase_dH_r)*dx_inv;
    jac.d_bed_l = -H_avg*grounded_l*dx_inv;
    jac.d_bed_r =  H_avg*grounded_r*dx_inv;
    return jac;
}

__device__ __forceinline__
DualFloat get_tau_dx_dual(
    TauDxStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    TauDxJacobian jac = get_tau_dx_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

struct TauDyStencil {
    float H_t, H_b;
    float bed_t, bed_b;
    float phi_t, phi_b;
    float sigmoid_c;
};

struct TauDyStencilDual {
    DualFloat H_t, H_b;
    float bed_t, bed_b;
    float phi_t, phi_b;
    float sigmoid_c;

    __device__ __forceinline__
    TauDyStencil get_primals() const {
        return {H_t.v,H_b.v,bed_t,bed_b,phi_t,phi_b,sigmoid_c};
    }

    __device__ __forceinline__
    TauDyStencil get_diffs() const {
        return {H_t.d,H_b.d,0.0f,0.0f,0.0f,0.0f,0.0f};
    }

};

struct TauDyJacobian {
    float res;
    float d_H_t, d_H_b;
    float d_bed_t, d_bed_b;

    __device__ __forceinline__
    float apply_jvp(const TauDyStencil& dot) const {
        return d_H_t * dot.H_t +
	       d_H_b * dot.H_b;
    }
};

__device__ __forceinline__
TauDyJacobian get_tau_dy_jac(
    TauDyStencil s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {

    TauDyJacobian jac = {0};
    if (i <= 0 || i >= ny) {
        return jac;
    }

    float H_avg = 0.5f*(s.H_t + s.H_b);
    //float grounded_t = sigmoid(0.917f*s.H_t + s.bed_t,s.sigmoid_c);
    //float grounded_b = sigmoid(0.917f*s.H_b + s.bed_b,s.sigmoid_c);
    float grounded_t = s.phi_t;//sigmoid(s.phi_t,s.sigmoid_c);
    float grounded_b = s.phi_b;//sigmoid(s.phi_b,s.sigmoid_c);

    float base_t = grounded_t * s.bed_t - (1.0f - grounded_t)*0.917f*s.H_t;
    float base_b = grounded_b * s.bed_b - (1.0f - grounded_b)*0.917f*s.H_b;

    float dbase_dH_t = -(1.0f - grounded_t)*0.917f;
    float dbase_dH_b = -(1.0f - grounded_b)*0.917f;

    float S_t = base_t + s.H_t;
    float S_b = base_b + s.H_b;

    jac.res = H_avg * (S_t - S_b) * dx_inv;

    jac.d_H_t = 0.5f*(S_t - S_b)*dx_inv + H_avg*(1.0f + dbase_dH_t)*dx_inv;
    jac.d_H_b = 0.5f*(S_t - S_b)*dx_inv - H_avg*(1.0f + dbase_dH_b)*dx_inv;
    jac.d_bed_t =  H_avg*grounded_t*dx_inv;
    jac.d_bed_b = -H_avg*grounded_b*dx_inv;
    return jac;

}

__device__ __forceinline__
DualFloat get_tau_dy_dual(
    TauDyStencilDual s,
    float dx_inv,
    int i, int j,
    int ny, int nx) {
    TauDyJacobian jac = get_tau_dy_jac(s.get_primals(),dx_inv,i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

