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
  ================== Basal Shear Stress ===================
  =========================================================*/

struct TauBxStencil {
    float u;
    float H_l, H_r;
    float bed_l, bed_r;
    float beta_l, beta_r;
    float water_drag;
};

struct TauBxStencilDual {
    DualFloat u;
    DualFloat H_l, H_r;
    float bed_l, bed_r;
    float beta_l, beta_r;
    float water_drag;

    __device__ __forceinline__
    TauBxStencil get_primals() const {
        return {u.v,H_l.v,H_r.v,bed_l,bed_r,beta_l,beta_r,water_drag};
    }

    __device__ __forceinline__
    TauBxStencil get_diffs() const {
        return {u.d,H_l.d,H_r.d,0.0f,0.0f,0.0f,0.0f,0.0f};
    }

};

struct TauBxJacobian {
    float res;
    float d_u;
    float d_H_l, d_H_r;
    float d_beta_l, d_beta_r;

    __device__ __forceinline__
    float apply_jvp(const TauBxStencil& dot) const {
        return d_u * dot.u +
	       d_H_l * dot.H_l +
	       d_H_r * dot.H_r;
    }
};

__device__ __forceinline__
TauBxJacobian get_tau_bx_jac(
   TauBxStencil s )
{
    TauBxJacobian jac = {0};

    float grounded_l = sigmoid(s.bed_l + 0.917f*s.H_l,0.1f);
    float grounded_r = sigmoid(s.bed_r + 0.917f*s.H_r,0.1f);

    float beta_eff_l = grounded_l*s.beta_l + (1.0f - grounded_l)*s.water_drag;
    float beta_eff_r = grounded_r*s.beta_r + (1.0f - grounded_r)*s.water_drag;

    float beta_eff = 0.5f*(beta_eff_l + beta_eff_r);

    jac.res = -beta_eff * s.u;
    jac.d_u = -beta_eff;
    jac.d_beta_l = -0.5f*grounded_l*s.u;
    jac.d_beta_r = -0.5f*grounded_r*s.u;

    return jac;
}

__device__ __forceinline__
DualFloat get_tau_bx_dual(TauBxStencilDual s) {
    TauBxJacobian jac = get_tau_bx_jac(s.get_primals());
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

struct TauByStencil {
    float v;
    float H_t, H_b;
    float bed_t, bed_b;
    float beta_t, beta_b;
    float water_drag;
};

struct TauByStencilDual {
    DualFloat v;
    DualFloat H_t, H_b;
    float bed_t, bed_b;
    float beta_t, beta_b;
    float water_drag;

    __device__ __forceinline__
    TauByStencil get_primals() const {
        return {v.v,H_t.v,H_b.v,bed_t,bed_b,beta_t,beta_b,water_drag};
    }

    __device__ __forceinline__
    TauByStencil get_diffs() const {
        return {v.d,H_t.d,H_b.d,0.0f,0.0f,0.0f,0.0f,0.0f};
    }

};


struct TauByJacobian {
    float res;
    float d_v;
    float d_H_t, d_H_b;
    float d_beta_t, d_beta_b;

    __device__ __forceinline__
    float apply_jvp(const TauByStencil& dot) const {
        return d_v * dot.v +
	       d_H_t * dot.H_t +
	       d_H_b * dot.H_b;
    }
};

__device__ __forceinline__
TauByJacobian get_tau_by_jac(
   TauByStencil s) {
    TauByJacobian jac = {0};

    float grounded_t = sigmoid(s.bed_t + 0.917f*s.H_t,0.1f);
    float grounded_b = sigmoid(s.bed_b + 0.917f*s.H_b,0.1f);

    float beta_eff_t = grounded_t*s.beta_t + (1.0f - grounded_t)*s.water_drag;
    float beta_eff_b = grounded_b*s.beta_b + (1.0f - grounded_b)*s.water_drag;

    float beta_eff = 0.5f*(beta_eff_t + beta_eff_b);

    jac.res = -beta_eff * s.v;
    jac.d_v = -beta_eff;
    jac.d_beta_t = -0.5f*grounded_t*s.v;
    jac.d_beta_b = -0.5f*grounded_b*s.v;

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
};

struct TauDxStencilDual {
    DualFloat H_l, H_r;
    float bed_l, bed_r;

    __device__ __forceinline__
    TauDxStencil get_primals() const {
        return {H_l.v,H_r.v,bed_l,bed_r};
    }

    __device__ __forceinline__
    TauDxStencil get_diffs() const {
        return {H_l.d,H_r.d,0.0f,0.0f};
    }

};

struct TauDxJacobian {
    float res;
    float d_H_l, d_H_r;

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

    float base_l = fmaxf(s.bed_l,-0.917f*s.H_l);
    float base_r = fmaxf(s.bed_r,-0.917f*s.H_r);

    float dbase_dH_l = s.bed_l > -0.917f*s.H_l ? 0.0f : -0.917f;
    float dbase_dH_r = s.bed_r > -0.917f*s.H_r ? 0.0f : -0.917f;

    float S_l = base_l + s.H_l;
    float S_r = base_r + s.H_r;

    jac.res = H_avg * (S_r - S_l) * dx_inv;

    jac.d_H_l = 0.5f*(S_r - S_l)*dx_inv - H_avg*(1.0f + dbase_dH_l)*dx_inv;
    jac.d_H_r = 0.5f*(S_r - S_l)*dx_inv + H_avg*(1.0f + dbase_dH_r)*dx_inv;
    //jac.d_H_l = 0.0f*(S_r - S_l)*dx_inv - H_avg*(1.0f + dbase_dH_l)*dx_inv;
    //jac.d_H_r = 0.0f*(S_r - S_l)*dx_inv + H_avg*(1.0f + dbase_dH_r)*dx_inv;
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
};

struct TauDyStencilDual {
    DualFloat H_t, H_b;
    float bed_t, bed_b;

    __device__ __forceinline__
    TauDyStencil get_primals() const {
        return {H_t.v,H_b.v,bed_t,bed_b};
    }

    __device__ __forceinline__
    TauDyStencil get_diffs() const {
        return {H_t.d,H_b.d,0.0f,0.0f};
    }

};

struct TauDyJacobian {
    float res;
    float d_H_t, d_H_b;

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

    float base_t = fmaxf(s.bed_t,-0.917f*s.H_t);
    float base_b = fmaxf(s.bed_b,-0.917f*s.H_b);

    float dbase_dH_t = s.bed_t > -0.917f*s.H_t ? 0.0f : -0.917f;
    float dbase_dH_b = s.bed_b > -0.917f*s.H_b ? 0.0f : -0.917f;

    float S_t = base_t + s.H_t;
    float S_b = base_b + s.H_b;

    jac.res = H_avg * (S_t - S_b) * dx_inv;

    jac.d_H_t = 0.5f*(S_t - S_b)*dx_inv + H_avg*(1.0f + dbase_dH_t)*dx_inv;
    jac.d_H_b = 0.5f*(S_t - S_b)*dx_inv - H_avg*(1.0f + dbase_dH_b)*dx_inv;
    //jac.d_H_t = 0.0f*(S_t - S_b)*dx_inv + H_avg*(1.0f + dbase_dH_t)*dx_inv;
    //jac.d_H_b = 0.0f*(S_t - S_b)*dx_inv - H_avg*(1.0f + dbase_dH_b)*dx_inv;
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

