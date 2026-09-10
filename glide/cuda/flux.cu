/*=========================================================
  ====================== Mass Flux ========================
  =========================================================*/

struct HorizontalFluxStencil {
    float u;
    float H_l, H_r;
};

struct HorizontalFluxStencilDual {
    DualFloat u;
    DualFloat H_l, H_r;

    __device__ __forceinline__
    HorizontalFluxStencil get_primals() const {
        return {u.v,H_l.v,H_r.v};
    }

    __device__ __forceinline__
    HorizontalFluxStencil get_diffs() const {
        return {u.d,H_l.d,H_r.d};
    }
};

struct HorizontalFluxJacobian {
    float res;
    float d_u;
    float d_H_l, d_H_r;

    __device__ __forceinline__
    float apply_jvp(const HorizontalFluxStencil& dot) const {
        return d_u * dot.u +
	       d_H_l * dot.H_l +
	       d_H_r * dot.H_r;
    }

};

__device__
HorizontalFluxJacobian get_horizontal_flux_jac(
    HorizontalFluxStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {

    HorizontalFluxJacobian jac = {0};

    // No flux on boundaries
    if (j <= 0 || j >= nx) {
	return jac;
    }

    float H_avg = 0.5f*(s.H_l + s.H_r);
    float u_mag = sqrtf(s.u * s.u + 10.0f);//fabsf(s.u);
    float u_sign = s.u / u_mag;//copysignf(1.0f, s.u);
    //float u_mag = fabsf(s.u);
    //float u_sign = copysignf(1.0f, s.u);
    jac.res = H_avg*s.u - 0.5f*u_mag*(s.H_r - s.H_l);

    jac.d_H_l = 0.5f*(s.u + u_mag);
    jac.d_H_r = 0.5f*(s.u - u_mag);
    jac.d_u   = H_avg - 0.5f*u_sign*(s.H_r - s.H_l);
    return jac;
}

__device__ __forceinline__
DualFloat get_horizontal_flux_dual(
    HorizontalFluxStencilDual s,
    int i, int j,
    int ny, int nx) {
    HorizontalFluxJacobian jac = get_horizontal_flux_jac(s.get_primals(),i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}


struct VerticalFluxStencil {
    float v;
    float H_t, H_b;
};

struct VerticalFluxStencilDual {
    DualFloat v;
    DualFloat H_t, H_b;

    __device__ __forceinline__
    VerticalFluxStencil get_primals() const {
        return {v.v,H_t.v,H_b.v};
    }

    __device__ __forceinline__
    VerticalFluxStencil get_diffs() const {
        return {v.d,H_t.d,H_b.d};
    }
};

struct VerticalFluxJacobian {
    float res;
    float d_v;
    float d_H_t, d_H_b;

    __device__ __forceinline__
    float apply_jvp(const VerticalFluxStencil& dot) const {
        return d_v * dot.v +
	       d_H_t * dot.H_t +
	       d_H_b * dot.H_b;
    }

};

__device__
VerticalFluxJacobian get_vertical_flux_jac(
    VerticalFluxStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {

    VerticalFluxJacobian jac = {0};

    // No flux on boundaries
    if (i <= 0 || i >= ny) {
	return jac;
    }

    float H_avg = 0.5f*(s.H_t + s.H_b);
    float v_mag = sqrtf(s.v * s.v + 10.0f);//fabsf(s.v);
    float v_sign = s.v / v_mag;//copysignf(1.0f, s.v);
    //float v_mag = fabsf(s.v);
    //float v_sign = copysignf(1.0f, s.v);
    jac.res = H_avg*s.v - 0.5f*v_mag*(s.H_t - s.H_b);

    jac.d_H_t = 0.5f*(s.v - v_mag);
    jac.d_H_b = 0.5f*(s.v + v_mag);
    jac.d_v   = H_avg - 0.5f*v_sign*(s.H_t - s.H_b);
    return jac;
}

__device__ __forceinline__
DualFloat get_vertical_flux_dual(
    VerticalFluxStencilDual s,
    int i, int j,
    int ny, int nx) {
    VerticalFluxJacobian jac = get_vertical_flux_jac(s.get_primals(),i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*==============================================
  ==========  CALVING ==========================
  =============================================*/

// Non-conservative calving sink through a facet: ice leaves the cell at
// speed calving_rate wherever both cells are below the height-above-
// buoyancy threshold carried by the calving flag psi (see common.cu).
struct FacetCalvingStencil {
    float H_this, H_other;
    float psi_this, psi_other;
    float calving_rate;
};

struct FacetCalvingStencilDual {
    DualFloat H_this, H_other;
    float psi_this, psi_other;
    float calving_rate;

    __device__ __forceinline__
    FacetCalvingStencil get_primals() const {
        return {H_this.v,H_other.v,psi_this,psi_other,calving_rate};
    }

    __device__ __forceinline__
    FacetCalvingStencil get_diffs() const {
        return {H_this.d,H_other.d,0.0f,0.0f,0.0f};
    }
};

struct FacetCalvingJacobian {
    float res;
    float d_H_this;

    __device__ __forceinline__
    float apply_jvp(const FacetCalvingStencil& dot) const {
        return d_H_this * dot.H_this;
    }

};

__device__
FacetCalvingJacobian get_facet_calving_jac(
    FacetCalvingStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {

    FacetCalvingJacobian jac = {0};

    float chi_this  = 1.0f - s.psi_this;
    float chi_other = 1.0f - s.psi_other;
    float coeff = chi_this*chi_other;
    jac.res = coeff * s.calving_rate * s.H_this;
    jac.d_H_this = coeff * s.calving_rate;
    
    return jac;
}

__device__ __forceinline__
DualFloat get_facet_calving_dual(
    FacetCalvingStencilDual s,
    int i, int j,
    int ny, int nx) {
    FacetCalvingJacobian jac = get_facet_calving_jac(s.get_primals(),i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}







