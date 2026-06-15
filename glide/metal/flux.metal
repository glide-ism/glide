// Auto-translated from cuda/flux.cu by the metal port (mechanical: pure helpers).
// Relies on common.metal (concatenated first) for DualFloat etc.

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

    inline
    HorizontalFluxStencil get_primals() const {
        return {u.v,H_l.v,H_r.v};
    }

    inline
    HorizontalFluxStencil get_diffs() const {
        return {u.d,H_l.d,H_r.d};
    }
};

struct HorizontalFluxJacobian {
    float res;
    float d_u;
    float d_H_l, d_H_r;

    inline
    float apply_jvp(const thread HorizontalFluxStencil& dot) const {
        return d_u * dot.u +
	       d_H_l * dot.H_l +
	       d_H_r * dot.H_r;
    }

};

inline
HorizontalFluxJacobian get_horizontal_flux_jac(
    HorizontalFluxStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {

    HorizontalFluxJacobian jac = {};

    // No flux on boundaries
    if (j <= 0 || j >= nx) {
	return jac;
    }

    float H_avg = 0.5f*(s.H_l + s.H_r);
    float u_mag = sqrt(s.u * s.u + 10.0f);//fabs(s.u);
    float u_sign = s.u / u_mag;//copysignf(1.0f, s.u);
    //float u_mag = fabs(s.u);
    //float u_sign = copysignf(1.0f, s.u);
    jac.res = H_avg*s.u - 0.5f*u_mag*(s.H_r - s.H_l);

    jac.d_H_l = 0.5f*(s.u + u_mag);
    jac.d_H_r = 0.5f*(s.u - u_mag);
    jac.d_u   = H_avg - 0.5f*u_sign*(s.H_r - s.H_l);
    return jac;
}

inline
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

    inline
    VerticalFluxStencil get_primals() const {
        return {v.v,H_t.v,H_b.v};
    }

    inline
    VerticalFluxStencil get_diffs() const {
        return {v.d,H_t.d,H_b.d};
    }
};

struct VerticalFluxJacobian {
    float res;
    float d_v;
    float d_H_t, d_H_b;

    inline
    float apply_jvp(const thread VerticalFluxStencil& dot) const {
        return d_v * dot.v +
	       d_H_t * dot.H_t +
	       d_H_b * dot.H_b;
    }

};

inline
VerticalFluxJacobian get_vertical_flux_jac(
    VerticalFluxStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {

    VerticalFluxJacobian jac = {};

    // No flux on boundaries
    if (i <= 0 || i >= ny) {
	return jac;
    }

    float H_avg = 0.5f*(s.H_t + s.H_b);
    float v_mag = sqrt(s.v * s.v + 10.0f);//fabs(s.v);
    float v_sign = s.v / v_mag;//copysignf(1.0f, s.v);
    //float v_mag = fabs(s.v);
    //float v_sign = copysignf(1.0f, s.v);
    jac.res = H_avg*s.v - 0.5f*v_mag*(s.H_t - s.H_b);

    jac.d_H_t = 0.5f*(s.v - v_mag);
    jac.d_H_b = 0.5f*(s.v + v_mag);
    jac.d_v   = H_avg - 0.5f*v_sign*(s.H_t - s.H_b);
    return jac;
}

inline
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

struct CellCalvingStencil {
    float H;
    float grounded;
    float calving_rate;
    float sigmoid_c;
};

struct CellCalvingStencilDual {
    DualFloat H;
    float grounded;
    float calving_rate;
    float sigmoid_c;

    inline
    CellCalvingStencil get_primals() const {
        return {H.v,grounded,calving_rate,sigmoid_c};
    }

    inline
    CellCalvingStencil get_diffs() const {
        return {H.d,0.0f,0.0f,0.0f};
    }
};

struct CellCalvingJacobian {
    float res;
    float d_H;

    inline
    float apply_jvp(const thread CellCalvingStencil& dot) const {
        return d_H * dot.H;
    }

};

inline
CellCalvingJacobian get_cell_calving_jac(
    CellCalvingStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {

    CellCalvingJacobian jac = {};

    jac.res = -s.calving_rate*(1.0f - s.grounded)*s.H;
    jac.d_H = -s.calving_rate*(1.0f - s.grounded);

    return jac;
}

inline
DualFloat get_cell_calving_dual(
    CellCalvingStencilDual s,
    int i, int j,
    int ny, int nx) {
    CellCalvingJacobian jac = get_cell_calving_jac(s.get_primals(),i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

struct FacetCalvingStencil {
    float H_this, H_other;
    float phi_this, phi_other;
    float calving_rate;
    float calving_length;
};

struct FacetCalvingStencilDual {
    DualFloat H_this, H_other;
    float phi_this, phi_other;
    float calving_rate;
    float calving_length;

    inline
    FacetCalvingStencil get_primals() const {
        return {H_this.v,H_other.v,phi_this,phi_other,calving_rate,calving_length};
    }

    inline
    FacetCalvingStencil get_diffs() const {
        return {H_this.d,H_other.d,0.0f,0.0f,0.0f,0.0f};
    }
};

struct FacetCalvingJacobian {
    float res;
    float d_H_this;

    inline
    float apply_jvp(const thread FacetCalvingStencil& dot) const {
        return d_H_this * dot.H_this;
    }

};

inline
FacetCalvingJacobian get_facet_calving_jac(
    FacetCalvingStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {

    FacetCalvingJacobian jac = {};


    //float phineg_this = fmax(-s.phi_this,0.0f)/s.calving_length;
    //float phineg_other = fmax(-s.phi_other,0.0f)/s.calving_length;

    //float phineg2_this = phineg_this * phineg_this;
    //float phineg2_other = phineg_other * phineg_other;

    //float chi_this = phineg2_this/(1.0f + phineg2_this);
    //float chi_other = phineg2_other/(1.0f + phineg2_other);

    //float chi_this = 1.0f - sigmoid(s.phi_this,s.calving_length);
    //float chi_other = 1.0f - sigmoid(s.phi_other,s.calving_length);
    float chi_this = 1.0f - s.phi_this;//sigmoid(s.phi_this,s.calving_length);
    float chi_other = 1.0f - s.phi_other;// sigmoid(s.phi_other,s.calving_length);

    float coeff = chi_this*chi_other;
    jac.res = coeff * s.calving_rate * s.H_this;
    jac.d_H_this = coeff * s.calving_rate;
    
    return jac;
}

inline
DualFloat get_facet_calving_dual(
    FacetCalvingStencilDual s,
    int i, int j,
    int ny, int nx) {
    FacetCalvingJacobian jac = get_facet_calving_jac(s.get_primals(),i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}







