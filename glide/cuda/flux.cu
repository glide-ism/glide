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

    float alpha = 0.5f;
    float H_avg = 0.5f*(s.H_l + s.H_r);
    //float u_mag = sqrtf(s.u * s.u + 10.0f);//fabsf(s.u);
    //float u_sign = s.u / u_mag;//copysignf(1.0f, s.u);
    float u_mag = fabsf(s.u);
    float u_sign = copysignf(1.0f, s.u);
    jac.res = H_avg*s.u - alpha*u_mag*(s.H_r - s.H_l);

    jac.d_H_l = 0.5f*s.u + alpha*u_mag;
    jac.d_H_r = 0.5f*s.u - alpha*u_mag;
    jac.d_u   = H_avg - alpha*u_sign*(s.H_r - s.H_l);
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

    float alpha = 0.5f;
    float H_avg = 0.5f*(s.H_t + s.H_b);
    //float v_mag = sqrtf(s.v * s.v + 10.0f);//fabsf(s.v);
    //float v_sign = s.v / v_mag;//copysignf(1.0f, s.v);
    float v_mag = fabsf(s.v);
    float v_sign = copysignf(1.0f, s.v);
    jac.res = H_avg*s.v - alpha*v_mag*(s.H_t - s.H_b);

    jac.d_H_t = 0.5f*s.v - alpha*v_mag;
    jac.d_H_b = 0.5f*s.v + alpha*v_mag;
    jac.d_v   = H_avg - alpha*v_sign*(s.H_t - s.H_b);
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

struct CellCalvingStencil {
    float H;
    float bed;
    float calving_rate;
    float sigmoid_c;
    int gl_derivatives;
};

struct CellCalvingStencilDual {
    DualFloat H;
    float bed;
    float calving_rate;
    float sigmoid_c;
    int gl_derivatives;

    __device__ __forceinline__
    CellCalvingStencil get_primals() const {
        return {H.v,bed,calving_rate,sigmoid_c,gl_derivatives};
    }

    __device__ __forceinline__
    CellCalvingStencil get_diffs() const {
        return {H.d,0.0f,0.0f,0.0f,0};
    }
};

struct CellCalvingJacobian {
    float res;
    float d_H;

    __device__ __forceinline__
    float apply_jvp(const CellCalvingStencil& dot) const {
        return d_H * dot.H;
    }

};

__device__
CellCalvingJacobian get_cell_calving_jac(
    CellCalvingStencil s,
    int i, int j,  // Defined on facets
    int ny, int nx
    ) {

    CellCalvingJacobian jac = {0};

    float z = s.bed + 0.917f*s.H;
    float grounded = sigmoid(z, s.sigmoid_c);

    // res = -calving_rate * (1 - grounded) * H
    jac.res = -s.calving_rate*(1.0f - grounded)*s.H;
    jac.d_H = -s.calving_rate*(1.0f - grounded);

    // Optional: include H derivative through grounding line sigmoid
    // d(res)/d(H) += calving_rate * H * dgrounded_dH
    if (s.gl_derivatives) {
        float dgrounded_dH = 0.917f * sigmoid_deriv(z, s.sigmoid_c);
        jac.d_H += s.calving_rate * s.H * dgrounded_dH;
    }

    return jac;
}

__device__ __forceinline__
DualFloat get_cell_calving_dual(
    CellCalvingStencilDual s,
    int i, int j,
    int ny, int nx) {
    CellCalvingJacobian jac = get_cell_calving_jac(s.get_primals(),i,j,ny,nx);
    return {jac.res,jac.apply_jvp(s.get_diffs())};
}

/*==============================================
  ==========  CALVING (Frozen) =================
  =============================================*/

// Frozen version: uses precomputed c_eff instead of computing grounding
struct CellCalvingFrozenStencil {
    float H;
    float c_eff;  // Precomputed effective calving rate = (1-grounded)*calving_rate
};

struct CellCalvingFrozenJacobian {
    float res;
    float d_H;

    __device__ __forceinline__
    float apply_jvp(const CellCalvingFrozenStencil& dot) const {
        return d_H * dot.H;
    }
};

__device__
CellCalvingFrozenJacobian get_cell_calving_frozen_jac(
    CellCalvingFrozenStencil s,
    int i, int j,
    int ny, int nx)
{
    CellCalvingFrozenJacobian jac = {0};
    // res = -c_eff * H  (c_eff already includes the (1-grounded) factor)
    jac.res = -s.c_eff * s.H;
    jac.d_H = -s.c_eff;
    return jac;
}

