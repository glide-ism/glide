# MIVA handoff — two-field depth-integrated ice dynamics in GLIDE

Context for debugging an indexing bug in the Vanka smoother or residual
assembly of the new shear-resolving model. Everything below is settled;
treat it as ground truth to test against.

## Model

Extension of GLIDE's SSA+transport to resolve vertical shear. Ansatz
(from MOLHO): u(x,σ) = ū + u_d·φ(σ), φ(σ) = (1 − (n+2)σ^{n+1})/(n+1),
σ = 0 at surface, 1 at bed, n = 3. Consequences used everywhere:

- ū is the exact depth average → transport flux is H·ū, unchanged from SSA.
- Bed velocity u_b = ū − u_d (surface velocity = ū + u_d/(n+1)).
- ∂u/∂z = (n+2)σⁿ u_d / H.

The BP dissipation is approximated by depth-averaging the *squared*
invariant before exponentiation ((·)^{(n+1)/(2n)} = (·)^{2/3}), then
renormalizing so both limits are exact (plug → SSA identically,
no-slip shear → SIA identically). Metric terms from ∇σ are neglected
(deliberate). Depth-averaged invariant (n = 3), up to the code's
factor-of-2 convention:

  Ē² = M(∇ū, ∇v̄) + κ₁·M(∇u_d, ∇v_d) + κ₂·[(u_d/H)² + (v_d/H)²]

  M(u,v) = u_x² + v_y² + u_x v_y + ¼(u_y + v_x)²
  κ₁ = ∫φ² dσ = 1/9
  κ₂ = c·S₁/4 = √5/4 = 0.5590170
      with S₁ = (n+2)²/(2n+1) = 25/7 (shape-function geometry)
      and  c  = (2n+1)/(n+2)^{2n/(n+1)} = 7/5^{3/2} = 0.6261000
      (limit renormalization; kept as a separate labeled constant in code)

No ∇ū·∇u_d cross terms exist (∫φ = 0). One scalar viscosity per cell:
η̄ ∝ B·(Ē²)^{(1−n)/(2n)}. Sliding: Weertman(+Budd N(H)) potential of
|u_b|; drag couples (ū, u_d) only through u_b.

Everything is derived by differentiating one discrete functional J
(exactly self-adjoint stress balance). If a residual disagrees with a
finite-difference gradient of J, the residual is wrong, not J.

## Discretization (MAC grid)

- H at cell centers; ū, u_d normal components on facets (u on vertical
  facets, v on horizontal). Vertical facet (i,j) sits between cells
  (i,j−1) and (i,j).
- Rule used everywhere: interpolate squared norms, never square
  interpolants. ε_xy: computed at corners, cell value = mean of 4 corner
  squares. Shear: cell value = (κ₂/H_c²)·½(u_d,l² + u_d,r² + v_d,t² +
  v_d,b²) — cell's own H_c, no facet-H averaging. Drag speed:
  |u_b|²_c = ½Σ(facet components of ū − u_d)², regularized +γ².
- Known-correct EL forms to check assembly against (up to Δx² and the
  η convention factor of 2):
  - Shear residual, facet (i,j):
    R_d = (κ₂-coefficient)·(η_{i,j−1}/H_{i,j−1} + η_{i,j}/H_{i,j})·u_d,ij
  - Drag residual, facet f: ½(β̄_left + β̄_right)·(ū − u_d)_f, with
    β̄_c = C_c N_c (|u_b|²_c + γ²)^{(1−m)/(2m)} cell-centered.
  - Drag sign structure: R_ū = +R_SSA-drag(ū − u_d), R_{u_d} = −same,
    entrywise. Jacobian drag block = (1,−1)(1,−1)ᵀ ⊗ J_SSA-drag,
    including all cross-facet softening (ggᵀ) terms.

## Vanka smoother

9×9 patch: [H_c; ū on 4 facets; u_d on 4 facets]. Transport row has no
u_d columns. LU: in-place, Doolittle, no pivoting, row-major A[i*N+j],
**diagonal of factored matrix stores 1/U[kk]** (backsolve multiplies;
any transpose solve must multiply by stored reciprocals too). RHS b is
consumed as scratch by the forward solve; x must not alias A.
Recommended DOF order: u_d first (dominant diagonals; trailing 5×5 =
condensed (H, ū) Schur complement).

## Exact tests (these are the bug bisectors — all must hold to machine precision, not truncation error)

1. **SSA reduction**: set u_d ≡ 0 fields and κ₂-term off → residuals and
   Jacobian must equal the existing (verified) SSA code exactly.
2. **Drag equivalence**: two-field drag assembly vs. SSA drag assembly
   evaluated at (ū − u_d), arbitrary random input fields: R_ū = +,
   R_{u_d} = −, identically. Drag Hessian must annihilate δū = δu_d.
3. **Uniform slab, frozen bed** (drag → large): ū = 2τ_d³H/(5B³),
   u_d = ū. All discrete averages are exact for uniform fields, so any
   deviation is a bug, and its rational multiple of the truth usually
   identifies the misplaced constant.
4. **Uniform slab with Weertman slip**: adds u_b = (τ_d/CN)^m; the
   deformational part is unchanged.
5. **Symmetry**: velocity-block Hessian-vector products satisfy
   ⟨w, Av⟩ = ⟨v, Aw⟩ for random v, w. Indexing bugs (wrong facet↔cell
   adjacency, transposed gathers) almost always break this first.
6. **Gradient check**: FD Taylor-remainder test on scalar J vs assembled
   residual.

## Likely suspects for an indexing bug

Facet↔cell adjacency conventions (off-by-one in the (i,j−1)/(i,j)
pairing, or x-facet vs y-facet role swap for v_d); corner gathers for
ε_xy of the u_d field (new code path, mirrors ū's); DOF-ordering
mismatch between patch assembly and the LU layout (especially if u_d
was reordered first in the solver but not in assembly); sign of
u_b = ū − u_d (σ increases downward; getting u_b = ū + u_d flips the
drag block's null direction and breaks test 2); forgetting the
reciprocal-diagonal convention in a hand-written transpose solve.

Debugging order: run test 1, then 2, then 5 on random fields, then 3.
Whichever fails first localizes the bug to a small code region.
