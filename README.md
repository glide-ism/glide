# GLIDE

**GPU-accelerated Lightweight Ice Dynamics Engine**

GLIDE is a CUDA ice-sheet model. It solves a shear-resolving, depth-integrated
momentum balance (MOLHO, with the shallow shelf approximation as an exact
special case) coupled implicitly to mass conservation, with optional enthalpy
thermodynamics. Every solve, forward and adjoint, runs on the GPU through a
nonlinear multigrid method, and gradients are available both from a built-in
adjoint and through PyTorch autograd.

## Features

- **Shear-resolving momentum balance**: a two-field mono-layer higher-order
  (MOLHO) model carries the depth-averaged velocity and a deformational
  component, so it recovers SSA for plug flow and SIA for a frozen bed.
  `stress_scheme='ssa'` compiles a pure-SSA build.
- **Fully coupled, implicit time stepping**: velocity and thickness are solved
  together each step, with $H \ge$ `thklim` enforced as an obstacle constraint,
  so large time steps (tens of years on continental grids) are stable.
- **FASCD multigrid**: Full Approximation Scheme with Constrained Descent, using
  a Newton–Vanka patch smoother. Every kernel is a custom CUDA kernel.
- **Marine margins**: one flotation excess drives a smoothed grounding line, an
  effective-pressure-dependent sliding law, and a height-above-buoyancy calving
  criterion.
- **Thermodynamics**: a conservative enthalpy model in sigma coordinates,
  coupled to the rheology through the rate factor.
- **Gradients**: a matching FAS adjoint solver gives gradients with respect to
  basal friction, bed, initial thickness, and surface mass balance.
  `glide.torch.GlideStep` wraps one time step as a `torch.autograd.Function`.

## Installation

```bash
git clone https://github.com/glide-ism/glide.git
cd glide
pip install -e ".[cuda12]"     # or cuda11 / cuda13, matching your CUDA toolkit
```

CuPy ships one wheel per CUDA major version, so pick the extra that matches
your toolkit. Other extras:

```bash
pip install -e ".[cuda12,torch]"      # + PyTorch, for GlideStep / inverse problems
pip install -e ".[cuda12,examples]"   # + torch, matplotlib, polars, rioxarray, shapely
```

### Requirements

- Python ≥ 3.10
- An NVIDIA GPU with CUDA 11, 12, or 13
- NumPy, SciPy, xarray, h5py, h5netcdf, zarr, pyproj, geopandas, and gdown
  (installed automatically)

## Getting started

The examples are the documentation. Start with
[`examples/greenland/greenland_forward.py`](examples/greenland/greenland_forward.py).
It shows the full workflow: load a preprocessed dataset, build an
`IceDynamics` model and its multigrid hierarchy, set fields through the
managers (`mg.state`, `mg.geometry`, `mg.rheology`, `mg.sliding`,
`mg.calving`, `mg.forcing`), configure the solver, step with
`model.forward(t, dt)`, and write output to VTI/PVD (ParaView) and Zarr.
[`greenland_inverse.py`](examples/greenland/greenland_inverse.py) shows the
inverse workflow: infer basal friction from observed velocities with
`GlideStep` and a `torch.optim` optimizer.

| Example | What it shows |
|---|---|
| `greenland/greenland_forward.py` | Time-dependent Greenland run (MOLHO, calving, VTI + Zarr output) |
| `greenland/greenland_inverse.py` | Basal friction inversion from surface velocities (PyTorch) |
| `greenland/greenland_thermal_forcing.py` | Fetch and regrid geothermal flux and surface temperature |
| `antarctica/antarctica_forward.py`, `antarctica_inverse.py` | The same two workflows for Antarctica |
| `antarctica/build_dataset.py` | Build a model dataset from BedMachine, RACMO, and velocity mosaics |
| `bitterroot/bitterroot.py` | Glaciating a mountain range from a DEM |
| `ismip-hom/ismip-hom.py` | ISMIP-HOM experiments A–D |
| `thermal/cold_column.py`, `stefan.py`, `horizontal_advection.py` | Enthalpy solver against analytical references |
| `thermal/column_only_dome.py`, `valley_glacier.py` | Enthalpy on prescribed geometry and flow |
| `thermal/coupled_dome.py`, `coupled_valley_glacier.py` | Two-way momentum–enthalpy coupling (`ThermalModel`) |

Preprocessed datasets for Greenland, Antarctica, and the Bitterroot DEM are
downloaded and cached on first use (`glide.data.load_*`).

## Physics

### Units

Velocities are in m a⁻¹, time in years, lengths in m. Stresses are measured
in **metres of ice head**, so the factor $\rho_i g$ is folded into the rate
factor `B` and the friction coefficient `beta`. The examples show the
conversion, e.g. `B = A**(-1/n) / (917 * 9.81)`. The ice/water density
ratio $r = \rho_i/\rho_w = 0.917$ is defined in one place (`common.cu`).

### Momentum balance (MOLHO)

The horizontal velocity is given a fixed vertical profile with two unknown
fields, the depth average $\bar{\mathbf u}$ and a deformational amplitude
$\mathbf u_d$:

$$
\mathbf u(x, y, \zeta) = \bar{\mathbf u} + \mathbf u_d\,\varphi(\zeta),
\qquad
\varphi(\zeta) = \frac{1 - (n+2)\,\zeta^{n+1}}{n+1},
$$

where $\zeta$ runs from 0 at the surface to 1 at the bed. Since
$\int_0^1 \varphi\,d\zeta = 0$, $\bar{\mathbf u}$ is the exact depth average,
so the transport flux is $H\bar{\mathbf u}$, as in SSA. The bed and surface
velocities are

$$
\mathbf u_b = \bar{\mathbf u} - \mathbf u_d, \qquad
\mathbf u_s = \bar{\mathbf u} + \frac{\mathbf u_d}{n+1}.
$$

The momentum equations are the Euler–Lagrange equations of one discrete
dissipation functional, so the velocity block is exactly symmetric. That
symmetry is what keeps the adjoint consistent. In continuous form:

$$
\begin{aligned}
\nabla\cdot\big(2\bar\eta H\,\dot{\boldsymbol\varepsilon}'(\bar{\mathbf u})\big) - \boldsymbol\tau_b
  &= \rho_i g H\,\nabla s, \\
\kappa_1\,\nabla\cdot\big(2\bar\eta H\,\dot{\boldsymbol\varepsilon}'(\mathbf u_d)\big)
  - 2\kappa_2\,\frac{\bar\eta}{H}\,\mathbf u_d + \boldsymbol\tau_b &= 0,
\end{aligned}
$$

with $\dot{\boldsymbol\varepsilon}'(\mathbf u) = \dot{\boldsymbol\varepsilon}(\mathbf u) + \operatorname{tr}(\dot{\boldsymbol\varepsilon}(\mathbf u))\,\mathbf I$
the usual membrane strain-rate tensor. Basal drag depends only on
$\mathbf u_b$. It enters the two equations with opposite signs, so the drag
block of the Jacobian annihilates $\delta\bar{\mathbf u} = \delta\mathbf u_d$.

- With $\mathbf u_d = 0$ (plug flow), the first equation is exactly the SSA.
  `stress_scheme='ssa'` pins $\mathbf u_d = 0$ through identity rows and
  compiles kernels that solve 5×5 instead of 9×9 Vanka patches.
- With a frozen bed and a uniform slab, the model gives the SIA solution
  exactly: $\bar u = 2\tau_d^{\,n}H/((n+2)B^n)$ with
  $\mathbf u_d = \bar{\mathbf u}$.

Metric terms from the sloping $\zeta$ coordinate are neglected by design.

### Rheology

Glen's flow law is applied with a single depth-averaged viscosity per cell.
The squared strain-rate invariant is depth-averaged *before* exponentiation:

$$
\bar\eta = \tfrac12 B\,\big(\bar E^2 + \varepsilon_\text{reg}\big)^{\frac{1-n}{2n}},
\qquad
\bar E^2 = M(\bar{\mathbf u}) + \kappa_1\,M(\mathbf u_d) + \kappa_2\,\frac{|\mathbf u_d|^2}{H^2},
$$

$$
M(\mathbf u) = u_x^2 + v_y^2 + u_x v_y + \tfrac14 (u_y + v_x)^2 .
$$

$\kappa_1 = \int_0^1\varphi^2\,d\zeta = 1/(2n+3)$. $\kappa_2 = c\,S_1/4$
combines the shape-function geometry $S_1 = (n+2)^2/(2n+1)$ with a constant
$c = (2n+1)/(n+2)^{2n/(n+1)}$, chosen so that the plug-flow and no-slip limits
are both exact. Parameters: `rheology.B`, `n`, `eps_reg`.

`rheology.H_reg` regularizes the shear term on thin ice:
$1/H^2 \to 1/(H^2 + H_\text{reg}^2)$, applied consistently in the viscosity
and the shear residual. A reactive spring is added that pins
$\mathbf u_d$ on columns thinner than about $H_\text{reg}$. Without it, bare
steep terrain at `thklim` would be free to deform.

### Grounded and floating ice

Every grounded/floating quantity derives from a single **flotation excess**

$$
z = r H - d,
$$

where $d$ (`geometry.depth`) is the signed head deficit: water level minus
bed, positive under water and negative on dry land. For a domain at sea level
set `depth = -bed`. $z > 0$ is grounded, $z < 0$ floating, and the effective
pressure is $N = \rho_w g z$. Three smooth flags are built from $z$:

- **Grounded flag** $\phi = \sigma(c\,z)$ (logistic, steepness
  `geometry.sigmoid_c`) blends the ice base between the bed and the
  flotation draft, $b_\text{ice} = \phi\,b - (1-\phi)\,rH$. The surface is
  $s = b_\text{ice} + H$. This form is exact at flotation.
- **Flotation fraction** $\xi = N/(\rho_i g H) = z/(rH)$, clipped to
  $[0, 1]$, scales basal drag (below).
- **Calving flag** $\psi$ (see Calving).

The flags are computed on the finest grid and *restricted* to coarser
multigrid levels, not recomputed there. Read as area fractions, restriction is
the consistent coarse-grid representation, and recomputing them from coarse
geometry degrades FAS convergence.

### Driving stress

The driving stress $\rho_i g H\,\nabla s$ is formed on facets from the
blended surface above. One correction is applied: **rock cannot push ice**.
Where one cell's base stands above its neighbour's surface (a fjord wall or a
terrace), the part of the base above that surface is removed from the step
before the gradient is formed. The clip is one-sided and $C^1$. Overlapping
columns, floating ice, and calving fronts are left exactly unchanged.

### Basal drag

$$
\boldsymbol\tau_b = \Big(\beta\,\xi^{p}\,K(|\mathbf u_b|) + \beta_w\Big)\,\mathbf u_b,
\qquad
K = \big(|\mathbf u_b|^2 + u_\text{reg}\big)^{\frac{m-1}{2}}
    \left(\frac{u_0}{|\mathbf u_b| + u_0}\right)^{m}.
$$

- `sliding.beta`, `m`: Weertman coefficient and exponent
  ($\tau_b \propto |\mathbf u_b|^m$). `u_reg` regularizes $|\mathbf u_b|$ near
  zero.
- `sliding.u0`: regularized-Coulomb transition speed. At 0 (the default) the
  last factor is 1 and the law is pure Weertman. For $u_0 > 0$ the drag is
  Weertman for $|\mathbf u_b| \ll u_0$ and caps at $\beta\,\xi^p u_0^m$ for
  $|\mathbf u_b| \gg u_0$.
- `sliding.p`: effective-pressure exponent (Leguy et al., 2014). $p = 1$
  assumes full hydraulic connection to the ocean. $p \to 0$ confines the
  drag reduction to the grounding line, and $\xi^0$ is taken as 0 where
  $\xi = 0$.
- `sliding.N_scale_H`, `N_floor_H`: when `N_scale_H > 0`, $\xi$ uses a
  dimensional effective pressure, $\xi = \xi_f\,(H + H_\text{floor})/H_\text{scale}$.
  This makes the Coulomb cap proportional to $N$. The floor keeps drag on thin
  grounded ice.
- `sliding.water_drag` ($\beta_w$): a small linear drag everywhere, which
  keeps floating ice and ice-free cells well posed.

### Mass conservation

Thickness lives at cell centres and is advanced with implicit Euler,
coupled to the momentum solve:

$$
\frac{H - H_\text{prev}}{\Delta t} + \nabla\cdot(H\bar{\mathbf u}) = \dot a - (1-\psi)\,\frac{H}{\tau_c},
\qquad H \ge H_\text{min},
$$

with $\dot a$ = `forcing.smb` and $H_\text{min}$ = `geometry.thklim`. The
flux is centred with an upwind-type dissipation proportional to a smoothed
$|u|$, so it is differentiable. The bound $H \ge H_\text{min}$ is a
complementarity constraint: cells where it is active form the active set
(`state.mask`), and their thickness is held at `thklim`. Ice margins advance
and retreat by entering and leaving the active set. They need no explicit
tracking.

### Calving

Calving is a non-conservative, cell-centred sink $(1-\psi)H/\tau_c$, where
$\tau_c$ is `calving.timescale` in years. The default, `inf`, turns calving
off. The flag $\psi = \sigma\big(c\,r\,(H - H_\text{calve})\big)$ implements
a height-above-buoyancy criterion. Grounded ice calves where

$$
H - H_f < qH + h_0, \qquad H_f = d/r,
$$

with `calving.q` (fractional) and `calving.h0` (absolute, m). Both are cell
fields, so they can vary along the coast, for example with ocean thermal
forcing. `calving.H_c` sets a minimum thickness for floating tongues in deep
water. Ice whose base has lifted off the bed blends from the grounded law to
the shelf law over a ramp that keeps the criterion monotone in $H$. Thinning
never reduces calving, which is what lets the implicit solve converge. With
$q = h_0 = 0$ and $H_c = \infty$, exactly the floating ice calves. Under
implicit Euler a flagged cell decays by $1/(1+\Delta t/\tau_c)$ per step until
the active set takes it to `thklim`.

### Thermodynamics

`glide.enthalpy` solves the conservative enthalpy equation of Aschwanden et
al. (2012) in terrain-following coordinates. The conserved variable is
$HE$, not $E$, so energy is conserved exactly as the column thickens and
thins:

$$
\rho_i\Big[\partial_t(HE) + \nabla\cdot(H\mathbf u E) + \partial_\sigma(E\omega)\Big]
= \partial_\sigma\!\Big(\frac{K}{H}\,\partial_\sigma E\Big) + H\Phi - H\rho_w L\,r_d\,\varpi(E).
$$

Diffusivity switches smoothly between the cold and temperate values. Water
content above the pressure-melting point drains at rate $r_d$. The scaled
vertical velocity $\omega$ is integrated from the *realized* thickness change
of the momentum step, which makes a uniform field an exact discrete solution.
The basal boundary condition is geothermal plus frictional heat
($\boldsymbol\tau_b\cdot\mathbf u_b$). Strain heating comes from the MOLHO
shear profile. `ThermalModel` (in `glide.model`) operator-splits this solve
with the momentum step. It rebuilds the 3D velocity from
$(\bar{\mathbf u}, \mathbf u_d)$, solves the enthalpy with column Newton/Thomas
sweeps, and writes the Arrhenius rate factor back to `rheology.B` on every
multigrid level. Derivations and verification tests are in
[`notes/`](notes/).

## Numerics

- **Grid**: a staggered (MAC) grid. $H$ is at cell centres; $u$ and $u_d$ are
  on vertical facets; $v$ and $v_d$ are on horizontal facets. `ny` and `nx`
  must both be divisible by $2^{\,n_\text{levels}-1}$.
- **Forward solver (`FASCDSolver`)**: FAS V-cycles with constrained descent
  (Bueler-style FASCD) for the obstacle $H \ge$ `thklim`. The smoother is a
  multiplicative Vanka sweep. Each patch couples a cell's thickness with the
  velocities on its four facets: 9 unknowns for MOLHO, 5 for SSA. Patches are
  solved by damped Newton iteration with in-kernel LU. Options are set through
  `model.forward_solver.fas_options` and `.vanka_options` (including
  `.newton_options`), globally or per level.
- **Robustness tools**: backtracking coarse corrections, a cold-start solve at
  a small `dt` from zero velocity, `coarse_thinning_fraction`,
  NaN detection, residual tracing by cell class, and save-and-replay of failed
  solves (`dump_dir`, `glide.dump.load_solve_state`).
- **Adjoint (`FASAdjointSolver`)**: solves the transposed linearized system
  with the same multigrid structure. Constrained rows follow a single
  identity-row convention, documented in `common.cu`. The constraint
  multipliers are therefore exact. `model.backward(t, dt, dJdu=..., dJdv=...,
  dJdH=...)` fills gradients with respect to `beta`, `bed`, `H_prev`, and `smb`.
- **PyTorch**: `glide.torch.GlideStep.apply(t, dt, model, level, H_prev, bed,
  beta, smb)` returns the new state as tensors and backpropagates through the
  adjoint. Time-varying inputs (`q`, `h0`, `B`) are checkpointed per step.

## Package layout

```
glide/
├── model.py       # IceDynamics (forward/backward), ThermalModel
├── multigrid.py   # Multigrid hierarchy, field managers, FASCDSolver, FASAdjointSolver, options
├── grid.py        # Per-level Grid; State/Geometry/Rheology/Sliding/Calving/Forcing dataclasses
├── field.py       # Field / SubgridField / Constant over CuPy arrays
├── operators.py   # ForwardOperators / AdjointOperators (kernel dispatch, Vanka config)
├── enthalpy.py    # EnthalpyOperators: conservative enthalpy solver
├── torch.py       # GlideStep autograd.Function
├── io.py          # write_vti, VTIWriter (VTI/PVD), ZarrWriter
├── data.py        # Dataset download/caching; BedMachine, velocity, MAR/RACMO ingest
├── dump.py        # Save/replay of single forward solves
└── cuda/
    ├── common.cu      # Dual numbers, flotation/calving flags, constraint convention
    ├── viscosity.cu   # Depth-averaged Glen's-law viscosity (SSA and MOLHO)
    ├── stress.cu      # Membrane, vertical shear, basal drag, driving stress
    ├── flux.cu        # Mass flux and calving sink
    ├── residuals.cu   # Coupled residual, JVP, VJP
    ├── vanka.cu       # Newton–Vanka patch smoother (forward and adjoint)
    ├── transfer.cu    # Restriction/prolongation
    ├── grad.cu        # Parameter-gradient kernels
    └── enthalpy.cu    # Enthalpy residual and column smoother
```

Kernels are compiled at runtime by CuPy, with `-DGLIDE_MOLHO=0/1` selecting
the stress scheme.

## Tests and notes

`tests/` contains the verification scripts. The momentum and adjoint tests
cover SSA reduction, JVP against finite differences, adjoint consistency, and
gradients in `beta` and `bed`. The enthalpy tests cover the cold column,
advection, Stefan, horizontal advection, and conservation cases. `notes/`
holds the enthalpy derivation, its discretization, and the documentation of
the enthalpy tests.

## License

GPL-3.0. See [LICENSE](LICENSE) for details.
