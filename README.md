# alpha-prime functions

Python tools for evaluating **α-prime functions** (and related Green-function kernels) in
**multiply connected circular domains**, using Schottky-group products of Möbius maps.

The geometry is a unit disk with circular holes removed. Each hole is paired with its
reflection by a Schottky generator. Finite products over these maps approximate the
α-prime / $X_\alpha$ building blocks used for harmonic problems: Green functions,
generalized Schwarz integrals, conformal maps, and nematic director fields.

Repository: [wp1306/alpha_prime_functions](https://github.com/wp1306/alpha_prime_functions)

## Features

- Schottky generators and truncated group products (`constmaps_alp`)
- α-prime kernels $\omega_\alpha$, $X_\alpha$ and their logarithmic derivatives
- Green-type kernels $G$, $G^{(m)}$, $\Pi$
- Domain membership tests for circular multiply connected regions
- Example notebooks for conformal maps, generalized Schwarz integrals, and nematic textures
- Optional MATLAB SK-prime comparison via the MATLAB Engine API

## Requirements

- Python 3.10+
- NumPy
- Matplotlib (notebooks)
- Jupyter (example notebooks)

Optional:

- MATLAB + `matlabengine` (only for `matlab_prime.py` and `nematic_plot.ipynb`)

## Repository layout

| File | Role |
|------|------|
| `maps_core.py` | Pure functions: Möbius maps, Schottky products, $X_\alpha$ and derivatives |
| `context.py` | Geometry / map context (`init_context`) |
| `alpha_prime.py` | High-level kernels: $\omega_\alpha$, $G$, $\Pi$, $\theta_m$, … |
| `matlab_prime.py` | MATLAB SK-prime wrapper (optional) |
| `conf_map01.ipynb` | Generalized Schwarz integrals / conformal maps |
| `conf_map_parallel.ipynb` | Parallel-slit maps |
| `conf_map_cayley.ipynb` | Cayley-type maps |
| `generalize_sch.ipynb` | Generalized Schwarz integral formulae |
| `nematic_plot.ipynb` | Nematic director textures from the harmonic $\alpha$-prime field |

Run the notebooks from this directory so that `maps_core`, `context`, and `alpha_prime` import as local modules.

## Quick start

```python
import numpy as np
from context import init_context
from alpha_prime import omega_alpha, G
from maps_core import is_inside_domain

# Circular holes: radii qv, centres dv (inside the unit disk)
qv = np.array([0.2, 0.2, 0.1])
dv = np.array([0.5 - 0.2j, -0.5 + 0.1j, 0.5j])

# Twist / period parameters for each hole
alpha = np.array([[np.pi / 2], [np.pi / 2], [0.0]])

# Truncation depth of the Schottky product
init_context(qv, dv, alpha, level_in=4)

z, a = 0.1 + 0.1j, -0.4j
w = omega_alpha(z, a)
g = G(z, a)
print(is_inside_domain(z, qv, dv), w, g)
```

`init_context` builds the generators, truncated Schottky words, and selected fixed
points, then stores them in `context`. Subsequent calls to `alpha_prime` use that
context.

## Mathematical outline

Let $D$ be the unit disk minus $M$ circular holes with radii $q_j$ and centres $d_j$.
The Schottky generators $\theta_j$ pair each hole with its inverse circle. A truncated
product over the group (depth `level`) approximates $\widehat{X}_\alpha(z,a)$, and

$$
\omega_\alpha(z,a)=(z-a)\sqrt{\widehat{X}_\alpha(z,a)},
\qquad
X_\alpha(z,a)=(z-a)^2\,\widehat{X}_\alpha(z,a).
$$

Green-type kernels are built from these, for example

$$
G(z,a)
=\frac{1}{2\pi i}
\log\frac{\omega_\alpha(z,a)}{|a|\,\omega_\alpha(z,1/\overline{a})}.
$$

In the harmonic / low-Ericksen regime, the 2-D nematic director angle is a harmonic
function. The same kernels therefore recover director textures in multiply connected
circular domains (`nematic_plot.ipynb`).

`first_app` labels each group element by its leftmost generator, i.e. which Schottky
disk the corresponding map (and its fixed points) lands in.

Larger `level` improves the product at higher cost. Typical values are `4`–`6`
(the notebooks sometimes use higher values).

## Typical workflow

1. Choose geometry (`qv`, `dv`) and periods (`alpha`).
2. Call `init_context(..., level_in=...)`.
3. Evaluate $\omega_\alpha$, $G$, a conformal map, or a nematic texture in a notebook.
4. Restrict evaluation points with `is_inside_domain`.

## Notes

- `matlab_prime.py` contains a local default path to an SK-prime MATLAB tree.
  Change `DEFAULT_SK_PATH` before using it.
- The MATLAB engine is optional. The core $\alpha$-prime evaluation is pure Python.

## References

- D. G. Crowdy, work on the Schottky–Klein prime function and multiply connected domains.
- $\alpha$-prime / twisted prime-function constructions for harmonic and conformal
  problems in circular domains.
