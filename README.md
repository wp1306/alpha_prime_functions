# alpha-prime functions

Python tools for evaluating **α-prime functions** (and related Green-function kernels) in
**multiply connected circular domains**, using Schottky-group products of Möbius maps.

The geometry is a unit disk with circular holes removed. Each hole is paired with its
reflection by a Schottky generator. Finite products over these maps approximate the
α-prime / $X_\alpha$ building blocks used for harmonic problems (Green functions,
generalized Schwarz integrals, conformal maps, nematic director fields).

## Features

- Schottky generators and truncated group products (`constmaps_alp`)
- α-prime kernels $\omega_\alpha$, $X_\alpha$ and logarithmic derivatives
- Green-type kernels $G$, $G^{(m)}$, $\Pi$
- Domain membership tests for circular multiply connected regions
- Optional MATLAB SK-prime comparison via the MATLAB Engine API

## Requirements

- Python 3.10+
- NumPy
- Matplotlib (notebooks)
- Jupyter (example notebooks)

Optional:

- MATLAB + `matlabengine` (only for `matlab_prime.py`)

## Repository layout

| File | Role |
|------|------|
| `maps_core.py` | Pure functions: Möbius maps, Schottky products, $X_\alpha$ and derivatives |
| `context.py` | Geometry / map context (`init_context`) |
| `alpha_prime.py` | High-level kernels: $\omega_\alpha$, $G$, $\Pi$, $\theta_m$, … |
| `matlab_prime.py` | MATLAB SK-prime wrapper (optional) |
| `conf_map01.ipynb` | Generalized Schwarz integrals / conformal maps |
| `conf_map_parallel.ipynb` | Parallel-slit type maps |
| `conf_map_cayley.ipynb` | Cayley-type maps |

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
