"""Alpha-prime / Green-function style APIs (a_prime4.ipynb cell 1).

Depends on context.init_context(...).
"""

from __future__ import annotations

import context as ctx
from maps_core import (
    abs,
    conj,
    dlogXda,
    dlogXdz,
    d2logXdz,
    d2logXdadz,
    d3logXdadz2,
    dthz,
    exp,
    imag,
    log,
    pi,
    psiz_inf,
    sin,
    thz,
    vec_to_map,
    xhat_alpha_inf,
)


def omega_alpha(z, a):
    ctx._require_context()
    return (z - a) * np_sqrt_xhat(z, a, ctx.alpha)


def omega_alpham(z, a):
    ctx._require_context()
    return (z - a) * np_sqrt_xhat(z, a, -ctx.alpha)


def np_sqrt_xhat(z, a, alpha_arr):
    import numpy as np

    return np.sqrt(
        xhat_alpha_inf(z, a, ctx.maps, ctx.mp_i, alpha_arr, ctx.selected_fps2)
    )


def X_alpha(z, a):
    ctx._require_context()
    return (z - a) ** 2 * xhat_alpha_inf(
        z, a, ctx.maps, ctx.mp_i, ctx.alpha, ctx.selected_fps2
    )


def X_alpham(z, a):
    ctx._require_context()
    return (z - a) ** 2 * xhat_alpha_inf(
        z, a, ctx.maps, ctx.mp_i, -ctx.alpha, ctx.selected_fps2
    )


def dlogX_da(z, a):
    ctx._require_context()
    return dlogXda(z, a, ctx.maps, ctx.mp_i, ctx.alpha, ctx.selected_fps2)

def d2logX_dz(z, a):
    ctx._require_context()
    return d2logXdz(z, a, ctx.maps, ctx.mp_i, ctx.alpha, ctx.selected_fps2)

def dlogX_dz(z, a):
    ctx._require_context()
    return dlogXdz(z, a, ctx.maps, ctx.mp_i, ctx.alpha, ctx.selected_fps2)

def d2logX_dadz(z, a):
    ctx._require_context()
    return d2logXdadz(z, a, ctx.maps, ctx.mp_i, ctx.alpha, ctx.selected_fps2)

def d3logX_dadz2(z, a):
    ctx._require_context()
    return d3logXdadz2(z, a, ctx.maps, ctx.mp_i, ctx.alpha, ctx.selected_fps2)


def dlogXm_da(z, a):
    ctx._require_context()
    return dlogXda(z, a, ctx.maps, ctx.mp_i, -ctx.alpha, ctx.selected_fps2)


def dlogXm_dz(z, a):
    ctx._require_context()
    return dlogXdz(z, a, ctx.maps, ctx.mp_i, -ctx.alpha, ctx.selected_fps2)


def thetam(z, m):
    ctx._require_context()
    mapm = vec_to_map(ctx.maps[:, m])
    return thz(mapm, z)


def thetam_inv(z, m):
    ctx._require_context()
    mapm = vec_to_map(ctx.maps[:, 2 * ctx.M - m - 1])
    return thz(mapm, z)


def dthetam(z, m):
    ctx._require_context()
    mapm = vec_to_map(ctx.maps[:, m])
    return dthz(mapm, z)


def dvalpha_dz(z, alpha, j, a=0.4 + 0.8j):
    return (
        exp(-1j * alpha[j]) * dlogX_da(thetam(1 / conj(a), j), z)
        - exp(1j * alpha[j]) * dlogX_da(1 / conj(a), z)
    ) / (4 * pi * 1j)


def dvalpham_dz(z, alpha, j, a=0.4 + 0.8j):
    return (
        exp(1j * alpha[j]) * dlogXm_da(thetam(1 / conj(a), j), z)
        - exp(-1j * alpha[j]) * dlogXm_da(1 / conj(a), z)
    ) / (4 * pi * 1j)


def dpsialpha_dz(z, alpha, j):
    return dvalpha_dz(z, alpha, j) / sin(alpha[j]) - dvalpha_dz(z, alpha, 0) / sin(
        alpha[0]
    )


def G(z, a):
    return (
        1
        / (2 * pi * 1j)
        * log(omega_alpha(z, a) / (abs(a) * omega_alpha(z, 1 / conj(a))))
    )


def Gm(z, a):
    return (
        1
        / (2 * pi * 1j)
        * log(omega_alpham(z, a) / (abs(a) * omega_alpham(z, 1 / conj(a))))
    )


def G1(z, a, j):
    return (
        1
        / (2 * pi * 1j)
        * log(omega_alpha(z, a) / omega_alpha(z, thetam(1 / conj(a), j)))
    )


def c1(a, m):
    ctx._require_context()
    return imag(
        exp(-1j * ctx.alpha[m])
        * G(ctx.qv[m] * ctx.zp[15] + ctx.dv[m], a)
    ) / sin(ctx.alpha[m])


def Gmod(z, a):
    return G(z, a) + c1(a, 0)


def Pi(z, w, a, b):
    return (
        omega_alpha(z, a)
        * omega_alpham(w, b)
        / omega_alpham(w, a)
        / omega_alpha(z, b)
    )


def psiz(z, m, a_pt=0.4 + 0.8j):
    ctx._require_context()
    return psiz_inf(z, m, ctx.maps, ctx.mp_i, ctx.alpha, a_pt=a_pt)


def X_alpha_ratio(z, a, b):
    return X_alpha(z, a) / X_alpha(z, b)


def X_alpham_ratio(z, a, b):
    return X_alpham(z, a) / X_alpham(z, b)
