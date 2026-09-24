"""Möbius / Schottky map utilities and alpha-prime building blocks (pure functions).

Extracted from a_prime4.ipynb cell 0. No notebook globals.
"""

from __future__ import annotations

import numpy as np

pi = np.pi


def real(z):
    return np.real(z)


def imag(z):
    return np.imag(z)


def abs(z):
    return np.abs(z)


def conj(z):
    return np.conj(z)


def exp(z):
    return np.exp(z)


def log(z):
    return np.log(z)


def sin(z):
    return np.sin(z)

def cos(z):
    return np.cos(z)


def sqrt(z):
    return np.sqrt(z)


def prod(z):
    return np.prod(z)


def vec_to_map(mm: np.ndarray) -> np.ndarray:
    """MM = [a;c;b;d] (MATLAB TT(:)) -> 2x2 [[a,b],[c,d]]."""
    return np.array([[mm[0], mm[2]], [mm[1], mm[3]]], dtype=np.complex128)

def diff(arr, n=1):
    """
    Mimics MATLAB's diff for 1D numpy arrays.
    Returns the n-th discrete difference along the given axis.

    Parameters:
    arr : array-like, input data.
    n : int, optional. The number of times values are differenced. Default is 1.

    Returns:
    diff_arr : array of length len(arr) - n
    """
    import numpy as np
    arr = np.asarray(arr)
    return np.diff(arr, n=n)

def thz(map_: np.ndarray, z: np.ndarray | complex) -> np.ndarray | complex:
    # (az + b) / (cz + d)
    return (map_[0, 0] * z + map_[0, 1]) / (map_[1, 0] * z + map_[1, 1])


def dthz(map_: np.ndarray, z: np.ndarray | complex) -> np.ndarray | complex:
    return 1.0 / (map_[1, 0] * z + map_[1, 1]) ** 2

def d2thz(map_: np.ndarray, z: np.ndarray | complex) -> np.ndarray | complex:
    return -2 * map_[1, 0] / (map_[1, 0] * z + map_[1, 1]) ** 3


def is_inside_domain(z, qv, dv):
    """
    多重連結領域 D の中に点 z が存在するか判定
    - qv: 各円の半径 (array-like)
    - dv: 各円の中心 (array-like)
    - z: 判定したい複素数または配列

    D = 単位円から重円を切り抜いた領域

    return: True (in D), False (out of D)
    """
    # 単位円
    zabs = np.abs(z)
    if np.ndim(zabs) == 0:
        # スカラーの場合
        inside_main = zabs < 1
        for q, d in zip(qv, dv):
            if np.abs(z - d) < q:
                return False
        return inside_main
    else:
        # 配列の場合
        z = np.asarray(z)
        mask = (np.abs(z) < 1)
        for q, d in zip(qv, dv):
            mask = mask & (np.abs(z - d) > q)
        return mask


def build_theta_mlab(qv, dv):
    theta_mlab: dict[int, np.ndarray] = {}
    m = len(qv)
    for k in range(1, m + 1):
        kk = k - 1
        thb = np.array(
            [
                [qv[kk] ** 2 - np.abs(dv[kk]) ** 2, dv[kk]],
                [-np.conj(dv[kk]), 1.0],
            ],
            dtype=np.complex128,
        ) / qv[kk]
        theta_mlab[k] = thb
        theta_mlab[2 * m - (k - 1)] = np.array(
            [[thb[1, 1], -thb[0, 1]], [-thb[1, 0], thb[0, 0]]],
            dtype=np.complex128,
        )
    return theta_mlab


def _generator_inverse_index(j: int, m: int) -> int:
    """theta_dict index of the inverse of generator j (1-based, j in 1..2m)."""
    return 2 * m - j + 1


def _mp_delta_for_generator(j: int, m: int) -> np.ndarray:
    """Net exponent contribution of generator j to the length-m mp vector."""
    delta = np.zeros((m,), dtype=float)
    if 1 <= j <= m:
        delta[j - 1] = 1.0
    else:
        # inverse of kk where j == 2*m - (kk - 1)  =>  kk = 2*m - j + 1
        kk = 2 * m - j + 1
        delta[kk - 1] = -1.0
    return delta


def _map_key(map_vec: np.ndarray, ndigits: int = 8) -> tuple:
    """Hashable key for Möbius matrix columns (tolerance ~ 1e-8)."""
    return tuple(np.round(map_vec.real, ndigits)) + tuple(np.round(map_vec.imag, ndigits))


def constmaps_alp(level: int, theta_dict: dict[int, np.ndarray], m: int):
    """[maps, mp, first_idx] = constmaps_alp(level, THETA, M).

    first_idx[j] is the theta_dict index applied first when building column j.

    Words are enumerated as reduced Schottky words: when appending a generator
    θ_j after a word ending in θ_last, skip j = last^{-1}.  This replaces the
    full g^{level} product tree by g(g-1)^{level-1} candidates per length.
    """
    g = 2 * m
    tol = 1e-8
    id_vec = np.array([1, 0, 0, 1], dtype=np.complex128)

    p_cols: list[np.ndarray] = [None] * g  # type: ignore[list-item]
    mp_cols: list[np.ndarray] = [None] * g  # type: ignore[list-item]
    first_list: list[int] = [0] * g

    # Frontier of reduced words at the previous length:
    # (map 2x2, mp vector, first_idx, last_idx)
    frontier: list[tuple[np.ndarray, np.ndarray, int, int]] = []

    # Same column layout as the original MATLAB/Python code:
    # [θ_1,...,θ_m, θ_m^{-1},...,θ_1^{-1}]
    for k in range(1, m + 1):
        tt = theta_dict[k]
        inv_tt = np.array(
            [[tt[1, 1], -tt[0, 1]], [-tt[1, 0], tt[0, 0]]], dtype=np.complex128
        )
        inv_idx = 2 * m - (k - 1)
        mp_k = _mp_delta_for_generator(k, m)
        mp_inv = _mp_delta_for_generator(inv_idx, m)

        p_cols[k - 1] = tt.reshape(-1, order="F").ravel()
        mp_cols[k - 1] = mp_k
        first_list[k - 1] = k

        p_cols[2 * m - k] = inv_tt.reshape(-1, order="F").ravel()
        mp_cols[2 * m - k] = mp_inv
        first_list[2 * m - k] = inv_idx

    seen = {_map_key(p_cols[j]) for j in range(g)}

    for j in range(g):
        mat2 = p_cols[j].reshape((2, 2), order="F")
        frontier.append((mat2.copy(), mp_cols[j].copy(), first_list[j], first_list[j]))

    for _lev in range(2, level + 1):
        new_frontier: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        for map_mat, mp_vec, first_idx, last_idx in frontier:
            inv_last = _generator_inverse_index(last_idx, m)
            for j in range(1, g + 1):
                if j == inv_last:
                    continue
                map_cand = map_mat @ theta_dict[j]
                map_vec = map_cand.reshape(-1, order="F").ravel()
                mp_new = mp_vec + _mp_delta_for_generator(j, m)
                new_frontier.append((map_cand, mp_new, first_idx, j))

                if np.sum(np.abs(id_vec - map_vec) > tol) == 0:
                    continue
                key = _map_key(map_vec)
                if key in seen:
                    continue
                seen.add(key)
                p_cols.append(map_vec)
                mp_cols.append(mp_new.copy())
                first_list.append(first_idx)

        frontier = new_frontier

    p = np.column_stack(p_cols)
    mp_i = np.column_stack(mp_cols)
    first_idx_arr = np.asarray(first_list, dtype=int)
    if first_idx_arr.shape[0] != p.shape[1]:
        raise RuntimeError(
            f"first_applied_map_indices length {first_idx_arr.shape[0]} != maps columns {p.shape[1]}"
        )
    return p, mp_i, first_idx_arr


def fixed_points_of_map_vec(matvec: np.ndarray) -> np.ndarray:
    """Fixed points of Möbius map given as MATLAB column-major [a,c,b,d]."""
    a, c, b, d = matvec[0], matvec[1], matvec[2], matvec[3]
    A = c
    B = d - a
    C = -b
    if np.abs(A) < 1e-14:
        if np.abs(B) < 1e-14:
            return np.array([np.nan + 0j, np.nan + 0j], dtype=np.complex128)
        z = C / B
        return np.array([z, z], dtype=np.complex128)
    disc = B * B - 4 * A * C
    sqrt_disc = np.sqrt(disc)
    z1 = (-B + sqrt_disc) / (2 * A)
    z2 = (-B - sqrt_disc) / (2 * A)
    return np.array([z1, z2], dtype=np.complex128)


def select_fixed_point(last_app_idx, fixed_points, m):
    """Choose z*_θ from a fixed-point pair (paper convention).

    Outermost generator θ_m → outside the unit disk; θ_m^{-1} → inside.
    This is the repelling fixed point of the corresponding Schottky map and
    makes the infinite-product factors tend to 1.
    """
    abs0 = np.abs(fixed_points[0])
    abs1 = np.abs(fixed_points[1])
    in_unit0 = abs0 < 1
    in_unit1 = abs1 < 1

    if last_app_idx > m:
        # θ_m^{-1}: prefer inside
        if in_unit0 and not in_unit1:
            return fixed_points[0]
        if in_unit1 and not in_unit0:
            return fixed_points[1]
        idx = 0 if abs0 < abs1 else 1
        return fixed_points[idx]
    # θ_m: prefer outside
    if not in_unit0 and in_unit1:
        return fixed_points[0]
    if not in_unit1 and in_unit0:
        return fixed_points[1]
    idx = 0 if abs0 > abs1 else 1
    return fixed_points[idx]


def build_selected_fps(maps: np.ndarray, first_app: np.ndarray, m: int) -> np.ndarray:
    """Build selected_fps used by xhat_alpha_inf.

    For m==2, keeps the notebook hard-coded choice on generator fixed points.
    For m!=2, uses select_fixed_point on the outermost (first-applied) generator:
    outside for θ_m, inside for θ_m^{-1}.
    """
    gen_fps = np.stack(
        [fixed_points_of_map_vec(maps[:, j]) for j in range(2 * m)]
    )
    for j in range(2 * m):
        if gen_fps[j, 0] == 0 and gen_fps[j, 1] == 0:
            gen_fps[j, 1] = 1e8

    selected: list[complex] = []
    if m == 2:
        cand = gen_fps[0:m, :]
        for j in range(maps.shape[1]):
            fa = int(first_app[j])
            if fa == 1:
                selected.append(cand[0, 1])
            elif fa == 2:
                selected.append(cand[1, 1])
            elif fa == 3:
                selected.append(cand[1, 0])
            elif fa == 4:
                selected.append(cand[0, 0])
            else:
                raise ValueError(f"unexpected first_app={fa} for m=2")
    else:
        for j in range(maps.shape[1]):
            fa = int(first_app[j])
            selected.append(select_fixed_point(fa, gen_fps[fa - 1], m))
    return np.asarray(selected, dtype=np.complex128)


def xhat_alpha_inf(z, a, maps, mp, alpha, selected_fps):
    l = maps.shape[1]
    xh = np.ones_like(z, dtype=np.complex128)
    powp = np.exp(2j * alpha * mp)
    powm = np.exp(-2j * alpha * mp)
    for j in range(l):
        map_ = vec_to_map(maps[:, j])
        th_zz = thz(map_, z)
        th_za = thz(map_, a)
        fp = selected_fps[j]
        xh = xh * (
            ((th_zz - a) / (th_za - a) * (th_za - fp) / (th_zz - fp)) ** np.prod(powm[:, j])
            * ((th_za - z) / (th_zz - z) * (th_zz - fp) / (th_za - fp)) ** np.prod(powp[:, j])
        )
    return xh


def dlogXda(z, a, maps, mp, alpha, selected_fps):
    l = maps.shape[1]
    dlogxda = -2 / (z - a)
    powp = np.exp(2j * alpha * mp)
    powm = np.exp(-2j * alpha * mp)
    for j in range(l):
        map_ = vec_to_map(maps[:, j])
        th_zz = thz(map_, z)
        th_za = thz(map_, a)
        dth_da = dthz(map_, a)
        fp = selected_fps[j]
        dlogxda = (
            dlogxda
            + (-1 / (th_zz - a) - (dth_da - 1) / (th_za - a) + dth_da / (th_za - fp))
            * np.prod(powm[:, j])
            + (dth_da / (th_za - z) - dth_da / (th_za - fp)) * np.prod(powp[:, j])
        )
    return dlogxda

def d2logXdadz(z, a, maps, mp, alpha, selected_fps):
    # Omega*2 = d2logXdadz
    l = maps.shape[1]
    d2logxdaz = 2 / (z - a)**2
    powp = np.exp(2j * alpha * mp)
    powm = np.exp(-2j * alpha * mp)
    for j in range(l):
        map_ = vec_to_map(maps[:, j])
        th_zz = thz(map_, z)
        th_za = thz(map_, a)
        dth_da = dthz(map_, a)
        dth_dz = dthz(map_, z)
        fp = selected_fps[j]
        d2logxdaz = (
            d2logxdaz
            + (dth_dz* np.prod(powm[:, j]) / (th_zz - a)**2)
            + (dth_da* np.prod(powp[:, j]) / (th_za - z)**2)
        )   
    return d2logxdaz

def d3logXdadz2(z, a, maps, mp, alpha, selected_fps):
    """logX(z, a) の a で1回・z で2回の混合微分 ∂³logX/∂a∂z².

    d2logXdadz(z,a) を z で解析的に微分したもの。
    """
    l = maps.shape[1]
    d3 = -4.0 / (z - a) ** 3
    powp = np.exp(2j * alpha * mp)
    powm = np.exp(-2j * alpha * mp)
    for j in range(l):
        map_ = vec_to_map(maps[:, j])
        t = thz(map_, z)          # θ(z)
        tp = dthz(map_, z)        # θ'(z)
        tpp = d2thz(map_, z)      # θ''(z)
        ca = thz(map_, a)         # θ(a): z に非依存
        dth_da = dthz(map_, a)    # θ'(a): z に非依存
        d3 = (
            d3
            + (tpp * (t - a) - 2.0 * tp ** 2) / (t - a) ** 3 * np.prod(powm[:, j])
            + 2.0 * dth_da / (ca - z) ** 3 * np.prod(powp[:, j])
        )
    return d3


def dlogXdzh(z, a, maps, mp, alpha, selected_fps):
    l = maps.shape[1]
    dlogxdz = 2 / (z - a)
    powp = np.exp(2j * alpha * mp)
    powm = np.exp(-2j * alpha * mp)
    for j in range(l):
        map_ = vec_to_map(maps[:, j])
        th_zz = thz(map_, z)
        th_za = thz(map_, a)
        dth_da = dthz(map_, a)
        dth_dz = dthz(map_, z)
        fp = selected_fps[j]
        dlogxdz = (
            dlogxdz
            + (-1 / (th_zz - a) - (dth_da - 1) / (th_za - a) + dth_da / (th_za - fp))
            * np.prod(powm[:, j])
            + (dth_da / (th_za - z) - dth_da / (th_za - fp)) * np.prod(powp[:, j])
        )
    return dlogxdz


def dlogXdz(z, a, maps, mp, alpha, selected_fps):
    return dlogXda(a, z, maps, mp, -alpha, selected_fps)


def d2logXdz(z, a, maps, mp, alpha, selected_fps):
    """logX(z, a) の z に関する2階微分 ∂²logX/∂z².

    dlogXdz(z,a) = dlogXda(a, z, -alpha) を z で解析的に微分したもの。
    """
    l = maps.shape[1]
    powp = np.exp(2j * alpha * mp)
    powm = np.exp(-2j * alpha * mp)
    d2 = -2.0 / (z - a) ** 2
    for j in range(l):
        map_ = vec_to_map(maps[:, j])
        t = thz(map_, z)          # θ(z)
        tp = dthz(map_, z)        # θ'(z)
        tpp = d2thz(map_, z)      # θ''(z)
        ca = thz(map_, a)         # θ(a): z に依存しない定数
        fp = selected_fps[j]
        ap = (
            -1.0 / (ca - z) ** 2
            - (tpp * (t - z) - (tp - 1.0) ** 2) / (t - z) ** 2
            + (tpp * (t - fp) - tp ** 2) / (t - fp) ** 2
        )
        bp = (
            (tpp * (t - a) - tp ** 2) / (t - a) ** 2
            - (tpp * (t - fp) - tp ** 2) / (t - fp) ** 2
        )
        d2 = d2 + ap * np.prod(powp[:, j]) + bp * np.prod(powm[:, j])
    return d2


def psiz_inf(z, m, maps, mp, alpha, a_pt):
    l = maps.shape[1]
    thetam_map = vec_to_map(maps[:, m])
    psiz = (1 / (2j * pi)) * np.log(
        (z - 1 / conj(a_pt)) / (z - thz(thetam_map, 1 / conj(a_pt)))
    )
    powm = np.exp(-2j * alpha * mp)
    for k in range(l):
        map_ = vec_to_map(maps[:, k])
        psiz = psiz + (1 / (2j * pi)) * prod(powm[:, k]) * np.log(
            (thz(map_, z) - 1 / conj(a_pt))
            / (thz(map_, z) - thz(thetam_map, 1 / conj(a_pt)))
        )
    return psiz


def valpha_m(a, maps, mp, alpha, selected_fps, m, z_ref=None):
    if z_ref is None:
        z_ref = 1 / conj(0.5 + 0.4j)
    l = maps.shape[1]
    thm = vec_to_map(maps[:, m])
    powm = np.exp(-2j * alpha * mp)
    vja = 1 / (2 * pi * 1j) * log((a - thz(thm, z_ref)) / (a - z_ref))
    for j in range(l):
        map_ = vec_to_map(maps[:, j])
        th_zz = thz(map_, z_ref)
        th_zm = thz(map_, thz(thm, z_ref))
        fp = selected_fps[j]
        vja = vja + 1 / (2 * pi * 1j) * np.prod(powm[:, j]) * log(
            (a - th_zm) / (a - th_zz) * (fp - th_zz) / (fp - th_zm)
        )
    return vja
