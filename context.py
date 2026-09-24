"""Notebook-compatible global context for alpha-prime evaluations."""

from __future__ import annotations

import numpy as np

from maps_core import (
    build_selected_fps,
    build_theta_mlab,
    constmaps_alp,
    exp,
    pi,
)

# --- mutable context (notebook-style) ---
qv: np.ndarray | None = None
dv: np.ndarray | None = None
maps: np.ndarray | None = None
mp_i: np.ndarray | None = None
alpha: np.ndarray | None = None
selected_fps2: np.ndarray | None = None
first_app: np.ndarray | None = None
M: int | None = None
level: int | None = None
zp: np.ndarray | None = None


def _require_context() -> None:
    if any(
        v is None
        for v in (qv, dv, maps, mp_i, alpha, selected_fps2, M)
    ):
        raise ValueError(
            "context is not initialized; call init_context(...) or set_context(...) first"
        )


def set_context(
    qv_in: np.ndarray,
    dv_in: np.ndarray,
    maps_in: np.ndarray,
    mp_in: np.ndarray,
    alpha_in: np.ndarray,
    selected_fps_in: np.ndarray,
    first_app_in: np.ndarray | None = None,
    level_in: int | None = None,
) -> None:
    """Manually set the global computation context."""
    global qv, dv, maps, mp_i, alpha, selected_fps2, first_app, M, level, zp
    qv = np.asarray(qv_in, dtype=np.complex128)
    dv = np.asarray(dv_in, dtype=np.complex128)
    maps = np.asarray(maps_in, dtype=np.complex128)
    mp_i = np.asarray(mp_in)
    alpha = np.asarray(alpha_in, dtype=float)
    selected_fps2 = np.asarray(selected_fps_in, dtype=np.complex128)
    first_app = None if first_app_in is None else np.asarray(first_app_in, dtype=int)
    M = int(qv.size)
    level = level_in
    zp = unit_circle_samples()


def unit_circle_samples(n: int = 100) -> np.ndarray:
    ang_p = np.linspace(-pi, pi, n)
    return exp(1j * ang_p)


def init_context(
    qv_in: np.ndarray,
    dv_in: np.ndarray,
    alpha_in: np.ndarray,
    level_in: int = 5,
) -> None:
    """Build maps / mp / selected_fps from geometry and set globals."""
    global qv, dv, maps, mp_i, alpha, selected_fps2, first_app, M, level, zp
    qv = np.asarray(qv_in, dtype=np.complex128)
    dv = np.asarray(dv_in, dtype=np.complex128)
    alpha = np.asarray(alpha_in, dtype=float)
    M = int(qv.size)
    level = int(level_in)

    theta_dict = build_theta_mlab(qv, dv)
    maps, mp_i, first_app = constmaps_alp(level, theta_dict, M)
    selected_fps2 = build_selected_fps(maps, first_app, M)
    zp = unit_circle_samples()

