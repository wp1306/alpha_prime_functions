"""MATLAB SKPrime helpers via matlab.engine.

Typical usage (kernel: Python (matlab_engine))::

    from matlab_prime import vj, diff, dvj

    vj.init(qv, dv)
    w = vj.v1(z)
    wp = diff(vj).v1(z)   # MATLAB-like: diff(vj)
    wp = dvj.v1(z)        # same object, direct import
    wp = dvj(1, z)        # general index

Requires the MATLAB Engine API for Python (matlabengine) and an Apple-silicon
MATLAB install matching the engine version.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

DEFAULT_SK_PATH = (
    "/Users/miyoshihiroyuki/Library/Mobile Documents/"
    "com~apple~CloudDocs/1_Journal/sk_function"
)


class _EvalMixin:
    """Shared MATLAB evaluation helpers."""

    eng = None
    m = 0

    def _to_ml_complex(self, z):
        import matlab

        flat = np.asarray(z, dtype=complex).reshape(-1)
        return matlab.double([complex(v) for v in flat], is_complex=True)

    def _from_ml(self, val, shape, scalar):
        out = np.asarray(val, dtype=complex).reshape(shape)
        return complex(out) if scalar else out

    def _require_ready(self):
        if self.eng is None or self.m == 0:
            raise RuntimeError("Call vj.init(qv, dv) first.")

    def _eval_named(self, name: str, z):
        self._require_ready()
        scalar = np.ndim(z) == 0 and not isinstance(z, (list, tuple, np.ndarray))
        z_arr = np.asarray(z, dtype=complex)
        self.eng.workspace["z_py"] = self._to_ml_complex(z_arr)
        w = self.eng.eval(f"{name}(z_py)", nargout=1)
        return self._from_ml(w, z_arr.shape, scalar)


class DVjAPI(_EvalMixin):
    """Derivative handles: MATLAB ``dvj = diff(vj)``."""

    def __init__(self, parent: "VjAPI"):
        self._parent = parent

    @property
    def eng(self):
        return self._parent.eng

    @property
    def m(self):
        return self._parent.m

    def __call__(self, j: int, z):
        return self.v(j, z)

    def v(self, j: int, z):
        """Evaluate (diff v_j)(z) for 1-based boundary index j."""
        j = int(j)
        if j < 1 or j > self.m:
            raise IndexError(f"j must be in 1..{self.m}, got {j}")
        return self._eval_named(f"dv{j}", z)

    def v1(self, z):
        return self.v(1, z)

    def v2(self, z):
        return self.v(2, z)


class VjAPI(_EvalMixin):
    """Session wrapper around MATLAB ``vjFirstKind``."""

    def __init__(self, sk_path: str = DEFAULT_SK_PATH):
        self.sk_path = sk_path
        self.eng = None
        self.qv = None
        self.dv_centers = None  # domain centers (avoid clash with derivative API)
        self.m = 0
        self._diff = DVjAPI(self)

    def init(self, qv, dv, *, sk_path: Optional[str] = None, force: bool = False):
        """Start MATLAB if needed and build vjFirstKind / diff(vj) for each circle."""
        import matlab
        import matlab.engine

        if sk_path is not None:
            self.sk_path = sk_path

        qv_use = np.asarray(qv, dtype=float).ravel()
        dv_use = np.asarray(dv, dtype=complex).ravel()
        if qv_use.size != dv_use.size:
            raise ValueError("qv and dv must have the same length")

        if (
            not force
            and self.eng is not None
            and self.qv is not None
            and self.dv_centers is not None
            and np.array_equal(self.qv, qv_use)
            and np.array_equal(self.dv_centers, dv_use)
        ):
            return self

        if self.eng is None:
            self.eng = matlab.engine.start_matlab()
            self.eng.addpath(self.sk_path, nargout=0)

        eng = self.eng
        eng.workspace["qv_py"] = matlab.double(qv_use.tolist())
        eng.workspace["dv_py"] = matlab.double(
            [complex(v) for v in dv_use], is_complex=True
        )
        eng.eval("D = skpDomain(dv_py, qv_py);", nargout=0)

        self.m = int(qv_use.size)
        for j in range(1, self.m + 1):
            eng.eval(f"v{j} = vjFirstKind({j}, D);", nargout=0)
            eng.eval(f"dv{j} = diff(v{j});", nargout=0)

        self.qv = qv_use.copy()
        self.dv_centers = dv_use.copy()
        return self

    def quit(self):
        """Shut down the MATLAB engine session."""
        if self.eng is not None:
            self.eng.quit()
        self.eng = None
        self.qv = None
        self.dv_centers = None
        self.m = 0

    def diff(self) -> DVjAPI:
        """MATLAB-like ``diff(vj)``; returns derivative callables."""
        return self._diff

    def __call__(self, j: int, z):
        return self.v(j, z)

    def v(self, j: int, z):
        """Evaluate v_j(z) for 1-based boundary index j."""
        j = int(j)
        if j < 1 or j > self.m:
            raise IndexError(f"j must be in 1..{self.m}, got {j}")
        return self._eval_named(f"v{j}", z)

    def v1(self, z):
        return self.v(1, z)

    def v2(self, z):
        return self.v(2, z)

    # Keep short aliases used earlier
    def dv(self, j: int, z):
        return self._diff.v(j, z)

    def dv1(self, z):
        return self._diff.v1(z)

    def dv2(self, z):
        return self._diff.v2(z)


def diff(obj: Union[VjAPI, DVjAPI]) -> DVjAPI:
    """MATLAB-like ``diff(vj)``.

    Prefer ``diff_vj`` if ``maps_core.diff`` is already imported in the notebook::

        from matlab_prime import vj, diff_vj
        vj.init(qv, dv)
        diff_vj(vj).v1(z)
    """
    if isinstance(obj, DVjAPI):
        return obj
    if isinstance(obj, VjAPI):
        return obj.diff()
    raise TypeError(f"diff() expects vj (VjAPI), got {type(obj)!r}")


# Alias that does not collide with maps_core.diff
diff_vj = diff

# Module-level singletons
vj = VjAPI()
dvj = vj._diff

__all__ = [
    "VjAPI",
    "DVjAPI",
    "vj",
    "dvj",
    "diff",
    "diff_vj",
    "DEFAULT_SK_PATH",
]