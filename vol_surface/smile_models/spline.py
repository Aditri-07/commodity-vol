"""
Monotone cubic (PCHIP) interpolation directly on (log-moneyness, IV) points
-- a deliberately model-free baseline.

PCHIP (not a plain natural cubic spline) is used because it's shape-
preserving: with only 7-11 strikes per maturity, a naive cubic spline can
overshoot/oscillate between points (Runge-phenomenon-style wiggle),
especially at the wings where quotes are sparsest.

Unlike SABR/SVI, this has NO structural arbitrage-free guarantee -- nothing
stops it from curving the wrong way in the second derivative between two
widely spaced strikes. That's intentional: it's the natural "this can fail
the butterfly check" case for exercising arb_filters.py, in contrast to
SVI's closed-form guarantee and SABR's generally well-behaved asymptotics.

Extrapolation beyond the quoted strike range is flat (clipped to the
nearest quoted log-moneyness), since a spline has no principled basis for
extrapolation the way a parametric model's asymptotics do.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from .base import SmileModel


class SplineSmile(SmileModel):
    def __init__(self, T: float, F: float):
        self.T = T
        self.F = F
        self.params: dict = {}
        self._interp: PchipInterpolator | None = None
        self._k_min: float = 0.0
        self._k_max: float = 0.0

    def fit(self, k, iv, weights=None) -> "SplineSmile":
        # weights are accepted for interface parity but PCHIP is a pure
        # interpolant (passes through every point) and has no weighting
        # concept -- duplicate k's are collapsed to their first IV instead.
        k = np.asarray(k, dtype=float)
        iv = np.asarray(iv, dtype=float)
        order = np.argsort(k)
        k_sorted, iv_sorted = k[order], iv[order]
        k_unique, first_idx = np.unique(k_sorted, return_index=True)

        self._interp = PchipInterpolator(k_unique, iv_sorted[first_idx], extrapolate=False)
        self._k_min, self._k_max = float(k_unique[0]), float(k_unique[-1])
        self.params = {"n_points": int(len(k_unique))}
        return self

    def iv(self, k):
        k = np.asarray(k, dtype=float)
        k_clipped = np.clip(k, self._k_min, self._k_max)
        return np.asarray(self._interp(k_clipped), dtype=float)
