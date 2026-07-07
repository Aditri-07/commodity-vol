"""
Quadratic-in-log-moneyness smile: sigma(k) = a + b*k + c*k^2.

The simplest possible parametric fit -- and deliberately the same
functional form data/adapters/synthetic.py:_smile_vol uses to generate the
"true" smile, so this is a fast, cheap sanity check that a calibration
pipeline is wired correctly before trusting SABR/SVI results on the same
data. It carries no structural arbitrage-free guarantee (a fitted parabola
can easily curve the wrong way at the wings if strikes are sparse); that's
what arb_filters.py is for.
"""

from __future__ import annotations

import numpy as np

from .base import SmileModel


class QuadraticSmile(SmileModel):
    def __init__(self, T: float, F: float):
        self.T = T
        self.F = F
        self.params: dict = {}

    def fit(self, k, iv, weights=None) -> "QuadraticSmile":
        k = np.asarray(k, dtype=float)
        iv = np.asarray(iv, dtype=float)
        w = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)

        A = np.column_stack([np.ones_like(k), k, k ** 2])
        wsqrt = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * wsqrt[:, None], iv * wsqrt, rcond=None)
        self.params = {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2])}
        return self

    def iv(self, k):
        k = np.asarray(k, dtype=float)
        a, b, c = self.params["a"], self.params["b"], self.params["c"]
        return np.maximum(a + b * k + c * k ** 2, 1e-4)
