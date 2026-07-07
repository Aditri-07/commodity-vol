"""
Common interface for per-maturity smile models.

Every model operates in log-moneyness k = ln(K/F) space and is fit against
Black-76 implied vols already extracted via pricing.iv_solver. Keeping the
interface uniform means surface.py can swap models without caring which one
is active, and arb_filters.py can probe any of them with the same
finite-difference machinery regardless of parametrization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class SmileModel(ABC):
    """A fitted (or fittable) smile for a single maturity slice."""

    T: float    # time to expiry, ACT/365 year fraction
    F: float    # forward/futures price for this maturity
    params: dict  # fitted parameters, kept for inspection/reporting

    @abstractmethod
    def fit(self, k: np.ndarray, iv: np.ndarray, weights: np.ndarray | None = None) -> "SmileModel":
        """Fit to market log-moneyness/implied-vol pairs. Returns self."""
        raise NotImplementedError

    @abstractmethod
    def iv(self, k: np.ndarray) -> np.ndarray:
        """Black-76 (lognormal) implied vol at log-moneyness k. Vectorized."""
        raise NotImplementedError

    def iv_at_strike(self, K: np.ndarray) -> np.ndarray:
        """Convenience: implied vol at strike(s) K, converting to log-moneyness internally."""
        K = np.asarray(K, dtype=float)
        k = np.log(K / self.F)
        return self.iv(k)
