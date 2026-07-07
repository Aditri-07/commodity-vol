"""
Raw SVI (Gatheral 2004), fit in total-variance space:

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

where k = ln(K/F) is log-moneyness and w = sigma_BS(k)^2 * T is total
implied variance.

Chosen alongside SABR as the other industry-standard smile parametrization
(dominant in equity index vol, also used in commodities). The main reason
to keep it in this project: Gatheral & Jacquier (2014) give closed-form
NECESSARY conditions for a single slice to be free of butterfly arbitrage,
directly on the fitted parameters --

    b >= 0,  |rho| < 1,  sigma > 0,  a + b*sigma*sqrt(1-rho^2) >= 0

-- so arb_filters.py can cross-check SVI slices two ways (this closed-form
condition AND the numerical Breeden-Litzenberger check used for every
model), which is a useful validation of the numerical check itself.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .base import SmileModel


class SVISmile(SmileModel):
    def __init__(self, T: float, F: float):
        self.T = T
        self.F = F
        self.params: dict = {}

    def fit(self, k, iv, weights=None) -> "SVISmile":
        k = np.asarray(k, dtype=float)
        iv = np.asarray(iv, dtype=float)
        w_mkt = iv ** 2 * self.T
        w = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)

        a0 = max(float(np.min(w_mkt)), 1e-6)
        b0 = 0.1
        rho0 = -0.1
        m0 = float(k[np.argmin(np.abs(k))])
        sigma0 = max(float(np.std(k)), 0.05)

        def model_w(x, k):
            a, b, rho, m, sigma = x
            return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

        def resid(x):
            return (model_w(x, k) - w_mkt) * np.sqrt(w)

        x0 = np.array([a0, b0, rho0, m0, sigma0])
        bounds = ([-5.0, 1e-6, -0.999, -3.0, 1e-4], [5.0, 5.0, 0.999, 3.0, 5.0])
        result = least_squares(resid, x0, bounds=bounds, max_nfev=3000)
        a, b, rho, m, sigma = result.x
        rmse_w = float(np.sqrt(np.mean(result.fun ** 2))) if len(result.fun) else float("nan")
        self.params = {
            "a": float(a), "b": float(b), "rho": float(rho),
            "m": float(m), "sigma": float(sigma), "rmse_w": rmse_w,
        }
        return self

    def total_variance(self, k):
        k = np.asarray(k, dtype=float)
        a, b, rho, m, sigma = (self.params[key] for key in ("a", "b", "rho", "m", "sigma"))
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def iv(self, k):
        w = np.maximum(self.total_variance(k), 1e-10)
        return np.sqrt(w / self.T)

    def butterfly_arbitrage_free(self) -> bool:
        """Gatheral-Jacquier (2014) closed-form necessary condition for this slice."""
        a, b, rho, sigma = (self.params[key] for key in ("a", "b", "rho", "sigma"))
        return bool(b >= 0 and abs(rho) < 1.0 and sigma > 0 and a + b * sigma * np.sqrt(1 - rho ** 2) >= -1e-8)


if __name__ == "__main__":
    F, T = 74.0, 0.5
    k = np.linspace(-0.4, 0.4, 9)
    true_params = dict(a=0.02, b=0.15, rho=-0.3, m=0.02, sigma=0.15)
    w_true = true_params["a"] + true_params["b"] * (
        true_params["rho"] * (k - true_params["m"])
        + np.sqrt((k - true_params["m"]) ** 2 + true_params["sigma"] ** 2)
    )
    iv_true = np.sqrt(w_true / T)

    fitted = SVISmile(T, F).fit(k, iv_true)
    err = np.abs(fitted.iv(k) - iv_true)
    print("SVI self-consistency fit:")
    print(f"  params: {fitted.params}")
    print(f"  max abs vol error vs synthetic truth: {err.max():.2e}")
    print(f"  butterfly-arb-free (closed form): {fitted.butterfly_arbitrage_free()}")
