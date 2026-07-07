"""
SABR (Hagan, Kumar, Lesniewski, Woodward 2002), targeting the Black-76
lognormal implied vol via Hagan's asymptotic formula.

beta is a CHOICE, not fit, because it is underdetermined jointly with alpha
from a single smile snapshot -- fixing beta and calibrating (alpha, rho, nu)
by least squares is standard market practice:

    beta = 1    lognormal backbone  (closest to Black-76's own assumption)
    beta = 0.5  CIR-like backbone   (common commodities default)
    beta = 0    normal backbone     (closest to Bachelier) -- relevant for
                                     forwards stressed toward/through zero

This project defaults beta=0.5 for WTI and beta=0.7 for Henry Hub (crude's
2020 collapse showed the backbone can behave more normal-like under stress;
gas retains more lognormal character even in its vol spikes), but callers
can override per-commodity in VolSurface.

Shifted SABR (F -> F+shift, K -> K+shift) is supported so the same lognormal
formula stays usable when the forward is near zero, at the cost of
introducing an extra free parameter. For forwards that go materially
negative (e.g. WTI April 2020), prefer the Bachelier module directly rather
than pushing the shift very large -- shifted-lognormal SABR is a patch, not
a substitute for a genuinely normal model in that regime.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .base import SmileModel


def hagan_lognormal_vol(F, K, T, alpha, beta, rho, nu, shift: float = 0.0):
    """
    Hagan et al.'s asymptotic lognormal-vol approximation. Vectorized over K
    (F, T, alpha, beta, rho, nu, shift are scalars for a single slice fit).
    """
    F_s = float(F) + shift
    K_s = np.asarray(K, dtype=float) + shift
    T = float(T)

    F_s = max(F_s, 1e-8)
    K_s = np.maximum(K_s, 1e-8)

    logFK = np.log(F_s / K_s)
    FK_beta = (F_s * K_s) ** ((1.0 - beta) / 2.0)

    atm = np.isclose(F_s, K_s, rtol=1e-10, atol=1e-10)

    z = (nu / alpha) * FK_beta * logFK
    with np.errstate(invalid="ignore", divide="ignore"):
        xz = np.log((np.sqrt(1.0 - 2.0 * rho * z + z ** 2) + z - rho) / (1.0 - rho))
        zx = np.where(np.abs(z) < 1e-12, 1.0, z / xz)

    A = alpha / (
        FK_beta
        * (1.0 + (1.0 - beta) ** 2 / 24.0 * logFK ** 2 + (1.0 - beta) ** 4 / 1920.0 * logFK ** 4)
    )
    B = 1.0 + T * (
        (1.0 - beta) ** 2 / 24.0 * alpha ** 2 / (F_s * K_s) ** (1.0 - beta)
        + rho * beta * nu * alpha / (4.0 * FK_beta)
        + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    )
    vol_general = A * zx * B

    vol_atm = (alpha / F_s ** (1.0 - beta)) * (
        1.0
        + T
        * (
            (1.0 - beta) ** 2 / 24.0 * alpha ** 2 / F_s ** (2.0 - 2.0 * beta)
            + rho * beta * nu * alpha / (4.0 * F_s ** (1.0 - beta))
            + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
        )
    )
    return np.where(atm, vol_atm, vol_general)


class SABRSmile(SmileModel):
    def __init__(self, T: float, F: float, beta: float = 0.5, shift: float = 0.0):
        self.T = T
        self.F = F
        self.beta = beta
        self.shift = shift
        self.params: dict = {}

    def fit(self, k, iv, weights=None) -> "SABRSmile":
        k = np.asarray(k, dtype=float)
        iv = np.asarray(iv, dtype=float)
        K = self.F * np.exp(k)
        w = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)

        atm_iv_guess = float(np.interp(0.0, k, iv))
        alpha0 = max(atm_iv_guess * (self.F + self.shift) ** (1.0 - self.beta), 1e-4)

        def resid(x):
            alpha, rho, nu = x
            model_iv = hagan_lognormal_vol(self.F, K, self.T, alpha, self.beta, rho, nu, self.shift)
            return (model_iv - iv) * np.sqrt(w)

        x0 = np.array([alpha0, -0.1, 0.5])
        bounds = ([1e-6, -0.999, 1e-6], [10.0, 0.999, 5.0])
        result = least_squares(resid, x0, bounds=bounds, max_nfev=3000)
        alpha, rho, nu = result.x
        rmse = float(np.sqrt(np.mean(result.fun ** 2))) if len(result.fun) else float("nan")
        self.params = {
            "alpha": float(alpha), "beta": self.beta, "rho": float(rho),
            "nu": float(nu), "shift": self.shift, "rmse": rmse,
        }
        return self

    def iv(self, k):
        k = np.asarray(k, dtype=float)
        K = self.F * np.exp(k)
        return hagan_lognormal_vol(
            self.F, K, self.T, self.params["alpha"], self.beta,
            self.params["rho"], self.params["nu"], self.shift,
        )


if __name__ == "__main__":
    # Fit SABR to a hand-built smile and check it recovers a sane shape.
    rng = np.random.default_rng(0)
    F, T = 74.0, 0.5
    true = dict(alpha=6.0, beta=0.5, rho=-0.25, nu=0.8)
    k = np.linspace(-0.4, 0.4, 9)
    iv_true = hagan_lognormal_vol(F, F * np.exp(k), T, **true)

    fitted = SABRSmile(T, F, beta=0.5).fit(k, iv_true)
    err = np.abs(fitted.iv(k) - iv_true)
    print("SABR self-consistency fit:")
    print(f"  params: {fitted.params}")
    print(f"  max abs vol error vs synthetic truth: {err.max():.2e}")
