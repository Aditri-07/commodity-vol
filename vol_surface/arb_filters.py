"""
Arbitrage flagging for a calibrated VolSurface. DETECTION ONLY -- no
auto-repair; a violation is reported for a human (or a later project phase)
to decide what to do about.

Two checks:

  * Butterfly (per slice): the risk-neutral density implied by the smile
    must be non-negative everywhere. Checked numerically via the discrete
    second derivative of the Black-76 call price w.r.t. strike
    (Breeden-Litzenberger: d2C/dK2 proportional to the density) on a fine
    strike grid. For SVI slices specifically, this is cross-checked against
    the closed-form Gatheral-Jacquier necessary condition on the fitted
    parameters, since a numerical second derivative can be noisy right at
    a domain boundary while the closed form is exact.

  * Calendar (across slices): total variance w(k, T) = iv(k,T)^2 * T must
    be non-decreasing in T at fixed log-moneyness k. Checked on a shared
    k-grid across every consecutive pair of calibrated maturities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from pricing.black76 import price as black76_price
from vol_surface.smile_models.svi import SVISmile


@dataclass
class ArbReport:
    butterfly_violations: list = field(default_factory=list)  # (expiry, K, density)
    calendar_violations: list = field(default_factory=list)   # (k, T_lo, T_hi, w_lo, w_hi)
    svi_closed_form_flags: list = field(default_factory=list)  # (expiry,) slices failing Gatheral-Jacquier

    @property
    def clean(self) -> bool:
        return not self.butterfly_violations and not self.calendar_violations and not self.svi_closed_form_flags

    def summary(self) -> str:
        lines = [
            f"Butterfly violations : {len(self.butterfly_violations)}",
            f"Calendar violations  : {len(self.calendar_violations)}",
        ]
        if self.svi_closed_form_flags:
            lines.append(f"SVI closed-form flags: {len(self.svi_closed_form_flags)} slice(s): "
                         f"{[e.isoformat() if isinstance(e, date) else e for e in self.svi_closed_form_flags]}")
        return "\n".join(lines)


def check_butterfly(surface, n_grid: int = 61, k_width: float = 0.6,
                     dK_frac: float = 1e-3, tol: float = -1e-6) -> list:
    """
    Numerical Breeden-Litzenberger check per slice. Returns a list of
    (expiry, K, density) tuples for grid points where the implied density
    goes meaningfully negative (below `tol`, to allow for finite-difference
    noise rather than flagging on floating-point dust).
    """
    violations = []
    for s in surface.slices:
        F, T = s.F, s.T
        k_grid = np.linspace(-k_width, k_width, n_grid)
        K_grid = F * np.exp(k_grid)

        dK = np.maximum(K_grid * dK_frac, 1e-6)
        K_dn = np.maximum(K_grid - dK, 1e-6)
        K_up = K_grid + dK

        iv_mid = s.model.iv_at_strike(K_grid)
        iv_dn = s.model.iv_at_strike(K_dn)
        iv_up = s.model.iv_at_strike(K_up)

        C_mid = black76_price(F, K_grid, T, iv_mid, surface.r, True)
        C_dn = black76_price(F, K_dn, T, iv_dn, surface.r, True)
        C_up = black76_price(F, K_up, T, iv_up, surface.r, True)

        density = (C_up - 2.0 * C_mid + C_dn) / dK ** 2
        bad = density < tol
        for K_bad, d_bad in zip(K_grid[bad], density[bad]):
            violations.append((s.expiry, float(K_bad), float(d_bad)))
    return violations


def check_svi_closed_form(surface) -> list:
    """Gatheral-Jacquier necessary condition, checked directly on SVI slices' fitted params."""
    flags = []
    for s in surface.slices:
        if isinstance(s.model, SVISmile) and not s.model.butterfly_arbitrage_free():
            flags.append(s.expiry)
    return flags


def check_calendar(surface, n_grid: int = 41, k_width: float = 0.5, tol: float = -1e-8) -> list:
    """Total variance must be non-decreasing in T at fixed log-moneyness, slice to slice."""
    violations = []
    if len(surface.slices) < 2:
        return violations
    k_grid = np.linspace(-k_width, k_width, n_grid)
    for lo, hi in zip(surface.slices[:-1], surface.slices[1:]):
        iv_lo = lo.model.iv(k_grid)
        iv_hi = hi.model.iv(k_grid)
        w_lo = iv_lo ** 2 * lo.T
        w_hi = iv_hi ** 2 * hi.T
        bad = (w_hi - w_lo) < tol
        for kk, wl, wh in zip(k_grid[bad], w_lo[bad], w_hi[bad]):
            violations.append((float(kk), lo.T, hi.T, float(wl), float(wh)))
    return violations


def check_surface(surface) -> ArbReport:
    """Run all flag-only checks against an already-calibrated VolSurface."""
    return ArbReport(
        butterfly_violations=check_butterfly(surface),
        calendar_violations=check_calendar(surface),
        svi_closed_form_flags=check_svi_closed_form(surface),
    )


if __name__ == "__main__":
    from data.adapters.synthetic import SyntheticAdapter
    from data.schema import Commodity
    from vol_surface.surface import VolSurface

    adapter = SyntheticAdapter(inject_noise=False)
    snap = adapter.fetch([Commodity.WTI, Commodity.HENRY_HUB])

    for commodity in (Commodity.WTI, Commodity.HENRY_HUB):
        for model_name in ("quadratic", "sabr", "svi", "spline"):
            surf = VolSurface(commodity, model=model_name).calibrate(snap)
            report = check_surface(surf)
            print(f"{commodity.value:10s} [{model_name:10s}] clean={report.clean}  "
                  f"butterfly={len(report.butterfly_violations)}  "
                  f"calendar={len(report.calendar_violations)}  "
                  f"svi_flags={len(report.svi_closed_form_flags)}")
