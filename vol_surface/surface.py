"""
VolSurface: end-to-end from a QuoteSnapshot to a queryable iv(K, T) surface.

Pipeline per commodity:
  1. Invert quoted mids to Black-76 IV via pricing.iv_solver, one maturity
     at a time (a bad quote in one slice shouldn't poison another).
  2. Fit the chosen smile model independently to each maturity slice.
  3. Interpolate ACROSS maturities in total-variance space
     (w = iv^2 * T, linear in T at fixed log-moneyness k) -- the standard
     way to interpolate a term structure of smiles without introducing
     calendar arbitrage AT THE INTERPOLATION NODES themselves. The fitted
     slices can still individually or jointly violate no-arb; that's what
     vol_surface.arb_filters checks, separately and explicitly.
  4. Flat-extrapolate in T beyond the first/last quoted maturity (hold the
     nearest slice's fitted shape), since extrapolating a parametric smile
     beyond its calibration window has no principled basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from data.schema import Commodity, OptionType, QuoteSnapshot, year_fraction
from pricing.iv_solver import implied_vol_surface
from vol_surface.smile_models.quadratic import QuadraticSmile
from vol_surface.smile_models.sabr import SABRSmile
from vol_surface.smile_models.svi import SVISmile
from vol_surface.smile_models.spline import SplineSmile

_MODEL_REGISTRY = {
    "quadratic": QuadraticSmile,
    "sabr": SABRSmile,
    "svi": SVISmile,
    "spline": SplineSmile,
}

# Default SABR backbone per commodity -- see smile_models/sabr.py docstring
# for the reasoning (crude's stress behavior skews more normal, gas stays
# more lognormal even in its vol spikes).
_DEFAULT_SABR_BETA = {
    Commodity.WTI: 0.5,
    Commodity.HENRY_HUB: 0.7,
}


@dataclass
class MaturitySlice:
    T: float
    F: float
    expiry: date
    k: np.ndarray            # log-moneyness of the quoted points that inverted cleanly
    iv_market: np.ndarray    # their Black-76 IVs
    n_dropped: int           # quotes in this slice that failed inversion (bad price/no-arb bound)
    model: object            # fitted SmileModel


class VolSurface:
    def __init__(self, commodity: Commodity, model: str = "sabr", sabr_beta: float | None = None):
        if model not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown smile model '{model}'; choices: {sorted(_MODEL_REGISTRY)}")
        self.commodity = commodity
        self.model_name = model
        self.sabr_beta = sabr_beta
        self.slices: list[MaturitySlice] = []
        self.r: float = 0.045
        self.as_of: date | None = None

    def calibrate(self, snap: QuoteSnapshot) -> "VolSurface":
        self.as_of = snap.as_of
        self.r = float(snap.meta.get("r", 0.045))
        df = snap.for_commodity(self.commodity)
        if df.empty:
            raise ValueError(f"No quotes for {self.commodity} in this snapshot")

        slices: list[MaturitySlice] = []
        for expiry, sub in df.groupby("expiry"):
            T = year_fraction(self.as_of, expiry)
            if T <= 0:
                continue
            F = float(sub["forward"].iloc[0])
            is_call = (sub["option_type"] == OptionType.CALL).to_numpy()
            T_arr = np.full(len(sub), T)

            iv = implied_vol_surface(
                sub["mid"].to_numpy(), sub["forward"].to_numpy(), sub["strike"].to_numpy(),
                T_arr, self.r, is_call, model="black76",
            )
            ok = ~np.isnan(iv)
            n_dropped = int((~ok).sum())
            if ok.sum() < 3:
                continue  # not enough clean points to fit any of these models

            k = np.log(sub["strike"].to_numpy()[ok] / F)
            iv_ok = iv[ok]

            if self.model_name == "sabr":
                beta = self.sabr_beta if self.sabr_beta is not None else _DEFAULT_SABR_BETA.get(self.commodity, 0.5)
                fitted = SABRSmile(T, F, beta=beta).fit(k, iv_ok)
            else:
                model_cls = _MODEL_REGISTRY[self.model_name]
                fitted = model_cls(T, F).fit(k, iv_ok)

            slices.append(MaturitySlice(T=T, F=F, expiry=expiry, k=k, iv_market=iv_ok,
                                         n_dropped=n_dropped, model=fitted))

        slices.sort(key=lambda s: s.T)
        if not slices:
            raise ValueError(f"No maturity slice for {self.commodity} had >= 3 clean quotes to calibrate")
        self.slices = slices
        return self

    def _bracket(self, T: float) -> tuple[MaturitySlice, MaturitySlice]:
        earlier = [s for s in self.slices if s.T <= T]
        later = [s for s in self.slices if s.T >= T]
        lo = max(earlier, key=lambda s: s.T)
        hi = min(later, key=lambda s: s.T)
        return lo, hi

    def iv(self, K: float, T: float) -> float:
        """Implied vol at an arbitrary strike/maturity, via total-variance
        interpolation across calibrated slices (flat extrapolation past the ends)."""
        if not self.slices:
            raise RuntimeError("Surface not calibrated -- call .calibrate() first")
        K, T = float(K), float(T)

        if T <= self.slices[0].T:
            return float(self.slices[0].model.iv_at_strike(np.array([K]))[0])
        if T >= self.slices[-1].T:
            return float(self.slices[-1].model.iv_at_strike(np.array([K]))[0])

        lo, hi = self._bracket(T)
        if lo.T == hi.T:
            return float(lo.model.iv_at_strike(np.array([K]))[0])

        iv_lo = float(lo.model.iv_at_strike(np.array([K]))[0])
        iv_hi = float(hi.model.iv_at_strike(np.array([K]))[0])
        w_lo, w_hi = iv_lo ** 2 * lo.T, iv_hi ** 2 * hi.T
        weight = (T - lo.T) / (hi.T - lo.T)
        w_T = w_lo + weight * (w_hi - w_lo)
        return float(np.sqrt(max(w_T, 1e-12) / T))

    def summary(self) -> str:
        lines = [f"VolSurface({self.commodity.value}, model={self.model_name})"]
        for s in self.slices:
            extra = f", dropped={s.n_dropped}" if s.n_dropped else ""
            lines.append(f"  T={s.T:.4f}  F={s.F:.4f}  n_pts={len(s.k)}{extra}  params={s.model.params}")
        return "\n".join(lines)


if __name__ == "__main__":
    from data.adapters.synthetic import SyntheticAdapter

    adapter = SyntheticAdapter(inject_noise=False)
    snap = adapter.fetch([Commodity.WTI, Commodity.HENRY_HUB])

    for model_name in ("quadratic", "sabr", "svi", "spline"):
        surf = VolSurface(Commodity.WTI, model=model_name).calibrate(snap)
        atm_T = surf.slices[2].T  # a middle maturity
        atm_iv = surf.iv(surf.slices[2].F, atm_T)
        print(f"[{model_name:10s}] WTI ATM IV at T={atm_T:.3f}: {atm_iv:.4f}")
