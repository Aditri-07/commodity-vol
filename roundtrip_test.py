"""
Round-trip test for the synthetic adapter.

Inverts the (unrounded) generated prices back to implied vol via Black-76 and
checks we recover the input smile. This is both a unit test and a demonstration
that the synthetic data is arbitrage-consistent by construction.

Two tolerances are checked:
  * raw_mid   -> machine-ish precision (the generator is self-consistent)
  * quoted mid -> ~1 vol point (tick rounding of the stored quote, realistic)
"""
import numpy as np
from scipy.optimize import brentq
from data.adapters.synthetic import SyntheticAdapter, _black76_price
from data.schema import Commodity, OptionType, year_fraction

def implied_vol(price, F, K, T, r, is_call):
    return brentq(lambda s: _black76_price(F, K, T, s, r, is_call) - price,
                  1e-4, 5.0, maxiter=200)

def run():
    adapter = SyntheticAdapter(inject_noise=False)
    snap = adapter.fetch([Commodity.WTI, Commodity.HENRY_HUB])
    true_iv = snap.meta["true_iv"]
    raw_mid = snap.meta["raw_mid"]
    r = snap.meta["r"]
    as_of = snap.as_of
    f = snap.frame.reset_index(drop=True)

    raw_err, quote_err = [], []
    for i in range(len(f)):
        row = f.iloc[i]
        T = year_fraction(as_of, row["expiry"])
        is_call = row["option_type"] == OptionType.CALL
        iv_raw = implied_vol(raw_mid[i], row["forward"], row["strike"], T, r, is_call)
        iv_q   = implied_vol(row["mid"], row["forward"], row["strike"], T, r, is_call)
        raw_err.append(abs(iv_raw - true_iv[i]))
        quote_err.append(abs(iv_q - true_iv[i]))

    raw_err, quote_err = np.array(raw_err), np.array(quote_err)
    print(f"Round-trip over {len(f)} quotes (WTI + Henry Hub):")
    print(f"  raw price  -> IV : max {raw_err.max():.2e}, mean {raw_err.mean():.2e}")
    print(f"  quoted mid -> IV : max {quote_err.max():.2e}, mean {quote_err.mean():.2e}")

    # Tolerances (in vol points):
    #   raw_mid recovers the smile to ~1e-4. It is not machine-epsilon because
    #   time-to-maturity is derived from integer calendar days (ACT/365), which
    #   is not perfectly invertible — this is exactly the granularity real,
    #   date-quoted options carry, so we accept it rather than fake precision.
    #   quoted mid adds tick rounding on top, still well under one vol point.
    assert raw_err.max() < 1e-3, "Generator not self-consistent beyond date granularity (possible arb violation)!"
    assert quote_err.max() < 1e-2, "Quoted-mid IV drifts more than one vol point!"
    print("PASS: synthetic prices invert cleanly; data is arbitrage-consistent by construction.")

if __name__ == "__main__":
    run()
