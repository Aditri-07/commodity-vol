# Commodity Options Pricing & Volatility Surface Analytics Engine

A Python engine for pricing commodity **futures options** (WTI crude, Henry Hub
natural gas) with Black-76 and Bachelier models, and building/validating implied
volatility surfaces.

## Data access & honesty note
This project is designed around a **source-agnostic quote schema**. Live vendor
data (Bloomberg BQuant, CME public settlements, yfinance, EIA) plugs into the
same schema, but the **primary runnable path is a synthetic fallback generator**
so the whole pipeline runs end-to-end with zero external access. Synthetic
prices are generated *from* a known volatility surface via Black-76, so they are
arbitrage-consistent by construction — inverting them recovers the input smile
(see `roundtrip_test.py`). This is documented rather than hidden.

## What's built so far (Step 1: data layer)
- `data/schema.py` — canonical quote schema + shared ACT/365 `year_fraction`
  (single day-count convention for the whole project).
- `data/adapters/base.py` — `QuoteAdapter` protocol; downstream code depends
  only on this interface, never on a specific vendor.
- `data/adapters/synthetic.py` — arb-consistent synthetic chain generator in the
  futures-option (Black-76) shape: WTI backwardation + mild put skew, Henry Hub
  contango + call skew, maturity-dependent smile, widening wings, optional noise
  injection to exercise downstream validation/arb filters.
- `roundtrip_test.py` — inverts generated prices back to IV; proves the data is
  arbitrage-consistent.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
Always run from the project root so the `data` package imports resolve.

```bash
# Generate a snapshot and print a summary (12 maturities x 9 strikes = 108 quotes)
python -m data.adapters.synthetic

# Validate arbitrage-consistency (round-trip prices -> implied vol)
python roundtrip_test.py
```

Expected round-trip output:
```
Round-trip over 108 quotes (WTI + Henry Hub):
  raw price  -> IV : max ~5e-05, mean ~7e-06
  quoted mid -> IV : max ~6e-04, mean ~2e-05
PASS: synthetic prices invert cleanly; data is arbitrage-consistent by construction.
```

## Roadmap
- Pricing core: `black76.py`, `bachelier.py`, `iv_solver.py` (SciPy brentq)
- Forward curve construction (no-arb interpolation)
- Vol surface calibration + arbitrage filters (calendar/butterfly)
- Greeks, scenario/stress engine (2020 oil collapse, gas spikes, curve shocks)
- Streamlit dashboard
