"""
Adapter protocol. Every data source implements `fetch()` returning a
QuoteSnapshot. Downstream code depends only on this interface, never on a
specific vendor, so swapping yfinance -> Bloomberg BQuant is a one-line change.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, Sequence

from data.schema import Commodity, QuoteSnapshot


class QuoteAdapter(Protocol):
    name: str

    def fetch(
        self,
        commodities: Sequence[Commodity],
        as_of: date | None = None,
    ) -> QuoteSnapshot:
        """Return a validated QuoteSnapshot for the requested commodities."""
        ...
