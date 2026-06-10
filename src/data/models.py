"""
Data models for market data.

Lightweight containers for the price history and option-chain snapshots
returned by data providers and the cache layer.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class PriceData:
    """OHLCV price history for a single symbol."""

    symbol: str
    data: pd.DataFrame
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    interval: str = "1d"
    source: str = "yfinance"

    @property
    def is_valid(self) -> bool:
        """True when the frame holds at least one row with a Close column."""
        return (
            self.data is not None
            and not self.data.empty
            and "Close" in self.data.columns
        )

    @property
    def latest_price(self) -> Optional[float]:
        """Most recent close price, or None when no data is present."""
        if not self.is_valid:
            return None
        return float(self.data["Close"].iloc[-1])


@dataclass
class OptionChainData:
    """Calls and puts for a single symbol and expiration."""

    symbol: str
    expiration: Optional[datetime]
    calls: pd.DataFrame
    puts: pd.DataFrame
    underlying_price: Optional[float] = None
    source: str = "yfinance"

    @property
    def is_valid(self) -> bool:
        """True when both the call and put frames hold at least one row."""
        return (
            self.calls is not None
            and self.puts is not None
            and not self.calls.empty
            and not self.puts.empty
        )
