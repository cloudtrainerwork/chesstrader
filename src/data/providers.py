"""
Market data providers.

A yfinance-backed implementation plus the provider error hierarchy, a sliding
window rate limiter, and a request descriptor used across the data layer.
"""

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import yfinance as yf

from .models import OptionChainData, PriceData


class DataProviderError(Exception):
    """Base error for all data provider failures."""


class InvalidSymbol(DataProviderError):
    """Raised when a symbol is malformed or empty."""


class DataNotAvailable(DataProviderError):
    """Raised when a provider returns no data for an otherwise valid request."""


# Tickers are letters/digits with optional dot or hyphen (e.g. BRK.B, RDS-A).
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,12}$")


@dataclass
class MarketDataRequest:
    """Description of a price-history request."""

    symbol: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    interval: str = "1d"


class RateLimiter:
    """Sliding-window rate limiter capping requests per minute."""

    def __init__(self, max_requests_per_minute: int = 30):
        self.max_requests_per_minute = max_requests_per_minute
        self.requests: List[float] = []
        self._lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Block until issuing another request stays within the limit."""
        with self._lock:
            now = time.monotonic()
            self.requests = [t for t in self.requests if t > now - 60.0]

            if len(self.requests) >= self.max_requests_per_minute:
                sleep_for = self.requests[0] + 60.0 - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                self.requests = [t for t in self.requests if t > now - 60.0]

            self.requests.append(now)


def _parse_expiration(value: str) -> datetime:
    """Parse a yfinance expiration string (YYYY-MM-DD) to a datetime."""
    return datetime.strptime(value, "%Y-%m-%d")


class YFinanceProvider:
    """Market data provider backed by yfinance."""

    def __init__(self, rate_limiter: Optional[RateLimiter] = None,
                 requests_per_minute: int = 30):
        self.rate_limiter = rate_limiter or RateLimiter(requests_per_minute)

    def _validate_symbol(self, symbol: Optional[str]) -> str:
        """Normalize and validate a ticker symbol.

        Returns the upper-cased, stripped symbol. Raises InvalidSymbol for
        empty, non-string, or malformed input.
        """
        if not symbol or not isinstance(symbol, str):
            raise InvalidSymbol(f"Invalid symbol: {symbol!r}")
        cleaned = symbol.strip().upper()
        if not _SYMBOL_PATTERN.match(cleaned):
            raise InvalidSymbol(f"Invalid symbol: {symbol!r}")
        return cleaned

    def get_price_history(self, symbol: str, start: Optional[datetime] = None,
                          end: Optional[datetime] = None,
                          interval: str = "1d") -> PriceData:
        """Fetch OHLCV history. Raises DataNotAvailable when empty."""
        symbol = self._validate_symbol(symbol)
        self.rate_limiter.wait_if_needed()

        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start, end=end, interval=interval)

        if data is None or data.empty:
            raise DataNotAvailable(f"No price data available for {symbol}")

        return PriceData(
            symbol=symbol,
            data=data,
            start_date=start,
            end_date=end,
            interval=interval,
            source="yfinance",
        )

    def get_current_price(self, symbol: str) -> float:
        """Return the latest price, falling back to recent history."""
        symbol = self._validate_symbol(symbol)
        self.rate_limiter.wait_if_needed()

        ticker = yf.Ticker(symbol)
        last_price = getattr(ticker.fast_info, "last_price", None)
        if last_price is not None:
            return float(last_price)

        history = ticker.history(period="1d")
        if history is None or history.empty:
            raise DataNotAvailable(f"No current price available for {symbol}")
        return float(history["Close"].iloc[-1])

    def batch_get_prices(self, symbols: List[str], days: int = 730) -> Dict[str, PriceData]:
        """Fetch price history for several symbols, skipping failures."""
        start = datetime.now() - timedelta(days=days)
        results: Dict[str, PriceData] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_price_history(symbol, start=start)
            except DataProviderError:
                continue
        return results

    def get_options_chain(self, symbol: str,
                          expiration: Optional[datetime] = None) -> OptionChainData:
        """Fetch the option chain for a symbol and expiration."""
        symbol = self._validate_symbol(symbol)
        self.rate_limiter.wait_if_needed()

        ticker = yf.Ticker(symbol)
        available = ticker.options
        if not available:
            raise DataNotAvailable(f"No options available for {symbol}")

        if expiration is None:
            expiration = _parse_expiration(available[0])
            expiration_str = available[0]
        elif isinstance(expiration, datetime):
            expiration_str = expiration.strftime("%Y-%m-%d")
        else:
            expiration_str = str(expiration)
            expiration = _parse_expiration(expiration_str)

        chain = ticker.option_chain(expiration_str)
        underlying = getattr(ticker.fast_info, "last_price", None)

        return OptionChainData(
            symbol=symbol,
            expiration=expiration,
            calls=chain.calls,
            puts=chain.puts,
            underlying_price=float(underlying) if underlying is not None else None,
            source="yfinance",
        )


_default_provider: Optional[YFinanceProvider] = None


def get_default_provider() -> YFinanceProvider:
    """Return a process-wide default YFinanceProvider instance."""
    global _default_provider
    if _default_provider is None:
        requests_per_minute = 30
        try:
            from ..config import config
            requests_per_minute = config.data.yfinance_requests_per_minute
        except Exception:
            pass
        _default_provider = YFinanceProvider(requests_per_minute=requests_per_minute)
    return _default_provider
