"""
SQLite-backed cache for market data.

Wraps a data provider, persisting price history and option chains in a local
SQLite database via SQLAlchemy and serving cached results within a TTL.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from .models import OptionChainData, PriceData
from .schema import Base, CacheMetadata, OptionsData, PriceHistory

_DEFAULT_PRICE_TTL = 24 * 60 * 60   # 24 hours
_DEFAULT_OPTIONS_TTL = 15 * 60      # 15 minutes


class CacheError(Exception):
    """Raised when a cache operation fails."""


class CacheManager:
    """Cache market data in SQLite, refreshing from a provider on miss/expiry."""

    def __init__(self, database_path=None, provider=None):
        if database_path is None:
            database_path = self._default_db_path()
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        if provider is None:
            from .providers import get_default_provider
            provider = get_default_provider()
        self.provider = provider

        self._engine = create_engine(f"sqlite:///{self.database_path}")
        Base.metadata.create_all(self._engine)
        self.Session = sessionmaker(bind=self._engine)

        price_ttl, options_ttl = _DEFAULT_PRICE_TTL, _DEFAULT_OPTIONS_TTL
        try:
            from ..config import config
            price_ttl = config.cache.price_data_ttl
            options_ttl = config.cache.options_data_ttl
        except Exception:
            pass
        self._ttl_cache = {
            "price_data_ttl_seconds": price_ttl,
            "options_data_ttl_seconds": options_ttl,
        }

        # In-memory hit/miss counters keyed by cache key.
        self._stats: Dict[str, Dict[str, int]] = {}

    @staticmethod
    def _default_db_path() -> Path:
        try:
            from ..config import config
            return Path(config.cache.cache_db_path)
        except Exception:
            return Path("data/cache.db")

    # ------------------------------------------------------------------ price

    def get_price_history(self, symbol: str, start: Optional[datetime] = None,
                          end: Optional[datetime] = None,
                          interval: str = "1d") -> PriceData:
        """Return cached price history when fresh, else refresh from provider."""
        key = f"price:{symbol}:{interval}"

        if self._price_is_fresh(symbol, interval):
            self._record(key, "hit")
            return self._load_price_data(symbol, interval)

        data = self.provider.get_price_history(
            symbol, start=start, end=end, interval=interval
        )

        session = self.Session()
        try:
            self._delete_price_rows(session, symbol, interval)
            self._store_price_data(session, data)
            session.commit()
        except Exception as exc:
            session.rollback()
            raise CacheError(f"Failed to cache price data for {symbol}: {exc}") from exc
        finally:
            session.close()

        self._record(key, "miss")
        return data

    def _price_is_fresh(self, symbol: str, interval: str) -> bool:
        session = self.Session()
        try:
            latest = session.query(func.max(PriceHistory.updated_at)).filter(
                PriceHistory.symbol == symbol,
                PriceHistory.interval == interval,
            ).scalar()
        finally:
            session.close()

        if latest is None:
            return False
        ttl = self._ttl_cache.get("price_data_ttl_seconds", _DEFAULT_PRICE_TTL)
        return (datetime.utcnow() - latest).total_seconds() < ttl

    def _load_price_data(self, symbol: str, interval: str) -> PriceData:
        session = self.Session()
        try:
            rows = session.query(PriceHistory).filter(
                PriceHistory.symbol == symbol,
                PriceHistory.interval == interval,
            ).order_by(PriceHistory.date).all()
        finally:
            session.close()

        frame = pd.DataFrame(
            {
                "Open": [r.open for r in rows],
                "High": [r.high for r in rows],
                "Low": [r.low for r in rows],
                "Close": [r.close for r in rows],
                "Volume": [r.volume for r in rows],
            },
            index=pd.DatetimeIndex([r.date for r in rows], name="Date"),
        )
        return PriceData(
            symbol=symbol,
            data=frame,
            start_date=rows[0].date if rows else None,
            end_date=rows[-1].date if rows else None,
            interval=interval,
            source="cache",
        )

    def _store_price_data(self, session, price_data: PriceData) -> None:
        now = datetime.utcnow()
        for idx, row in price_data.data.iterrows():
            timestamp = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            session.add(PriceHistory(
                symbol=price_data.symbol,
                date=timestamp,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
                interval=price_data.interval,
                source=price_data.source,
                updated_at=now,
            ))

    def _delete_price_rows(self, session, symbol: str, interval: str) -> None:
        session.query(PriceHistory).filter(
            PriceHistory.symbol == symbol,
            PriceHistory.interval == interval,
        ).delete(synchronize_session=False)

    # ---------------------------------------------------------------- options

    def get_options_chain(self, symbol: str,
                          expiration: Optional[datetime] = None) -> OptionChainData:
        """Return cached option chain when fresh, else refresh from provider."""
        key = f"options:{symbol}:{expiration}"

        if self._options_is_fresh(symbol, expiration):
            self._record(key, "hit")
            return self._load_options_data(symbol, expiration)

        data = self.provider.get_options_chain(symbol, expiration)

        session = self.Session()
        try:
            self._delete_options_rows(session, symbol, expiration)
            self._store_options_data(session, data)
            session.commit()
        except Exception as exc:
            session.rollback()
            raise CacheError(f"Failed to cache options for {symbol}: {exc}") from exc
        finally:
            session.close()

        self._record(key, "miss")
        return data

    def _options_is_fresh(self, symbol: str,
                          expiration: Optional[datetime]) -> bool:
        session = self.Session()
        try:
            query = session.query(func.max(OptionsData.updated_at)).filter(
                OptionsData.symbol == symbol,
            )
            if expiration is not None:
                query = query.filter(OptionsData.expiration == expiration)
            latest = query.scalar()
        finally:
            session.close()

        if latest is None:
            return False
        ttl = self._ttl_cache.get("options_data_ttl_seconds", _DEFAULT_OPTIONS_TTL)
        return (datetime.utcnow() - latest).total_seconds() < ttl

    def _load_options_data(self, symbol: str,
                           expiration: Optional[datetime]) -> OptionChainData:
        session = self.Session()
        try:
            query = session.query(OptionsData).filter(OptionsData.symbol == symbol)
            if expiration is not None:
                query = query.filter(OptionsData.expiration == expiration)
            rows = query.all()
        finally:
            session.close()

        calls = [r for r in rows if r.option_type == "call"]
        puts = [r for r in rows if r.option_type == "put"]
        underlying = rows[0].underlying_price if rows else None

        return OptionChainData(
            symbol=symbol,
            expiration=expiration,
            calls=self._options_frame(calls),
            puts=self._options_frame(puts),
            underlying_price=underlying,
            source="cache",
        )

    @staticmethod
    def _options_frame(rows) -> pd.DataFrame:
        return pd.DataFrame({
            "strike": [r.strike for r in rows],
            "bid": [r.bid for r in rows],
            "ask": [r.ask for r in rows],
            "volume": [r.volume for r in rows],
            "openInterest": [r.open_interest for r in rows],
            "impliedVolatility": [r.implied_volatility for r in rows],
        })

    def _store_options_data(self, session, options_data: OptionChainData) -> None:
        now = datetime.utcnow()
        frames = (("call", options_data.calls), ("put", options_data.puts))
        for option_type, frame in frames:
            for _, row in frame.iterrows():
                session.add(OptionsData(
                    symbol=options_data.symbol,
                    expiration=options_data.expiration,
                    option_type=option_type,
                    strike=float(row["strike"]),
                    bid=self._opt_float(row, "bid"),
                    ask=self._opt_float(row, "ask"),
                    volume=self._opt_float(row, "volume"),
                    open_interest=self._opt_float(row, "openInterest"),
                    implied_volatility=self._opt_float(row, "impliedVolatility"),
                    underlying_price=options_data.underlying_price,
                    source=options_data.source,
                    updated_at=now,
                ))

    @staticmethod
    def _opt_float(row, column: str) -> Optional[float]:
        if column in row and pd.notna(row[column]):
            return float(row[column])
        return None

    def _delete_options_rows(self, session, symbol: str,
                             expiration: Optional[datetime]) -> None:
        query = session.query(OptionsData).filter(OptionsData.symbol == symbol)
        if expiration is not None:
            query = query.filter(OptionsData.expiration == expiration)
        query.delete(synchronize_session=False)

    # ------------------------------------------------------------ maintenance

    def get_cache_statistics(self) -> dict:
        """Return database row counts and per-key hit/miss counters."""
        session = self.Session()
        try:
            price_records = session.query(func.count(PriceHistory.id)).scalar() or 0
            options_records = session.query(func.count(OptionsData.id)).scalar() or 0
        finally:
            session.close()

        stats: dict = {
            "database": {
                "price_records": price_records,
                "options_records": options_records,
            },
        }
        for key, counters in self._stats.items():
            stats[key] = dict(counters)
        return stats

    def cleanup_old_data(self, max_age_days: int = 30) -> int:
        """Delete cached rows older than max_age_days. Returns rows removed."""
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        session = self.Session()
        try:
            deleted = session.query(PriceHistory).filter(
                PriceHistory.updated_at < cutoff
            ).delete(synchronize_session=False)
            deleted += session.query(OptionsData).filter(
                OptionsData.updated_at < cutoff
            ).delete(synchronize_session=False)
            session.commit()
        except Exception as exc:
            session.rollback()
            raise CacheError(f"Failed to clean up cache: {exc}") from exc
        finally:
            session.close()
        return deleted

    def _record(self, key: str, outcome: str) -> None:
        counters = self._stats.setdefault(key, {"hits": 0, "misses": 0})
        counters["hits" if outcome == "hit" else "misses"] += 1
