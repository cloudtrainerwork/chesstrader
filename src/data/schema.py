"""
SQLAlchemy ORM schema for the local market-data cache.

Three tables back the cache: persisted price history, persisted option-chain
rows, and per-key cache metadata used for statistics.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PriceHistory(Base):
    """One OHLCV bar for a symbol/interval."""

    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False, default=0.0)
    interval = Column(String(8), nullable=False, default="1d")
    source = Column(String(32), nullable=False, default="yfinance")
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "date", "interval",
                         name="uq_price_symbol_date_interval"),
    )


class OptionsData(Base):
    """One option contract row (call or put) for a symbol/expiration."""

    __tablename__ = "options_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), nullable=False, index=True)
    expiration = Column(DateTime, nullable=True)
    option_type = Column(String(4), nullable=False)  # 'call' or 'put'
    strike = Column(Float, nullable=False)
    bid = Column(Float)
    ask = Column(Float)
    volume = Column(Float)
    open_interest = Column(Float)
    implied_volatility = Column(Float)
    underlying_price = Column(Float)
    source = Column(String(32), nullable=False, default="yfinance")
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class CacheMetadata(Base):
    """Per-key cache bookkeeping (last refresh, hit/miss counters)."""

    __tablename__ = "cache_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(128), nullable=False, unique=True)
    symbol = Column(String(16), index=True)
    data_type = Column(String(16))  # 'price' or 'options'
    hits = Column(Integer, nullable=False, default=0)
    misses = Column(Integer, nullable=False, default=0)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)
