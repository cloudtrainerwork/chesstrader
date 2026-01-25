"""
Options trading strategies module.

Provides base strategy interface and implementations for various options strategies
including neutral, directional, and volatility strategies.
"""

from .base import BaseStrategy, StrategyMetadata

__all__ = [
    'BaseStrategy',
    'StrategyMetadata',
]