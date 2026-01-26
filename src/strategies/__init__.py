"""
Options trading strategies module.

Provides base strategy interface and implementations for various options strategies
including neutral, directional, volatility, and advanced strategies with factory pattern.
"""

from .base import BaseStrategy, StrategyMetadata, StrategyType, StrategyCategory, RiskLevel
from .factory import StrategyFactory, StrategyRecommendation

# Strategy implementations
from .neutral import IronCondorStrategy, IronButterflyStrategy
from .directional import (
    BullCallSpreadStrategy, BearCallSpreadStrategy,
    BullPutSpreadStrategy, BearPutSpreadStrategy
)
from .volatility import (
    LongStraddleStrategy, ShortStraddleStrategy,
    LongStrangleStrategy, ShortStrangleStrategy
)
from .advanced import CalendarCallStrategy, CalendarPutStrategy
from .equity import CoveredCallStrategy, CollarStrategy

__all__ = [
    # Base classes and enums
    'BaseStrategy',
    'StrategyMetadata',
    'StrategyType',
    'StrategyCategory',
    'RiskLevel',

    # Factory and recommendations
    'StrategyFactory',
    'StrategyRecommendation',

    # Neutral strategies
    'IronCondorStrategy',
    'IronButterflyStrategy',

    # Directional strategies
    'BullCallSpreadStrategy',
    'BearCallSpreadStrategy',
    'BullPutSpreadStrategy',
    'BearPutSpreadStrategy',

    # Volatility strategies
    'LongStraddleStrategy',
    'ShortStraddleStrategy',
    'LongStrangleStrategy',
    'ShortStrangleStrategy',

    # Advanced strategies
    'CalendarCallStrategy',
    'CalendarPutStrategy',

    # Equity-based strategies
    'CoveredCallStrategy',
    'CollarStrategy',
]