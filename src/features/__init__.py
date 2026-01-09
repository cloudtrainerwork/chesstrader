"""
Feature engineering module for ChessTrader.

Provides comprehensive feature extraction for market regime detection
and position sizing algorithms.
"""

from .base import FeatureEngineering
from .regime_features import (
    PriceStructureFeatures,
    TrendIndicators,
    MomentumIndicators,
    RegimeStateVector
)

__all__ = [
    "FeatureEngineering",
    "PriceStructureFeatures",
    "TrendIndicators",
    "MomentumIndicators",
    "RegimeStateVector",
]