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
    VolatilityFeatures,
    VolumeFeatures,
    SupportResistanceFeatures,
    MarketContextFeatures,
    EventFeatures,
    RegimeStateVector
)

__all__ = [
    "FeatureEngineering",
    "PriceStructureFeatures",
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityFeatures",
    "VolumeFeatures",
    "SupportResistanceFeatures",
    "MarketContextFeatures",
    "EventFeatures",
    "RegimeStateVector",
]