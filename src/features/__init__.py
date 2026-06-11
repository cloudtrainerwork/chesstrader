"""
Feature engineering module for ChessTrader.

Provides comprehensive feature extraction for market regime detection
and position sizing algorithms.

Public names are imported lazily (PEP 562). A broken dependency in one
module (for example base.py's data-layer import) therefore no longer
prevents importing unrelated, self-contained modules in this package
(greeks, pnl, position_models, ...).
"""

import importlib

# Map each public name to the submodule that defines it.
_EXPORTS = {
    "FeatureEngineering": ".base",
    "PriceStructureFeatures": ".regime_features",
    "TrendIndicators": ".regime_features",
    "MomentumIndicators": ".regime_features",
    "VolatilityFeatures": ".regime_features",
    "VolumeFeatures": ".regime_features",
    "SupportResistanceFeatures": ".regime_features",
    "MarketContextFeatures": ".regime_features",
    "EventFeatures": ".regime_features",
    "RegimeStateVector": ".regime_features",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Lazily import and return a public attribute (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
