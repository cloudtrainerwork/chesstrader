"""
Compatibility wrapper for regime detection components.

Exposes RegimeDetector and RegimeType from their canonical locations.
"""

from ..models.regime_detector import RegimeDetector
from ..data.regime_labeler import RegimeType

__all__ = ["RegimeDetector", "RegimeType"]
