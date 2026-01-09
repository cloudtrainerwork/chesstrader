"""
Position state feature engineering for options trading.

Provides comprehensive position state representation combining position models,
Greeks calculations, and feature vector assembly for RL-based management.
"""

from typing import Union, List
import pandas as pd
import numpy as np
from datetime import datetime

from .base import FeatureEngineering
from .position_models import Position, PositionZones, StrategyType


class PositionState(FeatureEngineering):
    """
    Position state feature engineering for options positions.

    Calculates position-specific features including price zones, timing,
    and spatial relationships for neural network input.
    """

    def calculate(self, position: Position) -> pd.Series:
        """
        Calculate position state features.

        Args:
            position: Position object to analyze

        Returns:
            Series with position state features
        """
        features = {}

        # Price zone and spatial features
        current_zone = position.calculate_price_zone()
        features['price_zone'] = float(current_zone.value)

        # Calculate zone velocity (simplified for now)
        features['zone_velocity'] = 0.0  # TODO: Implement based on historical zones

        # Distance calculations
        breakevens = position.calculate_breakevens()
        current_price = position.current_underlying_price

        if len(breakevens) >= 1:
            features['distance_to_lower_breakeven'] = (current_price - min(breakevens)) / current_price
        else:
            features['distance_to_lower_breakeven'] = 0.0

        if len(breakevens) >= 2:
            features['distance_to_upper_breakeven'] = (max(breakevens) - current_price) / current_price
        else:
            features['distance_to_upper_breakeven'] = features['distance_to_lower_breakeven']

        # Strike distance calculations
        strikes = position.strikes
        if strikes:
            features['distance_to_lower_strike'] = (current_price - min(strikes)) / current_price
            features['distance_to_upper_strike'] = (max(strikes) - current_price) / current_price
        else:
            features['distance_to_lower_strike'] = 0.0
            features['distance_to_upper_strike'] = 0.0

        # Normalize and validate
        result = pd.Series(features)
        result = self.standardize(result, method='zscore')

        return result