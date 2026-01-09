"""
Position state feature engineering for options trading.

Provides comprehensive position state representation combining position models,
Greeks calculations, and feature vector assembly for RL-based management.
"""

from typing import Union, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from .position_models import Position, PositionZones, StrategyType
from .position_vector import PositionStateVector


class PositionState:
    """
    Position state feature engineering for options positions.

    Provides both legacy interface and new 24-dimensional vector assembly
    for neural network compatibility.
    """

    def __init__(self):
        """Initialize position state calculator."""
        self.vector_assembler = PositionStateVector()

    def calculate(self, position: Position) -> pd.Series:
        """
        Calculate legacy position state features.

        Args:
            position: Position object to analyze

        Returns:
            Series with basic position state features
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

        return pd.Series(features)

    def calculate_vector(self, position: Position,
                        current_prices: Optional[List[int]] = None,
                        iv_estimates: Optional[List[float]] = None) -> np.ndarray:
        """
        Calculate 24-dimensional position state vector.

        Args:
            position: Position object to analyze
            current_prices: Current market prices (uses position.current_prices if None)
            iv_estimates: IV estimates for each leg (estimated if None)

        Returns:
            24-dimensional numpy array with normalized features
        """
        return self.vector_assembler.calculate(position, current_prices, iv_estimates)