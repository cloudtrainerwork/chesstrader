"""
Position state vector assembler for 24-dimensional feature representation.

Creates comprehensive position state vectors combining strategy identity,
board position, timing, volatility, Greeks, P/L status, and metadata.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime

from .position_models import Position, StrategyType, PositionZones
from .greeks import GreeksCalculator, ImpliedVolatilityEstimator
from .pnl import PnLCalculator


class PositionStateVector:
    """
    24-dimensional position state vector assembler.

    Combines all position features into standardized neural network input:
    - Strategy identity (1): one-hot index for strategy type
    - Board position (6): spatial relationships and zones
    - Time (3): expiration and time decay features
    - Volatility (3): implied volatility dynamics
    - Greeks (4): position Greeks from Black-Scholes
    - P/L status (5): profit/loss metrics
    - Meta (2): position management metadata
    """

    def __init__(self):
        """Initialize feature calculators."""
        self.greeks_calc = GreeksCalculator()
        self.iv_estimator = ImpliedVolatilityEstimator()
        self.pnl_calc = PnLCalculator()

        # Strategy type mapping to indices
        self.strategy_mapping = {
            strategy: idx for idx, strategy in enumerate(StrategyType)
        }

    def calculate(self, position: Position, current_prices: Optional[List[int]] = None,
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
        if current_prices is None:
            current_prices = position.current_prices

        if iv_estimates is None:
            iv_estimates = self.iv_estimator.get_iv_for_position(position)

        # Initialize feature vector
        features = np.zeros(24)

        # 1. Strategy Identity (1 feature)
        features[0] = self._calculate_strategy_identity(position)

        # 2. Board Position (6 features)
        board_features = self._calculate_board_position(position)
        features[1:7] = board_features

        # 3. Time (3 features)
        time_features = self._calculate_time_features(position)
        features[7:10] = time_features

        # 4. Volatility (3 features)
        vol_features = self._calculate_volatility_features(position, iv_estimates)
        features[10:13] = vol_features

        # 5. Greeks (4 features)
        greeks_features = self._calculate_greeks_features(position, iv_estimates)
        features[13:17] = greeks_features

        # 6. P/L Status (5 features)
        pnl_features = self._calculate_pnl_features(position, current_prices)
        features[17:22] = pnl_features

        # 7. Meta (2 features)
        meta_features = self._calculate_meta_features(position)
        features[22:24] = meta_features

        # Validate and return
        self.validate(features)
        return features

    def calculate_batch(self, positions: List[Position],
                       current_prices_list: Optional[List[List[int]]] = None,
                       iv_estimates_list: Optional[List[List[float]]] = None) -> np.ndarray:
        """
        Calculate state vectors for multiple positions.

        Args:
            positions: List of Position objects
            current_prices_list: List of current prices for each position
            iv_estimates_list: List of IV estimates for each position

        Returns:
            Array of shape (n_positions, 24) with state vectors
        """
        n_positions = len(positions)
        batch_features = np.zeros((n_positions, 24))

        for i, position in enumerate(positions):
            current_prices = None
            if current_prices_list is not None:
                current_prices = current_prices_list[i]

            iv_estimates = None
            if iv_estimates_list is not None:
                iv_estimates = iv_estimates_list[i]

            batch_features[i] = self.calculate(position, current_prices, iv_estimates)

        return batch_features

    def _calculate_strategy_identity(self, position: Position) -> float:
        """Calculate strategy identity as normalized index."""
        strategy_idx = self.strategy_mapping[position.strategy_type]
        # Normalize to [-1, 1] range
        return (strategy_idx / (len(StrategyType) - 1)) * 2 - 1

    def _calculate_board_position(self, position: Position) -> np.ndarray:
        """Calculate 6-dimensional board position features."""
        features = np.zeros(6)

        # Feature 0: Price zone
        price_zone = position.calculate_price_zone()
        features[0] = price_zone.value / 3.0  # Normalize to [-1, 1]

        # Feature 1: Zone velocity (simplified - would need historical data)
        features[1] = 0.0  # TODO: Implement with position history

        # Calculate strike and breakeven distances
        current_price = position.current_underlying_price
        strikes = position.strikes
        breakevens = position.calculate_breakevens()

        # Feature 2: Distance to upper strike
        if strikes:
            upper_strike = max(strikes)
            features[2] = np.tanh((upper_strike - current_price) / current_price)
        else:
            features[2] = 0.0

        # Feature 3: Distance to lower strike
        if strikes:
            lower_strike = min(strikes)
            features[3] = np.tanh((current_price - lower_strike) / current_price)
        else:
            features[3] = 0.0

        # Feature 4: Distance to upper breakeven
        if breakevens:
            upper_be = max(breakevens)
            features[4] = np.tanh((upper_be - current_price) / current_price)
        else:
            features[4] = 0.0

        # Feature 5: Distance to lower breakeven
        if breakevens:
            lower_be = min(breakevens)
            features[5] = np.tanh((current_price - lower_be) / current_price)
        else:
            features[5] = 0.0

        return features

    def _calculate_time_features(self, position: Position) -> np.ndarray:
        """Calculate 3-dimensional time features."""
        features = np.zeros(3)

        days_to_exp = position.days_to_expiration
        days_held = position.days_held

        # Feature 0: Days to expiration (normalized)
        features[0] = np.tanh(days_to_exp / 30.0)  # Normalize around 30-day options

        # Feature 1: Percent time remaining
        total_days = (position.expiration_date - position.entry_date).days
        if total_days > 0:
            pct_time_remaining = max(days_to_exp / total_days, 0.0)
            features[1] = pct_time_remaining * 2 - 1  # Map [0,1] to [-1,1]
        else:
            features[1] = -1.0

        # Feature 2: Theta pressure (time decay acceleration)
        if days_to_exp > 0:
            theta_pressure = min(1.0, max(0.0, (30 - days_to_exp) / 30.0))
            features[2] = theta_pressure * 2 - 1  # Map [0,1] to [-1,1]
        else:
            features[2] = 1.0  # Maximum theta pressure

        return features

    def _calculate_volatility_features(self, position: Position,
                                     iv_estimates: List[float]) -> np.ndarray:
        """Calculate 3-dimensional volatility features."""
        features = np.zeros(3)

        # Use weighted average IV based on position quantities
        if iv_estimates and position.quantities:
            weights = [abs(qty) for qty in position.quantities]
            total_weight = sum(weights)
            if total_weight > 0:
                avg_iv = sum(iv * w for iv, w in zip(iv_estimates, weights)) / total_weight
            else:
                avg_iv = np.mean(iv_estimates)
        else:
            avg_iv = 0.30  # Default 30% volatility

        # Feature 0: IV at entry (estimated)
        entry_iv = avg_iv  # In practice, would store actual entry IV
        features[0] = np.tanh((entry_iv - 0.30) / 0.30)  # Normalize around 30%

        # Feature 1: Current IV
        current_iv = avg_iv
        features[1] = np.tanh((current_iv - 0.30) / 0.30)

        # Feature 2: IV change
        iv_change = current_iv - entry_iv
        features[2] = np.tanh(iv_change / 0.10)  # Normalize around 10% vol changes

        return features

    def _calculate_greeks_features(self, position: Position,
                                 iv_estimates: List[float]) -> np.ndarray:
        """Calculate 4-dimensional Greeks features."""
        features = np.zeros(4)

        try:
            # Get position Greeks
            greeks = self.greeks_calc.position_greeks(position, iv_estimates)

            # Feature 0: Position delta
            features[0] = np.tanh(greeks['delta'])

            # Feature 1: Position gamma (scaled)
            features[1] = np.tanh(greeks['gamma'] / 10.0)  # Scale for typical range

            # Feature 2: Position theta (daily)
            features[2] = np.tanh(greeks['theta'] * 1000)  # Scale for visibility

            # Feature 3: Position vega (scaled)
            features[3] = np.tanh(greeks['vega'] * 100)  # Scale for typical range

        except Exception:
            # Fallback to zeros if Greeks calculation fails
            features[:] = 0.0

        return features

    def _calculate_pnl_features(self, position: Position,
                              current_prices: List[int]) -> np.ndarray:
        """Calculate 5-dimensional P/L status features."""
        features = np.zeros(5)

        try:
            # Get P/L metrics
            pnl_metrics = self.pnl_calc.calculate_pnl_metrics(position, current_prices)

            # Feature 0: Entry credit (normalized by underlying price)
            entry_credit = pnl_metrics['entry_credit']
            features[0] = np.tanh(entry_credit / position.current_underlying_price)

            # Feature 1: Current value (normalized)
            current_value = pnl_metrics['current_value']
            features[1] = np.tanh(current_value / position.current_underlying_price)

            # Feature 2: Unrealized P/L (normalized)
            unrealized_pnl = pnl_metrics['unrealized_pnl']
            features[2] = np.tanh(unrealized_pnl / position.current_underlying_price)

            # Feature 3: Percent of max profit
            pct_max_profit = pnl_metrics['percent_of_max_profit']
            features[3] = np.tanh(pct_max_profit)

            # Feature 4: Percent of max loss
            pct_max_loss = pnl_metrics['percent_of_max_loss']
            features[4] = np.tanh(pct_max_loss)

        except Exception:
            # Fallback to zeros if P/L calculation fails
            features[:] = 0.0

        return features

    def _calculate_meta_features(self, position: Position) -> np.ndarray:
        """Calculate 2-dimensional metadata features."""
        features = np.zeros(2)

        # Feature 0: Days held (normalized)
        days_held = position.days_held
        features[0] = np.tanh(days_held / 30.0)  # Normalize around 30-day holds

        # Feature 1: Adjustments made
        adjustments = position.adjustments_made
        features[1] = np.tanh(adjustments / 3.0)  # Normalize around 3 adjustments

        return features

    def validate(self, state_vector: np.ndarray) -> None:
        """
        Validate 24-dimensional state vector for quality and completeness.

        Args:
            state_vector: 24-dimensional feature vector

        Raises:
            ValueError: If validation fails
        """
        if state_vector.shape != (24,):
            raise ValueError(f"State vector must be 24-dimensional, got {state_vector.shape}")

        # Check for NaN or infinite values
        if np.any(np.isnan(state_vector)):
            raise ValueError("State vector contains NaN values")

        if np.any(np.isinf(state_vector)):
            raise ValueError("State vector contains infinite values")

        # Check reasonable ranges (most features should be in [-3, 3])
        extreme_values = np.where(np.abs(state_vector) > 5.0)[0]
        if len(extreme_values) > 0:
            print(f"Warning: Extreme values in features {extreme_values}: "
                  f"{state_vector[extreme_values]}")

        # All checks passed
        return

    def get_feature_names(self) -> List[str]:
        """
        Get descriptive names for all 24 features.

        Returns:
            List of feature names in order
        """
        return [
            # Strategy identity (1)
            'strategy_identity',

            # Board position (6)
            'price_zone',
            'zone_velocity',
            'distance_to_upper_strike',
            'distance_to_lower_strike',
            'distance_to_upper_breakeven',
            'distance_to_lower_breakeven',

            # Time (3)
            'days_to_expiration',
            'percent_time_remaining',
            'theta_pressure',

            # Volatility (3)
            'iv_at_entry',
            'current_iv',
            'iv_change',

            # Greeks (4)
            'position_delta',
            'position_gamma',
            'position_theta',
            'position_vega',

            # P/L status (5)
            'entry_credit',
            'current_value',
            'unrealized_pnl',
            'percent_of_max_profit',
            'percent_of_max_loss',

            # Meta (2)
            'days_held',
            'adjustments_made'
        ]

    def describe_vector(self, state_vector: np.ndarray) -> Dict[str, Any]:
        """
        Create human-readable description of state vector.

        Args:
            state_vector: 24-dimensional feature vector

        Returns:
            Dictionary with feature descriptions and statistics
        """
        feature_names = self.get_feature_names()

        description = {
            'features': {
                name: float(value) for name, value in zip(feature_names, state_vector)
            },
            'statistics': {
                'min': float(np.min(state_vector)),
                'max': float(np.max(state_vector)),
                'mean': float(np.mean(state_vector)),
                'std': float(np.std(state_vector)),
                'range': float(np.max(state_vector) - np.min(state_vector))
            },
            'validation': {
                'shape': state_vector.shape,
                'has_nan': bool(np.any(np.isnan(state_vector))),
                'has_inf': bool(np.any(np.isinf(state_vector))),
                'extreme_count': int(np.sum(np.abs(state_vector) > 3.0))
            }
        }

        return description