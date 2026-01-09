"""
Tests for 24-dimensional position state vector assembler.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from src.features.position_models import Position, StrategyType, OptionType
from src.features.position_vector import PositionStateVector
from src.features.position_state import PositionState


class TestPositionStateVector(unittest.TestCase):
    """Test 24-dimensional position state vector assembly."""

    def setUp(self):
        """Set up test positions and vector assembler."""
        self.psv = PositionStateVector()

        # Create test positions
        self.entry_date = datetime(2024, 1, 1)
        self.expiration_date = datetime(2024, 2, 16)

        # Long call position
        self.long_call = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=self.entry_date,
            expiration_date=self.expiration_date,
            strikes=[10000],  # $100 strike
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[500],  # $5.00 premium
            current_prices=[300],  # $3.00 current
            underlying_price_at_entry=9500,
            current_underlying_price=10200,
            adjustments_made=0
        )

        # Iron Condor position
        self.iron_condor = Position(
            strategy_type=StrategyType.IRON_CONDOR,
            entry_date=self.entry_date,
            expiration_date=self.expiration_date,
            strikes=[9500, 10000, 11000, 11500],
            option_types=[OptionType.PUT, OptionType.PUT, OptionType.CALL, OptionType.CALL],
            quantities=[1, -1, -1, 1],
            entry_prices=[200, 500, 600, 250],
            current_prices=[100, 300, 400, 150],
            underlying_price_at_entry=10500,
            current_underlying_price=10500,
            adjustments_made=1
        )

    def test_vector_dimension(self):
        """Test that vector is exactly 24 dimensions."""
        state = self.psv.calculate(self.long_call)

        self.assertEqual(state.shape, (24,))
        self.assertIsInstance(state, np.ndarray)

    def test_vector_validation(self):
        """Test vector validation for NaN and infinite values."""
        state = self.psv.calculate(self.long_call)

        # Should not contain NaN or infinite values
        self.assertFalse(np.any(np.isnan(state)))
        self.assertFalse(np.any(np.isinf(state)))

        # Should pass validation
        self.psv.validate(state)

    def test_strategy_identity_feature(self):
        """Test strategy identity encoding."""
        lc_state = self.psv.calculate(self.long_call)
        ic_state = self.psv.calculate(self.iron_condor)

        # Strategy identity should be different for different strategies
        self.assertNotEqual(lc_state[0], ic_state[0])

        # Should be in [-1, 1] range
        self.assertGreaterEqual(lc_state[0], -1.0)
        self.assertLessEqual(lc_state[0], 1.0)

    def test_board_position_features(self):
        """Test board position features (indices 1-6)."""
        state = self.psv.calculate(self.iron_condor)

        # Should have 6 board position features
        board_features = state[1:7]

        # All should be finite and in reasonable range
        for i, feature in enumerate(board_features):
            self.assertFalse(np.isnan(feature))
            self.assertFalse(np.isinf(feature))
            self.assertGreaterEqual(feature, -5.0)  # Reasonable bounds
            self.assertLessEqual(feature, 5.0)

    def test_time_features(self):
        """Test time features (indices 7-9)."""
        state = self.psv.calculate(self.long_call)

        time_features = state[7:10]

        # Should have 3 time features
        self.assertEqual(len(time_features), 3)

        # All should be finite
        for feature in time_features:
            self.assertFalse(np.isnan(feature))
            self.assertFalse(np.isinf(feature))

    def test_volatility_features(self):
        """Test volatility features (indices 10-12)."""
        state = self.psv.calculate(self.long_call)

        vol_features = state[10:13]

        # Should have 3 volatility features
        self.assertEqual(len(vol_features), 3)

        # All should be finite
        for feature in vol_features:
            self.assertFalse(np.isnan(feature))
            self.assertFalse(np.isinf(feature))

    def test_greeks_features(self):
        """Test Greeks features (indices 13-16)."""
        state = self.psv.calculate(self.long_call)

        greeks_features = state[13:17]

        # Should have 4 Greeks features
        self.assertEqual(len(greeks_features), 4)

        # All should be finite
        for feature in greeks_features:
            self.assertFalse(np.isnan(feature))
            self.assertFalse(np.isinf(feature))

    def test_pnl_features(self):
        """Test P/L features (indices 17-21)."""
        state = self.psv.calculate(self.long_call)

        pnl_features = state[17:22]

        # Should have 5 P/L features
        self.assertEqual(len(pnl_features), 5)

        # All should be finite
        for feature in pnl_features:
            self.assertFalse(np.isnan(feature))
            self.assertFalse(np.isinf(feature))

    def test_meta_features(self):
        """Test metadata features (indices 22-23)."""
        state = self.psv.calculate(self.iron_condor)

        meta_features = state[22:24]

        # Should have 2 meta features
        self.assertEqual(len(meta_features), 2)

        # All should be finite
        for feature in meta_features:
            self.assertFalse(np.isnan(feature))
            self.assertFalse(np.isinf(feature))

    def test_batch_processing(self):
        """Test batch processing of multiple positions."""
        positions = [self.long_call, self.iron_condor]

        batch_states = self.psv.calculate_batch(positions)

        # Should be (2, 24) shape
        self.assertEqual(batch_states.shape, (2, 24))

        # Each row should be a valid 24-dimensional vector
        for i in range(batch_states.shape[0]):
            state = batch_states[i]
            self.assertEqual(state.shape, (24,))
            self.assertFalse(np.any(np.isnan(state)))
            self.assertFalse(np.any(np.isinf(state)))

    def test_feature_names(self):
        """Test feature name mapping."""
        feature_names = self.psv.get_feature_names()

        # Should have exactly 24 feature names
        self.assertEqual(len(feature_names), 24)

        # Each name should be a non-empty string
        for name in feature_names:
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)

    def test_vector_description(self):
        """Test vector description functionality."""
        state = self.psv.calculate(self.long_call)
        description = self.psv.describe_vector(state)

        # Should have main sections
        self.assertIn('features', description)
        self.assertIn('statistics', description)
        self.assertIn('validation', description)

        # Features section should have all 24 features
        self.assertEqual(len(description['features']), 24)

        # Statistics should be reasonable
        stats = description['statistics']
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)

        # Validation should show clean data
        validation = description['validation']
        self.assertFalse(validation['has_nan'])
        self.assertFalse(validation['has_inf'])

    def test_custom_current_prices(self):
        """Test vector calculation with custom current prices."""
        custom_prices = [400]  # Different from position.current_prices
        state = self.psv.calculate(self.long_call, current_prices=custom_prices)

        # Should still be valid 24-dimensional vector
        self.assertEqual(state.shape, (24,))
        self.assertFalse(np.any(np.isnan(state)))

    def test_custom_iv_estimates(self):
        """Test vector calculation with custom IV estimates."""
        custom_iv = [0.35]  # 35% volatility
        state = self.psv.calculate(self.long_call, iv_estimates=custom_iv)

        # Should still be valid 24-dimensional vector
        self.assertEqual(state.shape, (24,))
        self.assertFalse(np.any(np.isnan(state)))

    def test_edge_case_positions(self):
        """Test vector calculation with edge case positions."""
        # Position with zero time to expiration
        expired_position = Position(
            strategy_type=StrategyType.LONG_PUT,
            entry_date=datetime.now() - timedelta(days=30),
            expiration_date=datetime.now() - timedelta(days=1),
            strikes=[10000],
            option_types=[OptionType.PUT],
            quantities=[1],
            entry_prices=[500],
            current_prices=[100],
            underlying_price_at_entry=10500,
            current_underlying_price=10200
        )

        state = self.psv.calculate(expired_position)

        # Should still be valid despite edge case
        self.assertEqual(state.shape, (24,))
        self.assertFalse(np.any(np.isnan(state)))
        self.assertFalse(np.any(np.isinf(state)))

    def test_validation_errors(self):
        """Test validation error handling."""
        # Wrong dimension vector
        wrong_dim = np.array([1, 2, 3])  # Only 3 dimensions

        with self.assertRaises(ValueError):
            self.psv.validate(wrong_dim)

        # NaN values
        nan_vector = np.full(24, np.nan)
        with self.assertRaises(ValueError):
            self.psv.validate(nan_vector)

        # Infinite values
        inf_vector = np.full(24, np.inf)
        with self.assertRaises(ValueError):
            self.psv.validate(inf_vector)


class TestPositionStateIntegration(unittest.TestCase):
    """Test integration between PositionState and PositionStateVector."""

    def setUp(self):
        """Set up test position and state calculator."""
        self.position_state = PositionState()

        self.test_position = Position(
            strategy_type=StrategyType.BULL_CALL_SPREAD,
            entry_date=datetime(2024, 1, 1),
            expiration_date=datetime(2024, 2, 16),
            strikes=[10000, 10500],
            option_types=[OptionType.CALL, OptionType.CALL],
            quantities=[1, -1],
            entry_prices=[500, 200],
            current_prices=[300, 100],
            underlying_price_at_entry=9800,
            current_underlying_price=10200
        )

    def test_legacy_calculate_method(self):
        """Test legacy calculate method still works."""
        features = self.position_state.calculate(self.test_position)

        # Should return pandas Series
        self.assertIsInstance(features, pd.Series)

        # Should have basic position features
        expected_features = [
            'price_zone', 'zone_velocity', 'distance_to_lower_breakeven',
            'distance_to_upper_breakeven', 'distance_to_lower_strike',
            'distance_to_upper_strike'
        ]

        for feature in expected_features:
            self.assertIn(feature, features.index)

    def test_new_vector_method(self):
        """Test new 24-dimensional vector method."""
        vector = self.position_state.calculate_vector(self.test_position)

        # Should return numpy array
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (24,))

        # Should be valid
        self.assertFalse(np.any(np.isnan(vector)))
        self.assertFalse(np.any(np.isinf(vector)))


if __name__ == '__main__':
    import pandas as pd
    unittest.main()