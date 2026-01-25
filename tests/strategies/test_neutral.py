"""
Comprehensive tests for neutral options strategies.

Tests Iron Condor and Iron Butterfly strategies including validation,
entry/exit criteria, risk metrics calculation, and edge cases.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategies.neutral import IronCondorStrategy, IronButterflyStrategy
from src.strategies.base import (
    MarketConditions, EntrySignal, ExitSignal, RiskMetrics,
    PositionLeg, OptionType, StrategyType, StrategyCategory, RiskLevel
)


class TestIronCondorStrategy:
    """Test suite for Iron Condor strategy."""

    def setup_method(self):
        """Setup test fixtures."""
        self.strategy = IronCondorStrategy()
        self.base_conditions = MarketConditions(
            regime=2,  # Low volatility regime
            volatility_rank=0.5,
            trend_strength=0.3,
            time_to_expiration=35,
            underlying_price=50000,  # $500 in cents
            risk_free_rate=0.05
        )
        self.test_strikes = [48000, 49000, 51000, 52000]  # $480, $490, $510, $520

    def test_strategy_metadata(self):
        """Test strategy metadata is correctly configured."""
        metadata = self.strategy.metadata
        assert metadata.name == "Iron Condor"
        assert metadata.category == StrategyCategory.NEUTRAL
        assert metadata.risk_level == RiskLevel.MEDIUM
        assert metadata.capital_requirement == 1.5
        assert "range-bound profit" in metadata.description.lower()
        assert len(metadata.typical_market_conditions) > 0

    def test_get_strategy_type(self):
        """Test strategy type enum is correct."""
        assert self.strategy.get_strategy_type() == StrategyType.IRON_CONDOR

    def test_validate_market_conditions_favorable(self):
        """Test validation with favorable market conditions."""
        # Test favorable regime 2
        assert self.strategy.validate_market_conditions(self.base_conditions) == True

        # Test favorable regime 4
        conditions_regime4 = MarketConditions(
            regime=4, volatility_rank=0.5, trend_strength=0.3,
            time_to_expiration=35, underlying_price=50000
        )
        assert self.strategy.validate_market_conditions(conditions_regime4) == True

    def test_validate_market_conditions_unfavorable(self):
        """Test validation with unfavorable market conditions."""
        # Unfavorable regime
        bad_regime = MarketConditions(
            regime=0, volatility_rank=0.5, trend_strength=0.3,
            time_to_expiration=35, underlying_price=50000
        )
        assert self.strategy.validate_market_conditions(bad_regime) == False

        # Low volatility
        low_vol = MarketConditions(
            regime=2, volatility_rank=0.2, trend_strength=0.3,
            time_to_expiration=35, underlying_price=50000
        )
        assert self.strategy.validate_market_conditions(low_vol) == False

        # Too short expiration
        short_exp = MarketConditions(
            regime=2, volatility_rank=0.5, trend_strength=0.3,
            time_to_expiration=15, underlying_price=50000
        )
        assert self.strategy.validate_market_conditions(short_exp) == False

        # Strong trend
        strong_trend = MarketConditions(
            regime=2, volatility_rank=0.5, trend_strength=0.8,
            time_to_expiration=35, underlying_price=50000
        )
        assert self.strategy.validate_market_conditions(strong_trend) == False

    def test_calculate_entry_criteria_favorable(self):
        """Test entry signal calculation with favorable conditions."""
        signal = self.strategy.calculate_entry_criteria(self.base_conditions)

        assert isinstance(signal, EntrySignal)
        assert signal.should_enter == True
        assert signal.confidence >= 0.6
        assert len(signal.reasons) > 0
        assert signal.recommended_size > 0

    def test_calculate_entry_criteria_unfavorable(self):
        """Test entry signal with unfavorable conditions."""
        bad_conditions = MarketConditions(
            regime=0, volatility_rank=0.2, trend_strength=0.8,
            time_to_expiration=10, underlying_price=50000
        )
        signal = self.strategy.calculate_entry_criteria(bad_conditions)

        assert signal.should_enter == False
        assert signal.confidence == 0.0
        assert "not suitable" in signal.reasons[0].lower()

    def test_calculate_exit_criteria_profit_target(self):
        """Test exit signal when profit target is reached."""
        # Mock position with profit
        mock_position = Mock()
        mock_position.days_to_expiration = 20
        mock_position.calculate_unrealized_pnl.return_value = 500  # $5 profit
        mock_position.calculate_max_profit.return_value = 1000   # $10 max profit
        mock_position.calculate_max_loss.return_value = 2000     # $20 max loss
        mock_position.calculate_breakevens.return_value = [49000, 51000]
        mock_position.current_underlying_price = 50000

        signal = self.strategy.calculate_exit_criteria(mock_position, self.base_conditions)

        assert isinstance(signal, ExitSignal)
        assert signal.should_exit == True
        assert signal.urgency > 0.6
        assert signal.exit_type == "profit"

    def test_calculate_exit_criteria_loss_limit(self):
        """Test exit signal when loss limit is breached."""
        # Mock position with significant loss
        mock_position = Mock()
        mock_position.days_to_expiration = 20
        mock_position.calculate_unrealized_pnl.return_value = -4000  # $40 loss
        mock_position.calculate_max_profit.return_value = 1000      # $10 max profit
        mock_position.calculate_max_loss.return_value = 2000        # $20 max loss
        mock_position.calculate_breakevens.return_value = [49000, 51000]
        mock_position.current_underlying_price = 50000

        signal = self.strategy.calculate_exit_criteria(mock_position, self.base_conditions)

        assert signal.should_exit == True
        assert signal.urgency == 1.0
        assert signal.exit_type == "loss"

    def test_calculate_exit_criteria_time_decay(self):
        """Test exit signal approaching expiration."""
        # Mock position close to expiration
        mock_position = Mock()
        mock_position.days_to_expiration = 5  # Close to expiration
        mock_position.calculate_unrealized_pnl.return_value = 100
        mock_position.calculate_max_profit.return_value = 1000
        mock_position.calculate_max_loss.return_value = 2000
        mock_position.calculate_breakevens.return_value = [49000, 51000]
        mock_position.current_underlying_price = 50000

        signal = self.strategy.calculate_exit_criteria(mock_position, self.base_conditions)

        assert signal.urgency >= 0.5
        assert "expiration" in " ".join(signal.reasons).lower()

    def test_get_risk_metrics_valid_strikes(self):
        """Test risk metrics calculation with valid strikes."""
        risk_metrics = self.strategy.get_risk_metrics(
            strikes=self.test_strikes,
            underlying_price=50000,
            time_to_expiration=35,
            volatility=0.3
        )

        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.max_profit > 0
        assert risk_metrics.max_loss > 0
        assert len(risk_metrics.breakeven_points) == 2
        assert 0 <= risk_metrics.profit_probability <= 1
        assert risk_metrics.risk_reward_ratio > 0
        assert risk_metrics.capital_requirement > 0
        assert risk_metrics.margin_requirement > 0

    def test_get_risk_metrics_invalid_strikes(self):
        """Test risk metrics with invalid strike count."""
        with pytest.raises(ValueError, match="exactly 4 strikes"):
            self.strategy.get_risk_metrics([1, 2, 3], 50000, 35, 0.3)

    def test_get_position_legs(self):
        """Test position legs construction."""
        expiration = datetime.now() + timedelta(days=35)
        legs = self.strategy.get_position_legs(self.test_strikes, expiration)

        assert len(legs) == 4
        assert all(isinstance(leg, PositionLeg) for leg in legs)

        # Check position structure: Long Put, Short Put, Short Call, Long Call
        assert legs[0].option_type == OptionType.PUT and legs[0].quantity == 1    # Long put
        assert legs[1].option_type == OptionType.PUT and legs[1].quantity == -1   # Short put
        assert legs[2].option_type == OptionType.CALL and legs[2].quantity == -1  # Short call
        assert legs[3].option_type == OptionType.CALL and legs[3].quantity == 1   # Long call

        # Check strikes are correct
        assert legs[0].strike == self.test_strikes[1]  # Long put at short put strike
        assert legs[1].strike == self.test_strikes[0]  # Short put at long put strike
        assert legs[2].strike == self.test_strikes[2]  # Short call at long call strike
        assert legs[3].strike == self.test_strikes[3]  # Long call at short call strike

    def test_get_position_legs_invalid_strikes(self):
        """Test position legs with invalid strike count."""
        expiration = datetime.now() + timedelta(days=35)
        with pytest.raises(ValueError, match="exactly 4 strikes"):
            self.strategy.get_position_legs([1, 2, 3], expiration)

    def test_validate_strategy_strikes_valid(self):
        """Test strike validation with valid configuration."""
        assert self.strategy._validate_strategy_strikes(self.test_strikes, 50000) == True

    def test_validate_strategy_strikes_invalid_count(self):
        """Test strike validation with invalid count."""
        assert self.strategy._validate_strategy_strikes([1, 2, 3], 50000) == False

    def test_validate_strategy_strikes_invalid_ordering(self):
        """Test strike validation with invalid ordering."""
        bad_strikes = [52000, 49000, 51000, 48000]  # Wrong order
        assert self.strategy._validate_strategy_strikes(bad_strikes, 50000) == False

    def test_validate_strategy_strikes_invalid_positioning(self):
        """Test strike validation with bad positioning relative to underlying."""
        bad_strikes = [46000, 47000, 48000, 49000]  # All below current price
        assert self.strategy._validate_strategy_strikes(bad_strikes, 50000) == False

    def test_validate_strikes_base_validation(self):
        """Test base strike validation."""
        # Valid strikes
        assert self.strategy.validate_strikes(self.test_strikes, 50000) == True

        # Empty strikes
        assert self.strategy.validate_strikes([], 50000) == False

        # Negative strikes
        assert self.strategy.validate_strikes([-100, 200, 300, 400], 50000) == False

        # Strikes too far OTM
        extreme_strikes = [10000, 20000, 150000, 160000]
        assert self.strategy.validate_strikes(extreme_strikes, 50000) == False

    def test_validate_expiration_date(self):
        """Test expiration date validation."""
        # Valid future date
        valid_exp = datetime.now() + timedelta(days=30)
        assert self.strategy.validate_expiration_date(valid_exp) == True

        # Past date
        past_exp = datetime.now() - timedelta(days=1)
        assert self.strategy.validate_expiration_date(past_exp) == False

        # Too far in future
        far_future = datetime.now() + timedelta(days=400)
        assert self.strategy.validate_expiration_date(far_future) == False


class TestIronButterflyStrategy:
    """Test suite for Iron Butterfly strategy."""

    def setup_method(self):
        """Setup test fixtures."""
        self.strategy = IronButterflyStrategy()
        self.base_conditions = MarketConditions(
            regime=4,  # Very low volatility regime
            volatility_rank=0.6,
            trend_strength=0.2,
            time_to_expiration=30,
            underlying_price=50000,  # $500 in cents
            risk_free_rate=0.05
        )
        self.test_strikes = [48000, 50000, 52000]  # $480, $500 (body), $520

    def test_strategy_metadata(self):
        """Test strategy metadata is correctly configured."""
        metadata = self.strategy.metadata
        assert metadata.name == "Iron Butterfly"
        assert metadata.category == StrategyCategory.NEUTRAL
        assert metadata.risk_level == RiskLevel.MEDIUM
        assert metadata.capital_requirement == 1.3
        assert "tight profit zone" in metadata.description.lower()
        assert len(metadata.typical_market_conditions) > 0

    def test_get_strategy_type(self):
        """Test strategy type enum is correct."""
        assert self.strategy.get_strategy_type() == StrategyType.BUTTERFLY

    def test_validate_market_conditions_favorable(self):
        """Test validation with favorable market conditions (regime 4 only)."""
        assert self.strategy.validate_market_conditions(self.base_conditions) == True

    def test_validate_market_conditions_unfavorable_regime(self):
        """Test validation fails with non-regime-4 conditions."""
        bad_regime = MarketConditions(
            regime=2, volatility_rank=0.6, trend_strength=0.2,
            time_to_expiration=30, underlying_price=50000
        )
        assert self.strategy.validate_market_conditions(bad_regime) == False

    def test_validate_market_conditions_unfavorable_volatility(self):
        """Test validation fails with low volatility."""
        low_vol = MarketConditions(
            regime=4, volatility_rank=0.3, trend_strength=0.2,
            time_to_expiration=30, underlying_price=50000
        )
        assert self.strategy.validate_market_conditions(low_vol) == False

    def test_validate_market_conditions_unfavorable_trend(self):
        """Test validation fails with strong trend."""
        strong_trend = MarketConditions(
            regime=4, volatility_rank=0.6, trend_strength=0.6,
            time_to_expiration=30, underlying_price=50000
        )
        assert self.strategy.validate_market_conditions(strong_trend) == False

    def test_calculate_entry_criteria_favorable(self):
        """Test entry signal with favorable conditions."""
        signal = self.strategy.calculate_entry_criteria(self.base_conditions)

        assert isinstance(signal, EntrySignal)
        assert signal.should_enter == True
        assert signal.confidence >= 0.7  # Higher threshold than Iron Condor
        assert len(signal.reasons) > 0
        assert signal.recommended_size > 0

    def test_calculate_entry_criteria_high_confidence(self):
        """Test entry signal with very favorable conditions."""
        optimal_conditions = MarketConditions(
            regime=4, volatility_rank=0.8, trend_strength=0.1,
            time_to_expiration=30, underlying_price=50000
        )
        signal = self.strategy.calculate_entry_criteria(optimal_conditions)

        assert signal.confidence > 0.8
        assert signal.recommended_size >= 1.0

    def test_calculate_exit_criteria_more_aggressive_than_condor(self):
        """Test that butterfly exits are more aggressive than condor."""
        # Mock position with moderate profit (40% should trigger exit)
        mock_position = Mock()
        mock_position.days_to_expiration = 15
        mock_position.calculate_unrealized_pnl.return_value = 400  # $4 profit
        mock_position.calculate_max_profit.return_value = 1000    # $10 max profit (40%)
        mock_position.calculate_max_loss.return_value = 2000      # $20 max loss
        mock_position.calculate_breakevens.return_value = [49500, 50500]
        mock_position.current_underlying_price = 50000

        signal = self.strategy.calculate_exit_criteria(mock_position, self.base_conditions)

        assert signal.should_exit == True
        assert signal.urgency >= 0.5
        assert signal.exit_type == "profit"

    def test_calculate_exit_criteria_time_more_aggressive(self):
        """Test that butterfly exits earlier than condor on time."""
        # Mock position with 5 DTE (should trigger exit)
        mock_position = Mock()
        mock_position.days_to_expiration = 5
        mock_position.calculate_unrealized_pnl.return_value = 100
        mock_position.calculate_max_profit.return_value = 1000
        mock_position.calculate_max_loss.return_value = 2000
        mock_position.calculate_breakevens.return_value = [49500, 50500]
        mock_position.current_underlying_price = 50000

        signal = self.strategy.calculate_exit_criteria(mock_position, self.base_conditions)

        assert signal.urgency >= 0.7
        assert "gamma risk" in " ".join(signal.reasons).lower()

    def test_get_risk_metrics_valid_strikes(self):
        """Test risk metrics calculation with valid strikes."""
        risk_metrics = self.strategy.get_risk_metrics(
            strikes=self.test_strikes,
            underlying_price=50000,
            time_to_expiration=30,
            volatility=0.3
        )

        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.max_profit > 0
        assert risk_metrics.max_loss > 0
        assert len(risk_metrics.breakeven_points) == 2
        assert risk_metrics.profit_probability <= 0.6  # Lower than condor
        assert risk_metrics.capital_requirement > 0

    def test_get_risk_metrics_invalid_strikes(self):
        """Test risk metrics with invalid strike count."""
        with pytest.raises(ValueError, match="exactly 3 strikes"):
            self.strategy.get_risk_metrics([1, 2], 50000, 30, 0.3)

    def test_get_position_legs(self):
        """Test position legs construction."""
        expiration = datetime.now() + timedelta(days=30)
        legs = self.strategy.get_position_legs(self.test_strikes, expiration)

        assert len(legs) == 4
        assert all(isinstance(leg, PositionLeg) for leg in legs)

        # Check position structure
        assert legs[0].option_type == OptionType.PUT and legs[0].quantity == 1     # Long put
        assert legs[1].option_type == OptionType.PUT and legs[1].quantity == -1    # Short put (body)
        assert legs[2].option_type == OptionType.CALL and legs[2].quantity == -1   # Short call (body)
        assert legs[3].option_type == OptionType.CALL and legs[3].quantity == 1    # Long call

        # Check body strikes are the same
        assert legs[1].strike == legs[2].strike == self.test_strikes[1]

    def test_get_position_legs_invalid_strikes(self):
        """Test position legs with invalid strike count."""
        expiration = datetime.now() + timedelta(days=30)
        with pytest.raises(ValueError, match="exactly 3 strikes"):
            self.strategy.get_position_legs([1, 2], expiration)

    def test_validate_strategy_strikes_valid(self):
        """Test strike validation with valid configuration."""
        assert self.strategy._validate_strategy_strikes(self.test_strikes, 50000) == True

    def test_validate_strategy_strikes_invalid_count(self):
        """Test strike validation with invalid count."""
        assert self.strategy._validate_strategy_strikes([1, 2], 50000) == False

    def test_validate_strategy_strikes_invalid_ordering(self):
        """Test strike validation with invalid ordering."""
        bad_strikes = [52000, 50000, 48000]  # Wrong order
        assert self.strategy._validate_strategy_strikes(bad_strikes, 50000) == False

    def test_validate_strategy_strikes_body_not_atm(self):
        """Test strike validation when body strike is not ATM."""
        # Body strike too far from ATM (6% away)
        far_strikes = [47000, 47000, 49000]  # Body at $470, underlying at $500
        assert self.strategy._validate_strategy_strikes(far_strikes, 50000) == False

    def test_validate_strategy_strikes_unequal_spreads(self):
        """Test strike validation with unequal spread widths."""
        unequal_strikes = [48000, 50000, 53000]  # $20 put spread, $30 call spread
        assert self.strategy._validate_strategy_strikes(unequal_strikes, 50000) == False


class TestStrategyIntegration:
    """Integration tests for strategies with position models."""

    def test_iron_condor_with_position_mock(self):
        """Test Iron Condor integration with position-like object."""
        strategy = IronCondorStrategy()

        # Create a mock position that behaves like the real Position class
        mock_position = Mock()
        mock_position.days_to_expiration = 25
        mock_position.calculate_unrealized_pnl.return_value = 200
        mock_position.calculate_max_profit.return_value = 800
        mock_position.calculate_max_loss.return_value = 1200
        mock_position.calculate_breakevens.return_value = [48500, 51500]
        mock_position.current_underlying_price = 50000

        conditions = MarketConditions(
            regime=2, volatility_rank=0.6, trend_strength=0.4,
            time_to_expiration=25, underlying_price=50000
        )

        # Test exit criteria calculation
        signal = strategy.calculate_exit_criteria(mock_position, conditions)
        assert isinstance(signal, ExitSignal)
        assert signal.urgency >= 0.0

    def test_iron_butterfly_with_position_mock(self):
        """Test Iron Butterfly integration with position-like object."""
        strategy = IronButterflyStrategy()

        # Create a mock position
        mock_position = Mock()
        mock_position.days_to_expiration = 20
        mock_position.calculate_unrealized_pnl.return_value = 150
        mock_position.calculate_max_profit.return_value = 600
        mock_position.calculate_max_loss.return_value = 1000
        mock_position.calculate_breakevens.return_value = [49800, 50200]
        mock_position.current_underlying_price = 50100  # Slightly off center

        conditions = MarketConditions(
            regime=4, volatility_rank=0.7, trend_strength=0.1,
            time_to_expiration=20, underlying_price=50100
        )

        # Test exit criteria calculation
        signal = strategy.calculate_exit_criteria(mock_position, conditions)
        assert isinstance(signal, ExitSignal)

    def test_strategy_type_enum_compatibility(self):
        """Test that strategy types match position model enums."""
        condor = IronCondorStrategy()
        butterfly = IronButterflyStrategy()

        assert condor.get_strategy_type() == StrategyType.IRON_CONDOR
        assert butterfly.get_strategy_type() == StrategyType.BUTTERFLY

    def test_calculate_margin_requirement(self):
        """Test margin requirement calculation."""
        strategy = IronCondorStrategy()
        expiration = datetime.now() + timedelta(days=30)

        legs = strategy.get_position_legs([48000, 49000, 51000, 52000], expiration)
        margin = strategy.calculate_margin_requirement(legs, 50000)

        assert margin > 0
        assert isinstance(margin, int)


if __name__ == "__main__":
    pytest.main([__file__])