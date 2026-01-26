"""
Comprehensive test suite for directional options strategies.

Tests all four directional spread strategies: Bull Call, Bear Call, Bull Put, Bear Put.
Validates strategy mechanics, risk calculations, regime integration, and edge cases.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.strategies.directional import (
    BullCallSpreadStrategy,
    BearCallSpreadStrategy,
    BullPutSpreadStrategy,
    BearPutSpreadStrategy
)
from src.strategies.base import (
    MarketConditions, StrategyType, StrategyCategory, RiskLevel,
    OptionType, EntrySignal, ExitSignal
)


class TestBullCallSpreadStrategy:
    """Test suite for Bull Call Spread strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = BullCallSpreadStrategy()
        self.base_conditions = MarketConditions(
            regime=1,  # Bull trending
            volatility_rank=0.5,
            trend_strength=0.4,
            time_to_expiration=35,
            underlying_price=10000,  # $100.00 in cents
            risk_free_rate=0.05
        )

    def test_strategy_metadata(self):
        """Test strategy metadata is correctly configured."""
        assert self.strategy.name == "Bull Call Spread"
        assert self.strategy.category == StrategyCategory.DIRECTIONAL
        assert self.strategy.risk_level == RiskLevel.LOW
        assert self.strategy.get_strategy_type() == StrategyType.BULL_CALL_SPREAD
        assert "bullish debit spread" in self.strategy.metadata.description.lower()

    def test_validate_market_conditions_favorable(self):
        """Test market condition validation for favorable conditions."""
        # Bull trending regime (1)
        conditions = self.base_conditions
        assert self.strategy.validate_market_conditions(conditions) is True

        # Recovery regime (6)
        conditions = MarketConditions(
            regime=6, volatility_rank=0.4, trend_strength=0.3,
            time_to_expiration=40, underlying_price=10000
        )
        assert self.strategy.validate_market_conditions(conditions) is True

    def test_validate_market_conditions_unfavorable(self):
        """Test market condition validation for unfavorable conditions."""
        # Bear trending regime
        bad_conditions = MarketConditions(
            regime=2, volatility_rank=0.5, trend_strength=-0.4,
            time_to_expiration=35, underlying_price=10000
        )
        assert self.strategy.validate_market_conditions(bad_conditions) is False

        # Too bearish trend
        bad_conditions = MarketConditions(
            regime=1, volatility_rank=0.5, trend_strength=-0.3,
            time_to_expiration=35, underlying_price=10000
        )
        assert self.strategy.validate_market_conditions(bad_conditions) is False

        # Too low volatility
        bad_conditions = MarketConditions(
            regime=1, volatility_rank=0.1, trend_strength=0.4,
            time_to_expiration=35, underlying_price=10000
        )
        assert self.strategy.validate_market_conditions(bad_conditions) is False

        # Too short expiration
        bad_conditions = MarketConditions(
            regime=1, volatility_rank=0.5, trend_strength=0.4,
            time_to_expiration=10, underlying_price=10000
        )
        assert self.strategy.validate_market_conditions(bad_conditions) is False

    def test_calculate_entry_criteria_high_confidence(self):
        """Test entry criteria calculation for high confidence setup."""
        # Perfect conditions: Bull trending, strong trend, optimal volatility/time
        conditions = MarketConditions(
            regime=1, volatility_rank=0.5, trend_strength=0.7,
            time_to_expiration=40, underlying_price=10000
        )

        signal = self.strategy.calculate_entry_criteria(conditions)

        assert signal.should_enter is True
        assert signal.confidence >= 0.8  # High confidence
        assert signal.recommended_size >= 1.0
        assert len(signal.reasons) > 0
        assert any("bull trending" in reason.lower() for reason in signal.reasons)

    def test_calculate_entry_criteria_low_confidence(self):
        """Test entry criteria calculation for marginal conditions."""
        # Marginal conditions
        conditions = MarketConditions(
            regime=6, volatility_rank=0.3, trend_strength=0.2,
            time_to_expiration=70, underlying_price=10000
        )

        signal = self.strategy.calculate_entry_criteria(conditions)

        # Should still enter due to acceptable conditions but with lower confidence
        assert signal.confidence < 0.7
        assert signal.recommended_size < 1.0  # Reduced size

    def test_calculate_entry_criteria_rejection(self):
        """Test entry criteria rejects unfavorable conditions."""
        bad_conditions = MarketConditions(
            regime=2, volatility_rank=0.5, trend_strength=-0.4,
            time_to_expiration=35, underlying_price=10000
        )

        signal = self.strategy.calculate_entry_criteria(bad_conditions)

        assert signal.should_enter is False
        assert signal.confidence == 0.0
        assert len(signal.reasons) > 0

    def test_get_risk_metrics_valid_strikes(self):
        """Test risk metrics calculation with valid strikes."""
        strikes = [9500, 10500]  # $95-$105 spread, $10 wide
        underlying_price = 10000

        metrics = self.strategy.get_risk_metrics(strikes, underlying_price, 35, 0.25)

        # Bull call spread: Max loss = debit paid, Max profit = spread width - debit
        assert metrics.max_loss > 0
        assert metrics.max_profit > 0
        assert metrics.max_profit < 1000  # Less than spread width
        assert len(metrics.breakeven_points) == 1
        assert metrics.breakeven_points[0] > strikes[0]  # Above long strike
        assert metrics.profit_probability > 0.0
        assert metrics.risk_reward_ratio > 0.0

    def test_get_risk_metrics_invalid_strikes(self):
        """Test risk metrics with invalid strike configurations."""
        with pytest.raises(ValueError, match="requires exactly 2 strikes"):
            self.strategy.get_risk_metrics([9500], 10000, 35, 0.25)

        with pytest.raises(ValueError, match="Long strike must be lower"):
            self.strategy.get_risk_metrics([10500, 9500], 10000, 35, 0.25)

    def test_get_position_legs(self):
        """Test position leg construction."""
        strikes = [9500, 10500]
        expiration = datetime.now() + timedelta(days=35)

        legs = self.strategy.get_position_legs(strikes, expiration)

        assert len(legs) == 2

        # Long call at lower strike
        assert legs[0].option_type == OptionType.CALL
        assert legs[0].strike == 9500
        assert legs[0].quantity == 1
        assert legs[0].expiration_date == expiration

        # Short call at higher strike
        assert legs[1].option_type == OptionType.CALL
        assert legs[1].strike == 10500
        assert legs[1].quantity == -1
        assert legs[1].expiration_date == expiration

    def test_validate_strategy_strikes(self):
        """Test strike validation logic."""
        underlying_price = 10000

        # Valid strikes
        assert self.strategy._validate_strategy_strikes([9500, 10500], underlying_price) is True

        # Invalid: wrong number of strikes
        assert self.strategy._validate_strategy_strikes([9500], underlying_price) is False
        assert self.strategy._validate_strategy_strikes([9500, 10000, 10500], underlying_price) is False

        # Invalid: long strike >= short strike
        assert self.strategy._validate_strategy_strikes([10500, 9500], underlying_price) is False
        assert self.strategy._validate_strategy_strikes([10000, 10000], underlying_price) is False

        # Invalid: spread too narrow
        assert self.strategy._validate_strategy_strikes([9990, 10000], underlying_price) is False

        # Invalid: spread too wide
        assert self.strategy._validate_strategy_strikes([8000, 12500], underlying_price) is False

        # Invalid: long strike too far from current price
        assert self.strategy._validate_strategy_strikes([7500, 8500], underlying_price) is False

        # Invalid: short strike below current price
        assert self.strategy._validate_strategy_strikes([9000, 9500], underlying_price) is False

    def test_calculate_exit_criteria_profit_target(self):
        """Test exit criteria for profit target scenarios."""
        # Mock position with good profit
        position = Mock()
        position.days_to_expiration = 20
        position.calculate_unrealized_pnl.return_value = 750  # Good profit
        position.calculate_max_profit.return_value = 1000
        position.calculate_max_loss.return_value = 500
        position.strikes = [9500, 10500]
        position.current_underlying_price = 10300

        signal = self.strategy.calculate_exit_criteria(position, self.base_conditions)

        assert signal.should_exit is True
        assert signal.exit_type == "profit"
        assert signal.urgency > 0.5
        assert any("profit" in reason.lower() for reason in signal.reasons)

    def test_calculate_exit_criteria_loss_limit(self):
        """Test exit criteria for loss limit scenarios."""
        # Mock position with significant loss
        position = Mock()
        position.days_to_expiration = 20
        position.calculate_unrealized_pnl.return_value = -400  # 80% of max loss
        position.calculate_max_profit.return_value = 1000
        position.calculate_max_loss.return_value = 500
        position.strikes = [9500, 10500]
        position.current_underlying_price = 9200

        signal = self.strategy.calculate_exit_criteria(position, self.base_conditions)

        assert signal.should_exit is True
        assert signal.exit_type == "loss"
        assert signal.urgency >= 0.8
        assert any("loss" in reason.lower() for reason in signal.reasons)

    def test_calculate_exit_criteria_regime_change(self):
        """Test exit criteria for unfavorable regime changes."""
        position = Mock()
        position.days_to_expiration = 20
        position.calculate_unrealized_pnl.return_value = 100
        position.calculate_max_profit.return_value = 1000
        position.calculate_max_loss.return_value = 500
        position.strikes = [9500, 10500]
        position.current_underlying_price = 10000

        # Bear trending regime
        bear_conditions = MarketConditions(
            regime=2, volatility_rank=0.5, trend_strength=-0.4,
            time_to_expiration=20, underlying_price=10000
        )

        signal = self.strategy.calculate_exit_criteria(position, bear_conditions)

        assert any("regime" in reason.lower() for reason in signal.reasons)
        assert signal.urgency > 0.0


class TestBearCallSpreadStrategy:
    """Test suite for Bear Call Spread strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = BearCallSpreadStrategy()
        self.base_conditions = MarketConditions(
            regime=2,  # Bear trending
            volatility_rank=0.7,
            trend_strength=-0.4,
            time_to_expiration=35,
            underlying_price=10000,
            risk_free_rate=0.05
        )

    def test_strategy_metadata(self):
        """Test strategy metadata is correctly configured."""
        assert self.strategy.name == "Bear Call Spread"
        assert self.strategy.category == StrategyCategory.DIRECTIONAL
        assert self.strategy.risk_level == RiskLevel.MEDIUM
        assert self.strategy.get_strategy_type() == StrategyType.BEAR_CALL_SPREAD
        assert "bearish credit spread" in self.strategy.metadata.description.lower()

    def test_validate_market_conditions_favorable(self):
        """Test market condition validation for favorable conditions."""
        # Bear trending regime (2)
        assert self.strategy.validate_market_conditions(self.base_conditions) is True

        # Distribution regime (7)
        conditions = MarketConditions(
            regime=7, volatility_rank=0.6, trend_strength=-0.1,
            time_to_expiration=40, underlying_price=10000
        )
        assert self.strategy.validate_market_conditions(conditions) is True

    def test_get_risk_metrics_credit_spread(self):
        """Test risk metrics for credit spread characteristics."""
        strikes = [10500, 11500]  # Short $105, Long $115 call spread
        underlying_price = 10000

        metrics = self.strategy.get_risk_metrics(strikes, underlying_price, 35, 0.3)

        # Bear call spread: Max profit = credit received, Max loss = spread width - credit
        assert metrics.max_profit > 0  # Credit received
        assert metrics.max_loss > metrics.max_profit  # Loss > credit received
        assert len(metrics.breakeven_points) == 1
        assert metrics.breakeven_points[0] > strikes[0]  # Above short strike

    def test_get_position_legs_credit_spread(self):
        """Test position leg construction for credit spread."""
        strikes = [10500, 11500]  # Short lower, long higher
        expiration = datetime.now() + timedelta(days=35)

        legs = self.strategy.get_position_legs(strikes, expiration)

        assert len(legs) == 2

        # Short call at lower strike
        assert legs[0].option_type == OptionType.CALL
        assert legs[0].strike == 10500
        assert legs[0].quantity == -1  # Short position

        # Long call at higher strike
        assert legs[1].option_type == OptionType.CALL
        assert legs[1].strike == 11500
        assert legs[1].quantity == 1  # Long position

    def test_assignment_risk_exit_criteria(self):
        """Test exit criteria considers assignment risk."""
        # Position where price is above short strike
        position = Mock()
        position.days_to_expiration = 20
        position.calculate_unrealized_pnl.return_value = -200
        position.calculate_max_profit.return_value = 300
        position.calculate_max_loss.return_value = 700
        position.strikes = [10500, 11500]
        position.current_underlying_price = 10600  # Above short strike

        signal = self.strategy.calculate_exit_criteria(position, self.base_conditions)

        assert signal.urgency > 0.5
        assert any("assignment" in reason.lower() for reason in signal.reasons)


class TestBullPutSpreadStrategy:
    """Test suite for Bull Put Spread strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = BullPutSpreadStrategy()
        self.base_conditions = MarketConditions(
            regime=1,  # Bull trending
            volatility_rank=0.7,
            trend_strength=0.4,
            time_to_expiration=35,
            underlying_price=10000,
            risk_free_rate=0.05
        )

    def test_strategy_metadata(self):
        """Test strategy metadata is correctly configured."""
        assert self.strategy.name == "Bull Put Spread"
        assert self.strategy.category == StrategyCategory.DIRECTIONAL
        assert self.strategy.risk_level == RiskLevel.MEDIUM
        assert self.strategy.get_strategy_type() == StrategyType.BULL_PUT_SPREAD
        assert "bullish credit spread" in self.strategy.metadata.description.lower()

    def test_get_risk_metrics_put_credit_spread(self):
        """Test risk metrics for put credit spread."""
        strikes = [9500, 8500]  # Short $95, Long $85 put spread
        underlying_price = 10000

        metrics = self.strategy.get_risk_metrics(strikes, underlying_price, 35, 0.3)

        # Bull put spread: Max profit = credit, Max loss = spread width - credit
        assert metrics.max_profit > 0
        assert metrics.max_loss > 0
        assert len(metrics.breakeven_points) == 1
        assert metrics.breakeven_points[0] < strikes[0]  # Below short strike

    def test_get_position_legs_put_spread(self):
        """Test position leg construction for put spread."""
        strikes = [9500, 8500]  # Short higher, long lower
        expiration = datetime.now() + timedelta(days=35)

        legs = self.strategy.get_position_legs(strikes, expiration)

        assert len(legs) == 2

        # Short put at higher strike
        assert legs[0].option_type == OptionType.PUT
        assert legs[0].strike == 9500
        assert legs[0].quantity == -1  # Short position

        # Long put at lower strike
        assert legs[1].option_type == OptionType.PUT
        assert legs[1].strike == 8500
        assert legs[1].quantity == 1  # Long position

    def test_put_assignment_risk(self):
        """Test put assignment risk evaluation."""
        position = Mock()
        position.days_to_expiration = 20
        position.calculate_unrealized_pnl.return_value = -150
        position.calculate_max_profit.return_value = 200
        position.calculate_max_loss.return_value = 800
        position.strikes = [9500, 8500]
        position.current_underlying_price = 9400  # Below short strike

        signal = self.strategy.calculate_exit_criteria(position, self.base_conditions)

        assert signal.urgency > 0.5
        assert any("assignment" in reason.lower() for reason in signal.reasons)


class TestBearPutSpreadStrategy:
    """Test suite for Bear Put Spread strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = BearPutSpreadStrategy()
        self.base_conditions = MarketConditions(
            regime=2,  # Bear trending
            volatility_rank=0.5,
            trend_strength=-0.5,
            time_to_expiration=35,
            underlying_price=10000,
            risk_free_rate=0.05
        )

    def test_strategy_metadata(self):
        """Test strategy metadata is correctly configured."""
        assert self.strategy.name == "Bear Put Spread"
        assert self.strategy.category == StrategyCategory.DIRECTIONAL
        assert self.strategy.risk_level == RiskLevel.LOW
        assert self.strategy.get_strategy_type() == StrategyType.BEAR_PUT_SPREAD
        assert "bearish debit spread" in self.strategy.metadata.description.lower()

    def test_get_risk_metrics_put_debit_spread(self):
        """Test risk metrics for put debit spread."""
        strikes = [10500, 9500]  # Long $105, Short $95 put spread
        underlying_price = 10000

        metrics = self.strategy.get_risk_metrics(strikes, underlying_price, 35, 0.3)

        # Bear put spread: Max loss = debit, Max profit = spread width - debit
        assert metrics.max_loss > 0
        assert metrics.max_profit > 0
        assert len(metrics.breakeven_points) == 1
        assert metrics.breakeven_points[0] < strikes[0]  # Below long strike

    def test_get_position_legs_put_debit_spread(self):
        """Test position leg construction for put debit spread."""
        strikes = [10500, 9500]  # Long higher, short lower
        expiration = datetime.now() + timedelta(days=35)

        legs = self.strategy.get_position_legs(strikes, expiration)

        assert len(legs) == 2

        # Long put at higher strike
        assert legs[0].option_type == OptionType.PUT
        assert legs[0].strike == 10500
        assert legs[0].quantity == 1  # Long position

        # Short put at lower strike
        assert legs[1].option_type == OptionType.PUT
        assert legs[1].strike == 9500
        assert legs[1].quantity == -1  # Short position


class TestDirectionalStrategiesIntegration:
    """Integration tests for all directional strategies."""

    def setup_method(self):
        """Set up all strategies for integration testing."""
        self.bull_call = BullCallSpreadStrategy()
        self.bear_call = BearCallSpreadStrategy()
        self.bull_put = BullPutSpreadStrategy()
        self.bear_put = BearPutSpreadStrategy()

        self.all_strategies = [
            self.bull_call, self.bear_call, self.bull_put, self.bear_put
        ]

    def test_all_strategies_implement_base_interface(self):
        """Test all strategies properly implement BaseStrategy interface."""
        for strategy in self.all_strategies:
            # Test required methods exist and return expected types
            assert hasattr(strategy, '_create_metadata')
            assert hasattr(strategy, 'validate_market_conditions')
            assert hasattr(strategy, 'calculate_entry_criteria')
            assert hasattr(strategy, 'calculate_exit_criteria')
            assert hasattr(strategy, 'get_risk_metrics')
            assert hasattr(strategy, 'get_position_legs')
            assert hasattr(strategy, '_validate_strategy_strikes')
            assert hasattr(strategy, 'get_strategy_type')

            # Test metadata properties
            assert strategy.name is not None
            assert strategy.category == StrategyCategory.DIRECTIONAL
            assert strategy.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]

            # Test strategy type mapping
            strategy_type = strategy.get_strategy_type()
            assert strategy_type in [
                StrategyType.BULL_CALL_SPREAD,
                StrategyType.BEAR_CALL_SPREAD,
                StrategyType.BULL_PUT_SPREAD,
                StrategyType.BEAR_PUT_SPREAD
            ]

    def test_regime_based_entry_logic(self):
        """Test that strategies respect regime-based entry criteria."""
        # Bull strategies should prefer regimes 1,6
        bull_conditions_good = MarketConditions(
            regime=1, volatility_rank=0.5, trend_strength=0.4,
            time_to_expiration=35, underlying_price=10000
        )

        bull_conditions_bad = MarketConditions(
            regime=2, volatility_rank=0.5, trend_strength=-0.4,
            time_to_expiration=35, underlying_price=10000
        )

        # Bull Call Spread
        assert self.bull_call.validate_market_conditions(bull_conditions_good) is True
        assert self.bull_call.validate_market_conditions(bull_conditions_bad) is False

        # Bull Put Spread
        bull_put_good = MarketConditions(
            regime=6, volatility_rank=0.6, trend_strength=0.2,
            time_to_expiration=35, underlying_price=10000
        )
        assert self.bull_put.validate_market_conditions(bull_put_good) is True
        assert self.bull_put.validate_market_conditions(bull_conditions_bad) is False

        # Bear strategies should prefer regimes 2,7
        bear_conditions_good = MarketConditions(
            regime=2, volatility_rank=0.6, trend_strength=-0.3,
            time_to_expiration=35, underlying_price=10000
        )

        bear_conditions_bad = MarketConditions(
            regime=1, volatility_rank=0.6, trend_strength=0.5,
            time_to_expiration=35, underlying_price=10000
        )

        # Bear Call Spread
        assert self.bear_call.validate_market_conditions(bear_conditions_good) is True
        assert self.bear_call.validate_market_conditions(bear_conditions_bad) is False

        # Bear Put Spread
        bear_put_good = MarketConditions(
            regime=7, volatility_rank=0.4, trend_strength=-0.4,
            time_to_expiration=35, underlying_price=10000
        )
        assert self.bear_put.validate_market_conditions(bear_put_good) is True
        assert self.bear_put.validate_market_conditions(bear_conditions_bad) is False

    def test_position_leg_consistency(self):
        """Test that position legs are constructed consistently."""
        expiration = datetime.now() + timedelta(days=35)
        underlying_price = 10000

        # Bull Call: Long low call, Short high call
        bull_call_strikes = [9500, 10500]
        bull_call_legs = self.bull_call.get_position_legs(bull_call_strikes, expiration)
        assert len(bull_call_legs) == 2
        assert all(leg.option_type == OptionType.CALL for leg in bull_call_legs)
        assert bull_call_legs[0].quantity == 1  # Long
        assert bull_call_legs[1].quantity == -1  # Short

        # Bear Call: Short low call, Long high call
        bear_call_strikes = [10500, 11500]
        bear_call_legs = self.bear_call.get_position_legs(bear_call_strikes, expiration)
        assert len(bear_call_legs) == 2
        assert all(leg.option_type == OptionType.CALL for leg in bear_call_legs)
        assert bear_call_legs[0].quantity == -1  # Short
        assert bear_call_legs[1].quantity == 1  # Long

        # Bull Put: Short high put, Long low put
        bull_put_strikes = [9500, 8500]
        bull_put_legs = self.bull_put.get_position_legs(bull_put_strikes, expiration)
        assert len(bull_put_legs) == 2
        assert all(leg.option_type == OptionType.PUT for leg in bull_put_legs)
        assert bull_put_legs[0].quantity == -1  # Short high strike
        assert bull_put_legs[1].quantity == 1   # Long low strike

        # Bear Put: Long high put, Short low put
        bear_put_strikes = [10500, 9500]
        bear_put_legs = self.bear_put.get_position_legs(bear_put_strikes, expiration)
        assert len(bear_put_legs) == 2
        assert all(leg.option_type == OptionType.PUT for leg in bear_put_legs)
        assert bear_put_legs[0].quantity == 1   # Long high strike
        assert bear_put_legs[1].quantity == -1  # Short low strike

    def test_risk_metrics_consistency(self):
        """Test risk metrics are calculated consistently across strategies."""
        underlying_price = 10000

        for strategy in self.all_strategies:
            if "Call" in strategy.name:
                if "Bull" in strategy.name:
                    strikes = [9500, 10500]  # Long low, short high
                else:
                    strikes = [10500, 11500]  # Short low, long high
            else:  # Put strategies
                if "Bull" in strategy.name:
                    strikes = [9500, 8500]  # Short high, long low
                else:
                    strikes = [10500, 9500]  # Long high, short low

            metrics = strategy.get_risk_metrics(strikes, underlying_price, 35, 0.3)

            # All strategies should have positive max profit and max loss
            assert metrics.max_profit > 0, f"{strategy.name} should have positive max profit"
            assert metrics.max_loss > 0, f"{strategy.name} should have positive max loss"

            # Should have exactly one breakeven point
            assert len(metrics.breakeven_points) == 1, f"{strategy.name} should have one breakeven"

            # Risk/reward ratio should be reasonable
            assert 0 < metrics.risk_reward_ratio < 10, f"{strategy.name} risk/reward ratio unreasonable"

            # Margin requirement should be >= capital requirement
            assert metrics.margin_requirement >= metrics.capital_requirement

    def test_comprehensive_edge_cases(self):
        """Test edge cases and error conditions."""
        underlying_price = 10000
        expiration = datetime.now() + timedelta(days=35)

        for strategy in self.all_strategies:
            # Test invalid strike counts
            with pytest.raises(ValueError):
                strategy.get_risk_metrics([9500], underlying_price, 35, 0.3)

            with pytest.raises(ValueError):
                strategy.get_position_legs([9500], expiration)

            # Test invalid strike relationships
            if "Bull Call" in strategy.name or "Bear Put" in strategy.name:
                # These expect long strike < short strike or long strike > short strike
                continue

            # Test strike validation edge cases
            assert strategy._validate_strategy_strikes([], underlying_price) is False
            assert strategy._validate_strategy_strikes([9500, 9500], underlying_price) is False

    def test_all_import_successfully(self):
        """Test that all directional strategies can be imported successfully."""
        # This test validates the verification command from the plan
        from src.strategies.directional import (
            BullCallSpreadStrategy,
            BearCallSpreadStrategy,
            BullPutSpreadStrategy,
            BearPutSpreadStrategy
        )

        # Instantiate all strategies
        strategies = [
            BullCallSpreadStrategy(),
            BearCallSpreadStrategy(),
            BullPutSpreadStrategy(),
            BearPutSpreadStrategy()
        ]

        # Verify they all have names
        names = [s.name for s in strategies]
        expected_names = [
            "Bull Call Spread",
            "Bear Call Spread",
            "Bull Put Spread",
            "Bear Put Spread"
        ]

        assert names == expected_names
        print("All directional strategies imported successfully")