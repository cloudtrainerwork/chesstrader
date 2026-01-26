"""
Comprehensive test suite for volatility-based options strategies.

Tests all four volatility strategies (Long/Short Straddle/Strangle) and
advanced volatility analysis tools for regime-based entry/exit criteria.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.strategies.volatility import (
    LongStraddleStrategy, ShortStraddleStrategy,
    LongStrangleStrategy, ShortStrangleStrategy,
    VolatilityAnalysis
)
from src.strategies.base import (
    MarketConditions, OptionType, StrategyCategory, RiskLevel
)


class TestVolatilityAnalysis:
    """Test suite for advanced volatility analysis tools."""

    def test_calculate_volatility_percentile(self):
        """Test volatility percentile calculation."""
        vol_history = [0.15, 0.18, 0.20, 0.25, 0.30, 0.28, 0.22, 0.19, 0.16, 0.21]

        # Test current vol in middle of range
        percentile = VolatilityAnalysis.calculate_volatility_percentile(0.20, vol_history)
        assert 0.2 <= percentile <= 0.4  # Should be in lower portion

        # Test current vol at high end
        percentile = VolatilityAnalysis.calculate_volatility_percentile(0.35, vol_history)
        assert percentile >= 0.9  # Should be very high percentile

        # Test current vol at low end
        percentile = VolatilityAnalysis.calculate_volatility_percentile(0.10, vol_history)
        assert percentile <= 0.1  # Should be very low percentile

    def test_detect_volatility_regime_transition(self):
        """Test volatility regime transition detection."""
        # Test transition from low vol to high vol
        vol_ranks = [0.2, 0.3, 0.5, 0.7, 0.8]
        transition, new_regime = VolatilityAnalysis.detect_volatility_regime_transition(
            vol_ranks, current_regime=4
        )
        assert transition is True
        assert new_regime == 3  # Should transition to high vol regime

        # Test no transition in stable conditions
        vol_ranks = [0.3, 0.35, 0.32, 0.28, 0.31]
        transition, new_regime = VolatilityAnalysis.detect_volatility_regime_transition(
            vol_ranks, current_regime=4
        )
        assert transition is False
        assert new_regime == 4  # Should stay in same regime

    def test_calculate_vega_exposure_limit(self):
        """Test vega exposure limit calculation."""
        portfolio_size = 1000000  # $10,000 in cents

        # Test in low volatility
        vega_limit = VolatilityAnalysis.calculate_vega_exposure_limit(
            portfolio_size, volatility_rank=0.2
        )
        assert vega_limit > 40000  # Should allow higher exposure in low vol

        # Test in high volatility
        vega_limit_high = VolatilityAnalysis.calculate_vega_exposure_limit(
            portfolio_size, volatility_rank=0.8
        )
        assert vega_limit_high < vega_limit  # Should be lower in high vol

    def test_estimate_gamma_risk(self):
        """Test gamma risk estimation."""
        strikes = [95000, 100000, 105000]  # $950, $1000, $1050
        underlying_price = 100000  # $1000

        gamma_metrics = VolatilityAnalysis.estimate_gamma_risk(
            strikes, underlying_price, time_to_expiration=30, volatility=0.25
        )

        assert "total_gamma_risk" in gamma_metrics
        assert "max_gamma_risk" in gamma_metrics
        assert gamma_metrics["total_gamma_risk"] > 0
        assert gamma_metrics["max_gamma_risk"] > 0

    def test_forecast_volatility_mean_reversion(self):
        """Test volatility mean reversion forecasting."""
        # Test high volatility mean reversion
        vol_history = [0.3] * 30 + [0.6]  # Spike in volatility
        forecast = VolatilityAnalysis.forecast_volatility_mean_reversion(0.8, vol_history)

        assert forecast["mean_reversion_prob"] > 0.5
        assert forecast["expected_direction"] < 0  # Should expect reversion down

        # Test low volatility expansion expectation
        vol_history = [0.6] * 30 + [0.2]  # Drop in volatility
        forecast = VolatilityAnalysis.forecast_volatility_mean_reversion(0.2, vol_history)

        assert forecast["expected_direction"] > 0  # Should expect expansion up

    def test_calculate_volatility_skew_impact(self):
        """Test volatility skew impact calculation."""
        put_vol = 0.25
        call_vol = 0.20
        strikes = [95000, 105000]  # Put and call strikes
        underlying_price = 100000

        skew_analysis = VolatilityAnalysis.calculate_volatility_skew_impact(
            put_vol, call_vol, underlying_price, strikes
        )

        assert skew_analysis["vol_skew"] == 0.05  # 5% skew
        assert skew_analysis["put_advantage"] is True
        assert skew_analysis["call_advantage"] is False


class TestLongStraddleStrategy:
    """Test suite for Long Straddle strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = LongStraddleStrategy()
        self.market_conditions = MarketConditions(
            regime=4,  # Low volatility regime
            volatility_rank=0.3,
            trend_strength=0.1,
            time_to_expiration=30,
            underlying_price=100000,
            risk_free_rate=0.05
        )

    def test_strategy_metadata(self):
        """Test strategy metadata."""
        assert self.strategy.name == "Long Straddle"
        assert self.strategy.category == StrategyCategory.VOLATILITY
        assert self.strategy.risk_level == RiskLevel.MEDIUM
        assert self.strategy.metadata.capital_requirement == 1.4

    def test_validate_market_conditions(self):
        """Test market condition validation."""
        # Valid conditions (low vol regime)
        assert self.strategy.validate_market_conditions(self.market_conditions) is True

        # Invalid conditions (high vol regime)
        bad_conditions = MarketConditions(
            regime=3, volatility_rank=0.8, trend_strength=0.1,
            time_to_expiration=30, underlying_price=100000
        )
        assert self.strategy.validate_market_conditions(bad_conditions) is False

        # Invalid conditions (too high volatility)
        high_vol_conditions = MarketConditions(
            regime=4, volatility_rank=0.7, trend_strength=0.1,
            time_to_expiration=30, underlying_price=100000
        )
        assert self.strategy.validate_market_conditions(high_vol_conditions) is False

    def test_calculate_entry_criteria(self):
        """Test entry criteria calculation."""
        entry_signal = self.strategy.calculate_entry_criteria(self.market_conditions)

        assert entry_signal.confidence > 0.0
        assert len(entry_signal.reasons) > 0
        assert entry_signal.recommended_size > 0.0

        # Test with optimal conditions
        optimal_conditions = MarketConditions(
            regime=4, volatility_rank=0.25, trend_strength=0.15,
            time_to_expiration=35, underlying_price=100000
        )
        optimal_entry = self.strategy.calculate_entry_criteria(optimal_conditions)
        assert optimal_entry.confidence > entry_signal.confidence

    def test_get_position_legs(self):
        """Test position leg construction."""
        strikes = [100000]  # ATM straddle
        expiration = datetime.now() + timedelta(days=30)

        legs = self.strategy.get_position_legs(strikes, expiration)

        assert len(legs) == 2  # Call and put
        assert legs[0].option_type in [OptionType.CALL, OptionType.PUT]
        assert legs[1].option_type in [OptionType.CALL, OptionType.PUT]
        assert legs[0].option_type != legs[1].option_type  # Different types
        assert all(leg.quantity == 1 for leg in legs)  # Both long
        assert all(leg.strike == 100000 for leg in legs)  # Same strike

    def test_get_risk_metrics(self):
        """Test risk metrics calculation."""
        strikes = [100000]
        risk_metrics = self.strategy.get_risk_metrics(
            strikes, underlying_price=100000, time_to_expiration=30, volatility=0.25
        )

        assert risk_metrics.max_profit == 2147483647  # Unlimited
        assert risk_metrics.max_loss > 0  # Premium paid
        assert len(risk_metrics.breakeven_points) == 2  # Upper and lower
        assert risk_metrics.profit_probability > 0.0
        assert risk_metrics.capital_requirement == risk_metrics.max_loss

    def test_validate_strategy_strikes(self):
        """Test strike validation."""
        underlying_price = 100000

        # Valid ATM strike
        assert self.strategy._validate_strategy_strikes([100000], underlying_price) is True

        # Valid near-ATM strike
        assert self.strategy._validate_strategy_strikes([102000], underlying_price) is True

        # Invalid far OTM strike
        assert self.strategy._validate_strategy_strikes([120000], underlying_price) is False

        # Invalid multiple strikes
        assert self.strategy._validate_strategy_strikes([100000, 105000], underlying_price) is False


class TestShortStraddleStrategy:
    """Test suite for Short Straddle strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ShortStraddleStrategy()
        self.market_conditions = MarketConditions(
            regime=3,  # High volatility regime
            volatility_rank=0.8,
            trend_strength=0.1,
            time_to_expiration=20,
            underlying_price=100000,
            risk_free_rate=0.05
        )

    def test_strategy_metadata(self):
        """Test strategy metadata."""
        assert self.strategy.name == "Short Straddle"
        assert self.strategy.category == StrategyCategory.VOLATILITY
        assert self.strategy.risk_level == RiskLevel.VERY_HIGH
        assert self.strategy.metadata.capital_requirement == 2.5

    def test_validate_market_conditions(self):
        """Test market condition validation."""
        # Valid conditions (high vol regime)
        assert self.strategy.validate_market_conditions(self.market_conditions) is True

        # Invalid conditions (low vol regime)
        bad_conditions = MarketConditions(
            regime=4, volatility_rank=0.3, trend_strength=0.1,
            time_to_expiration=20, underlying_price=100000
        )
        assert self.strategy.validate_market_conditions(bad_conditions) is False

        # Invalid conditions (crisis regime)
        crisis_conditions = MarketConditions(
            regime=8, volatility_rank=0.9, trend_strength=0.1,
            time_to_expiration=20, underlying_price=100000
        )
        assert self.strategy.validate_market_conditions(crisis_conditions) is False

    def test_calculate_entry_criteria_conservative(self):
        """Test entry criteria with high threshold due to unlimited risk."""
        entry_signal = self.strategy.calculate_entry_criteria(self.market_conditions)

        # Should have conservative position sizing
        assert entry_signal.recommended_size <= 0.8

        # Should require high confidence due to unlimited risk
        if entry_signal.should_enter:
            assert entry_signal.confidence >= 0.7

    def test_get_position_legs(self):
        """Test position leg construction."""
        strikes = [100000]  # ATM straddle
        expiration = datetime.now() + timedelta(days=20)

        legs = self.strategy.get_position_legs(strikes, expiration)

        assert len(legs) == 2  # Call and put
        assert all(leg.quantity == -1 for leg in legs)  # Both short
        assert all(leg.strike == 100000 for leg in legs)  # Same strike

    def test_get_risk_metrics(self):
        """Test risk metrics calculation."""
        strikes = [100000]
        risk_metrics = self.strategy.get_risk_metrics(
            strikes, underlying_price=100000, time_to_expiration=20, volatility=0.30
        )

        assert risk_metrics.max_profit > 0  # Premium collected
        assert risk_metrics.max_loss == 2147483647  # Unlimited
        assert len(risk_metrics.breakeven_points) == 2
        assert risk_metrics.margin_requirement > risk_metrics.capital_requirement


class TestLongStrangleStrategy:
    """Test suite for Long Strangle strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = LongStrangleStrategy()
        self.market_conditions = MarketConditions(
            regime=4, volatility_rank=0.25, trend_strength=0.1,
            time_to_expiration=45, underlying_price=100000
        )

    def test_strategy_metadata(self):
        """Test strategy metadata."""
        assert self.strategy.name == "Long Strangle"
        assert self.strategy.category == StrategyCategory.VOLATILITY
        assert self.strategy.risk_level == RiskLevel.MEDIUM
        assert self.strategy.metadata.capital_requirement == 1.2  # Lower than straddle

    def test_get_position_legs(self):
        """Test position leg construction for asymmetric strikes."""
        strikes = [95000, 105000]  # OTM put and call
        expiration = datetime.now() + timedelta(days=45)

        legs = self.strategy.get_position_legs(strikes, expiration)

        assert len(legs) == 2
        assert all(leg.quantity == 1 for leg in legs)  # Both long
        assert legs[0].strike == 95000  # Put strike
        assert legs[1].strike == 105000  # Call strike
        assert legs[0].option_type == OptionType.PUT
        assert legs[1].option_type == OptionType.CALL

    def test_validate_strategy_strikes(self):
        """Test strike validation for strangle."""
        underlying_price = 100000

        # Valid OTM strikes
        assert self.strategy._validate_strategy_strikes([95000, 105000], underlying_price) is True

        # Invalid - put strike higher than call strike
        assert self.strategy._validate_strategy_strikes([105000, 95000], underlying_price) is False

        # Invalid - current price not between strikes
        assert self.strategy._validate_strategy_strikes([105000, 110000], underlying_price) is False

        # Invalid - strikes too close (not OTM enough)
        assert self.strategy._validate_strategy_strikes([99000, 101000], underlying_price) is False

    def test_get_risk_metrics(self):
        """Test risk metrics with asymmetric strikes."""
        strikes = [95000, 105000]
        risk_metrics = self.strategy.get_risk_metrics(
            strikes, underlying_price=100000, time_to_expiration=45, volatility=0.22
        )

        assert risk_metrics.max_profit == 2147483647  # Unlimited
        assert risk_metrics.max_loss > 0  # Premium paid
        assert len(risk_metrics.breakeven_points) == 2
        # Breakevens should be wider than strangle strikes
        assert risk_metrics.breakeven_points[0] < 95000
        assert risk_metrics.breakeven_points[1] > 105000


class TestShortStrangleStrategy:
    """Test suite for Short Strangle strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ShortStrangleStrategy()
        self.market_conditions = MarketConditions(
            regime=3, volatility_rank=0.7, trend_strength=0.2,
            time_to_expiration=25, underlying_price=100000
        )

    def test_strategy_metadata(self):
        """Test strategy metadata."""
        assert self.strategy.name == "Short Strangle"
        assert self.strategy.category == StrategyCategory.VOLATILITY
        assert self.strategy.risk_level == RiskLevel.VERY_HIGH
        assert self.strategy.metadata.capital_requirement == 2.0

    def test_validate_market_conditions_tolerance(self):
        """Test strangle has more tolerance than straddle."""
        # Slightly lower vol rank should be acceptable for strangle
        moderate_vol_conditions = MarketConditions(
            regime=3, volatility_rank=0.55, trend_strength=0.2,
            time_to_expiration=25, underlying_price=100000
        )
        assert self.strategy.validate_market_conditions(moderate_vol_conditions) is True

    def test_get_position_legs(self):
        """Test position leg construction for short strangle."""
        strikes = [92000, 108000]  # Wide OTM strikes
        expiration = datetime.now() + timedelta(days=25)

        legs = self.strategy.get_position_legs(strikes, expiration)

        assert len(legs) == 2
        assert all(leg.quantity == -1 for leg in legs)  # Both short
        assert legs[0].option_type == OptionType.PUT
        assert legs[1].option_type == OptionType.CALL

    def test_calculate_entry_criteria_wider_tolerance(self):
        """Test that strangle has wider trend tolerance than straddle."""
        moderate_trend_conditions = MarketConditions(
            regime=3, volatility_rank=0.7, trend_strength=0.4,
            time_to_expiration=25, underlying_price=100000
        )
        entry_signal = self.strategy.calculate_entry_criteria(moderate_trend_conditions)

        # Should still get some confidence with moderate trend
        assert entry_signal.confidence > 0.0


class TestVolatilityStrategiesIntegration:
    """Integration tests for volatility strategies with regime system."""

    def test_regime_based_strategy_selection(self):
        """Test that strategies are selected based on appropriate regimes."""
        low_vol_conditions = MarketConditions(
            regime=4, volatility_rank=0.3, trend_strength=0.1,
            time_to_expiration=30, underlying_price=100000
        )

        high_vol_conditions = MarketConditions(
            regime=3, volatility_rank=0.8, trend_strength=0.1,
            time_to_expiration=30, underlying_price=100000
        )

        # Long strategies should be valid in low vol regime
        long_straddle = LongStraddleStrategy()
        long_strangle = LongStrangleStrategy()

        assert long_straddle.validate_market_conditions(low_vol_conditions) is True
        assert long_strangle.validate_market_conditions(low_vol_conditions) is True

        # Short strategies should be valid in high vol regime
        short_straddle = ShortStraddleStrategy()
        short_strangle = ShortStrangleStrategy()

        assert short_straddle.validate_market_conditions(high_vol_conditions) is True
        assert short_strangle.validate_market_conditions(high_vol_conditions) is True

        # Cross-validation: long strategies should reject high vol
        assert long_straddle.validate_market_conditions(high_vol_conditions) is False
        assert long_strangle.validate_market_conditions(high_vol_conditions) is False

        # Short strategies should reject low vol
        assert short_straddle.validate_market_conditions(low_vol_conditions) is False
        assert short_strangle.validate_market_conditions(low_vol_conditions) is False

    def test_volatility_strategy_risk_progression(self):
        """Test that risk levels progress logically across strategies."""
        long_straddle = LongStraddleStrategy()
        long_strangle = LongStrangleStrategy()
        short_straddle = ShortStraddleStrategy()
        short_strangle = ShortStrangleStrategy()

        # Long strategies should have lower capital requirements
        assert long_strangle.metadata.capital_requirement < long_straddle.metadata.capital_requirement

        # Short strategies should have higher capital requirements
        assert short_straddle.metadata.capital_requirement > long_straddle.metadata.capital_requirement
        assert short_strangle.metadata.capital_requirement > long_strangle.metadata.capital_requirement

        # Risk levels should progress appropriately
        assert long_straddle.risk_level == RiskLevel.MEDIUM
        assert long_strangle.risk_level == RiskLevel.MEDIUM
        assert short_straddle.risk_level == RiskLevel.VERY_HIGH
        assert short_strangle.risk_level == RiskLevel.VERY_HIGH

    def test_breakeven_width_comparison(self):
        """Test that strangles have wider breakevens than straddles."""
        straddle_strikes = [100000]
        strangle_strikes = [95000, 105000]

        long_straddle = LongStraddleStrategy()
        long_strangle = LongStrangleStrategy()

        straddle_metrics = long_straddle.get_risk_metrics(
            straddle_strikes, underlying_price=100000,
            time_to_expiration=30, volatility=0.25
        )

        strangle_metrics = long_strangle.get_risk_metrics(
            strangle_strikes, underlying_price=100000,
            time_to_expiration=30, volatility=0.25
        )

        # Strangle should have wider breakeven range
        straddle_range = (straddle_metrics.breakeven_points[1] -
                         straddle_metrics.breakeven_points[0])
        strangle_range = (strangle_metrics.breakeven_points[1] -
                         strangle_metrics.breakeven_points[0])

        assert strangle_range > straddle_range

    @patch('src.strategies.volatility.VolatilityAnalysis')
    def test_volatility_analysis_integration(self, mock_vol_analysis):
        """Test integration with volatility analysis tools."""
        # Mock volatility analysis responses
        mock_vol_analysis.calculate_volatility_percentile.return_value = 0.3
        mock_vol_analysis.detect_volatility_regime_transition.return_value = (False, 4)
        mock_vol_analysis.forecast_volatility_mean_reversion.return_value = {
            "mean_reversion_prob": 0.7,
            "expected_direction": 0.2
        }

        strategy = LongStraddleStrategy()
        conditions = MarketConditions(
            regime=4, volatility_rank=0.3, trend_strength=0.1,
            time_to_expiration=30, underlying_price=100000
        )

        entry_signal = strategy.calculate_entry_criteria(conditions)

        # Should successfully calculate entry criteria
        assert entry_signal.confidence > 0.0
        assert len(entry_signal.reasons) > 0


class TestVolatilityScenarios:
    """Test volatility strategies under various market scenarios."""

    def test_volatility_expansion_scenario(self):
        """Test strategies during volatility expansion."""
        # Initial low volatility
        initial_conditions = MarketConditions(
            regime=4, volatility_rank=0.2, trend_strength=0.1,
            time_to_expiration=30, underlying_price=100000
        )

        # Volatility expansion
        expanded_conditions = MarketConditions(
            regime=3, volatility_rank=0.8, trend_strength=0.1,
            time_to_expiration=25, underlying_price=100000
        )

        long_straddle = LongStraddleStrategy()

        # Should enter in low vol
        initial_entry = long_straddle.calculate_entry_criteria(initial_conditions)
        assert initial_entry.should_enter is True

        # Should consider exit in high vol (mocked position)
        mock_position = Mock()
        mock_position.days_to_expiration = 25
        mock_position.calculate_unrealized_pnl.return_value = 5000  # Profitable
        mock_position.calculate_max_loss.return_value = 10000  # Original premium

        exit_signal = long_straddle.calculate_exit_criteria(mock_position, expanded_conditions)
        # Should suggest taking profits due to vol expansion
        assert "volatility" in " ".join(exit_signal.reasons).lower()

    def test_volatility_crush_scenario(self):
        """Test strategies during volatility crush."""
        # High volatility environment
        high_vol_conditions = MarketConditions(
            regime=3, volatility_rank=0.8, trend_strength=0.1,
            time_to_expiration=20, underlying_price=100000
        )

        # Volatility crush
        crushed_conditions = MarketConditions(
            regime=4, volatility_rank=0.2, trend_strength=0.1,
            time_to_expiration=15, underlying_price=100000
        )

        short_straddle = ShortStraddleStrategy()

        # Should enter in high vol
        entry_signal = short_straddle.calculate_entry_criteria(high_vol_conditions)
        assert entry_signal.should_enter is True

        # Should consider profit taking after vol crush (mocked position)
        mock_position = Mock()
        mock_position.days_to_expiration = 15
        mock_position.calculate_unrealized_pnl.return_value = 3000  # Profitable
        mock_position.calculate_max_profit.return_value = 5000  # Premium collected
        mock_position.strikes = [100000]
        mock_position.current_underlying_price = 100000

        exit_signal = short_straddle.calculate_exit_criteria(mock_position, crushed_conditions)
        # Should suggest profit taking due to vol contraction
        assert exit_signal.exit_type == "profit" or "profit" in " ".join(exit_signal.reasons).lower()

    def test_assignment_risk_scenarios(self):
        """Test assignment risk handling in short strategies."""
        expiration_conditions = MarketConditions(
            regime=3, volatility_rank=0.7, trend_strength=0.1,
            time_to_expiration=3, underlying_price=100000  # 3 days to expiration
        )

        short_strangle = ShortStrangleStrategy()

        # Mock position approaching expiration
        mock_position = Mock()
        mock_position.days_to_expiration = 3
        mock_position.calculate_unrealized_pnl.return_value = 1000
        mock_position.calculate_max_profit.return_value = 3000
        mock_position.strikes = [95000, 105000]
        mock_position.current_underlying_price = 100000

        exit_signal = short_strangle.calculate_exit_criteria(mock_position, expiration_conditions)

        # Should have high urgency due to assignment risk
        assert exit_signal.urgency >= 0.5
        assert "assignment" in " ".join(exit_signal.reasons).lower()

    def test_gamma_risk_scenarios(self):
        """Test gamma risk in near-expiration scenarios."""
        conditions = MarketConditions(
            regime=4, volatility_rank=0.4, trend_strength=0.1,
            time_to_expiration=7, underlying_price=100000
        )

        long_straddle = LongStraddleStrategy()

        # Mock position near expiration
        mock_position = Mock()
        mock_position.days_to_expiration = 7
        mock_position.calculate_unrealized_pnl.return_value = -2000  # Losing
        mock_position.calculate_max_loss.return_value = 8000

        exit_signal = long_straddle.calculate_exit_criteria(mock_position, conditions)

        # Should consider exit due to time decay/gamma risk
        assert exit_signal.urgency > 0.0
        assert any("time" in reason.lower() or "gamma" in reason.lower()
                  for reason in exit_signal.reasons)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])