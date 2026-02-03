"""Tests for strategy-specific reward calculators."""

import pytest
import numpy as np

from src.environments.strategy_rewards import (
    NeutralStrategyReward,
    DirectionalStrategyReward,
    VolatilityStrategyReward,
    get_strategy_reward_calculator
)
from src.environments.actions import ActionType
from src.strategies.base import StrategyType


class TestNeutralStrategyReward:
    """Test neutral strategy reward calculator."""

    def test_theta_bonus_in_profit_zone(self):
        """Test theta bonus when in profit zone."""
        calc = NeutralStrategyReward()

        # Iron Condor position in profit zone
        position_state = {
            'current_price': 105,  # Between inner strikes
            'strikes': [95, 100, 110, 115],  # Iron Condor strikes
            'days_to_expiry': 15,
            'initial_days_to_expiry': 30,
            'unrealized_pnl': 100,
            'max_profit': 500,
            'max_loss': -1000
        }

        # Calculate theta bonus
        theta_bonus = calc.calculate_theta_bonus(position_state)
        assert theta_bonus > 0  # Should get bonus for time decay in profit zone

    def test_theta_bonus_outside_profit_zone(self):
        """Test no theta bonus when outside profit zone."""
        calc = NeutralStrategyReward()

        # Position outside profit zone
        position_state = {
            'current_price': 120,  # Above upper strike
            'strikes': [95, 100, 110, 115],
            'days_to_expiry': 15,
            'initial_days_to_expiry': 30,
            'unrealized_pnl': -200,
            'max_profit': 500,
            'max_loss': -1000
        }

        theta_bonus = calc.calculate_theta_bonus(position_state)
        assert theta_bonus == 0  # No bonus outside profit zone

    def test_breach_penalty_below_strikes(self):
        """Test breach penalty when price below profit zone."""
        calc = NeutralStrategyReward()

        position_state = {
            'current_price': 95,  # Below lower inner strike
            'strikes': [95, 100, 110, 115],
            'unrealized_pnl': -150,
            'max_profit': 500,
            'max_loss': -1000
        }

        breach_penalty = calc.calculate_breach_penalty(position_state)
        assert breach_penalty < 0  # Should be negative penalty

    def test_breach_penalty_above_strikes(self):
        """Test breach penalty when price above profit zone."""
        calc = NeutralStrategyReward()

        position_state = {
            'current_price': 112,  # Above upper inner strike
            'strikes': [95, 100, 110, 115],
            'unrealized_pnl': -150,
            'max_profit': 500,
            'max_loss': -1000
        }

        breach_penalty = calc.calculate_breach_penalty(position_state)
        assert breach_penalty < 0  # Should be negative penalty

    def test_no_breach_penalty_in_range(self):
        """Test no breach penalty when price in profit zone."""
        calc = NeutralStrategyReward()

        position_state = {
            'current_price': 105,  # In profit zone
            'strikes': [95, 100, 110, 115],
            'unrealized_pnl': 150,
            'max_profit': 500,
            'max_loss': -1000
        }

        breach_penalty = calc.calculate_breach_penalty(position_state)
        assert breach_penalty == 0  # No penalty in range

    def test_strategy_specific_reward_integration(self):
        """Test full strategy-specific reward calculation."""
        calc = NeutralStrategyReward()

        position_state = {
            'current_price': 105,
            'strikes': [95, 100, 110, 115],
            'days_to_expiry': 10,
            'initial_days_to_expiry': 30,
            'unrealized_pnl': 200,
            'prev_pnl': 150,
            'max_profit': 500,
            'max_loss': -1000
        }

        reward = calc.calculate_reward(position_state, ActionType.HOLD)
        assert isinstance(reward, float)
        assert reward > 0  # Should be positive with theta bonus


class TestDirectionalStrategyReward:
    """Test directional strategy reward calculator."""

    def test_momentum_bonus_bullish(self):
        """Test momentum bonus for bullish strategy."""
        calc = DirectionalStrategyReward()

        position_state = {
            'current_price': 110,
            'entry_price': 100,
            'strategy_type': StrategyType.BULL_CALL_SPREAD,
            'unrealized_pnl': 300,
            'max_profit': 500,
            'max_loss': -200
        }

        momentum_bonus = calc.calculate_momentum_bonus(position_state)
        assert momentum_bonus > 0  # Positive for bullish move

    def test_momentum_bonus_bearish(self):
        """Test momentum bonus for bearish strategy."""
        calc = DirectionalStrategyReward()

        position_state = {
            'current_price': 95,
            'entry_price': 100,
            'strategy_type': StrategyType.BEAR_PUT_SPREAD,
            'unrealized_pnl': 250,
            'max_profit': 500,
            'max_loss': -200
        }

        momentum_bonus = calc.calculate_momentum_bonus(position_state)
        assert momentum_bonus > 0  # Positive for bearish move

    def test_no_momentum_bonus_wrong_direction(self):
        """Test no momentum bonus when price moves wrong way."""
        calc = DirectionalStrategyReward()

        position_state = {
            'current_price': 95,
            'entry_price': 100,
            'strategy_type': StrategyType.BULL_CALL_SPREAD,  # Bullish
            'unrealized_pnl': -100,
            'max_profit': 500,
            'max_loss': -200
        }

        momentum_bonus = calc.calculate_momentum_bonus(position_state)
        assert momentum_bonus == 0  # No bonus for wrong direction

    def test_early_exit_penalty(self):
        """Test penalty for early exit of profitable position."""
        calc = DirectionalStrategyReward()

        position_state = {
            'unrealized_pnl': 200,
            'days_to_expiry': 25,  # Very early
            'initial_days_to_expiry': 30,
            'max_profit': 500,
            'max_loss': -200
        }

        penalty = calc.calculate_early_exit_penalty(position_state)
        assert penalty < 0  # Should be negative penalty

    def test_no_early_exit_penalty_late(self):
        """Test no penalty for late exit."""
        calc = DirectionalStrategyReward()

        position_state = {
            'unrealized_pnl': 200,
            'days_to_expiry': 5,  # Late in cycle
            'initial_days_to_expiry': 30,
            'max_profit': 500,
            'max_loss': -200
        }

        penalty = calc.calculate_early_exit_penalty(position_state)
        assert penalty == 0  # No penalty for late exit

    def test_is_bullish_strategy(self):
        """Test bullish strategy identification."""
        calc = DirectionalStrategyReward()

        assert calc._is_bullish_strategy(StrategyType.LONG_CALL)
        assert calc._is_bullish_strategy(StrategyType.BULL_CALL_SPREAD)
        assert not calc._is_bullish_strategy(StrategyType.BEAR_CALL_SPREAD)
        assert not calc._is_bullish_strategy(StrategyType.LONG_PUT)


class TestVolatilityStrategyReward:
    """Test volatility strategy reward calculator."""

    def test_volatility_capture_long_vol(self):
        """Test volatility capture for long vol strategy."""
        calc = VolatilityStrategyReward()

        position_state = {
            'current_iv': 0.35,
            'entry_iv': 0.25,
            'strategy_type': StrategyType.LONG_STRADDLE,
            'unrealized_pnl': 400,
            'max_profit': 1000,
            'max_loss': -500
        }

        vol_reward = calc.calculate_volatility_capture(position_state)
        assert vol_reward > 0  # Positive for IV increase

    def test_volatility_capture_short_vol(self):
        """Test volatility capture for short vol strategy."""
        calc = VolatilityStrategyReward()

        position_state = {
            'current_iv': 0.20,
            'entry_iv': 0.30,
            'strategy_type': StrategyType.SHORT_STRADDLE,
            'unrealized_pnl': 300,
            'max_profit': 500,
            'max_loss': -2000
        }

        vol_reward = calc.calculate_volatility_capture(position_state)
        assert vol_reward > 0  # Positive for IV decrease

    def test_time_decay_penalty_long_premium(self):
        """Test time decay penalty for long premium strategies."""
        calc = VolatilityStrategyReward()

        position_state = {
            'days_to_expiry': 5,
            'initial_days_to_expiry': 30,
            'strategy_type': StrategyType.LONG_STRADDLE,
            'unrealized_pnl': -100,
            'max_profit': 1000,
            'max_loss': -500
        }

        decay_penalty = calc.calculate_time_decay_penalty(position_state)
        assert decay_penalty < 0  # Should be negative penalty

    def test_no_time_decay_penalty_short_premium(self):
        """Test no time decay penalty for short premium strategies."""
        calc = VolatilityStrategyReward()

        position_state = {
            'days_to_expiry': 5,
            'initial_days_to_expiry': 30,
            'strategy_type': StrategyType.SHORT_STRADDLE,
            'unrealized_pnl': 300,
            'max_profit': 500,
            'max_loss': -2000
        }

        # Short premium doesn't get time decay penalty
        reward = calc._calculate_strategy_specific_reward(
            position_state, ActionType.HOLD, None
        )
        # Should only have vol capture component, no decay penalty
        assert reward >= 0

    def test_is_long_volatility(self):
        """Test long volatility strategy identification."""
        calc = VolatilityStrategyReward()

        assert calc._is_long_volatility(StrategyType.LONG_STRADDLE)
        assert calc._is_long_volatility(StrategyType.LONG_STRANGLE)
        assert not calc._is_long_volatility(StrategyType.SHORT_STRADDLE)
        assert not calc._is_long_volatility(StrategyType.SHORT_STRANGLE)


class TestRewardCalculatorFactory:
    """Test reward calculator factory function."""

    def test_get_neutral_calculator(self):
        """Test getting calculator for neutral strategy."""
        calc = get_strategy_reward_calculator(StrategyType.IRON_CONDOR)
        assert isinstance(calc, NeutralStrategyReward)

        calc = get_strategy_reward_calculator(StrategyType.BUTTERFLY)
        assert isinstance(calc, NeutralStrategyReward)

    def test_get_directional_calculator(self):
        """Test getting calculator for directional strategy."""
        calc = get_strategy_reward_calculator(StrategyType.BULL_CALL_SPREAD)
        assert isinstance(calc, DirectionalStrategyReward)

        calc = get_strategy_reward_calculator(StrategyType.LONG_CALL)
        assert isinstance(calc, DirectionalStrategyReward)

    def test_get_volatility_calculator(self):
        """Test getting calculator for volatility strategy."""
        calc = get_strategy_reward_calculator(StrategyType.LONG_STRADDLE)
        assert isinstance(calc, VolatilityStrategyReward)

        calc = get_strategy_reward_calculator(StrategyType.CALENDAR_CALL)
        assert isinstance(calc, VolatilityStrategyReward)

    def test_hasattr_checks(self):
        """Test that strategy calculators have expected methods."""
        n = NeutralStrategyReward()
        assert hasattr(n, 'calculate_theta_bonus')
        assert hasattr(n, 'calculate_breach_penalty')

        d = DirectionalStrategyReward()
        assert hasattr(d, 'calculate_momentum_bonus')
        assert hasattr(d, 'calculate_early_exit_penalty')

        v = VolatilityStrategyReward()
        assert hasattr(v, 'calculate_volatility_capture')
        assert hasattr(v, 'calculate_time_decay_penalty')