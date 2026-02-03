"""Tests for base reward calculator."""

import pytest
import numpy as np

from src.environments.rewards import RewardCalculator
from src.environments.actions import ActionType


class TestRewardCalculator:
    """Test base reward calculator functionality."""

    class ConcreteRewardCalculator(RewardCalculator):
        """Concrete implementation for testing."""
        pass

    def test_reward_calculator_creation(self):
        """Test reward calculator can be instantiated."""
        calc = self.ConcreteRewardCalculator()
        assert calc.sharpe_scaling is True
        assert calc.drawdown_threshold == 0.10
        assert calc.drawdown_penalty_factor == 0.5

    def test_pnl_reward_on_close(self):
        """Test P/L reward when closing position."""
        calc = self.ConcreteRewardCalculator()

        position_state = {
            'unrealized_pnl': 500,
            'max_profit': 1000,
            'max_loss': -500
        }

        reward = calc.calculate_reward(position_state, 'CLOSE')
        assert isinstance(reward, float)
        # Should be normalized P/L
        assert reward > 0

    def test_pnl_reward_on_hold(self):
        """Test P/L change reward when holding."""
        calc = self.ConcreteRewardCalculator()

        position_state = {
            'unrealized_pnl': 200,
            'prev_pnl': 150,
            'max_profit': 1000,
            'max_loss': -500
        }

        reward = calc.calculate_reward(position_state, 'HOLD')
        assert isinstance(reward, float)
        # Should be positive for P/L improvement
        assert reward > 0

    def test_adjustment_cost_penalty(self):
        """Test adjustment action includes cost penalty."""
        calc = self.ConcreteRewardCalculator()

        position_state = {
            'unrealized_pnl': 200,
            'prev_pnl': 200,  # No P/L change
            'adjustment_cost': 10,
            'max_profit': 1000,
            'max_loss': -500
        }

        reward = calc.calculate_reward(position_state, 'ADJUST')
        assert reward < 0  # Should be negative due to cost

    def test_roll_cost_penalty(self):
        """Test rolling action includes cost penalty."""
        calc = self.ConcreteRewardCalculator()

        position_state = {
            'unrealized_pnl': 200,
            'prev_pnl': 200,  # No P/L change
            'roll_cost': 15,
            'max_profit': 1000,
            'max_loss': -500
        }

        reward = calc.calculate_reward(position_state, 'ROLL')
        assert reward < 0  # Should be negative due to cost

    def test_drawdown_penalty_applied(self):
        """Test drawdown penalty when loss exceeds threshold."""
        calc = self.ConcreteRewardCalculator(
            drawdown_threshold=0.10,
            drawdown_penalty_factor=0.5
        )

        # Position with small loss (no penalty)
        normal_state = {
            'unrealized_pnl': -50,
            'max_loss': -1000,
            'max_profit': 1000
        }
        normal_reward = calc.calculate_reward(normal_state, 'CLOSE')

        # Position with large loss (penalty applied)
        drawdown_state = {
            'unrealized_pnl': -150,  # 15% of max loss
            'max_loss': -1000,
            'max_profit': 1000
        }
        drawdown_reward = calc.calculate_reward(drawdown_state, 'CLOSE')

        # Drawdown reward should be penalized (more negative)
        assert drawdown_reward < normal_reward

    def test_position_size_normalization(self):
        """Test rewards are normalized by position size."""
        calc = self.ConcreteRewardCalculator(sharpe_scaling=False)

        # Small position
        small_position = {
            'unrealized_pnl': 100,
            'max_profit': 200,
            'max_loss': -100
        }
        small_reward = calc.calculate_reward(small_position, 'CLOSE')

        # Large position with proportional P/L
        large_position = {
            'unrealized_pnl': 1000,
            'max_profit': 2000,
            'max_loss': -1000
        }
        large_reward = calc.calculate_reward(large_position, 'CLOSE')

        # Normalized rewards should be similar
        assert abs(small_reward - large_reward) < 10  # Within 10% after normalization

    def test_sharpe_scaling(self):
        """Test Sharpe ratio scaling of rewards."""
        calc = self.ConcreteRewardCalculator(sharpe_scaling=True)

        # Build up returns history with varying volatility
        position_state = {
            'unrealized_pnl': 100,
            'prev_pnl': 0,
            'max_profit': 1000,
            'max_loss': -500
        }

        # Add some historical returns
        for i in range(10):
            calc._update_returns_history(i * 10)  # Low volatility

        low_vol_reward = calc.calculate_reward(position_state, 'HOLD')

        # Reset and add high volatility returns
        calc.reset()
        for i in range(10):
            calc._update_returns_history(i * 50 * (-1)**i)  # High volatility

        high_vol_reward = calc.calculate_reward(position_state, 'HOLD')

        # High volatility should result in scaled down reward
        # (Sharpe scaling reduces rewards in high volatility)
        assert abs(high_vol_reward) < abs(low_vol_reward)

    def test_returns_history_maintained(self):
        """Test returns history is properly maintained."""
        calc = self.ConcreteRewardCalculator()

        position_state = {
            'unrealized_pnl': 100,
            'max_profit': 1000,
            'max_loss': -500
        }

        # Generate many rewards
        for i in range(150):
            calc.calculate_reward(position_state, 'HOLD')

        # History should be capped at max size
        assert len(calc.returns_history) <= calc.max_history_size

    def test_reset_clears_history(self):
        """Test reset clears returns history."""
        calc = self.ConcreteRewardCalculator()

        position_state = {
            'unrealized_pnl': 100,
            'max_profit': 1000,
            'max_loss': -500
        }

        # Generate some rewards
        for i in range(10):
            calc.calculate_reward(position_state, 'HOLD')

        assert len(calc.returns_history) > 0

        # Reset should clear history
        calc.reset()
        assert len(calc.returns_history) == 0

    def test_action_type_enum_conversion(self):
        """Test string actions are converted to ActionType."""
        calc = self.ConcreteRewardCalculator()

        position_state = {
            'unrealized_pnl': 100,
            'max_profit': 1000,
            'max_loss': -500
        }

        # Test with string
        reward_str = calc.calculate_reward(position_state, 'HOLD')

        # Test with enum
        reward_enum = calc.calculate_reward(position_state, ActionType.HOLD)

        # Both should work
        assert isinstance(reward_str, float)
        assert isinstance(reward_enum, float)