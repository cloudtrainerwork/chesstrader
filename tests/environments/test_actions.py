"""Tests for action space and step function."""

import pytest
import numpy as np
from gym import spaces

from src.environments.base import OptionsEnvironment
from src.environments.actions import ActionType, ActionValidator, PositionAdjustment, calculate_action_cost
from src.strategies.base import StrategyType


class TestActionSpace:
    """Test action space definition and functionality."""

    def test_action_space_defined(self):
        """Test action space is properly defined."""
        env = OptionsEnvironment()
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 4

    def test_action_type_enum(self):
        """Test ActionType enum values."""
        assert ActionType.HOLD == 0
        assert ActionType.CLOSE == 1
        assert ActionType.ADJUST == 2
        assert ActionType.ROLL == 3

    def test_step_function_returns(self):
        """Test step function returns correct tuple."""
        env = OptionsEnvironment()
        env.reset()

        obs, reward, done, info = env.step(0)  # HOLD action

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (30,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_updates_state(self):
        """Test step function updates environment state."""
        env = OptionsEnvironment()
        env.reset()
        initial_step = env.current_step

        env.step(0)  # HOLD action

        assert env.current_step == initial_step + 1

    def test_hold_action(self):
        """Test HOLD action execution."""
        env = OptionsEnvironment()
        env.reset()

        initial_pnl = env.position_state.unrealized_pnl
        obs, reward, done, info = env.step(ActionType.HOLD)

        # Position should still exist
        assert env.position_state is not None
        assert not done  # Holding shouldn't end episode early

    def test_close_action(self):
        """Test CLOSE action execution."""
        env = OptionsEnvironment()
        env.reset()

        # Set some P/L
        env.position_state.unrealized_pnl = 100.0
        initial_capital = env.capital

        obs, reward, done, info = env.step(ActionType.CLOSE)

        # Position should be closed
        assert env.position_state is None
        assert done  # Closing ends episode
        # Capital should be updated
        assert env.capital > initial_capital

    def test_adjust_action(self):
        """Test ADJUST action execution."""
        env = OptionsEnvironment()
        env.reset()

        initial_strikes = env.position_state.strikes.copy()
        obs, reward, done, info = env.step(ActionType.ADJUST)

        # Position should still exist
        assert env.position_state is not None
        # Reward should reflect adjustment cost
        assert reward <= 0  # Adjustment has cost

    def test_roll_action(self):
        """Test ROLL action execution."""
        env = OptionsEnvironment()
        env.reset()

        # Set close to expiration to make rolling valid
        env.position_state.days_to_expiry = 10
        initial_expiry = env.position_state.days_to_expiry

        obs, reward, done, info = env.step(ActionType.ROLL)

        # Check if roll was executed (may be invalid if >14 days)
        if not info.get('invalid_action', False):
            assert env.position_state.days_to_expiry > initial_expiry
            assert reward < 0  # Rolling has cost


class TestActionValidation:
    """Test action validation logic."""

    def test_hold_always_valid(self):
        """Test HOLD is always valid."""
        env = OptionsEnvironment()
        env.reset()

        is_valid, reason = ActionValidator.is_action_valid(
            ActionType.HOLD, env.position_state, 0, 100
        )
        assert is_valid
        assert reason == ""

    def test_close_valid_with_position(self):
        """Test CLOSE is valid when position exists."""
        env = OptionsEnvironment()
        env.reset()

        is_valid, reason = ActionValidator.is_action_valid(
            ActionType.CLOSE, env.position_state, 0, 100
        )
        assert is_valid

    def test_adjust_invalid_near_expiry(self):
        """Test ADJUST is invalid close to expiration."""
        env = OptionsEnvironment()
        env.reset()
        env.position_state.days_to_expiry = 5  # Too close

        is_valid, reason = ActionValidator.is_action_valid(
            ActionType.ADJUST, env.position_state, 0, 100
        )
        assert not is_valid
        assert "close to expiration" in reason

    def test_roll_valid_near_expiry(self):
        """Test ROLL is valid only near expiration."""
        env = OptionsEnvironment()
        env.reset()

        # Too early to roll
        env.position_state.days_to_expiry = 20
        is_valid, reason = ActionValidator.is_action_valid(
            ActionType.ROLL, env.position_state, 0, 100
        )
        assert not is_valid
        assert "Too early" in reason

        # Good time to roll
        env.position_state.days_to_expiry = 10
        is_valid, reason = ActionValidator.is_action_valid(
            ActionType.ROLL, env.position_state, 0, 100
        )
        assert is_valid


class TestPositionAdjustment:
    """Test position adjustment logic."""

    def test_adjust_position_logic(self):
        """Test position adjustment calculations."""
        env = OptionsEnvironment()
        env.reset()

        # Simulate price move
        env.position_state.current_price = 110  # Moved from 105

        adjustment = PositionAdjustment.adjust_position(
            env.position_state, env.position_state.current_price
        )

        assert adjustment['type'] == 'adjust'
        assert 'new_strikes' in adjustment
        assert 'cost' in adjustment

    def test_roll_position_logic(self):
        """Test position rolling calculations."""
        env = OptionsEnvironment()
        env.reset()

        roll = PositionAdjustment.roll_position(env.position_state)

        assert roll['type'] == 'roll'
        assert roll['new_expiry'] > roll['original_expiry']
        assert roll['cost'] > 0


class TestActionCosts:
    """Test action cost calculations."""

    def test_action_costs(self):
        """Test cost calculation for different actions."""
        env = OptionsEnvironment()
        env.reset()

        hold_cost = calculate_action_cost(ActionType.HOLD, env.position_state)
        assert hold_cost == 0.0

        close_cost = calculate_action_cost(ActionType.CLOSE, env.position_state)
        assert close_cost < 0  # Negative cost

        adjust_cost = calculate_action_cost(ActionType.ADJUST, env.position_state)
        assert adjust_cost < 0

        roll_cost = calculate_action_cost(ActionType.ROLL, env.position_state)
        assert roll_cost < 0


class TestTerminalConditions:
    """Test episode terminal conditions."""

    def test_terminal_on_close(self):
        """Test episode ends when position closed."""
        env = OptionsEnvironment()
        env.reset()

        obs, reward, done, info = env.step(ActionType.CLOSE)
        assert done

    def test_terminal_on_max_steps(self):
        """Test episode ends at max steps."""
        env = OptionsEnvironment(max_steps=5)
        env.reset()

        for i in range(5):
            obs, reward, done, info = env.step(ActionType.HOLD)
            if i < 4:
                assert not done
            else:
                assert done  # Should be done at step 5

    def test_terminal_on_expiry(self):
        """Test episode ends when option expires."""
        env = OptionsEnvironment()
        env.reset()
        env.position_state.days_to_expiry = 1

        obs, reward, done, info = env.step(ActionType.HOLD)
        assert done  # Should be done when days_to_expiry hits 0

    def test_terminal_on_max_loss(self):
        """Test episode ends when max loss exceeded."""
        env = OptionsEnvironment()
        env.reset()

        # Set large loss
        env.position_state.unrealized_pnl = env.position_state.max_loss * 2

        obs, reward, done, info = env.step(ActionType.HOLD)
        assert done  # Should be done when loss too large