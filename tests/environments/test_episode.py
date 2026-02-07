"""Tests for episode management system."""

import pytest
from dataclasses import dataclass

from src.environments.episode import EpisodeManager, TerminalReason, EpisodeStats
from src.environments.base import PositionState
from src.strategies.base import StrategyType


@dataclass
class MockPositionState:
    """Mock position state for testing."""
    days_to_expiry: int = 30
    unrealized_pnl: float = 0.0
    max_loss: float = -500.0


class TestEpisodeManager:
    """Test episode management functionality."""

    def test_episode_manager_creation(self):
        """Test episode manager can be created."""
        em = EpisodeManager()
        assert em.max_steps == 252
        assert em.max_loss_threshold == 0.20
        assert isinstance(em.stats, EpisodeStats)

    def test_reset_clears_stats(self):
        """Test reset clears episode statistics."""
        em = EpisodeManager()
        em.stats.total_pnl = 100
        em.stats.steps_taken = 5

        em.reset()

        assert em.stats.total_pnl == 0
        assert em.stats.steps_taken == 0

    def test_terminal_position_closed(self):
        """Test terminal condition when position is closed."""
        em = EpisodeManager()
        state = {'position_closed': True}

        is_terminal, reason = em.is_terminal(state)

        assert is_terminal
        assert reason == TerminalReason.POSITION_CLOSED

    def test_terminal_no_position(self):
        """Test terminal condition when no position state."""
        em = EpisodeManager()
        state = {'position_state': None}

        is_terminal, reason = em.is_terminal(state)

        assert is_terminal
        assert reason == TerminalReason.POSITION_CLOSED

    def test_terminal_expiration_reached(self):
        """Test terminal condition when expiration reached."""
        em = EpisodeManager()
        position = MockPositionState(days_to_expiry=0)
        state = {'position_state': position}

        is_terminal, reason = em.is_terminal(state)

        assert is_terminal
        assert reason == TerminalReason.EXPIRATION_REACHED

    def test_terminal_max_loss_exceeded(self):
        """Test terminal condition when max loss exceeded."""
        em = EpisodeManager()
        position = MockPositionState(
            unrealized_pnl=-700,  # Exceeds max_loss * 1.2 = -600
            max_loss=-500
        )
        state = {'position_state': position}

        is_terminal, reason = em.is_terminal(state)

        assert is_terminal
        assert reason == TerminalReason.MAX_LOSS_EXCEEDED

    def test_terminal_max_steps_reached(self):
        """Test terminal condition when max steps reached."""
        em = EpisodeManager(max_steps=10)
        position = MockPositionState()
        state = {
            'position_state': position,
            'current_step': 10
        }

        is_terminal, reason = em.is_terminal(state)

        assert is_terminal
        assert reason == TerminalReason.MAX_STEPS_REACHED

    def test_not_terminal(self):
        """Test non-terminal condition."""
        em = EpisodeManager()
        position = MockPositionState(
            days_to_expiry=20,
            unrealized_pnl=-100,  # Within acceptable loss
            max_loss=-500
        )
        state = {
            'position_state': position,
            'current_step': 50
        }

        is_terminal, reason = em.is_terminal(state)

        assert not is_terminal
        assert reason == TerminalReason.NOT_TERMINAL

    def test_update_stats_basic(self):
        """Test basic statistics updating."""
        em = EpisodeManager()

        em.update_stats("HOLD", 1.0, 50.0)

        assert em.stats.steps_taken == 1
        assert em.stats.total_pnl == 50.0
        assert em.stats.max_pnl == 50.0
        assert em.stats.min_pnl == 50.0
        assert "HOLD" in em.stats.actions_taken

    def test_update_stats_action_counting(self):
        """Test action counting in statistics."""
        em = EpisodeManager()

        em.update_stats("ADJUST", 0, 0)
        em.update_stats("ROLL", 0, 0)
        em.update_stats("HOLD", 0, 0)

        assert em.stats.num_adjustments == 1
        assert em.stats.num_rolls == 1
        assert em.stats.steps_taken == 3

    def test_update_stats_drawdown_calculation(self):
        """Test drawdown calculation in statistics."""
        em = EpisodeManager()

        # Start with profit
        em.update_stats("HOLD", 0, 100)
        assert em.stats.max_drawdown == 0

        # Then lose some
        em.update_stats("HOLD", 0, 50)
        assert em.stats.max_drawdown == 0.5  # 50% drawdown from peak

        # Recover partially
        em.update_stats("HOLD", 0, 75)
        assert em.stats.max_drawdown == 0.5  # Max drawdown stays

    def test_terminal_reward_profitable_close(self):
        """Test terminal reward for profitable early close."""
        em = EpisodeManager()
        em.stats.steps_taken = 50  # Early in episode

        reward = em.calculate_terminal_reward(
            TerminalReason.POSITION_CLOSED, 100.0
        )

        assert reward == em.early_stop_bonus
        assert em.stats.win_flag is True

    def test_terminal_reward_loss_close(self):
        """Test terminal reward for loss close."""
        em = EpisodeManager()

        reward = em.calculate_terminal_reward(
            TerminalReason.POSITION_CLOSED, -100.0
        )

        assert reward == 0  # No bonus for loss
        assert em.stats.win_flag is False

    def test_terminal_reward_stop_loss(self):
        """Test terminal reward for stop loss."""
        em = EpisodeManager()

        reward = em.calculate_terminal_reward(
            TerminalReason.MAX_LOSS_EXCEEDED, -500.0
        )

        assert reward == -5.0  # Penalty for stop loss
        assert em.stats.win_flag is False

    def test_terminal_reward_expiration(self):
        """Test terminal reward for expiration."""
        em = EpisodeManager()

        # Profitable expiration
        reward1 = em.calculate_terminal_reward(
            TerminalReason.EXPIRATION_REACHED, 50.0
        )
        assert reward1 == 0  # No bonus/penalty
        assert em.stats.win_flag is True

        em.reset()

        # Loss at expiration
        reward2 = em.calculate_terminal_reward(
            TerminalReason.EXPIRATION_REACHED, -50.0
        )
        assert reward2 == 0  # No bonus/penalty
        assert em.stats.win_flag is False

    def test_episode_summary(self):
        """Test episode summary generation."""
        em = EpisodeManager()

        # Simulate some activity
        em.update_stats("HOLD", 0, 50)
        em.update_stats("ADJUST", -10, 40)
        em.stats.win_flag = True
        em.stats.terminal_reason = TerminalReason.POSITION_CLOSED

        summary = em.get_episode_summary()

        assert summary['total_pnl'] == 40
        assert summary['max_pnl'] == 50
        assert summary['min_pnl'] == 40
        assert summary['num_adjustments'] == 1
        assert summary['steps_taken'] == 2
        assert summary['win_flag'] is True
        assert summary['terminal_reason'] == 'position_closed'
        assert 'HOLD' in summary['actions_taken']
        assert 'ADJUST' in summary['actions_taken']

    def test_custom_parameters(self):
        """Test episode manager with custom parameters."""
        em = EpisodeManager(
            max_steps=100,
            max_loss_threshold=0.15,
            early_stop_bonus=10.0
        )

        assert em.max_steps == 100
        assert em.max_loss_threshold == 0.15
        assert em.early_stop_bonus == 10.0