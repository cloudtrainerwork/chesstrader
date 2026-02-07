"""
Episode management system for options trading environment.

Handles episode lifecycle, terminal conditions, and statistics tracking.
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class TerminalReason(Enum):
    """Reasons why an episode can terminate."""
    POSITION_CLOSED = "position_closed"
    EXPIRATION_REACHED = "expiration_reached"
    MAX_LOSS_EXCEEDED = "max_loss_exceeded"
    MAX_STEPS_REACHED = "max_steps_reached"
    NOT_TERMINAL = "not_terminal"


@dataclass
class EpisodeStats:
    """Statistics tracked during an episode."""
    total_pnl: float = 0.0
    max_pnl: float = 0.0
    min_pnl: float = 0.0
    max_drawdown: float = 0.0
    num_adjustments: int = 0
    num_rolls: int = 0
    steps_taken: int = 0
    win_flag: Optional[bool] = None
    terminal_reason: Optional[TerminalReason] = None
    actions_taken: List[str] = field(default_factory=list)


class EpisodeManager:
    """
    Manages episode lifecycle and terminal conditions.

    Tracks when episodes should end and maintains episode statistics.
    """

    def __init__(self,
                 max_steps: int = 252,
                 max_loss_threshold: float = 0.20,  # 20% max loss
                 early_stop_bonus: float = 5.0):
        """
        Initialize episode manager.

        Args:
            max_steps: Maximum steps in an episode (trading days)
            max_loss_threshold: Maximum loss threshold (fraction of max_loss)
            early_stop_bonus: Bonus reward for early profitable stopping
        """
        self.max_steps = max_steps
        self.max_loss_threshold = max_loss_threshold
        self.early_stop_bonus = early_stop_bonus

        # Current episode state
        self.stats = EpisodeStats()
        self.reset()

    def reset(self):
        """Reset for new episode."""
        self.stats = EpisodeStats()

    def is_terminal(self, state: Dict[str, Any]) -> Tuple[bool, TerminalReason]:
        """
        Check if current state is terminal.

        Args:
            state: Current environment state

        Returns:
            Tuple of (is_terminal, reason)
        """
        # Position closed
        if state.get('position_closed', False):
            return True, TerminalReason.POSITION_CLOSED

        # Check position state
        position_state = state.get('position_state')
        if position_state is None:
            return True, TerminalReason.POSITION_CLOSED

        # Days to expiry check
        days_to_expiry = getattr(position_state, 'days_to_expiry', 1)
        if days_to_expiry <= 0:
            return True, TerminalReason.EXPIRATION_REACHED

        # Max loss exceeded
        unrealized_pnl = getattr(position_state, 'unrealized_pnl', 0)
        max_loss = getattr(position_state, 'max_loss', -1000)
        if max_loss < 0 and unrealized_pnl < max_loss * (1 + self.max_loss_threshold):
            return True, TerminalReason.MAX_LOSS_EXCEEDED

        # Max steps reached
        current_step = state.get('current_step', 0)
        if current_step >= self.max_steps:
            return True, TerminalReason.MAX_STEPS_REACHED

        return False, TerminalReason.NOT_TERMINAL

    def update_stats(self, action_name: str, reward: float, pnl: float):
        """
        Update episode statistics.

        Args:
            action_name: Name of action taken
            reward: Reward received
            pnl: Current P/L
        """
        # Track actions
        self.stats.actions_taken.append(action_name)
        self.stats.steps_taken += 1

        # Count adjustments and rolls
        if action_name == "ADJUST":
            self.stats.num_adjustments += 1
        elif action_name == "ROLL":
            self.stats.num_rolls += 1

        # Update P/L tracking
        self.stats.total_pnl = pnl
        self.stats.max_pnl = max(self.stats.max_pnl, pnl)
        self.stats.min_pnl = min(self.stats.min_pnl, pnl)

        # Calculate drawdown
        if self.stats.max_pnl > 0:
            current_drawdown = (self.stats.max_pnl - pnl) / self.stats.max_pnl
            self.stats.max_drawdown = max(self.stats.max_drawdown, current_drawdown)

    def calculate_terminal_reward(self, reason: TerminalReason, final_pnl: float) -> float:
        """
        Calculate additional terminal reward.

        Args:
            reason: Reason for termination
            final_pnl: Final P/L of the position

        Returns:
            Terminal reward bonus
        """
        terminal_reward = 0.0

        if reason == TerminalReason.POSITION_CLOSED:
            # Determine win/loss
            self.stats.win_flag = final_pnl > 0
            self.stats.terminal_reason = reason

            # Early stop bonus if profitable
            if final_pnl > 0 and self.stats.steps_taken < self.max_steps * 0.5:
                terminal_reward = self.early_stop_bonus

        elif reason == TerminalReason.EXPIRATION_REACHED:
            # Natural expiration
            self.stats.win_flag = final_pnl > 0
            self.stats.terminal_reason = reason

        elif reason == TerminalReason.MAX_LOSS_EXCEEDED:
            # Risk management stop
            self.stats.win_flag = False
            self.stats.terminal_reason = reason
            terminal_reward = -5.0  # Penalty for hitting stop loss

        elif reason == TerminalReason.MAX_STEPS_REACHED:
            # Time limit reached
            self.stats.win_flag = final_pnl > 0
            self.stats.terminal_reason = reason

        return terminal_reward

    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Get summary of completed episode.

        Returns:
            Dictionary with episode statistics
        """
        return {
            'total_pnl': self.stats.total_pnl,
            'max_pnl': self.stats.max_pnl,
            'min_pnl': self.stats.min_pnl,
            'max_drawdown': self.stats.max_drawdown,
            'num_adjustments': self.stats.num_adjustments,
            'num_rolls': self.stats.num_rolls,
            'steps_taken': self.stats.steps_taken,
            'win_flag': self.stats.win_flag,
            'terminal_reason': self.stats.terminal_reason.value if self.stats.terminal_reason else None,
            'actions_taken': self.stats.actions_taken.copy()
        }