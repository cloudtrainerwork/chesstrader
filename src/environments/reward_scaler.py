"""
Reward scaling and normalization system for stable RL training.

Implements online normalization using running statistics with Welford's algorithm.
"""

import numpy as np
from typing import Optional


class RewardScaler:
    """
    Online reward scaler using running statistics.

    Uses Welford's algorithm for numerically stable computation of
    running mean and variance, with z-score normalization and clipping.
    """

    def __init__(self,
                 clip_range: tuple = (-3, 3),
                 temperature: float = 1.0,
                 exploration_bonus: float = 0.1,
                 exploration_decay: float = 0.995):
        """
        Initialize reward scaler.

        Args:
            clip_range: Range for clipping normalized rewards
            temperature: Temperature parameter for reward sensitivity
            exploration_bonus: Initial exploration bonus
            exploration_decay: Decay rate for exploration bonus per episode
        """
        self.clip_range = clip_range
        self.temperature = temperature
        self.exploration_bonus = exploration_bonus
        self.exploration_decay = exploration_decay

        # Running statistics (Welford's algorithm)
        self.n = 0  # Count of rewards seen
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared deviations
        self.variance = 1.0  # Initialize to 1 to avoid division by zero

        # Episode tracking for exploration decay
        self.episode_count = 0

    def scale_reward(self, reward: float, add_exploration: bool = True) -> float:
        """
        Scale reward using running statistics and normalization.

        Args:
            reward: Raw reward value
            add_exploration: Whether to add exploration bonus

        Returns:
            Scaled and normalized reward
        """
        # Update running statistics
        self._update_statistics(reward)

        # Apply z-score normalization if we have enough samples
        if self.n > 1:
            # Z-score normalization
            std_dev = np.sqrt(self.variance)
            if std_dev > 0:
                normalized = (reward - self.mean) / std_dev
            else:
                normalized = 0.0
        else:
            # Not enough samples yet, just scale by temperature
            normalized = reward

        # Apply temperature scaling
        scaled = normalized / self.temperature

        # Clip to prevent extreme values
        clipped = np.clip(scaled, self.clip_range[0], self.clip_range[1])

        # Add exploration bonus if requested
        if add_exploration and self.exploration_bonus > 0:
            current_bonus = self._get_exploration_bonus()
            clipped += current_bonus
            clipped = np.clip(clipped, self.clip_range[0], self.clip_range[1])

        return float(clipped)

    def _update_statistics(self, value: float):
        """
        Update running mean and variance using Welford's algorithm.

        Numerically stable method for computing running statistics.

        Args:
            value: New value to incorporate
        """
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2

        if self.n > 1:
            self.variance = self.M2 / (self.n - 1)
        else:
            self.variance = 1.0

    def _get_exploration_bonus(self) -> float:
        """
        Calculate current exploration bonus with decay.

        Returns:
            Current exploration bonus value
        """
        decayed_bonus = self.exploration_bonus * (self.exploration_decay ** self.episode_count)
        return max(0.0, decayed_bonus)  # Ensure non-negative

    def end_episode(self):
        """Mark end of episode for exploration decay."""
        self.episode_count += 1

    def reset_statistics(self):
        """Reset running statistics (for new training run)."""
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.variance = 1.0

    def reset_exploration(self):
        """Reset exploration tracking."""
        self.episode_count = 0

    def get_statistics(self) -> dict:
        """
        Get current running statistics.

        Returns:
            Dictionary with mean, variance, std_dev, and count
        """
        return {
            'mean': self.mean,
            'variance': self.variance,
            'std_dev': np.sqrt(self.variance),
            'count': self.n,
            'episode_count': self.episode_count,
            'current_exploration_bonus': self._get_exploration_bonus()
        }


class MultiStrategyScaler:
    """
    Manages multiple reward scalers for different strategy types.

    Maintains separate statistics for each strategy category to handle
    different reward magnitudes and distributions.
    """

    def __init__(self, **scaler_kwargs):
        """
        Initialize multi-strategy scaler.

        Args:
            **scaler_kwargs: Arguments passed to individual scalers
        """
        self.scalers = {}
        self.default_scaler = RewardScaler(**scaler_kwargs)
        self.scaler_kwargs = scaler_kwargs

    def scale_reward(self,
                    reward: float,
                    strategy_category: Optional[str] = None,
                    add_exploration: bool = True) -> float:
        """
        Scale reward using appropriate strategy-specific scaler.

        Args:
            reward: Raw reward value
            strategy_category: Category of strategy (neutral, directional, volatility)
            add_exploration: Whether to add exploration bonus

        Returns:
            Scaled reward
        """
        if strategy_category and strategy_category not in self.scalers:
            # Create new scaler for this category
            self.scalers[strategy_category] = RewardScaler(**self.scaler_kwargs)

        # Use strategy-specific scaler or default
        scaler = self.scalers.get(strategy_category, self.default_scaler)
        return scaler.scale_reward(reward, add_exploration)

    def end_episode(self, strategy_category: Optional[str] = None):
        """
        Mark end of episode for exploration decay.

        Args:
            strategy_category: Category to update, or all if None
        """
        if strategy_category:
            if strategy_category in self.scalers:
                self.scalers[strategy_category].end_episode()
        else:
            # End episode for all scalers
            for scaler in self.scalers.values():
                scaler.end_episode()
            self.default_scaler.end_episode()

    def reset_all(self):
        """Reset all scalers."""
        for scaler in self.scalers.values():
            scaler.reset_statistics()
            scaler.reset_exploration()
        self.default_scaler.reset_statistics()
        self.default_scaler.reset_exploration()

    def get_all_statistics(self) -> dict:
        """
        Get statistics for all scalers.

        Returns:
            Dictionary mapping category to statistics
        """
        stats = {
            'default': self.default_scaler.get_statistics()
        }
        for category, scaler in self.scalers.items():
            stats[category] = scaler.get_statistics()
        return stats
