"""Tests for reward scaling and normalization system."""

import pytest
import numpy as np

from src.environments.reward_scaler import RewardScaler, MultiStrategyScaler


class TestRewardScaler:
    """Test reward scaler functionality."""

    def test_scaler_creation(self):
        """Test reward scaler can be instantiated."""
        scaler = RewardScaler()
        assert scaler.clip_range == (-3, 3)
        assert scaler.temperature == 1.0
        assert scaler.exploration_bonus == 0.1

    def test_scale_reward_basic(self):
        """Test basic reward scaling."""
        scaler = RewardScaler()

        # Scale a reward
        scaled = scaler.scale_reward(100.0)
        assert isinstance(scaled, float)
        assert -3 <= scaled <= 3  # Should be clipped

    def test_running_statistics_update(self):
        """Test Welford's algorithm updates statistics correctly."""
        scaler = RewardScaler()

        # Add some rewards
        rewards = [10, 20, 30, 40, 50]
        for r in rewards:
            scaler._update_statistics(r)

        # Check statistics
        assert scaler.n == 5
        assert abs(scaler.mean - 30.0) < 0.01  # Mean should be 30
        assert scaler.variance > 0  # Should have positive variance

    def test_z_score_normalization(self):
        """Test z-score normalization after sufficient samples."""
        scaler = RewardScaler()

        # Build up statistics with known distribution
        np.random.seed(42)
        for _ in range(100):
            reward = np.random.normal(50, 10)  # Mean 50, std 10
            scaler._update_statistics(reward)

        # Now scale a reward at the mean
        scaled_mean = scaler.scale_reward(scaler.mean, add_exploration=False)
        assert abs(scaled_mean) < 0.5  # Should be close to 0 after normalization

        # Scale a reward 2 std deviations above mean
        std_dev = np.sqrt(scaler.variance)
        scaled_high = scaler.scale_reward(scaler.mean + 2 * std_dev, add_exploration=False)
        assert 1.5 < scaled_high < 2.5  # Should be around 2

    def test_clipping(self):
        """Test extreme values are clipped."""
        scaler = RewardScaler(clip_range=(-2, 2))

        # Add some baseline statistics
        for i in range(10):
            scaler._update_statistics(i * 10)

        # Test extreme positive
        scaled = scaler.scale_reward(10000, add_exploration=False)
        assert scaled == 2.0  # Should be clipped to max

        # Test extreme negative
        scaled = scaler.scale_reward(-10000, add_exploration=False)
        assert scaled == -2.0  # Should be clipped to min

    def test_temperature_scaling(self):
        """Test temperature parameter affects scaling."""
        scaler_low_temp = RewardScaler(temperature=0.5)
        scaler_high_temp = RewardScaler(temperature=2.0)

        # Add same statistics to both
        for i in range(10):
            scaler_low_temp._update_statistics(i * 10)
            scaler_high_temp._update_statistics(i * 10)

        # Scale same reward
        reward = 100.0
        scaled_low = scaler_low_temp.scale_reward(reward, add_exploration=False)
        scaled_high = scaler_high_temp.scale_reward(reward, add_exploration=False)

        # Low temperature should give more extreme scaling
        assert abs(scaled_low) > abs(scaled_high)

    def test_exploration_bonus(self):
        """Test exploration bonus is added and decays."""
        scaler = RewardScaler(exploration_bonus=0.5, exploration_decay=0.9)

        # First episode - full bonus
        reward = 0.0  # Use zero to isolate bonus
        scaled = scaler.scale_reward(reward, add_exploration=True)
        assert scaled > 0  # Should have positive bonus

        initial_bonus = scaled

        # End episode and check decay
        scaler.end_episode()
        scaled = scaler.scale_reward(reward, add_exploration=True)
        assert scaled < initial_bonus  # Bonus should decay
        assert scaled > 0  # But still positive

        # Many episodes later
        for _ in range(20):
            scaler.end_episode()

        scaled = scaler.scale_reward(reward, add_exploration=True)
        assert scaled < 0.1  # Bonus should be very small

    def test_no_exploration_bonus_when_disabled(self):
        """Test exploration bonus can be disabled."""
        scaler = RewardScaler(exploration_bonus=0.5)

        reward = 0.0
        scaled_with = scaler.scale_reward(reward, add_exploration=True)
        scaled_without = scaler.scale_reward(reward, add_exploration=False)

        assert scaled_with > scaled_without  # With bonus should be higher

    def test_reset_statistics(self):
        """Test statistics can be reset."""
        scaler = RewardScaler()

        # Build up statistics
        for i in range(50):
            scaler._update_statistics(i * 10)

        assert scaler.n > 0
        assert scaler.mean != 0

        # Reset
        scaler.reset_statistics()

        assert scaler.n == 0
        assert scaler.mean == 0.0
        assert scaler.variance == 1.0

    def test_reset_exploration(self):
        """Test exploration tracking can be reset."""
        scaler = RewardScaler()

        # Advance episodes
        for _ in range(10):
            scaler.end_episode()

        assert scaler.episode_count == 10

        # Reset exploration
        scaler.reset_exploration()
        assert scaler.episode_count == 0

    def test_get_statistics(self):
        """Test statistics retrieval."""
        scaler = RewardScaler()

        # Add some data
        for i in range(10):
            scaler._update_statistics(i * 10)

        scaler.end_episode()
        scaler.end_episode()

        stats = scaler.get_statistics()

        assert 'mean' in stats
        assert 'variance' in stats
        assert 'std_dev' in stats
        assert 'count' in stats
        assert stats['count'] == 10
        assert stats['episode_count'] == 2


class TestMultiStrategyScaler:
    """Test multi-strategy scaler functionality."""

    def test_multi_scaler_creation(self):
        """Test multi-strategy scaler can be created."""
        scaler = MultiStrategyScaler(temperature=1.5)
        assert scaler.default_scaler.temperature == 1.5
        assert len(scaler.scalers) == 0  # No strategy-specific scalers yet

    def test_strategy_specific_scaling(self):
        """Test different strategies get different scalers."""
        scaler = MultiStrategyScaler()

        # Scale rewards for different strategies
        neutral_scaled = scaler.scale_reward(100, 'neutral')
        directional_scaled = scaler.scale_reward(100, 'directional')

        # Should create separate scalers
        assert 'neutral' in scaler.scalers
        assert 'directional' in scaler.scalers

        # Add more data to neutral
        for i in range(20):
            scaler.scale_reward(i * 10, 'neutral')

        # Statistics should be separate
        neutral_stats = scaler.scalers['neutral'].get_statistics()
        directional_stats = scaler.scalers['directional'].get_statistics()

        assert neutral_stats['count'] > directional_stats['count']

    def test_default_scaler_used(self):
        """Test default scaler used when no category specified."""
        scaler = MultiStrategyScaler()

        # Scale without category
        scaled = scaler.scale_reward(100, None)
        assert isinstance(scaled, float)

        # Default scaler should have updated
        assert scaler.default_scaler.n == 1

    def test_end_episode_all_scalers(self):
        """Test ending episode for all scalers."""
        scaler = MultiStrategyScaler()

        # Create some scalers
        scaler.scale_reward(100, 'neutral')
        scaler.scale_reward(100, 'directional')
        scaler.scale_reward(100, None)  # Default

        # End episode for all
        scaler.end_episode()

        # All should have incremented episode count
        assert scaler.scalers['neutral'].episode_count == 1
        assert scaler.scalers['directional'].episode_count == 1
        assert scaler.default_scaler.episode_count == 1

    def test_end_episode_specific(self):
        """Test ending episode for specific strategy."""
        scaler = MultiStrategyScaler()

        # Create scalers
        scaler.scale_reward(100, 'neutral')
        scaler.scale_reward(100, 'directional')

        # End episode for neutral only
        scaler.end_episode('neutral')

        assert scaler.scalers['neutral'].episode_count == 1
        assert scaler.scalers['directional'].episode_count == 0

    def test_reset_all(self):
        """Test resetting all scalers."""
        scaler = MultiStrategyScaler()

        # Build up data in multiple scalers
        for i in range(10):
            scaler.scale_reward(i * 10, 'neutral')
            scaler.scale_reward(i * 20, 'directional')
            scaler.scale_reward(i * 30, None)

        # All should have data
        assert scaler.scalers['neutral'].n > 0
        assert scaler.scalers['directional'].n > 0
        assert scaler.default_scaler.n > 0

        # Reset all
        scaler.reset_all()

        # All should be reset
        assert scaler.scalers['neutral'].n == 0
        assert scaler.scalers['directional'].n == 0
        assert scaler.default_scaler.n == 0

    def test_get_all_statistics(self):
        """Test getting statistics for all scalers."""
        scaler = MultiStrategyScaler()

        # Add data to different scalers
        for i in range(5):
            scaler.scale_reward(i * 10, 'neutral')
            scaler.scale_reward(i * 20, 'volatility')

        stats = scaler.get_all_statistics()

        assert 'default' in stats
        assert 'neutral' in stats
        assert 'volatility' in stats
        assert stats['neutral']['count'] == 5
        assert stats['volatility']['count'] == 5
        assert stats['default']['count'] == 0  # Unused