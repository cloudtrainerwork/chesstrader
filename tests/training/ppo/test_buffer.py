"""
Tests for PPO experience buffer implementation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from src.training.ppo.buffer import PPOBuffer
from src.training.ppo.gae import GAECalculator


class TestPPOBuffer:
    """Test cases for PPO experience buffer."""

    @pytest.fixture
    def device(self):
        """Test device."""
        return 'cpu'

    @pytest.fixture
    def buffer(self, device):
        """Create test buffer."""
        return PPOBuffer(capacity=100, obs_dim=35, device=device)

    @pytest.fixture
    def gae_calculator(self, device):
        """Create test GAE calculator."""
        return GAECalculator(gamma=0.99, lambda_=0.95, device=device)

    def test_initialization(self, device):
        """Test buffer initialization."""
        buffer = PPOBuffer(capacity=50, obs_dim=10, device=device)

        assert buffer.capacity == 50
        assert buffer.obs_dim == 10
        assert buffer.device == device
        assert buffer.ptr == 0
        assert buffer.size == 0
        assert len(buffer.episode_starts) == 0

        # Check tensor shapes
        assert buffer.observations.shape == (50, 10)
        assert buffer.actions.shape == (50,)
        assert buffer.rewards.shape == (50,)

    def test_store_transition(self, buffer):
        """Test storing a single transition."""
        obs = torch.randn(35)
        action = 2
        reward = 1.5
        value = 0.8
        log_prob = -1.2
        done = False

        buffer.store_transition(obs, action, reward, value, log_prob, done)

        assert buffer.ptr == 1
        assert buffer.size == 1
        assert torch.equal(buffer.observations[0], obs)
        assert buffer.actions[0] == action
        assert buffer.rewards[0] == reward
        assert buffer.values[0] == value
        assert buffer.log_probs[0] == log_prob
        assert buffer.dones[0] == 0.0

    def test_store_multiple_transitions(self, buffer):
        """Test storing multiple transitions."""
        for i in range(5):
            obs = torch.randn(35)
            buffer.store_transition(obs, i % 4, float(i), float(i) * 0.1, -float(i), False)

        assert buffer.ptr == 5
        assert buffer.size == 5

        # Check that data is stored correctly
        for i in range(5):
            assert buffer.actions[i] == i % 4
            assert buffer.rewards[i] == float(i)

    def test_episode_boundary_tracking(self, buffer):
        """Test episode boundary tracking."""
        # Store transitions for first episode
        for i in range(3):
            obs = torch.randn(35)
            done = (i == 2)  # End episode on last step
            buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, done)

        assert len(buffer.episode_starts) == 1
        assert buffer.episode_starts[0] == (0, 3)
        assert len(buffer.episode_lengths) == 1
        assert buffer.episode_lengths[0] == 3

        # Store transitions for second episode
        for i in range(2):
            obs = torch.randn(35)
            done = (i == 1)  # End episode on last step
            buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, done)

        assert len(buffer.episode_starts) == 2
        assert buffer.episode_starts[1] == (3, 5)
        assert buffer.episode_lengths[1] == 2

    def test_finish_episode_manually(self, buffer):
        """Test manually finishing an episode."""
        # Store some transitions without done=True
        for i in range(3):
            obs = torch.randn(35)
            buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, False)

        # Manually finish episode
        buffer.finish_episode(next_value=1.2)

        assert len(buffer.episode_starts) == 1
        assert buffer.episode_starts[0] == (0, 3)

    def test_buffer_overflow(self, device):
        """Test buffer behavior when capacity is exceeded."""
        buffer = PPOBuffer(capacity=3, obs_dim=2, device=device)

        # Fill buffer beyond capacity
        for i in range(5):
            obs = torch.randn(2)
            buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, False)

        assert buffer.size == 3  # Should not exceed capacity
        assert buffer.ptr == 2   # Should wrap around (0, 1, 2, 0, 1)

    def test_compute_advantages_and_returns(self, buffer, gae_calculator):
        """Test advantage and return computation."""
        # Store a complete episode
        for i in range(3):
            obs = torch.randn(35)
            done = (i == 2)
            buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, done)

        # Compute advantages and returns
        buffer.compute_advantages_and_returns(gae_calculator)

        # Check that advantages and returns are computed
        episode_start, episode_end = buffer.episode_starts[0]
        assert not torch.allclose(buffer.advantages[episode_start:episode_end],
                                 torch.zeros(3))
        assert not torch.allclose(buffer.returns[episode_start:episode_end],
                                 torch.zeros(3))

    def test_get_batches(self, buffer, gae_calculator):
        """Test mini-batch generation."""
        # Store a complete episode
        for i in range(6):
            obs = torch.randn(35)
            done = (i == 5)
            buffer.store_transition(obs, i % 4, 1.0, 0.5, -0.5, done)

        # Compute advantages and returns
        buffer.compute_advantages_and_returns(gae_calculator)

        # Generate batches
        batches = list(buffer.get_batches(batch_size=3, shuffle=False))

        assert len(batches) == 2  # 6 samples / 3 batch_size = 2 batches

        for batch in batches:
            assert isinstance(batch, dict)
            required_keys = ['observations', 'actions', 'rewards', 'values',
                           'log_probs', 'dones', 'advantages', 'returns']
            for key in required_keys:
                assert key in batch
                assert batch[key].shape[0] == 3  # batch_size

    def test_get_batches_empty_buffer(self, buffer):
        """Test batch generation with empty buffer."""
        batches = list(buffer.get_batches(batch_size=4))
        assert len(batches) == 0

    def test_get_batches_no_complete_episodes(self, buffer):
        """Test batch generation with no complete episodes."""
        # Store transitions but don't complete episode
        for i in range(3):
            obs = torch.randn(35)
            buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, False)

        batches = list(buffer.get_batches(batch_size=2))
        assert len(batches) == 0

    def test_get_batches_insufficient_data(self, buffer, gae_calculator):
        """Test batch generation with insufficient data."""
        # Store only 2 transitions but request batch_size=5
        for i in range(2):
            obs = torch.randn(35)
            done = (i == 1)
            buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, done)

        buffer.compute_advantages_and_returns(gae_calculator)

        batches = list(buffer.get_batches(batch_size=5))
        assert len(batches) == 0

    def test_get_all_data(self, buffer, gae_calculator):
        """Test getting all buffer data."""
        # Store complete episode
        for i in range(4):
            obs = torch.randn(35)
            done = (i == 3)
            buffer.store_transition(obs, i, 1.0, 0.5, -0.5, done)

        buffer.compute_advantages_and_returns(gae_calculator)

        all_data = buffer.get_all_data()

        assert isinstance(all_data, dict)
        assert all_data['observations'].shape == (4, 35)
        assert all_data['actions'].shape == (4,)

        # Check that all required keys are present
        required_keys = ['observations', 'actions', 'rewards', 'values',
                        'log_probs', 'dones', 'advantages', 'returns']
        for key in required_keys:
            assert key in all_data

    def test_clear_buffer(self, buffer):
        """Test buffer clearing."""
        # Fill buffer with some data
        for i in range(3):
            obs = torch.randn(35)
            buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, True)

        assert buffer.size > 0
        assert len(buffer.episode_starts) > 0

        # Clear buffer
        buffer.clear()

        assert buffer.size == 0
        assert buffer.ptr == 0
        assert len(buffer.episode_starts) == 0
        assert len(buffer.episode_lengths) == 0

        # Check that tensors are zeroed
        assert torch.allclose(buffer.observations, torch.zeros_like(buffer.observations))
        assert torch.allclose(buffer.rewards, torch.zeros_like(buffer.rewards))

    def test_get_statistics_empty(self, buffer):
        """Test statistics for empty buffer."""
        stats = buffer.get_statistics()

        assert stats['size'] == 0
        assert stats['episodes'] == 0

    def test_get_statistics_with_data(self, buffer, gae_calculator):
        """Test statistics with data."""
        # Store complete episode
        rewards = [1.0, 2.0, -0.5]
        for i, reward in enumerate(rewards):
            obs = torch.randn(35)
            done = (i == 2)
            buffer.store_transition(obs, 0, reward, 0.5, -0.5, done)

        buffer.compute_advantages_and_returns(gae_calculator)

        stats = buffer.get_statistics()

        assert stats['size'] == 3
        assert stats['episodes'] == 1
        assert stats['average_episode_length'] == 3.0
        assert stats['total_reward'] == sum(rewards)
        assert 'average_advantage' in stats  # Should be present after GAE computation

    def test_save_and_load_buffer(self, buffer, tmp_path):
        """Test buffer saving and loading."""
        # Fill buffer with data
        for i in range(3):
            obs = torch.randn(35)
            done = (i == 2)
            buffer.store_transition(obs, i, float(i), 0.5, -0.5, done)

        # Save buffer
        save_path = tmp_path / "test_buffer.pt"
        buffer.save_buffer(str(save_path))

        assert save_path.exists()

        # Create new buffer and load data
        new_buffer = PPOBuffer(capacity=100, obs_dim=35, device=buffer.device)
        new_buffer.load_buffer(str(save_path))

        # Check that data is preserved
        assert new_buffer.size == buffer.size
        assert len(new_buffer.episode_starts) == len(buffer.episode_starts)
        assert torch.equal(new_buffer.rewards[:3], buffer.rewards[:3])

    def test_buffer_length(self, buffer):
        """Test buffer length property."""
        assert len(buffer) == 0

        # Add some transitions
        for i in range(5):
            obs = torch.randn(35)
            buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, False)

        assert len(buffer) == 5

    def test_is_full(self, device):
        """Test buffer full detection."""
        buffer = PPOBuffer(capacity=3, obs_dim=2, device=device)

        assert not buffer.is_full()

        # Fill buffer
        for i in range(3):
            obs = torch.randn(2)
            buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, False)

        assert buffer.is_full()

    def test_is_ready_for_update(self, buffer):
        """Test update readiness detection."""
        assert not buffer.is_ready_for_update(min_episodes=1)

        # Store incomplete episode
        for i in range(2):
            obs = torch.randn(35)
            buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, False)

        assert not buffer.is_ready_for_update(min_episodes=1)

        # Complete episode
        obs = torch.randn(35)
        buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, True)

        assert buffer.is_ready_for_update(min_episodes=1)

    def test_batch_shuffling(self, buffer, gae_calculator):
        """Test that batch shuffling works correctly."""
        # Store episode with sequential actions
        for i in range(6):
            obs = torch.randn(35)
            done = (i == 5)
            buffer.store_transition(obs, i, 1.0, 0.5, -0.5, done)

        buffer.compute_advantages_and_returns(gae_calculator)

        # Generate batches with and without shuffling
        batches_no_shuffle = list(buffer.get_batches(batch_size=3, shuffle=False))
        batches_shuffle = list(buffer.get_batches(batch_size=3, shuffle=True))

        # Without shuffling, should get sequential actions
        assert torch.equal(batches_no_shuffle[0]['actions'], torch.tensor([0, 1, 2]))

        # Note: We can't guarantee shuffling will change order in a test,
        # but we can check that the function accepts the parameter
        assert len(batches_shuffle) == len(batches_no_shuffle)

    def test_multiple_episodes_batching(self, buffer, gae_calculator):
        """Test batching with multiple episodes."""
        # Store first episode
        for i in range(3):
            obs = torch.randn(35)
            done = (i == 2)
            buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, done)

        # Store second episode
        for i in range(2):
            obs = torch.randn(35)
            done = (i == 1)
            buffer.store_transition(obs, 1, 2.0, 1.0, -1.0, done)

        buffer.compute_advantages_and_returns(gae_calculator)

        all_data = buffer.get_all_data()

        # Should have data from both episodes
        assert all_data['observations'].shape[0] == 5  # 3 + 2 transitions
        assert torch.all((all_data['actions'] == 0) | (all_data['actions'] == 1))

    def test_device_consistency(self, device):
        """Test that all tensors are on correct device."""
        buffer = PPOBuffer(capacity=10, obs_dim=5, device=device)

        # Add some data
        obs = torch.randn(5, device=device)
        buffer.store_transition(obs, 0, 1.0, 0.5, -0.5, True)

        # Check that all tensors are on correct device
        assert buffer.observations.device.type == device
        assert buffer.actions.device.type == device
        assert buffer.rewards.device.type == device

        # Check batch data device
        gae = GAECalculator(device=device)
        buffer.compute_advantages_and_returns(gae)

        all_data = buffer.get_all_data()
        for tensor in all_data.values():
            assert tensor.device.type == device