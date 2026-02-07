"""
Tests for PPO algorithm implementation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.training.ppo.algorithm import PPOAlgorithm
from src.training.ppo.networks import ActorCritic


class TestPPOAlgorithm:
    """Test cases for PPO algorithm."""

    @pytest.fixture
    def device(self):
        """Test device."""
        return 'cpu'

    @pytest.fixture
    def actor_critic(self, device):
        """Create test actor-critic network."""
        return ActorCritic(obs_dim=35, action_dim=4, device=device)

    @pytest.fixture
    def ppo_algorithm(self, actor_critic, device):
        """Create test PPO algorithm."""
        return PPOAlgorithm(
            actor_critic=actor_critic,
            policy_lr=3e-4,
            value_lr=3e-4,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            max_grad_norm=0.5,
            device=device
        )

    @pytest.fixture
    def sample_data(self, device):
        """Create sample training data."""
        batch_size = 32
        obs_dim = 35

        return {
            'observations': torch.randn(batch_size, obs_dim, device=device),
            'actions': torch.randint(0, 4, (batch_size,), device=device),
            'log_probs': torch.randn(batch_size, device=device),
            'advantages': torch.randn(batch_size, device=device),
            'returns': torch.randn(batch_size, device=device),
            'values': torch.randn(batch_size, device=device)
        }

    def test_initialization(self, actor_critic, device):
        """Test PPO algorithm initialization."""
        ppo = PPOAlgorithm(
            actor_critic=actor_critic,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            device=device
        )

        assert ppo.clip_epsilon == 0.2
        assert ppo.entropy_coef == 0.01
        assert ppo.device == device
        assert hasattr(ppo, 'policy_optimizer')
        assert hasattr(ppo, 'value_optimizer')
        assert isinstance(ppo.training_stats, dict)

    def test_compute_ppo_loss(self, ppo_algorithm, sample_data):
        """Test PPO loss computation."""
        observations = sample_data['observations']
        actions = sample_data['actions']
        old_log_probs = sample_data['log_probs']
        advantages = sample_data['advantages']

        loss, stats = ppo_algorithm.compute_ppo_loss(
            observations, actions, old_log_probs, advantages
        )

        # Check loss is a tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

        # Check statistics
        assert isinstance(stats, dict)
        expected_keys = ['policy_loss', 'entropy_loss', 'total_loss',
                        'kl_divergence', 'clip_fraction', 'mean_entropy']
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))

    def test_compute_value_loss(self, ppo_algorithm, sample_data):
        """Test value function loss computation."""
        observations = sample_data['observations']
        returns = sample_data['returns']
        old_values = sample_data['values']

        loss, stats = ppo_algorithm.compute_value_loss(
            observations, returns, old_values
        )

        # Check loss is a tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

        # Check statistics
        assert isinstance(stats, dict)
        expected_keys = ['value_loss', 'mean_value', 'mean_return', 'value_std']
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))

    def test_get_action_log_prob(self, ppo_algorithm, sample_data):
        """Test action log probability calculation."""
        observations = sample_data['observations']
        actions = sample_data['actions']

        log_probs = ppo_algorithm.get_action_log_prob(observations, actions)

        assert isinstance(log_probs, torch.Tensor)
        assert log_probs.shape == actions.shape
        assert torch.all(log_probs <= 0)  # Log probabilities should be negative

    def test_update_policy(self, ppo_algorithm, sample_data):
        """Test policy update."""
        initial_params = {name: param.clone()
                         for name, param in ppo_algorithm.actor_critic.named_parameters()}

        stats = ppo_algorithm.update_policy(sample_data, n_epochs=2)

        # Check that parameters have changed
        params_changed = False
        for name, param in ppo_algorithm.actor_critic.named_parameters():
            if not torch.equal(param, initial_params[name]):
                params_changed = True
                break

        assert params_changed, "Parameters should change after update"

        # Check statistics structure
        assert isinstance(stats, dict)
        expected_keys = ['policy_loss', 'value_loss', 'entropy_loss',
                        'total_loss', 'kl_divergence', 'clip_fraction']
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], list)
            assert len(stats[key]) <= 2  # n_epochs

    def test_ppo_clipping(self, ppo_algorithm, sample_data):
        """Test that PPO clipping works correctly."""
        observations = sample_data['observations']
        actions = sample_data['actions']

        # Create log probs that would result in large ratio
        old_log_probs = torch.full_like(sample_data['log_probs'], -10.0)  # Very small prob
        advantages = torch.ones_like(sample_data['advantages'])

        loss, stats = ppo_algorithm.compute_ppo_loss(
            observations, actions, old_log_probs, advantages
        )

        # Should have significant clipping when ratios are large
        assert stats['clip_fraction'] > 0, "Expected some clipping with large ratio"

    def test_advantage_normalization(self, ppo_algorithm, sample_data):
        """Test that advantages are normalized during policy update."""
        # Create advantages with specific mean and std
        sample_data['advantages'] = torch.tensor([1.0, 2.0, 3.0, 4.0] * 8, device=ppo_algorithm.device)

        with patch.object(ppo_algorithm, 'compute_ppo_loss') as mock_ppo_loss:
            mock_ppo_loss.return_value = (torch.tensor(0.0), {
                'policy_loss': 0.0, 'entropy_loss': 0.0, 'total_loss': 0.0,
                'kl_divergence': 0.0, 'clip_fraction': 0.0, 'mean_entropy': 0.0
            })

            with patch.object(ppo_algorithm, 'compute_value_loss') as mock_value_loss:
                mock_value_loss.return_value = (torch.tensor(0.0), {
                    'value_loss': 0.0, 'mean_value': 0.0, 'mean_return': 0.0, 'value_std': 0.0
                })

                ppo_algorithm.update_policy(sample_data, n_epochs=1)

                # Check that normalized advantages were passed to compute_ppo_loss
                called_advantages = mock_ppo_loss.call_args[0][3]
                assert abs(called_advantages.mean().item()) < 1e-6, "Advantages should be normalized to mean 0"
                assert abs(called_advantages.std().item() - 1.0) < 1e-6, "Advantages should be normalized to std 1"

    def test_early_stopping_on_high_kl(self, ppo_algorithm, sample_data):
        """Test early stopping when KL divergence is too high."""
        with patch.object(ppo_algorithm, 'compute_ppo_loss') as mock_ppo_loss:
            # Return high KL divergence
            mock_ppo_loss.return_value = (torch.tensor(0.0), {
                'policy_loss': 0.0, 'entropy_loss': 0.0, 'total_loss': 0.0,
                'kl_divergence': 0.05, 'clip_fraction': 0.0, 'mean_entropy': 0.0
            })

            with patch.object(ppo_algorithm, 'compute_value_loss') as mock_value_loss:
                mock_value_loss.return_value = (torch.tensor(0.0), {
                    'value_loss': 0.0, 'mean_value': 0.0, 'mean_return': 0.0, 'value_std': 0.0
                })

                stats = ppo_algorithm.update_policy(sample_data, n_epochs=4)

                # Should stop early due to high KL
                assert len(stats['kl_divergence']) < 4, "Should stop early due to high KL divergence"

    def test_gradient_clipping(self, ppo_algorithm, sample_data):
        """Test that gradients are clipped."""
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            ppo_algorithm.update_policy(sample_data, n_epochs=1)

            # Should have been called for gradient clipping
            assert mock_clip.called, "Gradient clipping should be applied"

            # Check that max_grad_norm was used
            call_args = mock_clip.call_args
            assert call_args[1]['max_norm'] == ppo_algorithm.max_grad_norm

    def test_training_stats_accumulation(self, ppo_algorithm, sample_data):
        """Test that training statistics are properly accumulated."""
        # Clear existing stats
        ppo_algorithm.reset_stats()

        # Run multiple updates
        ppo_algorithm.update_policy(sample_data, n_epochs=1)
        ppo_algorithm.update_policy(sample_data, n_epochs=1)

        stats = ppo_algorithm.get_training_stats()

        # Should have 2 entries for each metric
        for key in ['policy_loss', 'value_loss', 'total_loss']:
            assert len(stats[key]) == 2

    def test_get_training_stats(self, ppo_algorithm):
        """Test getting training statistics."""
        stats = ppo_algorithm.get_training_stats()

        assert isinstance(stats, dict)
        expected_keys = ['policy_loss', 'value_loss', 'entropy_loss',
                        'total_loss', 'kl_divergence', 'clip_fraction']
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], list)

    def test_reset_stats(self, ppo_algorithm, sample_data):
        """Test resetting training statistics."""
        # Add some stats
        ppo_algorithm.update_policy(sample_data, n_epochs=1)

        # Reset stats
        ppo_algorithm.reset_stats()

        stats = ppo_algorithm.get_training_stats()
        for key in stats:
            assert len(stats[key]) == 0, f"Stats for {key} should be empty after reset"