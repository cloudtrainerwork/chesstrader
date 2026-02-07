"""
Tests for GAE (Generalized Advantage Estimation) implementation.
"""

import pytest
import torch
import numpy as np

from src.training.ppo.gae import GAECalculator


class TestGAECalculator:
    """Test cases for GAE calculator."""

    @pytest.fixture
    def device(self):
        """Test device."""
        return 'cpu'

    @pytest.fixture
    def gae_calculator(self, device):
        """Create test GAE calculator."""
        return GAECalculator(gamma=0.99, lambda_=0.95, device=device)

    @pytest.fixture
    def sample_trajectory(self, device):
        """Create sample trajectory data."""
        return {
            'rewards': torch.tensor([1.0, 0.5, 2.0, -0.5, 1.5], device=device),
            'values': torch.tensor([2.0, 1.5, 3.0, 1.0, 2.5], device=device),
            'dones': torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], device=device)
        }

    def test_initialization(self):
        """Test GAE calculator initialization."""
        gae = GAECalculator(gamma=0.99, lambda_=0.95)

        assert gae.gamma == 0.99
        assert gae.lambda_ == 0.95
        assert gae.device == 'cpu'

    def test_initialization_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid gamma
        with pytest.raises(ValueError, match="Gamma must be between 0 and 1"):
            GAECalculator(gamma=1.5)

        with pytest.raises(ValueError, match="Gamma must be between 0 and 1"):
            GAECalculator(gamma=-0.1)

        # Test invalid lambda
        with pytest.raises(ValueError, match="Lambda must be between 0 and 1"):
            GAECalculator(lambda_=1.5)

        with pytest.raises(ValueError, match="Lambda must be between 0 and 1"):
            GAECalculator(lambda_=-0.1)

    def test_compute_gae_shapes(self, gae_calculator, sample_trajectory):
        """Test that GAE computation produces correct shapes."""
        advantages, returns = gae_calculator.compute_gae(
            sample_trajectory['rewards'],
            sample_trajectory['values'],
            sample_trajectory['dones'],
            next_value=0.0
        )

        assert advantages.shape == sample_trajectory['rewards'].shape
        assert returns.shape == sample_trajectory['rewards'].shape
        assert advantages.device == gae_calculator.device
        assert returns.device == gae_calculator.device

    def test_compute_gae_with_numpy_inputs(self, gae_calculator):
        """Test GAE computation with numpy array inputs."""
        rewards = np.array([1.0, 0.5, 2.0])
        values = np.array([2.0, 1.5, 3.0])
        dones = np.array([0.0, 0.0, 1.0])

        advantages, returns = gae_calculator.compute_gae(rewards, values, dones)

        assert isinstance(advantages, torch.Tensor)
        assert isinstance(returns, torch.Tensor)
        assert advantages.shape == torch.tensor(rewards).shape

    def test_compute_gae_with_list_inputs(self, gae_calculator):
        """Test GAE computation with list inputs."""
        rewards = [1.0, 0.5, 2.0]
        values = [2.0, 1.5, 3.0]
        dones = [0.0, 0.0, 1.0]

        advantages, returns = gae_calculator.compute_gae(rewards, values, dones)

        assert isinstance(advantages, torch.Tensor)
        assert isinstance(returns, torch.Tensor)
        assert advantages.shape == (3,)

    def test_gae_terminal_state(self, gae_calculator):
        """Test GAE computation with terminal state."""
        # Single step ending in terminal state
        rewards = torch.tensor([1.0])
        values = torch.tensor([2.0])
        dones = torch.tensor([1.0])

        advantages, returns = gae_calculator.compute_gae(
            rewards, values, dones, next_value=0.0
        )

        # For terminal state: advantage = reward + gamma * 0 - value = 1.0 - 2.0 = -1.0
        expected_advantage = 1.0 - 2.0
        assert torch.allclose(advantages[0], torch.tensor(expected_advantage), atol=1e-6)

        # Return should be advantage + value
        expected_return = expected_advantage + 2.0
        assert torch.allclose(returns[0], torch.tensor(expected_return), atol=1e-6)

    def test_gae_non_terminal_state(self, gae_calculator):
        """Test GAE computation with non-terminal final state."""
        rewards = torch.tensor([1.0])
        values = torch.tensor([2.0])
        dones = torch.tensor([0.0])
        next_value = 1.5

        advantages, returns = gae_calculator.compute_gae(
            rewards, values, dones, next_value=next_value
        )

        # advantage = reward + gamma * next_value - value = 1.0 + 0.99 * 1.5 - 2.0
        expected_advantage = 1.0 + 0.99 * 1.5 - 2.0
        assert torch.allclose(advantages[0], torch.tensor(expected_advantage), atol=1e-6)

    def test_compute_returns_only(self, gae_calculator, sample_trajectory):
        """Test standalone returns computation."""
        returns = gae_calculator.compute_returns(
            sample_trajectory['rewards'],
            sample_trajectory['values'],
            sample_trajectory['dones'],
            next_value=0.0
        )

        assert returns.shape == sample_trajectory['rewards'].shape
        assert returns.device == gae_calculator.device

        # Returns should be computed using reverse iteration
        # Last return (terminal): reward[-1] + 0 = 1.5
        assert torch.allclose(returns[-1], torch.tensor(1.5), atol=1e-6)

    def test_normalize_advantages(self, gae_calculator):
        """Test advantage normalization."""
        # Create advantages with known mean and std
        advantages = torch.tensor([1.0, 3.0, 5.0, 7.0])  # mean=4, std≈2.58

        normalized = gae_calculator.normalize_advantages(advantages)

        # Should have mean ≈ 0 and std ≈ 1
        assert torch.allclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(normalized.std(), torch.tensor(1.0), atol=1e-6)

    def test_normalize_advantages_edge_cases(self, gae_calculator):
        """Test advantage normalization edge cases."""
        # Empty tensor
        empty_advantages = torch.tensor([])
        normalized_empty = gae_calculator.normalize_advantages(empty_advantages)
        assert normalized_empty.shape == empty_advantages.shape

        # Constant advantages (std = 0)
        constant_advantages = torch.tensor([2.0, 2.0, 2.0])
        normalized_constant = gae_calculator.normalize_advantages(constant_advantages)
        # Should handle division by zero gracefully
        assert torch.isfinite(normalized_constant).all()

    def test_compute_gae_batched(self, gae_calculator):
        """Test batched GAE computation for multiple episodes."""
        # Create multiple episodes
        rewards_list = [
            torch.tensor([1.0, 0.5]),
            torch.tensor([2.0, -0.5, 1.0])
        ]
        values_list = [
            torch.tensor([2.0, 1.5]),
            torch.tensor([3.0, 1.0, 2.0])
        ]
        dones_list = [
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0])
        ]
        next_values = [0.0, 0.0]

        advantages_list, returns_list = gae_calculator.compute_gae_batched(
            rewards_list, values_list, dones_list, next_values
        )

        assert len(advantages_list) == 2
        assert len(returns_list) == 2
        assert advantages_list[0].shape == (2,)
        assert advantages_list[1].shape == (3,)

    def test_td_lambda_returns(self, gae_calculator, sample_trajectory):
        """Test TD(λ) returns computation."""
        returns = gae_calculator.compute_td_lambda_returns(
            sample_trajectory['rewards'],
            sample_trajectory['values'],
            sample_trajectory['dones'],
            next_value=0.0
        )

        assert returns.shape == sample_trajectory['rewards'].shape
        assert returns.device == gae_calculator.device

    def test_gae_vs_monte_carlo(self, device):
        """Test GAE with λ=1 should approximate Monte Carlo returns."""
        gae_mc = GAECalculator(gamma=0.99, lambda_=1.0, device=device)

        rewards = torch.tensor([1.0, 1.0, 1.0], device=device)
        values = torch.tensor([0.0, 0.0, 0.0], device=device)  # Zero baseline
        dones = torch.tensor([0.0, 0.0, 1.0], device=device)

        advantages, returns = gae_mc.compute_gae(rewards, values, dones)

        # With λ=1 and zero baseline, advantages should approximate MC returns
        expected_mc_returns = torch.tensor([
            1.0 + 0.99 * 1.0 + 0.99**2 * 1.0,  # Full trajectory return
            1.0 + 0.99 * 1.0,                   # Two-step return
            1.0                                  # One-step return
        ], device=device)

        assert torch.allclose(advantages, expected_mc_returns, atol=1e-6)

    def test_gae_vs_td(self, device):
        """Test GAE with λ=0 should approximate TD errors."""
        gae_td = GAECalculator(gamma=0.99, lambda_=0.0, device=device)

        rewards = torch.tensor([1.0], device=device)
        values = torch.tensor([2.0], device=device)
        dones = torch.tensor([0.0], device=device)
        next_value = 1.5

        advantages, _ = gae_td.compute_gae(rewards, values, dones, next_value)

        # With λ=0, should be TD error: r + γV(s') - V(s)
        expected_td = 1.0 + 0.99 * 1.5 - 2.0
        assert torch.allclose(advantages[0], torch.tensor(expected_td), atol=1e-6)

    def test_get_statistics(self, gae_calculator):
        """Test statistics computation."""
        advantages = torch.tensor([1.0, 2.0, -1.0, 0.5])
        returns = torch.tensor([3.0, 4.0, 1.0, 2.5])

        stats = gae_calculator.get_statistics(advantages, returns)

        assert isinstance(stats, dict)
        expected_keys = [
            'advantage_mean', 'advantage_std', 'advantage_min', 'advantage_max',
            'return_mean', 'return_std', 'return_min', 'return_max',
            'trajectory_length'
        ]

        for key in expected_keys:
            assert key in stats

        assert stats['trajectory_length'] == 4
        assert stats['advantage_mean'] == advantages.mean().item()
        assert stats['return_mean'] == returns.mean().item()

    def test_consistency_across_devices(self):
        """Test that GAE computation is consistent across devices."""
        # Only test if CUDA is available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        rewards = torch.tensor([1.0, 0.5, 2.0])
        values = torch.tensor([2.0, 1.5, 3.0])
        dones = torch.tensor([0.0, 0.0, 1.0])

        # CPU computation
        gae_cpu = GAECalculator(device='cpu')
        adv_cpu, ret_cpu = gae_cpu.compute_gae(rewards, values, dones)

        # GPU computation
        gae_gpu = GAECalculator(device='cuda')
        adv_gpu, ret_gpu = gae_gpu.compute_gae(
            rewards.cuda(), values.cuda(), dones.cuda()
        )

        # Results should be the same
        assert torch.allclose(adv_cpu, adv_gpu.cpu(), atol=1e-6)
        assert torch.allclose(ret_cpu, ret_gpu.cpu(), atol=1e-6)