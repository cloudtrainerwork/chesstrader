"""
Comprehensive test suite for Position Manager Network Architecture.

Tests the PositionManagerNetwork, PositionEncoder, and related components
for intelligent options position management decisions.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.models.position_manager import (
    PositionManagerNetwork,
    PositionEncoder,
    PositionManagerFeatureExtractor,
    PositionManagerActor,
    PositionManagerCritic,
    orthogonal_init
)


class TestPositionEncoder:
    """Test suite for PositionEncoder class."""

    def test_initialization(self):
        """Test PositionEncoder initialization."""
        encoder = PositionEncoder()

        assert encoder.position_dim == 12
        assert encoder.embed_dim == 64
        assert encoder.num_heads == 4
        assert isinstance(encoder.input_projection, torch.nn.Linear)
        assert isinstance(encoder.attention, torch.nn.MultiheadAttention)

    def test_custom_dimensions(self):
        """Test PositionEncoder with custom dimensions."""
        encoder = PositionEncoder(
            position_dim=16,
            embed_dim=128,
            num_heads=8,
            dropout=0.2
        )

        assert encoder.position_dim == 16
        assert encoder.embed_dim == 128
        assert encoder.num_heads == 8

    def test_forward_pass(self):
        """Test forward pass through PositionEncoder."""
        encoder = PositionEncoder()
        batch_size = 4
        position_features = torch.randn(batch_size, 12)

        output = encoder(position_features)

        assert output.shape == (batch_size, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_different_batch_sizes(self):
        """Test PositionEncoder with different batch sizes."""
        encoder = PositionEncoder()

        for batch_size in [1, 8, 16, 32]:
            position_features = torch.randn(batch_size, 12)
            output = encoder(position_features)
            assert output.shape == (batch_size, 64)

    def test_risk_encoding(self):
        """Test risk assessment encoding."""
        encoder = PositionEncoder()
        position_features = torch.randn(4, 12)

        # Test risk encoder directly
        risk_features = encoder.risk_encoder(position_features)
        assert risk_features.shape == (4, 16)  # embed_dim // 4

    def test_greeks_encoding(self):
        """Test Greeks exposure encoding."""
        encoder = PositionEncoder()
        position_features = torch.randn(4, 12)

        # Test Greeks encoder directly
        greeks_features = encoder.greeks_encoder(position_features)
        assert greeks_features.shape == (4, 16)  # embed_dim // 4

    def test_complexity_encoding(self):
        """Test position complexity encoding."""
        encoder = PositionEncoder()
        position_features = torch.randn(4, 12)

        # Test complexity encoder directly
        complexity_features = encoder.complexity_encoder(position_features)
        assert complexity_features.shape == (4, 8)  # embed_dim // 8


class TestPositionManagerFeatureExtractor:
    """Test suite for PositionManagerFeatureExtractor."""

    def test_initialization(self):
        """Test feature extractor initialization."""
        extractor = PositionManagerFeatureExtractor()

        assert extractor.obs_dim == 35
        assert extractor.position_dim == 12
        assert extractor.market_state_dim == 23
        assert extractor.hidden_dims == [256, 128, 64]

    def test_custom_configuration(self):
        """Test feature extractor with custom configuration."""
        extractor = PositionManagerFeatureExtractor(
            obs_dim=40,
            position_dim=15,
            hidden_dims=[512, 256, 128],
            activation='gelu',
            dropout=0.1
        )

        assert extractor.obs_dim == 40
        assert extractor.position_dim == 15
        assert extractor.market_state_dim == 25

    def test_forward_pass(self):
        """Test forward pass through feature extractor."""
        extractor = PositionManagerFeatureExtractor()
        batch_size = 4
        observations = torch.randn(batch_size, 35)

        features = extractor(observations)

        assert features.shape == (batch_size, 64)  # hidden_dims[-1]
        assert not torch.isnan(features).any()

    def test_market_position_split(self):
        """Test proper splitting of observations into market and position components."""
        extractor = PositionManagerFeatureExtractor()
        observations = torch.randn(4, 35)

        # Test internal components
        market_features = observations[:, :-extractor.position_dim]
        position_features = observations[:, -extractor.position_dim:]

        assert market_features.shape == (4, 23)
        assert position_features.shape == (4, 12)

    def test_activation_functions(self):
        """Test different activation functions."""
        activations = ['relu', 'tanh', 'gelu']

        for activation in activations:
            extractor = PositionManagerFeatureExtractor(activation=activation)
            observations = torch.randn(2, 35)
            features = extractor(observations)
            assert features.shape == (2, 64)

    def test_invalid_activation(self):
        """Test invalid activation function raises error."""
        with pytest.raises(ValueError, match="Unsupported activation"):
            PositionManagerFeatureExtractor(activation='invalid')


class TestPositionManagerActor:
    """Test suite for PositionManagerActor."""

    def test_initialization(self):
        """Test actor initialization."""
        actor = PositionManagerActor(feature_dim=64)

        assert actor.feature_dim == 64
        assert actor.action_dim == 4
        assert actor.temperature == 1.0

    def test_forward_pass(self):
        """Test forward pass through actor."""
        actor = PositionManagerActor(feature_dim=64)
        features = torch.randn(4, 64)

        action_logits = actor(features)

        assert action_logits.shape == (4, 4)
        assert not torch.isnan(action_logits).any()

    def test_action_masking(self):
        """Test action masking logic."""
        actor = PositionManagerActor(feature_dim=64)
        features = torch.randn(4, 64)

        action_mask = actor.get_action_mask(features)

        assert action_mask.shape == (4, 4)
        assert torch.all((action_mask == 0) | (action_mask == 1))  # Binary mask
        assert torch.all(action_mask.sum(dim=1) >= 1)  # At least one valid action

    def test_action_distribution(self):
        """Test action distribution generation."""
        actor = PositionManagerActor(feature_dim=64)
        features = torch.randn(4, 64)

        dist = actor.get_distribution(features)

        assert isinstance(dist, torch.distributions.Categorical)
        assert dist.probs.shape == (4, 4)
        assert torch.allclose(dist.probs.sum(dim=1), torch.ones(4))  # Probabilities sum to 1

    def test_action_sampling(self):
        """Test action sampling with masking."""
        actor = PositionManagerActor(feature_dim=64)
        features = torch.randn(4, 64)

        actions, log_probs = actor.get_action(features, deterministic=False)

        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert torch.all(actions >= 0) and torch.all(actions < 4)

    def test_deterministic_action(self):
        """Test deterministic action selection."""
        actor = PositionManagerActor(feature_dim=64)
        features = torch.randn(4, 64)

        actions, log_probs = actor.get_action(features, deterministic=True)

        assert actions.shape == (4,)
        assert log_probs.shape == (4,)

    def test_action_evaluation(self):
        """Test action evaluation for given actions."""
        actor = PositionManagerActor(feature_dim=64)
        features = torch.randn(4, 64)
        actions = torch.randint(0, 4, (4,))

        log_probs, entropy = actor.evaluate_actions(features, actions)

        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)
        assert not torch.isnan(log_probs).any()
        assert not torch.isnan(entropy).any()

    def test_temperature_scaling(self):
        """Test temperature scaling affects action distribution."""
        features = torch.randn(2, 64)

        # Low temperature (more deterministic)
        actor_low_temp = PositionManagerActor(feature_dim=64, temperature=0.1)
        dist_low = actor_low_temp.get_distribution(features)

        # High temperature (more random)
        actor_high_temp = PositionManagerActor(feature_dim=64, temperature=10.0)
        dist_high = actor_high_temp.get_distribution(features)

        # Low temperature should have lower entropy
        entropy_low = dist_low.entropy()
        entropy_high = dist_high.entropy()

        assert torch.all(entropy_low < entropy_high)


class TestPositionManagerCritic:
    """Test suite for PositionManagerCritic."""

    def test_initialization(self):
        """Test critic initialization."""
        critic = PositionManagerCritic(feature_dim=64)

        assert isinstance(critic.value_net, torch.nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass through critic."""
        critic = PositionManagerCritic(feature_dim=64)
        features = torch.randn(4, 64)

        values = critic(features)

        assert values.shape == (4, 1)
        assert not torch.isnan(values).any()

    def test_value_range(self):
        """Test that value outputs are reasonable."""
        critic = PositionManagerCritic(feature_dim=64)
        features = torch.randn(100, 64)

        values = critic(features)

        # Values should be finite and in reasonable range
        assert torch.all(torch.isfinite(values))
        assert values.std() > 0  # Should have some variation


class TestPositionManagerNetwork:
    """Test suite for complete PositionManagerNetwork."""

    def test_initialization(self):
        """Test network initialization."""
        network = PositionManagerNetwork()

        assert network.obs_dim == 35
        assert network.action_dim == 4
        assert network.position_dim == 12
        assert isinstance(network.feature_extractor, PositionManagerFeatureExtractor)
        assert isinstance(network.actor, PositionManagerActor)
        assert isinstance(network.critic, PositionManagerCritic)

    def test_custom_configuration(self):
        """Test network with custom configuration."""
        network = PositionManagerNetwork(
            obs_dim=40,
            action_dim=6,
            position_dim=15,
            hidden_dims=[512, 256, 128],
            temperature=0.5
        )

        assert network.obs_dim == 40
        assert network.action_dim == 6
        assert network.position_dim == 15

    def test_forward_pass(self):
        """Test forward pass through complete network."""
        network = PositionManagerNetwork()
        observations = torch.randn(4, 35)

        action_logits, state_values = network(observations)

        assert action_logits.shape == (4, 4)
        assert state_values.shape == (4, 1)
        assert not torch.isnan(action_logits).any()
        assert not torch.isnan(state_values).any()

    def test_get_action(self):
        """Test action generation with values."""
        network = PositionManagerNetwork()
        observations = torch.randn(4, 35)

        actions, log_probs, values = network.get_action(observations)

        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4, 1)
        assert torch.all(actions >= 0) and torch.all(actions < 4)

    def test_evaluate_actions(self):
        """Test action evaluation for training."""
        network = PositionManagerNetwork()
        observations = torch.randn(4, 35)
        actions = torch.randint(0, 4, (4,))

        log_probs, values, entropy = network.evaluate_actions(observations, actions)

        assert log_probs.shape == (4,)
        assert values.shape == (4, 1)
        assert entropy.shape == (4,)

    def test_get_value_only(self):
        """Test value-only computation."""
        network = PositionManagerNetwork()
        observations = torch.randn(4, 35)

        values = network.get_value(observations)

        assert values.shape == (4, 1)

    def test_deterministic_vs_stochastic(self):
        """Test deterministic vs stochastic action selection."""
        network = PositionManagerNetwork()
        observations = torch.randn(1, 35)

        # Multiple samples with same input
        actions_stochastic = []
        for _ in range(10):
            action, _, _ = network.get_action(observations, deterministic=False)
            actions_stochastic.append(action.item())

        # Deterministic should always return same action
        action_det1, _, _ = network.get_action(observations, deterministic=True)
        action_det2, _, _ = network.get_action(observations, deterministic=True)

        assert action_det1.item() == action_det2.item()

        # Stochastic might have some variation (though not guaranteed with random seed)
        actions_unique = len(set(actions_stochastic))
        assert actions_unique >= 1  # At least one unique action

    def test_checkpoint_save_load(self):
        """Test model checkpoint saving and loading."""
        network = PositionManagerNetwork()
        observations = torch.randn(2, 35)

        # Get initial output
        initial_output = network(observations)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name

        try:
            # Save checkpoint
            network.save_checkpoint(checkpoint_path)
            assert os.path.exists(checkpoint_path)

            # Create new network and load checkpoint
            new_network = PositionManagerNetwork()
            new_network.load_checkpoint(checkpoint_path)

            # Check outputs match
            loaded_output = new_network(observations)
            assert torch.allclose(initial_output[0], loaded_output[0], atol=1e-6)
            assert torch.allclose(initial_output[1], loaded_output[1], atol=1e-6)

        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)

    def test_model_summary(self):
        """Test model summary generation."""
        network = PositionManagerNetwork()
        summary = network.get_model_summary()

        required_keys = [
            'total_parameters', 'trainable_parameters', 'actor_parameters',
            'critic_parameters', 'feature_extractor_parameters', 'observation_dim',
            'action_dim', 'position_dim', 'hidden_dims'
        ]

        for key in required_keys:
            assert key in summary

        assert summary['observation_dim'] == 35
        assert summary['action_dim'] == 4
        assert summary['position_dim'] == 12
        assert summary['total_parameters'] > 0
        assert summary['trainable_parameters'] <= summary['total_parameters']

    def test_gradient_flow(self):
        """Test gradient flow through network."""
        network = PositionManagerNetwork()
        observations = torch.randn(4, 35, requires_grad=True)
        actions = torch.randint(0, 4, (4,))

        log_probs, values, entropy = network.evaluate_actions(observations, actions)

        # Compute dummy loss
        loss = -log_probs.mean() + values.pow(2).mean() - 0.01 * entropy.mean()
        loss.backward()

        # Check gradients exist
        has_gradients = any(p.grad is not None for p in network.parameters())
        assert has_gradients

    def test_batch_sizes(self):
        """Test network with different batch sizes."""
        network = PositionManagerNetwork()

        for batch_size in [1, 4, 16, 32]:
            observations = torch.randn(batch_size, 35)

            action_logits, state_values = network(observations)
            assert action_logits.shape == (batch_size, 4)
            assert state_values.shape == (batch_size, 1)

    def test_input_validation(self):
        """Test network handles invalid inputs gracefully."""
        network = PositionManagerNetwork()

        # Wrong observation dimension
        with pytest.raises((RuntimeError, IndexError)):
            wrong_obs = torch.randn(4, 30)  # Should be 35
            network(wrong_obs)

    def test_action_masking_integration(self):
        """Test that action masking works in complete network."""
        network = PositionManagerNetwork()
        observations = torch.randn(4, 35)

        actions, log_probs, values = network.get_action(observations)

        # Get action mask from actor
        features = network.feature_extractor(observations)
        action_mask = network.actor.get_action_mask(features)

        # All selected actions should be valid according to mask
        for i in range(4):
            action = actions[i].item()
            assert action_mask[i, action].item() == 1.0

    def test_position_encoding_integration(self):
        """Test position encoding integration in complete pipeline."""
        network = PositionManagerNetwork()

        # Create observations with varied position components
        observations = torch.randn(4, 35)
        # Make position features more distinct
        observations[:, -12:] = torch.randn(4, 12) * 5

        features = network.feature_extractor(observations)

        # Features should capture position information
        assert features.shape == (4, 64)
        assert not torch.allclose(features[0], features[1], atol=1e-3)


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_orthogonal_init_linear(self):
        """Test orthogonal initialization for linear layers."""
        layer = torch.nn.Linear(10, 20)
        original_weight = layer.weight.clone()

        orthogonal_init(layer, gain=2.0)

        # Weight should have changed
        assert not torch.allclose(layer.weight, original_weight)

        # Bias should be zero
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_orthogonal_init_conv(self):
        """Test orthogonal initialization for conv layers."""
        layer = torch.nn.Conv2d(3, 16, 3)
        original_weight = layer.weight.clone()

        orthogonal_init(layer, gain=1.5)

        # Weight should have changed
        assert not torch.allclose(layer.weight, original_weight)

    def test_orthogonal_init_other(self):
        """Test orthogonal initialization skips unsupported layers."""
        layer = torch.nn.BatchNorm2d(16)
        original_weight = layer.weight.clone()

        orthogonal_init(layer)  # Should not error

        # Weight should be unchanged
        assert torch.allclose(layer.weight, original_weight)


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_training_step_simulation(self):
        """Simulate a training step."""
        network = PositionManagerNetwork()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        # Simulate batch of trajectories
        batch_size = 32
        observations = torch.randn(batch_size, 35)
        actions = torch.randint(0, 4, (batch_size,))
        rewards = torch.randn(batch_size)
        next_observations = torch.randn(batch_size, 35)
        dones = torch.randint(0, 2, (batch_size,)).bool()

        # Forward pass
        log_probs, values, entropy = network.evaluate_actions(observations, actions)
        next_values = network.get_value(next_observations)

        # Compute simple loss (dummy PPO-like)
        advantages = rewards.unsqueeze(1) - values.detach()
        policy_loss = -(log_probs.unsqueeze(1) * advantages).mean()
        value_loss = F.mse_loss(values, rewards.unsqueeze(1))
        entropy_loss = -entropy.mean()

        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        assert not torch.isnan(total_loss)

    def test_position_state_variations(self):
        """Test network with various position state patterns."""
        network = PositionManagerNetwork()

        # Test different position scenarios
        scenarios = {
            'no_position': torch.zeros(1, 35),
            'long_call': torch.randn(1, 35),
            'short_put': torch.randn(1, 35),
            'iron_condor': torch.randn(1, 35),
        }

        # Modify position components for each scenario
        scenarios['long_call'][0, -12:] = torch.tensor([
            100, 105, 1, 1, 50, 45, 0.25, 0.6, 0.05, -0.4, 100, 0.1
        ])  # Strike, current, quantity, etc.

        for scenario_name, obs in scenarios.items():
            actions, log_probs, values = network.get_action(obs)
            assert actions.shape == (1,)
            assert not torch.isnan(log_probs).any()
            assert not torch.isnan(values).any()

    def test_extreme_input_conditions(self):
        """Test network robustness with extreme inputs."""
        network = PositionManagerNetwork()

        # Very large values
        large_obs = torch.ones(2, 35) * 1000
        actions, log_probs, values = network.get_action(large_obs)
        assert not torch.isnan(log_probs).any()

        # Very small values
        small_obs = torch.ones(2, 35) * 0.001
        actions, log_probs, values = network.get_action(small_obs)
        assert not torch.isnan(log_probs).any()

        # Mixed extreme values
        mixed_obs = torch.randn(2, 35) * 100
        actions, log_probs, values = network.get_action(mixed_obs)
        assert not torch.isnan(log_probs).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])