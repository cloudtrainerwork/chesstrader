"""
Tests for PPO neural networks implementation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from src.training.ppo.networks import (
    FeatureExtractor, Actor, Critic, ActorCritic, orthogonal_init
)


class TestOrthogonalInit:
    """Test orthogonal initialization function."""

    def test_linear_layer_init(self):
        """Test orthogonal initialization for linear layer."""
        layer = torch.nn.Linear(10, 5)
        original_weight = layer.weight.clone()

        orthogonal_init(layer, gain=1.0)

        # Weight should have changed
        assert not torch.equal(layer.weight, original_weight)

        # Bias should be zero
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_conv_layer_init(self):
        """Test orthogonal initialization for conv layer."""
        layer = torch.nn.Conv1d(3, 6, 3)
        original_weight = layer.weight.clone()

        orthogonal_init(layer, gain=2.0)

        # Weight should have changed
        assert not torch.equal(layer.weight, original_weight)

    def test_non_applicable_layer(self):
        """Test that non-applicable layers are not modified."""
        layer = torch.nn.BatchNorm1d(10)
        original_weight = layer.weight.clone()

        orthogonal_init(layer)

        # Weight should not have changed
        assert torch.equal(layer.weight, original_weight)


class TestFeatureExtractor:
    """Test cases for feature extractor."""

    @pytest.fixture
    def feature_extractor(self):
        """Create test feature extractor."""
        return FeatureExtractor(obs_dim=35, hidden_dims=[64, 32])

    def test_initialization(self):
        """Test feature extractor initialization."""
        fe = FeatureExtractor(obs_dim=35, hidden_dims=[64, 32], activation='relu')

        assert fe.obs_dim == 35
        assert fe.hidden_dims == [64, 32]
        assert isinstance(fe.activation, torch.nn.ReLU)

    def test_activation_functions(self):
        """Test different activation functions."""
        # ReLU
        fe_relu = FeatureExtractor(obs_dim=10, hidden_dims=[20], activation='relu')
        assert isinstance(fe_relu.activation, torch.nn.ReLU)

        # Tanh
        fe_tanh = FeatureExtractor(obs_dim=10, hidden_dims=[20], activation='tanh')
        assert isinstance(fe_tanh.activation, torch.nn.Tanh)

        # GELU
        fe_gelu = FeatureExtractor(obs_dim=10, hidden_dims=[20], activation='gelu')
        assert isinstance(fe_gelu.activation, torch.nn.GELU)

        # Invalid activation
        with pytest.raises(ValueError, match="Unsupported activation"):
            FeatureExtractor(obs_dim=10, hidden_dims=[20], activation='invalid')

    def test_forward_pass(self, feature_extractor):
        """Test forward pass through feature extractor."""
        batch_size = 16
        observations = torch.randn(batch_size, 35)

        features = feature_extractor(observations)

        assert features.shape == (batch_size, 32)  # Last hidden dim
        assert features.dtype == torch.float32

    def test_dropout(self):
        """Test feature extractor with dropout."""
        fe = FeatureExtractor(obs_dim=35, hidden_dims=[64], dropout=0.5)

        # Should contain dropout layers
        has_dropout = any(isinstance(layer, torch.nn.Dropout) for layer in fe.layers)
        assert has_dropout

    def test_no_dropout(self):
        """Test feature extractor without dropout."""
        fe = FeatureExtractor(obs_dim=35, hidden_dims=[64], dropout=0.0)

        # Should not contain dropout layers
        has_dropout = any(isinstance(layer, torch.nn.Dropout) for layer in fe.layers)
        assert not has_dropout


class TestActor:
    """Test cases for actor network."""

    @pytest.fixture
    def actor(self):
        """Create test actor."""
        return Actor(feature_dim=32, action_dim=4)

    def test_initialization(self, actor):
        """Test actor initialization."""
        assert actor.action_dim == 4
        assert isinstance(actor.action_logits, torch.nn.Linear)

    def test_forward_pass(self, actor):
        """Test forward pass through actor."""
        batch_size = 16
        features = torch.randn(batch_size, 32)

        action_logits = actor(features)

        assert action_logits.shape == (batch_size, 4)

    def test_get_distribution(self, actor):
        """Test action distribution generation."""
        features = torch.randn(8, 32)

        dist = actor.get_distribution(features)

        assert isinstance(dist, torch.distributions.Categorical)
        assert dist.probs.shape == (8, 4)

        # Probabilities should sum to 1
        assert torch.allclose(dist.probs.sum(dim=1), torch.ones(8))

    def test_get_action_deterministic(self, actor):
        """Test deterministic action selection."""
        features = torch.randn(1, 32)

        actions, log_probs = actor.get_action(features, deterministic=True)

        assert actions.shape == (1,)
        assert log_probs.shape == (1,)
        assert 0 <= actions.item() < 4

    def test_get_action_stochastic(self, actor):
        """Test stochastic action selection."""
        features = torch.randn(16, 32)

        actions, log_probs = actor.get_action(features, deterministic=False)

        assert actions.shape == (16,)
        assert log_probs.shape == (16,)
        assert torch.all((actions >= 0) & (actions < 4))
        assert torch.all(log_probs <= 0)  # Log probs should be negative

    def test_evaluate_actions(self, actor):
        """Test action evaluation."""
        features = torch.randn(8, 32)
        actions = torch.randint(0, 4, (8,))

        log_probs, entropy = actor.evaluate_actions(features, actions)

        assert log_probs.shape == (8,)
        assert entropy.shape == (8,)
        assert torch.all(log_probs <= 0)  # Log probs should be negative
        assert torch.all(entropy >= 0)    # Entropy should be positive


class TestCritic:
    """Test cases for critic network."""

    @pytest.fixture
    def critic(self):
        """Create test critic."""
        return Critic(feature_dim=32)

    def test_initialization(self, critic):
        """Test critic initialization."""
        assert isinstance(critic.value, torch.nn.Linear)

    def test_forward_pass(self, critic):
        """Test forward pass through critic."""
        batch_size = 16
        features = torch.randn(batch_size, 32)

        values = critic(features)

        assert values.shape == (batch_size, 1)


class TestActorCritic:
    """Test cases for combined actor-critic network."""

    @pytest.fixture
    def actor_critic(self):
        """Create test actor-critic network."""
        return ActorCritic(obs_dim=35, action_dim=4, hidden_dims=[64, 32])

    def test_initialization(self, actor_critic):
        """Test actor-critic initialization."""
        assert actor_critic.obs_dim == 35
        assert actor_critic.action_dim == 4
        assert isinstance(actor_critic.feature_extractor, FeatureExtractor)
        assert isinstance(actor_critic.actor, Actor)
        assert isinstance(actor_critic.critic, Critic)

    def test_forward_pass(self, actor_critic):
        """Test forward pass through actor-critic."""
        batch_size = 16
        observations = torch.randn(batch_size, 35)

        action_logits, values = actor_critic(observations)

        assert action_logits.shape == (batch_size, 4)
        assert values.shape == (batch_size, 1)

    def test_get_action(self, actor_critic):
        """Test action and value generation."""
        observations = torch.randn(8, 35)

        actions, log_probs, values = actor_critic.get_action(observations)

        assert actions.shape == (8,)
        assert log_probs.shape == (8,)
        assert values.shape == (8, 1)
        assert torch.all((actions >= 0) & (actions < 4))

    def test_get_action_deterministic(self, actor_critic):
        """Test deterministic action generation."""
        observations = torch.randn(1, 35)

        actions, log_probs, values = actor_critic.get_action(
            observations, deterministic=True
        )

        assert actions.shape == (1,)
        assert log_probs.shape == (1,)
        assert values.shape == (1, 1)

    def test_evaluate_actions(self, actor_critic):
        """Test action evaluation."""
        observations = torch.randn(8, 35)
        actions = torch.randint(0, 4, (8,))

        log_probs, values, entropy = actor_critic.evaluate_actions(observations, actions)

        assert log_probs.shape == (8,)
        assert values.shape == (8, 1)
        assert entropy.shape == (8,)
        assert torch.all(log_probs <= 0)
        assert torch.all(entropy >= 0)

    def test_get_value(self, actor_critic):
        """Test value-only computation."""
        observations = torch.randn(8, 35)

        values = actor_critic.get_value(observations)

        assert values.shape == (8, 1)

    def test_save_checkpoint(self, actor_critic, tmp_path):
        """Test model checkpoint saving."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"

        actor_critic.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_path)
        assert 'model_state_dict' in checkpoint
        assert 'obs_dim' in checkpoint
        assert 'action_dim' in checkpoint

    def test_load_checkpoint(self, actor_critic, tmp_path):
        """Test model checkpoint loading."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"

        # Save original state
        original_params = {name: param.clone()
                          for name, param in actor_critic.named_parameters()}

        # Save checkpoint
        actor_critic.save_checkpoint(str(checkpoint_path))

        # Modify parameters
        with torch.no_grad():
            for param in actor_critic.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Load checkpoint
        actor_critic.load_checkpoint(str(checkpoint_path))

        # Parameters should be restored
        for name, param in actor_critic.named_parameters():
            assert torch.allclose(param, original_params[name])

    def test_get_model_summary(self, actor_critic):
        """Test model summary generation."""
        summary = actor_critic.get_model_summary()

        assert isinstance(summary, dict)
        expected_keys = [
            'total_parameters', 'trainable_parameters', 'actor_parameters',
            'critic_parameters', 'shared_parameters', 'observation_dim',
            'action_dim', 'hidden_dims'
        ]

        for key in expected_keys:
            assert key in summary

        assert summary['observation_dim'] == 35
        assert summary['action_dim'] == 4
        assert summary['total_parameters'] > 0

    def test_parameter_sharing(self, actor_critic):
        """Test that feature extractor parameters are shared."""
        # Actor and critic should share feature extractor
        actor_features = id(actor_critic.feature_extractor)

        # Both actor and critic should use the same feature extractor
        assert hasattr(actor_critic, 'feature_extractor')

        # Parameters should be shared (same object)
        fe_params = list(actor_critic.feature_extractor.parameters())
        total_params = list(actor_critic.parameters())

        # Feature extractor parameters should be in total parameters
        assert len(fe_params) > 0
        assert all(any(torch.equal(fe_param, total_param)
                      for total_param in total_params)
                  for fe_param in fe_params)

    def test_gradient_flow(self, actor_critic):
        """Test that gradients flow through the network."""
        observations = torch.randn(4, 35, requires_grad=True)

        action_logits, values = actor_critic(observations)
        loss = action_logits.sum() + values.sum()
        loss.backward()

        # Check that gradients exist
        assert observations.grad is not None

        # Check that network parameters have gradients
        for name, param in actor_critic.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_different_hidden_dims(self):
        """Test actor-critic with different hidden dimensions."""
        # Single hidden layer
        ac1 = ActorCritic(obs_dim=35, action_dim=4, hidden_dims=[128])

        # Multiple hidden layers
        ac2 = ActorCritic(obs_dim=35, action_dim=4, hidden_dims=[256, 128, 64])

        # Both should work
        obs = torch.randn(1, 35)

        logits1, values1 = ac1(obs)
        logits2, values2 = ac2(obs)

        assert logits1.shape == logits2.shape == (1, 4)
        assert values1.shape == values2.shape == (1, 1)