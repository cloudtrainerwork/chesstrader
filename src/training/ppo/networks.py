"""
Actor-Critic networks for PPO in options trading environments.

This module implements neural networks optimized for options trading:
- Actor network for policy (action probability distribution)
- Critic network for value function estimation
- Shared feature extractor for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def orthogonal_init(layer: nn.Module, gain: float = 1.0):
    """
    Initialize layer weights using orthogonal initialization.

    Args:
        layer: Neural network layer to initialize
        gain: Scaling factor for initialization
    """
    if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class FeatureExtractor(nn.Module):
    """
    Shared feature extractor for actor and critic networks.

    Processes the 35-dimensional observation space into rich features
    for both policy and value function estimation.
    """

    def __init__(
        self,
        obs_dim: int = 35,
        hidden_dims: List[int] = [512, 256, 128],
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        """
        Initialize feature extractor.

        Args:
            obs_dim: Observation space dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'gelu')
            dropout: Dropout probability
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build layers
        layers = []
        input_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_dim

        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor.

        Args:
            observations: Input observations [batch_size, obs_dim]

        Returns:
            Feature representations [batch_size, hidden_dims[-1]]
        """
        return self.layers(observations)


class Actor(nn.Module):
    """
    Policy network (actor) for action selection.

    Outputs action probabilities for the 4 discrete actions in
    options trading environment.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int = 4,
        init_log_std: float = -0.5
    ):
        """
        Initialize actor network.

        Args:
            feature_dim: Input feature dimension from feature extractor
            action_dim: Number of discrete actions
            init_log_std: Initial log standard deviation (unused for discrete actions)
        """
        super().__init__()

        self.action_dim = action_dim

        # Action logits layer
        self.action_logits = nn.Linear(feature_dim, action_dim)

        # Initialize with smaller weights for final layer
        orthogonal_init(self.action_logits, gain=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network.

        Args:
            features: Feature representations [batch_size, feature_dim]

        Returns:
            Action logits [batch_size, action_dim]
        """
        action_logits = self.action_logits(features)
        return action_logits

    def get_distribution(self, features: torch.Tensor) -> torch.distributions.Categorical:
        """
        Get action probability distribution.

        Args:
            features: Feature representations

        Returns:
            Categorical distribution over actions
        """
        action_logits = self.forward(features)
        return torch.distributions.Categorical(logits=action_logits)

    def get_action(
        self,
        features: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from policy.

        Args:
            features: Feature representations
            deterministic: If True, take most probable action

        Returns:
            Tuple of (actions, log_probabilities)
        """
        dist = self.get_distribution(features)

        if deterministic:
            actions = dist.probs.argmax(dim=-1)
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)

        return actions, log_probs

    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.

        Args:
            features: Feature representations
            actions: Actions to evaluate

        Returns:
            Tuple of (log_probabilities, entropy)
        """
        dist = self.get_distribution(features)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy


class Critic(nn.Module):
    """
    Value network (critic) for state value estimation.

    Estimates the expected return from each state for policy evaluation.
    """

    def __init__(self, feature_dim: int):
        """
        Initialize critic network.

        Args:
            feature_dim: Input feature dimension from feature extractor
        """
        super().__init__()

        # Value estimation layer
        self.value = nn.Linear(feature_dim, 1)

        # Initialize with smaller weights
        orthogonal_init(self.value, gain=1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network.

        Args:
            features: Feature representations [batch_size, feature_dim]

        Returns:
            State values [batch_size, 1]
        """
        return self.value(features)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network with shared feature extractor.

    This architecture shares early layers between policy and value networks
    for improved sample efficiency and computational efficiency.
    """

    def __init__(
        self,
        obs_dim: int = 35,
        action_dim: int = 4,
        hidden_dims: Optional[List[int]] = None,
        hidden_dim: Optional[int] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
        init_log_std: float = -0.5,
        device: Optional[str] = None
    ):
        """
        Initialize Actor-Critic network.

        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout: Dropout probability
            init_log_std: Initial log standard deviation
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        if hidden_dim is not None:
            hidden_dims = [hidden_dim]

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared feature extractor
        self.feature_extractor = FeatureExtractor(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout
        )

        feature_dim = hidden_dims[-1]

        # Actor and critic networks
        self.actor = Actor(feature_dim, action_dim, init_log_std)
        self.critic = Critic(feature_dim)

        if device:
            self.to(device)

    def forward(
        self,
        observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both actor and critic.

        Args:
            observations: Input observations [batch_size, obs_dim]

        Returns:
            Tuple of (action_logits, state_values)
        """
        # Extract features
        features = self.feature_extractor(observations)

        # Get actor and critic outputs
        action_logits = self.actor(features)
        state_values = self.critic(features)

        return action_logits, state_values

    def get_action(
        self,
        observations: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions and values for given observations.

        Args:
            observations: Input observations
            deterministic: If True, take most probable actions

        Returns:
            Tuple of (actions, log_probabilities, state_values)
        """
        features = self.feature_extractor(observations)

        # Get actions from actor
        actions, log_probs = self.actor.get_action(features, deterministic)

        # Get values from critic
        values = self.critic(features)

        return actions, log_probs, values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training.

        Args:
            observations: Input observations
            actions: Actions to evaluate

        Returns:
            Tuple of (log_probabilities, state_values, entropy)
        """
        features = self.feature_extractor(observations)

        # Evaluate actions
        log_probs, entropy = self.actor.evaluate_actions(features, actions)

        # Get state values
        values = self.critic(features)

        return log_probs, values, entropy

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Get state values only (useful for trajectory collection).

        Args:
            observations: Input observations

        Returns:
            State values
        """
        features = self.feature_extractor(observations)
        return self.critic(features)

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.feature_extractor.hidden_dims
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str, device: str = 'cpu'):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model checkpoint loaded from {filepath}")

    def get_model_summary(self) -> dict:
        """Get summary of model architecture and parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        shared_params = sum(p.numel() for p in self.feature_extractor.parameters())

        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'actor_parameters': actor_params,
            'critic_parameters': critic_params,
            'shared_parameters': shared_params,
            'observation_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.feature_extractor.hidden_dims
        }

        return summary
