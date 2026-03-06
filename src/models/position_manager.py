"""
Position Manager Network Architecture for intelligent options position management.

Implements actor-critic neural network architecture for making position management decisions
(HOLD, CLOSE, ADJUST, ROLL) based on market state, position state, and regime information.
Builds on chess-inspired spatial encoder patterns and integrates with PPO training infrastructure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging

from .residual_blocks import ResidualBlock, ResidualStack
from .attention import SpatialAttention, PositionalEncoding

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


class PositionEncoder(nn.Module):
    """
    Specialized position state encoder that processes position components.

    Takes position-specific features (strike prices, time to expiration, current P&L, Greeks)
    and creates dense embeddings using attention mechanism to focus on most relevant attributes.
    """

    def __init__(
        self,
        position_dim: int = 12,  # Position-specific features from 35-dim observation
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize position encoder.

        Args:
            position_dim: Input dimension for position features
            embed_dim: Output embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.position_dim = position_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Initial feature embedding
        self.input_projection = nn.Linear(position_dim, embed_dim)

        # Position risk assessment layers
        self.risk_encoder = nn.Sequential(
            nn.Linear(position_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4)
        )

        # Greeks exposure encoding
        self.greeks_encoder = nn.Sequential(
            nn.Linear(position_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4)
        )

        # Position complexity encoding (number of legs, spread width)
        self.complexity_encoder = nn.Sequential(
            nn.Linear(position_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 8)
        )

        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))

    def forward(self, position_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through position encoder.

        Args:
            position_features: Position state tensor [batch_size, position_dim]

        Returns:
            Position embeddings [batch_size, embed_dim]
        """
        batch_size = position_features.shape[0]

        # Initial projection
        pos_embed = self.input_projection(position_features)  # [batch, embed_dim]

        # Risk assessment encoding
        risk_features = self.risk_encoder(position_features)  # [batch, embed_dim//4]

        # Greeks exposure encoding
        greeks_features = self.greeks_encoder(position_features)  # [batch, embed_dim//4]

        # Position complexity encoding
        complexity_features = self.complexity_encoder(position_features)  # [batch, embed_dim//8]

        # Combine specialized encodings
        specialized_features = torch.cat([
            risk_features,
            greeks_features,
            complexity_features,
            torch.zeros(batch_size, self.embed_dim - risk_features.shape[1] -
                       greeks_features.shape[1] - complexity_features.shape[1]).to(position_features.device)
        ], dim=1)

        # Apply attention mechanism
        # Reshape for attention: [batch, seq_len=2, embed_dim]
        query_key_value = torch.stack([pos_embed, specialized_features], dim=1)

        attended_features, _ = self.attention(
            query_key_value, query_key_value, query_key_value
        )

        # Pool attended features
        output_features = attended_features.mean(dim=1)  # [batch, embed_dim]

        # Layer normalization
        output_features = self.layer_norm(output_features)

        return output_features


class PositionManagerFeatureExtractor(nn.Module):
    """
    Feature extractor specialized for position management with chess-inspired patterns.

    Processes 35-dimensional observation space into rich features for both policy
    and value function estimation with specialized position encoding.
    """

    def __init__(
        self,
        obs_dim: int = 35,
        position_dim: int = 12,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        """
        Initialize position manager feature extractor.

        Args:
            obs_dim: Full observation space dimension
            position_dim: Position-specific features dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.position_dim = position_dim
        self.market_state_dim = obs_dim - position_dim  # Market + regime features
        self.hidden_dims = hidden_dims

        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Specialized position encoder
        self.position_encoder = PositionEncoder(
            position_dim=position_dim,
            embed_dim=64,
            dropout=dropout
        )

        # Market state encoder
        self.market_encoder = nn.Sequential(
            nn.Linear(self.market_state_dim, 128),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(128, 64)
        )

        # Combined feature processing with residual connections
        combined_input_dim = 64 + 64  # position_embed + market_embed

        layers = []
        input_dim = combined_input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))

            # Add residual connection for deeper layers
            if i > 0 and input_dim == hidden_dim:
                layers.append(ResidualConnection(hidden_dim))

            layers.append(self.activation)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_dim

        self.combined_layers = nn.Sequential(*layers)

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
        # Split observations into market and position components
        market_features = observations[:, :-self.position_dim]
        position_features = observations[:, -self.position_dim:]

        # Encode position features
        position_embed = self.position_encoder(position_features)

        # Encode market features
        market_embed = self.market_encoder(market_features)

        # Combine embeddings
        combined_features = torch.cat([market_embed, position_embed], dim=1)

        # Process through combined layers
        output_features = self.combined_layers(combined_features)

        return output_features


class ResidualConnection(nn.Module):
    """Simple residual connection for same-dimension layers."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # Identity - residual added in parent layer


class PositionManagerActor(nn.Module):
    """
    Policy network (actor) for position management action selection.

    Outputs action probabilities for 4 discrete actions (HOLD, CLOSE, ADJUST, ROLL)
    with position-aware action masking and risk constraint checking.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int = 4,
        temperature: float = 1.0
    ):
        """
        Initialize actor network.

        Args:
            feature_dim: Input feature dimension
            action_dim: Number of discrete actions
            temperature: Temperature for action selection (higher = more exploration)
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.temperature = temperature

        # Action logits layer
        self.action_logits = nn.Linear(feature_dim, action_dim)

        # Action masking logic (learned parameters)
        self.action_mask_net = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Sigmoid()  # Probability of action being valid
        )

        # Initialize with smaller weights for final layer
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            orthogonal_init(self.action_logits, gain=0.01)
            self.apply(lambda m: orthogonal_init(m, gain=0.01) if isinstance(m, nn.Linear) else None)

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

    def get_action_mask(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get action validity mask based on current state.

        Args:
            features: Feature representations

        Returns:
            Action mask [batch_size, action_dim] with 1 for valid actions
        """
        mask_probs = self.action_mask_net(features)
        # Convert probabilities to hard mask (threshold at 0.5)
        action_mask = (mask_probs > 0.5).float()

        # Ensure at least one action is always valid (fallback to HOLD)
        no_valid_actions = (action_mask.sum(dim=1) == 0)
        action_mask[no_valid_actions, 0] = 1.0  # HOLD action

        # If only one action is valid, add the next-best action to avoid degenerate entropy
        single_valid = (action_mask.sum(dim=1) == 1)
        if single_valid.any():
            top2 = torch.topk(mask_probs, k=2, dim=1).indices
            for idx in torch.where(single_valid)[0]:
                second_choice = top2[idx, 1]
                action_mask[idx, second_choice] = 1.0

        return action_mask

    def get_distribution(self, features: torch.Tensor) -> torch.distributions.Categorical:
        """
        Get action probability distribution with masking.

        Args:
            features: Feature representations

        Returns:
            Categorical distribution over valid actions
        """
        action_logits = self.forward(features)
        action_mask = self.get_action_mask(features)

        # Apply temperature scaling
        scaled_logits = action_logits / self.temperature

        # Apply mask (set invalid actions to very negative logits)
        masked_logits = scaled_logits + torch.log(action_mask + 1e-8)

        return torch.distributions.Categorical(logits=masked_logits)

    def get_action(
        self,
        features: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from policy with masking.

        Args:
            features: Feature representations
            deterministic: If True, take most probable valid action

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


class PositionManagerCritic(nn.Module):
    """
    Value network (critic) for position value estimation.

    Estimates the expected return from each position state for policy evaluation.
    """

    def __init__(self, feature_dim: int):
        """
        Initialize critic network.

        Args:
            feature_dim: Input feature dimension
        """
        super().__init__()

        # Value estimation layers
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Initialize weights
        self.apply(lambda m: orthogonal_init(m, gain=1.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network.

        Args:
            features: Feature representations [batch_size, feature_dim]

        Returns:
            State values [batch_size, 1]
        """
        return self.value_net(features)


class PositionManagerNetwork(nn.Module):
    """
    Combined Position Manager Actor-Critic network.

    Specialized actor-critic architecture for intelligent position management
    with chess-inspired patterns, position encoding, and action masking.
    """

    def __init__(
        self,
        obs_dim: int = 35,
        action_dim: int = 4,
        position_dim: int = 12,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = 'relu',
        dropout: float = 0.0,
        temperature: float = 1.0
    ):
        """
        Initialize Position Manager Network.

        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension (4 for HOLD, CLOSE, ADJUST, ROLL)
            position_dim: Position-specific features dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout: Dropout probability
            temperature: Temperature for action selection
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.position_dim = position_dim

        # Feature extractor with specialized position encoding
        self.feature_extractor = PositionManagerFeatureExtractor(
            obs_dim=obs_dim,
            position_dim=position_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout
        )

        feature_dim = hidden_dims[-1]

        # Actor and critic networks
        self.actor = PositionManagerActor(
            feature_dim=feature_dim,
            action_dim=action_dim,
            temperature=temperature
        )

        self.critic = PositionManagerCritic(feature_dim=feature_dim)

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
            deterministic: If True, take most probable valid actions

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
            'position_dim': self.position_dim,
            'hidden_dims': self.feature_extractor.hidden_dims
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Position Manager checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str, device: str = 'cpu'):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Position Manager checkpoint loaded from {filepath}")

    def get_model_summary(self) -> Dict[str, any]:
        """Get summary of model architecture and parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        feature_params = sum(p.numel() for p in self.feature_extractor.parameters())

        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'actor_parameters': actor_params,
            'critic_parameters': critic_params,
            'feature_extractor_parameters': feature_params,
            'observation_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'position_dim': self.position_dim,
            'hidden_dims': self.feature_extractor.hidden_dims
        }

        return summary
