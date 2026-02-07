"""
PPO (Proximal Policy Optimization) algorithm implementation.

This module implements the core PPO algorithm with clipped surrogate objective,
value function loss, and entropy regularization for stable policy optimization
in options trading environments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PPOAlgorithm:
    """
    Proximal Policy Optimization algorithm implementation.

    Implements the clipped surrogate objective function with value function loss
    and entropy regularization for stable policy optimization.
    """

    def __init__(
        self,
        actor_critic: nn.Module,
        policy_lr: float = 3e-4,
        value_lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Initialize PPO algorithm.

        Args:
            actor_critic: Actor-critic network
            policy_lr: Learning rate for policy optimizer
            value_lr: Learning rate for value optimizer
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient
            value_loss_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run computations on
        """
        self.actor_critic = actor_critic
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Separate optimizers for actor and critic (can be same learning rate)
        self.policy_optimizer = optim.Adam(
            self.actor_critic.actor.parameters(),
            lr=policy_lr
        )
        self.value_optimizer = optim.Adam(
            self.actor_critic.critic.parameters(),
            lr=value_lr
        )

        # Track training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': []
        }

    def compute_ppo_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO clipped surrogate loss.

        Args:
            observations: State observations
            actions: Actions taken
            old_log_probs: Log probabilities from old policy
            advantages: Advantage estimates

        Returns:
            Tuple of (loss_tensor, statistics_dict)
        """
        # Get current policy outputs
        action_logits, _ = self.actor_critic(observations)
        dist = torch.distributions.Categorical(logits=action_logits)

        # Calculate log probabilities and entropy
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Calculate probability ratios
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        # PPO clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus for exploration
        entropy_loss = -self.entropy_coef * entropy.mean()

        # Combined loss
        total_loss = policy_loss + entropy_loss

        # Calculate statistics for monitoring
        with torch.no_grad():
            kl_divergence = ((ratio - 1) - log_ratio).mean().item()
            clip_fraction = ((ratio > 1.0 + self.clip_epsilon) |
                           (ratio < 1.0 - self.clip_epsilon)).float().mean().item()

        stats = {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'kl_divergence': kl_divergence,
            'clip_fraction': clip_fraction,
            'mean_entropy': entropy.mean().item()
        }

        return total_loss, stats

    def compute_value_loss(
        self,
        observations: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute value function loss with optional clipping.

        Args:
            observations: State observations
            returns: Target returns
            old_values: Old value estimates

        Returns:
            Tuple of (loss_tensor, statistics_dict)
        """
        # Get current value estimates
        _, values = self.actor_critic(observations)
        values = values.squeeze(-1)

        # Clipped value loss (helps prevent large value function updates)
        value_pred_clipped = old_values + torch.clamp(
            values - old_values, -self.clip_epsilon, self.clip_epsilon
        )

        value_losses = (values - returns).pow(2)
        value_losses_clipped = (value_pred_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

        # Statistics
        stats = {
            'value_loss': value_loss.item(),
            'mean_value': values.mean().item(),
            'mean_return': returns.mean().item(),
            'value_std': values.std().item()
        }

        return value_loss, stats

    def get_action_log_prob(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate log probabilities for given actions.

        Args:
            observations: State observations
            actions: Actions to evaluate

        Returns:
            Log probabilities for the actions
        """
        action_logits, _ = self.actor_critic(observations)
        dist = torch.distributions.Categorical(logits=action_logits)
        return dist.log_prob(actions)

    def update_policy(
        self,
        batch_data: Dict[str, torch.Tensor],
        n_epochs: int = 4
    ) -> Dict[str, List[float]]:
        """
        Perform policy network updates using PPO.

        Args:
            batch_data: Dictionary containing batch data
            n_epochs: Number of optimization epochs

        Returns:
            Dictionary of training statistics
        """
        # Extract batch data
        observations = batch_data['observations'].to(self.device)
        actions = batch_data['actions'].to(self.device)
        old_log_probs = batch_data['log_probs'].to(self.device)
        advantages = batch_data['advantages'].to(self.device)
        returns = batch_data['returns'].to(self.device)
        old_values = batch_data['values'].to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        epoch_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': []
        }

        for epoch in range(n_epochs):
            # Compute policy loss
            policy_loss, policy_stats = self.compute_ppo_loss(
                observations, actions, old_log_probs, advantages
            )

            # Compute value loss
            value_loss, value_stats = self.compute_value_loss(
                observations, returns, old_values
            )

            # Combined loss
            total_loss = policy_loss + self.value_loss_coef * value_loss

            # Policy update
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.max_grad_norm
            )

            self.policy_optimizer.step()
            self.value_optimizer.step()

            # Collect statistics
            epoch_stats['policy_loss'].append(policy_stats['policy_loss'])
            epoch_stats['value_loss'].append(value_stats['value_loss'])
            epoch_stats['entropy_loss'].append(policy_stats['entropy_loss'])
            epoch_stats['total_loss'].append(total_loss.item())
            epoch_stats['kl_divergence'].append(policy_stats['kl_divergence'])
            epoch_stats['clip_fraction'].append(policy_stats['clip_fraction'])

            # Early stopping if KL divergence gets too large
            if policy_stats['kl_divergence'] > 0.02:
                logger.warning(f"Early stopping at epoch {epoch} due to high KL divergence: "
                             f"{policy_stats['kl_divergence']:.4f}")
                break

        # Update training statistics
        for key in epoch_stats:
            if epoch_stats[key]:
                self.training_stats[key].append(sum(epoch_stats[key]) / len(epoch_stats[key]))

        return epoch_stats

    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get accumulated training statistics."""
        return self.training_stats.copy()

    def reset_stats(self):
        """Reset training statistics."""
        for key in self.training_stats:
            self.training_stats[key] = []