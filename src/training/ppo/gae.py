"""
Generalized Advantage Estimation (GAE) implementation.

This module implements GAE for computing advantage estimates and returns
for PPO training in options trading environments.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class GAECalculator:
    """
    Generalized Advantage Estimation calculator.

    Computes advantage estimates using GAE algorithm with configurable
    discount factor (gamma) and GAE parameter (lambda).
    """

    def __init__(
        self,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        device: str = 'cpu'
    ):
        """
        Initialize GAE calculator.

        Args:
            gamma: Discount factor for future rewards
            lambda_: GAE parameter for bias-variance tradeoff
            device: Device to run computations on
        """
        self.gamma = gamma
        self.lambda_ = lambda_
        self.device = device

        # Validate parameters
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"Gamma must be between 0 and 1, got {gamma}")
        if not (0.0 <= lambda_ <= 1.0):
            raise ValueError(f"Lambda must be between 0 and 1, got {lambda_}")

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Reward sequence [T]
            values: Value estimates [T]
            dones: Done flags [T]
            next_value: Value estimate for state after final state

        Returns:
            Tuple of (advantages, returns)
        """
        # Convert to tensors if needed
        if isinstance(rewards, (list, np.ndarray)):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        if isinstance(values, (list, np.ndarray)):
            values = torch.tensor(values, dtype=torch.float32, device=self.device)
        if isinstance(dones, (list, np.ndarray)):
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Ensure tensors are on correct device
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)

        # Length of trajectory
        T = len(rewards)

        # Initialize arrays for advantages and returns
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Append next_value to values for bootstrapping
        next_values = torch.cat([
            values[1:],
            torch.tensor([next_value], device=self.device)
        ])

        # Compute TD errors (delta)
        deltas = rewards + self.gamma * next_values * (1.0 - dones) - values

        # Compute GAE advantages using reverse iteration
        gae_advantage = 0.0
        for t in reversed(range(T)):
            gae_advantage = deltas[t] + self.gamma * self.lambda_ * (1.0 - dones[t]) * gae_advantage
            advantages[t] = gae_advantage

        # Compute returns as advantages + values
        returns = advantages + values

        return advantages, returns

    def compute_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float = 0.0
    ) -> torch.Tensor:
        """
        Compute discounted returns (targets for value function).

        Args:
            rewards: Reward sequence [T]
            values: Value estimates [T] (unused, kept for compatibility)
            dones: Done flags [T]
            next_value: Value estimate for state after final state

        Returns:
            Discounted returns
        """
        # Convert to tensors if needed
        if isinstance(rewards, (list, np.ndarray)):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        if isinstance(dones, (list, np.ndarray)):
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Ensure tensors are on correct device
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        T = len(rewards)
        returns = torch.zeros_like(rewards)

        # Compute returns using reverse iteration
        running_return = next_value
        for t in reversed(range(T)):
            running_return = rewards[t] + self.gamma * running_return * (1.0 - dones[t])
            returns[t] = running_return

        return returns

    def normalize_advantages(
        self,
        advantages: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Normalize advantages for stable training.

        Args:
            advantages: Advantage estimates to normalize
            eps: Small epsilon for numerical stability

        Returns:
            Normalized advantages with mean 0 and std 1
        """
        if len(advantages) == 0:
            return advantages

        mean_adv = advantages.mean()
        std_adv = advantages.std()

        # Avoid division by zero
        if std_adv < eps:
            logger.warning(f"Advantage standard deviation is very small: {std_adv}")
            std_adv = eps

        normalized_advantages = (advantages - mean_adv) / (std_adv + eps)

        return normalized_advantages

    def compute_gae_batched(
        self,
        rewards_list: List[torch.Tensor],
        values_list: List[torch.Tensor],
        dones_list: List[torch.Tensor],
        next_values: List[float]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute GAE for multiple episodes in batch.

        Args:
            rewards_list: List of reward sequences
            values_list: List of value estimate sequences
            dones_list: List of done flag sequences
            next_values: List of next values for each episode

        Returns:
            Tuple of (advantages_list, returns_list)
        """
        advantages_list = []
        returns_list = []

        for rewards, values, dones, next_value in zip(
            rewards_list, values_list, dones_list, next_values
        ):
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            advantages_list.append(advantages)
            returns_list.append(returns)

        return advantages_list, returns_list

    def compute_td_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float = 0.0
    ) -> torch.Tensor:
        """
        Compute TD(λ) returns as an alternative to GAE.

        This provides a different way to compute returns that can be useful
        for comparison with standard GAE.

        Args:
            rewards: Reward sequence [T]
            values: Value estimates [T]
            dones: Done flags [T]
            next_value: Value estimate for state after final state

        Returns:
            TD(λ) returns
        """
        # Convert to tensors if needed
        if isinstance(rewards, (list, np.ndarray)):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        if isinstance(values, (list, np.ndarray)):
            values = torch.tensor(values, dtype=torch.float32, device=self.device)
        if isinstance(dones, (list, np.ndarray)):
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Ensure tensors are on correct device
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)

        T = len(rewards)
        returns = torch.zeros_like(rewards)

        # Append next_value to values for bootstrapping
        next_values = torch.cat([
            values[1:],
            torch.tensor([next_value], device=self.device)
        ])

        # Compute λ-returns using forward view
        for t in range(T):
            # Initialize return with 1-step TD target
            n_step_return = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t])
            lambda_return = (1.0 - self.lambda_) * n_step_return

            # Add weighted n-step returns
            discount = self.gamma * self.lambda_
            for k in range(1, T - t):
                if t + k >= T:
                    break

                # Check if any episode ended before this step
                episode_continues = torch.prod(1.0 - dones[t:t+k])
                if episode_continues == 0:
                    break

                # Compute k+1 step return
                k_step_reward = 0.0
                gamma_pow = 1.0
                for j in range(k + 1):
                    if t + j < T:
                        k_step_reward += gamma_pow * rewards[t + j]
                        gamma_pow *= self.gamma

                # Add bootstrapped value
                bootstrap_value = next_values[t + k] if t + k < T else next_value
                n_step_return = k_step_reward + (self.gamma ** (k + 1)) * bootstrap_value * episode_continues

                # Weight by λ^k
                lambda_return += (1.0 - self.lambda_) * (discount ** k) * n_step_return

            # Add the infinite horizon term
            if t < T - 1:
                monte_carlo_return = 0.0
                gamma_pow = 1.0
                for j in range(t, T):
                    monte_carlo_return += gamma_pow * rewards[j]
                    gamma_pow *= self.gamma

                # Weight by λ^(T-t-1)
                if T - t - 1 > 0:
                    lambda_return += (discount ** (T - t - 1)) * monte_carlo_return

            returns[t] = lambda_return

        return returns

    def get_statistics(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> dict:
        """
        Get statistics about computed advantages and returns.

        Args:
            advantages: Computed advantages
            returns: Computed returns

        Returns:
            Dictionary of statistics
        """
        stats = {
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            'advantage_min': advantages.min().item(),
            'advantage_max': advantages.max().item(),
            'return_mean': returns.mean().item(),
            'return_std': returns.std().item(),
            'return_min': returns.min().item(),
            'return_max': returns.max().item(),
            'trajectory_length': len(advantages)
        }

        return stats