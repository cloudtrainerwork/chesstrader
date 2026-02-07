"""
Experience buffer for PPO trajectory collection and batching.

This module implements an efficient experience buffer for storing trajectories
and generating mini-batches for PPO training in options trading environments.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Iterator
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class PPOBuffer:
    """
    Experience buffer for PPO training.

    Efficiently stores trajectories and provides mini-batch sampling
    for policy updates with proper handling of variable-length episodes.
    """

    def __init__(
        self,
        capacity: int = 10000,
        obs_dim: int = 35,
        device: str = 'cpu'
    ):
        """
        Initialize PPO buffer.

        Args:
            capacity: Maximum number of transitions to store
            obs_dim: Observation dimension
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.device = device

        # Storage for transitions
        self.observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)

        # GAE-related storage
        self.advantages = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.returns = torch.zeros(capacity, dtype=torch.float32, device=device)

        # Buffer management
        self.ptr = 0  # Current position in buffer
        self.size = 0  # Current buffer size
        self.episode_starts = []  # Track episode boundaries

        # Episode tracking
        self.current_episode_start = 0
        self.episode_lengths = []

    def store_transition(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """
        Store a single transition in the buffer.

        Args:
            obs: Observation [obs_dim]
            action: Action taken
            reward: Reward received
            value: Value estimate for observation
            log_prob: Log probability of action
            done: Whether episode ended
        """
        if self.ptr >= self.capacity:
            logger.warning("Buffer capacity exceeded, overwriting old data")
            self.ptr = 0

        # Store transition
        self.observations[self.ptr] = obs.to(self.device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)

        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

        # Track episode boundaries
        if done:
            self.finish_episode()

    def finish_episode(self, next_value: float = 0.0):
        """
        Process episode-end data and compute advantages/returns.

        Args:
            next_value: Value estimate for next state (bootstrap value)
        """
        if self.ptr == 0:
            return

        # Get episode data
        episode_end = self.ptr
        episode_start = self.current_episode_start

        if episode_end <= episode_start:
            return

        episode_length = episode_end - episode_start
        self.episode_lengths.append(episode_length)

        # Track episode boundaries
        self.episode_starts.append((episode_start, episode_end))
        self.current_episode_start = episode_end

        logger.debug(f"Finished episode of length {episode_length}")

    def compute_advantages_and_returns(
        self,
        gae_calculator,
        last_values: Optional[torch.Tensor] = None
    ):
        """
        Compute advantages and returns for all complete episodes using GAE.

        Args:
            gae_calculator: GAE calculator instance
            last_values: Values for last states of each episode
        """
        if not self.episode_starts:
            logger.warning("No complete episodes to process")
            return

        for i, (start, end) in enumerate(self.episode_starts):
            if end > self.size:
                continue

            # Extract episode data
            episode_rewards = self.rewards[start:end]
            episode_values = self.values[start:end]
            episode_dones = self.dones[start:end]

            # Get next value for bootstrapping
            if last_values is not None and i < len(last_values):
                next_value = last_values[i].item()
            else:
                next_value = 0.0

            # Compute GAE
            advantages, returns = gae_calculator.compute_gae(
                episode_rewards,
                episode_values,
                episode_dones,
                next_value
            )

            # Store computed values
            self.advantages[start:end] = advantages
            self.returns[start:end] = returns

        logger.debug(f"Computed advantages and returns for {len(self.episode_starts)} episodes")

    def get_batches(
        self,
        batch_size: int = 64,
        shuffle: bool = True
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Generate mini-batches for training.

        Args:
            batch_size: Size of each mini-batch
            shuffle: Whether to shuffle data

        Yields:
            Dictionary containing batch data
        """
        if self.size == 0:
            logger.warning("Buffer is empty, no batches to generate")
            return

        # Use only complete episodes for training
        if not self.episode_starts:
            logger.warning("No complete episodes, no batches to generate")
            return

        # Get indices of all transitions from complete episodes
        indices = []
        for start, end in self.episode_starts:
            if end <= self.size:
                indices.extend(range(start, end))

        if not indices:
            logger.warning("No valid transitions for batching")
            return

        indices = torch.tensor(indices, dtype=torch.long, device=self.device)

        # Shuffle if requested
        if shuffle:
            perm = torch.randperm(len(indices), device=self.device)
            indices = indices[perm]

        # Generate batches
        n_batches = len(indices) // batch_size
        if n_batches == 0:
            logger.warning(f"Not enough data for batch size {batch_size}")
            return

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_indices = indices[start_idx:end_idx]

            # Create batch
            batch = {
                'observations': self.observations[batch_indices],
                'actions': self.actions[batch_indices],
                'rewards': self.rewards[batch_indices],
                'values': self.values[batch_indices],
                'log_probs': self.log_probs[batch_indices],
                'dones': self.dones[batch_indices],
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices]
            }

            yield batch

    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """
        Get all stored data as a single batch.

        Returns:
            Dictionary containing all buffer data
        """
        # Get indices of all transitions from complete episodes
        indices = []
        for start, end in self.episode_starts:
            if end <= self.size:
                indices.extend(range(start, end))

        if not indices:
            logger.warning("No complete episodes to return")
            return {}

        indices = torch.tensor(indices, dtype=torch.long, device=self.device)

        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices],
            'dones': self.dones[indices],
            'advantages': self.advantages[indices],
            'returns': self.returns[indices]
        }

    def clear(self):
        """Reset buffer for next collection phase."""
        self.ptr = 0
        self.size = 0
        self.episode_starts = []
        self.current_episode_start = 0
        self.episode_lengths = []

        # Reset tensors
        self.observations.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.log_probs.zero_()
        self.dones.zero_()
        self.advantages.zero_()
        self.returns.zero_()

        logger.debug("Buffer cleared")

    def get_statistics(self) -> Dict[str, float]:
        """
        Get buffer statistics.

        Returns:
            Dictionary of buffer statistics
        """
        if self.size == 0:
            return {'size': 0, 'episodes': 0}

        # Get valid data indices
        indices = []
        for start, end in self.episode_starts:
            if end <= self.size:
                indices.extend(range(start, end))

        if not indices:
            return {'size': self.size, 'episodes': 0}

        indices = torch.tensor(indices, dtype=torch.long, device=self.device)

        stats = {
            'size': self.size,
            'episodes': len(self.episode_starts),
            'average_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'total_reward': self.rewards[indices].sum().item(),
            'average_reward': self.rewards[indices].mean().item(),
            'average_value': self.values[indices].mean().item(),
            'reward_std': self.rewards[indices].std().item(),
            'value_std': self.values[indices].std().item()
        }

        # Add advantage/return stats if computed
        if torch.any(self.advantages[indices] != 0):
            stats.update({
                'average_advantage': self.advantages[indices].mean().item(),
                'advantage_std': self.advantages[indices].std().item(),
                'average_return': self.returns[indices].mean().item(),
                'return_std': self.returns[indices].std().item()
            })

        return stats

    def save_buffer(self, filepath: str):
        """Save buffer contents to file."""
        data = {
            'observations': self.observations[:self.size].cpu(),
            'actions': self.actions[:self.size].cpu(),
            'rewards': self.rewards[:self.size].cpu(),
            'values': self.values[:self.size].cpu(),
            'log_probs': self.log_probs[:self.size].cpu(),
            'dones': self.dones[:self.size].cpu(),
            'advantages': self.advantages[:self.size].cpu(),
            'returns': self.returns[:self.size].cpu(),
            'episode_starts': self.episode_starts,
            'episode_lengths': self.episode_lengths,
            'size': self.size,
            'ptr': self.ptr
        }
        torch.save(data, filepath)
        logger.info(f"Buffer saved to {filepath}")

    def load_buffer(self, filepath: str):
        """Load buffer contents from file."""
        data = torch.load(filepath, map_location=self.device)

        size = data['size']
        if size > self.capacity:
            logger.warning(f"Loaded buffer size {size} exceeds capacity {self.capacity}")
            size = self.capacity

        self.observations[:size] = data['observations'][:size].to(self.device)
        self.actions[:size] = data['actions'][:size].to(self.device)
        self.rewards[:size] = data['rewards'][:size].to(self.device)
        self.values[:size] = data['values'][:size].to(self.device)
        self.log_probs[:size] = data['log_probs'][:size].to(self.device)
        self.dones[:size] = data['dones'][:size].to(self.device)
        self.advantages[:size] = data['advantages'][:size].to(self.device)
        self.returns[:size] = data['returns'][:size].to(self.device)

        self.episode_starts = data['episode_starts']
        self.episode_lengths = data['episode_lengths']
        self.size = size
        self.ptr = min(data['ptr'], self.capacity)

        logger.info(f"Buffer loaded from {filepath}")

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size >= self.capacity

    def is_ready_for_update(self, min_episodes: int = 1) -> bool:
        """
        Check if buffer has enough data for policy update.

        Args:
            min_episodes: Minimum number of complete episodes required

        Returns:
            True if ready for update
        """
        return len(self.episode_starts) >= min_episodes