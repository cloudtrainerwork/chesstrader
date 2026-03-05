"""
Position Manager Training System.

Extends PPO training for position management with specialized configuration,
curriculum learning for position complexity, and trading-specific metrics.
Integrates PositionManagerNetwork with existing PPO infrastructure.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import logging
import time
import os
from pathlib import Path
try:
    import gym
except ImportError:
    gym = None

from .ppo.trainer import PPOConfig
from .ppo.algorithm import PPOAlgorithm
from .curriculum import CurriculumScheduler
from .metrics import MetricsCalculator
from ..models.position_manager import PositionManagerNetwork
from ..environments import make_env

logger = logging.getLogger(__name__)


@dataclass
class PositionManagerConfig(PPOConfig):
    """Configuration for position manager training with trading-specific parameters."""

    # Position management specific
    action_masking: bool = True  # Enable invalid action masking
    risk_penalty: float = 0.1   # Penalty for excessive risk-taking
    position_diversity: float = 0.05  # Reward for diverse position types

    # Curriculum parameters
    start_simple: bool = True  # Begin with simple single-leg positions
    complexity_threshold: float = 0.6  # Performance threshold for progression

    # Trading metrics
    track_sharpe: bool = True
    track_drawdown: bool = True
    track_win_rate: bool = True

    # Model architecture
    position_embed_dim: int = 64
    state_embed_dim: int = 128
    num_attention_heads: int = 4

    # Training schedule
    warmup_steps: int = 10000  # Steps before curriculum advancement
    eval_frequency: int = 5000  # Steps between evaluations


class PositionManagerTrainer:
    """
    Position Manager training system that extends PPO for options trading.

    Provides specialized training for position management with:
    - Action masking for invalid trades
    - Position complexity curriculum learning
    - Trading-specific performance metrics
    - Risk-adjusted reward shaping
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        config: Optional[PositionManagerConfig] = None,
        strategy_types: Optional[List[str]] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize position manager trainer.

        Args:
            env_factory: Factory function that creates training environments
            config: Training configuration parameters
            strategy_types: List of strategy types to train on
            checkpoint_dir: Directory for saving checkpoints
        """
        self.config = config or PositionManagerConfig()
        self.env_factory = env_factory
        self.strategy_types = strategy_types or ['IronCondor', 'BullCallSpread', 'LongStrangle']
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('./checkpoints/position_manager')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize environment to get dimensions
        self.env = env_factory()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Initialize position manager network
        self.network = PositionManagerNetwork(
            obs_dim=self.observation_space.shape[0],
            action_dim=self.action_space.n,
            position_embed_dim=self.config.position_embed_dim,
            state_embed_dim=self.config.state_embed_dim,
            num_heads=self.config.num_attention_heads
        )

        # Initialize PPO algorithm with position manager network
        self.ppo_algorithm = PPOAlgorithm(
            network=self.network,
            learning_rate=self.config.learning_rate,
            clip_epsilon=self.config.clip_epsilon,
            entropy_coef=self.config.entropy_coef,
            value_loss_coef=self.config.value_loss_coef,
            max_grad_norm=self.config.max_grad_norm
        )

        # Initialize curriculum scheduler for position complexity
        self.curriculum = CurriculumScheduler(
            stages=[
                {'name': 'simple', 'min_performance': 0.4, 'strategies': ['BullCallSpread']},
                {'name': 'intermediate', 'min_performance': 0.6, 'strategies': ['IronCondor', 'LongStrangle']},
                {'name': 'complex', 'min_performance': 0.75, 'strategies': self.strategy_types}
            ],
            warmup_steps=self.config.warmup_steps,
            performance_window=50
        )

        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            track_sharpe=self.config.track_sharpe,
            track_drawdown=self.config.track_drawdown,
            track_win_rate=self.config.track_win_rate
        )

        # Training state
        self.global_step = 0
        self.current_stage = 'simple'
        self.episode_rewards = []
        self.episode_lengths = []

        # Position-specific metrics
        self.position_metrics = {
            'success_rate': [],
            'avg_hold_time': [],
            'adjustment_frequency': [],
            'risk_adjusted_returns': [],
            'strategy_performance': {strategy: [] for strategy in self.strategy_types}
        }

        logger.info(f"Position Manager Trainer initialized with {len(self.strategy_types)} strategy types")

    def train(
        self,
        total_steps: int,
        eval_callback: Optional[Callable] = None,
        save_frequency: int = 50000,
        log_frequency: int = 1000
    ) -> Dict[str, List[float]]:
        """
        Train the position manager with curriculum learning.

        Args:
            total_steps: Total training steps
            eval_callback: Optional evaluation callback
            save_frequency: Steps between checkpoint saves
            log_frequency: Steps between metric logging

        Returns:
            Training metrics dictionary
        """
        logger.info(f"Starting position manager training for {total_steps} steps")

        # Initialize multiple environments for parallel collection
        envs = [self.env_factory() for _ in range(self.config.n_envs)]
        observations = [env.reset() for env in envs]

        training_metrics = {
            'rewards': [],
            'policy_loss': [],
            'value_loss': [],
            'curriculum_stage': [],
            'position_success_rate': []
        }

        while self.global_step < total_steps:
            # Collect trajectories
            trajectories = self._collect_trajectories(envs, observations)

            # Update policy with PPO
            policy_loss, value_loss, entropy = self.ppo_algorithm.update(trajectories)

            # Update curriculum based on performance
            current_performance = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0.0
            stage_changed = self.curriculum.update(self.global_step, current_performance)

            if stage_changed:
                self.current_stage = self.curriculum.current_stage
                logger.info(f"Curriculum advanced to stage: {self.current_stage}")
                # Switch to new strategy mix
                self._update_environments(envs)

            # Calculate position-specific metrics
            self._update_position_metrics(trajectories)

            # Log metrics
            if self.global_step % log_frequency == 0:
                self._log_training_progress(policy_loss, value_loss, entropy)

            # Save checkpoint
            if self.global_step % save_frequency == 0:
                self._save_checkpoint()

            # Run evaluation
            if eval_callback and self.global_step % self.config.eval_frequency == 0:
                eval_callback(self)

            # Update training metrics
            training_metrics['rewards'].append(np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0)
            training_metrics['policy_loss'].append(policy_loss)
            training_metrics['value_loss'].append(value_loss)
            training_metrics['curriculum_stage'].append(self.current_stage)
            training_metrics['position_success_rate'].append(self._get_position_success_rate())

            self.global_step += self.config.n_steps * self.config.n_envs

        logger.info("Training completed")
        return training_metrics

    def _collect_trajectories(self, envs: List[Any], observations: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        """Collect trajectories from parallel environments with action masking."""
        trajectories = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'action_masks': []
        }

        for step in range(self.config.n_steps):
            # Convert observations to tensors
            obs_batch = torch.FloatTensor(np.array(observations))

            # Get action masks for valid actions
            action_masks = self._get_action_masks(envs, observations)

            # Get actions and values from network
            with torch.no_grad():
                actions, log_probs, values = self.network.act(obs_batch, action_masks if self.config.action_masking else None)

            # Execute actions in environments
            next_observations = []
            rewards = []
            dones = []

            for i, (env, action) in enumerate(zip(envs, actions)):
                obs, reward, done, info = env.step(action.item())

                # Apply risk penalty if needed
                if 'risk_violation' in info and info['risk_violation']:
                    reward -= self.config.risk_penalty

                # Apply position diversity reward
                if 'position_diversity' in info:
                    reward += self.config.position_diversity * info['position_diversity']

                next_observations.append(obs)
                rewards.append(reward)
                dones.append(done)

                # Track episode metrics
                if done:
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_lengths.append(info['episode']['l'])
                    observations[i] = env.reset()
                else:
                    observations[i] = obs

            # Store trajectory data
            trajectories['observations'].append(obs_batch)
            trajectories['actions'].append(actions)
            trajectories['rewards'].append(torch.FloatTensor(rewards))
            trajectories['values'].append(values)
            trajectories['log_probs'].append(log_probs)
            trajectories['dones'].append(torch.BoolTensor(dones))
            trajectories['action_masks'].append(torch.FloatTensor(action_masks))

        # Convert lists to tensors
        for key in trajectories:
            if key == 'observations':
                trajectories[key] = torch.stack(trajectories[key])
            else:
                trajectories[key] = torch.stack(trajectories[key])

        return trajectories

    def _get_action_masks(self, envs: List[Any], observations: List[np.ndarray]) -> np.ndarray:
        """Get action masks for valid actions in current state."""
        action_masks = []

        for env, obs in zip(envs, observations):
            # Get valid actions from environment
            if hasattr(env, 'get_valid_actions'):
                valid_actions = env.get_valid_actions()
                mask = np.zeros(self.action_space.n)
                mask[valid_actions] = 1.0
            else:
                # If no masking available, allow all actions
                mask = np.ones(self.action_space.n)

            action_masks.append(mask)

        return np.array(action_masks)

    def _update_environments(self, envs: List[Any]):
        """Update environments to match current curriculum stage."""
        current_strategies = self.curriculum.get_current_strategies()

        for env in envs:
            if hasattr(env, 'set_strategy_mix'):
                env.set_strategy_mix(current_strategies)

    def _update_position_metrics(self, trajectories: Dict[str, torch.Tensor]):
        """Update position-specific performance metrics."""
        # Extract episode information for metrics calculation
        # This would need access to episode-level information from trajectories
        pass

    def _get_position_success_rate(self) -> float:
        """Calculate current position success rate."""
        if not self.position_metrics['success_rate']:
            return 0.0
        return np.mean(self.position_metrics['success_rate'][-50:])

    def _log_training_progress(self, policy_loss: float, value_loss: float, entropy: float):
        """Log training progress and metrics."""
        avg_reward = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0.0
        avg_length = np.mean(self.episode_lengths[-50:]) if self.episode_lengths else 0.0

        logger.info(f"Step {self.global_step}: "
                   f"Reward={avg_reward:.3f}, "
                   f"Length={avg_length:.1f}, "
                   f"PolicyLoss={policy_loss:.3f}, "
                   f"ValueLoss={value_loss:.3f}, "
                   f"Entropy={entropy:.3f}, "
                   f"Stage={self.current_stage}")

    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.ppo_algorithm.optimizer.state_dict(),
            'curriculum_state': self.curriculum.get_state(),
            'config': self.config,
            'metrics': self.position_metrics
        }

        checkpoint_path = self.checkpoint_dir / f'checkpoint_{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        self.global_step = checkpoint['global_step']
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.ppo_algorithm.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.curriculum.set_state(checkpoint['curriculum_state'])
        self.position_metrics = checkpoint.get('metrics', self.position_metrics)

        logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def get_position_manager(self) -> PositionManagerNetwork:
        """Get the trained position manager network."""
        return self.network

    def export_model(self, export_path: str):
        """Export trained model for deployment."""
        model_data = {
            'network_state_dict': self.network.state_dict(),
            'config': self.config,
            'final_metrics': self.position_metrics,
            'training_steps': self.global_step
        }

        torch.save(model_data, export_path)
        logger.info(f"Model exported to: {export_path}")


def create_position_manager_trainer(
    strategy: str,
    config: Optional[PositionManagerConfig] = None,
    **kwargs
) -> PositionManagerTrainer:
    """
    Factory function to create position manager trainer for specific strategy.

    Args:
        strategy: Primary strategy type to train on
        config: Training configuration
        **kwargs: Additional arguments for trainer

    Returns:
        Configured PositionManagerTrainer instance
    """
    def env_factory():
        return make_env(strategy_type=strategy, **kwargs)

    return PositionManagerTrainer(
        env_factory=env_factory,
        config=config,
        strategy_types=[strategy],
        **kwargs
    )