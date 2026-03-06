"""
PPO Training Loop Implementation.

This module implements the main training orchestrator for PPO-based options trading
agents, including trajectory collection, policy updates, curriculum progression,
and performance evaluation with comprehensive monitoring.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import os
from pathlib import Path

from .algorithm import PPOAlgorithm
from .gae import GAECalculator
from .networks import ActorCritic
from .buffer import PPOBuffer
from ..curriculum import CurriculumScheduler, PerformanceMetrics
from ..metrics import MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration parameters for PPO training."""

    # Core PPO parameters
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training parameters
    n_steps: int = 2048  # steps per update
    n_epochs: int = 4    # optimization epochs per update
    batch_size: int = 64

    # Environment parameters
    n_envs: int = 8  # number of parallel environments
    gamma: float = 0.99  # discount factor
    gae_lambda: float = 0.95  # GAE lambda

    # Evaluation and checkpointing
    eval_freq: int = 10000
    checkpoint_freq: int = 5000
    log_freq: int = 100

    # Training control
    total_timesteps: int = 1000000
    early_stopping_patience: int = 50000
    target_performance: float = 0.3  # Sharpe ratio threshold

    # Device configuration
    device: str = 'cpu'

    # Advanced parameters
    normalize_advantages: bool = True
    clip_value_loss: bool = True
    use_adaptive_kl: bool = False
    target_kl: float = 0.01


class PPOTrainer:
    """
    Main PPO training orchestrator for options trading agents.

    Manages the complete training pipeline including trajectory collection,
    policy updates, curriculum progression, and performance evaluation.
    """

    def __init__(
        self,
        env_factory: Callable,
        policy_network: ActorCritic,
        curriculum_scheduler: CurriculumScheduler,
        config: PPOConfig,
        logger: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None,
        evaluator: Optional[Any] = None
    ):
        """
        Initialize PPO trainer.

        Args:
            env_factory: Function that creates environment instances
            policy_network: Actor-critic network
            curriculum_scheduler: Curriculum learning scheduler
            config: Training configuration
            logger: Training logger (PPOLogger)
            checkpoint_manager: Checkpoint management system
            evaluator: Agent evaluator for performance assessment
        """
        self.config = config
        self.env_factory = env_factory
        self.curriculum_scheduler = curriculum_scheduler
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.evaluator = evaluator

        # Set device
        self.device = torch.device(config.device)

        # Initialize policy network
        self.policy_network = policy_network.to(self.device)

        # Initialize PPO algorithm
        self.ppo_algorithm = PPOAlgorithm(
            actor_critic=policy_network,
            policy_lr=config.learning_rate,
            value_lr=config.learning_rate,
            clip_epsilon=config.clip_epsilon,
            entropy_coef=config.entropy_coef,
            value_loss_coef=config.value_loss_coef,
            max_grad_norm=config.max_grad_norm,
            device=config.device
        )

        # Initialize GAE calculator
        self.gae_calculator = GAECalculator(
            gamma=config.gamma,
            lambda_=config.gae_lambda
        )

        # Initialize environments
        self.envs = [env_factory() for _ in range(config.n_envs)]
        self.current_obs = [self._normalize_obs(env.reset()) for env in self.envs]

        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_performance = -float('inf')
        self.steps_without_improvement = 0

        # Performance tracking
        self.episode_returns = []
        self.episode_lengths = []
        self.performance_history = []

        # Metrics calculator
        self.metrics_calculator = MetricsCalculator()

        logging.getLogger(__name__).info(
            f"PPO Trainer initialized with {config.n_envs} environments"
        )
        logging.getLogger(__name__).info(
            f"Target timesteps: {config.total_timesteps:,}"
        )

    def _normalize_obs(self, obs: Any) -> np.ndarray:
        """
        Normalize observation shape to match policy network input.

        Args:
            obs: Raw observation from environment

        Returns:
            1D numpy array with expected observation dimension
        """
        obs_array = np.array(obs, dtype=np.float32).reshape(-1)
        target_dim = getattr(self.policy_network, 'obs_dim', obs_array.shape[0])

        if obs_array.shape[0] > target_dim:
            obs_array = obs_array[:target_dim]
        elif obs_array.shape[0] < target_dim:
            obs_array = np.pad(obs_array, (0, target_dim - obs_array.shape[0]))

        return obs_array

    def train(self, total_timesteps: Optional[int] = None, eval_freq: Optional[int] = None) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            total_timesteps: Total training timesteps (overrides config)
            eval_freq: Evaluation frequency (overrides config)

        Returns:
            Dict containing training statistics and final performance
        """
        total_timesteps = total_timesteps or self.config.total_timesteps
        eval_freq = eval_freq or self.config.eval_freq

        logging.getLogger(__name__).info(
            f"Starting PPO training for {total_timesteps:,} timesteps"
        )
        start_time = time.time()

        # Log hyperparameters
        if self.logger:
            self.logger.log_hyperparameters(self.config)

        while self.global_step < total_timesteps:
            # Check for early stopping
            if self.steps_without_improvement >= self.config.early_stopping_patience:
                logging.getLogger(__name__).info(
                    f"Early stopping triggered after {self.global_step} steps"
                )
                break

            # Collect trajectories
            buffer = self.collect_trajectories(self.config.n_steps)

            # Update global step count
            self.global_step += self.config.n_steps * self.config.n_envs

            # Update policy
            training_metrics = self.update_policy(buffer)

            # Log training metrics
            if self.logger and self.global_step % self.config.log_freq == 0:
                self.logger.log_training_metrics(self.global_step, training_metrics)

            # Evaluate performance
            if self.global_step % eval_freq == 0:
                eval_results = self.evaluate_performance()

                # Update curriculum based on performance
                curriculum_state = self.curriculum_scheduler.update_curriculum(
                    PerformanceMetrics(
                        sharpe_ratio=eval_results.get('sharpe_ratio', 0.0),
                        total_return=eval_results.get('total_return', 0.0),
                        max_drawdown=eval_results.get('max_drawdown', 0.0),
                        win_rate=eval_results.get('win_rate', 0.0),
                        episode_count=self.episode_count
                    )
                )

                # Log performance and curriculum metrics
                if self.logger:
                    self.logger.log_performance_metrics(self.global_step, eval_results)
                    self.logger.log_curriculum_metrics(self.global_step, curriculum_state)

                # Check for improvement
                current_performance = eval_results.get('sharpe_ratio', 0.0)
                if current_performance > self.best_performance:
                    self.best_performance = current_performance
                    self.steps_without_improvement = 0

                    # Save best model
                    if self.checkpoint_manager:
                        self.save_checkpoint(is_best=True)
                else:
                    self.steps_without_improvement += eval_freq

            # Save checkpoint
            if self.global_step % self.config.checkpoint_freq == 0:
                self.save_checkpoint()

        # Final evaluation
        final_results = self.evaluate_performance(n_episodes=50)

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final performance: Sharpe ratio = {final_results.get('sharpe_ratio', 0.0):.3f}")

        return {
            'global_step': self.global_step,
            'training_time': training_time,
            'best_performance': self.best_performance,
            'final_results': final_results,
            'steps_per_second': self.global_step / training_time
        }

    def collect_trajectories(self, n_steps: int) -> PPOBuffer:
        """
        Collect trajectories from parallel environments.

        Args:
            n_steps: Number of steps to collect per environment

        Returns:
            PPOBuffer containing collected trajectories
        """
        buffer = PPOBuffer(
            capacity=n_steps * self.config.n_envs,
            obs_dim=self.current_obs[0].shape[0],
            device=self.config.device
        )

        episode_returns = []
        episode_lengths = []

        for step in range(n_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(np.array(self.current_obs)).to(self.device)
                if hasattr(self.policy_network, 'get_action'):
                    actions, log_probs, values = self.policy_network.get_action(obs_tensor)
                else:
                    outputs = self.policy_network(obs_tensor)
                    if isinstance(outputs, (tuple, list)):
                        if len(outputs) >= 3:
                            actions, log_probs, values = outputs[:3]
                        else:
                            action_logits, values = outputs
                            dist = torch.distributions.Categorical(logits=action_logits)
                            actions = dist.sample()
                            log_probs = dist.log_prob(actions)
                    else:
                        dist = torch.distributions.Categorical(logits=outputs)
                        actions = dist.sample()
                        log_probs = dist.log_prob(actions)
                        values = torch.zeros(actions.shape, device=actions.device)

                actions_np = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions
                log_probs_np = log_probs.cpu().numpy()
                values_np = values.cpu().numpy().flatten()

                next_obs = []

                for i, (env, obs, action) in enumerate(zip(self.envs, self.current_obs, actions_np)):
                    next_ob, reward, done, info = env.step(action)
                    next_ob = self._normalize_obs(next_ob)

                    buffer.store_transition(
                        obs=torch.FloatTensor(obs),
                        action=int(action),
                        reward=float(reward),
                        value=float(values_np[i]),
                        log_prob=float(log_probs_np[i]),
                        done=done
                    )

                    if done:
                        if 'episode' in info:
                            episode_returns.append(info['episode']['r'])
                            episode_lengths.append(info['episode']['l'])
                        next_ob = self._normalize_obs(env.reset())
                        self.episode_count += 1

                    next_obs.append(next_ob)

                self.current_obs = next_obs

        buffer.compute_advantages_and_returns(self.gae_calculator)

        # Update episode statistics
        if episode_returns:
            self.episode_returns.extend(episode_returns)
            self.episode_lengths.extend(episode_lengths)

        logger.debug(f"Collected {buffer.size} transitions from {self.config.n_envs} environments")

        return buffer

    def update_policy(self, buffer: PPOBuffer) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.

        Args:
            buffer: Experience buffer with trajectories

        Returns:
            Dict containing training metrics
        """
        # Prepare data for training
        obs = torch.as_tensor(buffer.observations, device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(buffer.actions, device=self.device, dtype=torch.long)
        old_log_probs = torch.as_tensor(buffer.log_probs, device=self.device, dtype=torch.float32)
        values = torch.as_tensor(buffer.values, device=self.device, dtype=torch.float32)
        advantages = torch.as_tensor(buffer.advantages, device=self.device, dtype=torch.float32)
        returns = torch.as_tensor(buffer.returns, device=self.device, dtype=torch.float32)

        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update policy
        update_metrics = self.ppo_algorithm.update(
            obs=obs,
            actions=actions,
            old_log_probs=old_log_probs,
            values=values,
            advantages=advantages,
            returns=returns,
            n_epochs=self.config.n_epochs,
            batch_size=self.config.batch_size
        )

        # Add additional metrics
        update_metrics.update({
            'episode_count': self.episode_count,
            'mean_episode_return': np.mean(self.episode_returns[-100:]) if self.episode_returns else 0.0,
            'mean_episode_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0.0,
        })

        return update_metrics

    def evaluate_performance(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate current policy performance.

        Args:
            n_episodes: Number of episodes for evaluation

        Returns:
            Dict containing evaluation metrics
        """
        if self.evaluator:
            # Use dedicated evaluator
            results = self.evaluator.evaluate_agent(
                self.policy_network,
                deterministic=True
            )
            return {
                'total_return': results.total_return,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'win_rate': results.win_rate,
                'average_episode_length': results.average_episode_length
            }
        else:
            # Simple evaluation using training environment
            eval_env = self.env_factory()
            returns = []
            lengths = []

            for _ in range(n_episodes):
                obs = self._normalize_obs(eval_env.reset())
                episode_return = 0.0
                episode_length = 0
                done = False

                while not done:
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        if hasattr(self.policy_network, 'get_action'):
                            action, _, _ = self.policy_network.get_action(obs_tensor, deterministic=True)
                        else:
                            outputs = self.policy_network(obs_tensor)
                            if isinstance(outputs, (tuple, list)) and len(outputs) >= 1:
                                action = outputs[0]
                            else:
                                dist = torch.distributions.Categorical(logits=outputs)
                                action = dist.sample()

                        if isinstance(action, torch.Tensor):
                            action = action.cpu().numpy()[0]

                    obs, reward, done, _ = eval_env.step(action)
                    obs = self._normalize_obs(obs)
                    episode_return += reward
                    episode_length += 1

                returns.append(episode_return)
                lengths.append(episode_length)

            # Calculate basic metrics
            returns_array = np.array(returns)
            metrics = {
                'total_return': float(np.mean(returns)),
                'std_return': float(np.std(returns)),
                'max_return': float(np.max(returns)),
                'min_return': float(np.min(returns)),
                'average_episode_length': float(np.mean(lengths)),
                'win_rate': float(np.mean(returns_array > 0))
            }

            # Calculate Sharpe ratio if we have variance
            if len(returns) > 1 and np.std(returns) > 0:
                metrics['sharpe_ratio'] = float(np.mean(returns) / np.std(returns))
            else:
                metrics['sharpe_ratio'] = 0.0

            return metrics

    def save_checkpoint(self, filepath: Optional[str] = None, is_best: bool = False):
        """
        Save training checkpoint.

        Args:
            filepath: Custom filepath for checkpoint
            is_best: Whether this is the best performing model
        """
        if self.checkpoint_manager:
            trainer_state = {
                'step': self.global_step,
                'episode': self.episode_count,
                'model_state_dict': self.policy_network.state_dict(),
                'optimizer_state_dict': self.ppo_algorithm.policy_optimizer.state_dict(),
                'curriculum_state': self.curriculum_scheduler.get_state(),
                'performance_history': self.performance_history,
                'config': self.config,
                'best_performance': self.best_performance,
                'steps_without_improvement': self.steps_without_improvement
            }

            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                trainer_state=trainer_state,
                step=self.global_step,
                performance_metric=self.best_performance,
                is_best=is_best
            )

            logger.info(f"Checkpoint saved: {checkpoint_path}")
        elif filepath:
            # Simple checkpoint save
            checkpoint = {
                'step': self.global_step,
                'model_state_dict': self.policy_network.state_dict(),
                'optimizer_state_dict': self.ppo_algorithm.policy_optimizer.state_dict(),
                'config': self.config
            }
            torch.save(checkpoint, filepath)
            logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str) -> bool:
        """
        Load training checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            True if checkpoint loaded successfully
        """
        try:
            if self.checkpoint_manager:
                # Load using checkpoint manager
                if 'best' in filepath.lower():
                    trainer_state = self.checkpoint_manager.load_best_checkpoint()
                else:
                    trainer_state = self.checkpoint_manager.load_latest_checkpoint()

                if trainer_state is None:
                    return False

                # Restore training state
                self.global_step = trainer_state['step']
                self.episode_count = trainer_state['episode']
                self.policy_network.load_state_dict(trainer_state['model_state_dict'])
                self.ppo_algorithm.policy_optimizer.load_state_dict(trainer_state['optimizer_state_dict'])
                self.curriculum_scheduler.load_state(trainer_state['curriculum_state'])
                self.performance_history = trainer_state.get('performance_history', [])
                self.best_performance = trainer_state.get('best_performance', -float('inf'))
                self.steps_without_improvement = trainer_state.get('steps_without_improvement', 0)
            else:
                # Simple checkpoint load
                checkpoint = torch.load(filepath, map_location=self.device)
                self.global_step = checkpoint['step']
                self.policy_network.load_state_dict(checkpoint['model_state_dict'])
                self.ppo_algorithm.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logger.info(f"Checkpoint loaded successfully from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint from {filepath}: {e}")
            return False

    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state for monitoring."""
        curriculum_level = None
        if hasattr(self.curriculum_scheduler, 'current_level'):
            current_level = self.curriculum_scheduler.current_level
            curriculum_level = getattr(current_level, 'name', current_level)
        elif hasattr(self.curriculum_scheduler, 'get_state'):
            try:
                curriculum_level = self.curriculum_scheduler.get_state().get('level')
            except Exception:
                curriculum_level = None

        return {
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'best_performance': self.best_performance,
            'steps_without_improvement': self.steps_without_improvement,
            'curriculum_level': curriculum_level,
            'recent_episode_returns': self.episode_returns[-10:] if self.episode_returns else [],
            'recent_episode_lengths': self.episode_lengths[-10:] if self.episode_lengths else []
        }
