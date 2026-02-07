"""
Test PPO Training Loop Implementation.

Tests for the main training orchestrator including trajectory collection,
policy updates, curriculum progression, and performance evaluation.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from src.training.ppo.trainer import PPOTrainer, PPOConfig
from src.training.ppo.networks import ActorCritic
from src.training.ppo.algorithm import PPOAlgorithm
from src.training.ppo.buffer import PPOBuffer
from src.training.curriculum.scheduler import CurriculumScheduler, PerformanceMetrics


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self, obs_dim=10, action_dim=3):
        self.observation_space = Mock()
        self.observation_space.shape = (obs_dim,)
        self.action_space = Mock()
        self.action_space.n = action_dim

        self.step_count = 0
        self.max_steps = 100

    def reset(self):
        self.step_count = 0
        return np.random.randn(10)

    def step(self, action):
        self.step_count += 1
        next_obs = np.random.randn(10)
        reward = np.random.randn() * 0.1
        done = self.step_count >= self.max_steps
        info = {}

        if done:
            info['episode'] = {'r': sum([np.random.randn() * 0.1 for _ in range(self.step_count)]),
                              'l': self.step_count}

        return next_obs, reward, done, info


class MockActorCritic:
    """Mock actor-critic network for testing."""

    def __init__(self, obs_dim=10, action_dim=3):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def __call__(self, obs):
        batch_size = obs.shape[0]
        actions = torch.randint(0, self.action_dim, (batch_size,))
        log_probs = torch.randn(batch_size)
        values = torch.randn(batch_size, 1)
        aux_outputs = {}

        return actions, log_probs, values, aux_outputs

    def to(self, device):
        return self

    def state_dict(self):
        return {'param': torch.randn(10, 10)}

    def load_state_dict(self, state_dict):
        pass


class TestPPOConfig(unittest.TestCase):
    """Test PPO configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PPOConfig()

        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.clip_epsilon, 0.2)
        self.assertEqual(config.entropy_coef, 0.01)
        self.assertEqual(config.value_loss_coef, 0.5)
        self.assertEqual(config.max_grad_norm, 0.5)
        self.assertEqual(config.n_steps, 2048)
        self.assertEqual(config.n_epochs, 4)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.n_envs, 8)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.gae_lambda, 0.95)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PPOConfig(
            learning_rate=1e-3,
            clip_epsilon=0.1,
            n_envs=4,
            total_timesteps=500000
        )

        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.clip_epsilon, 0.1)
        self.assertEqual(config.n_envs, 4)
        self.assertEqual(config.total_timesteps, 500000)


class TestPPOTrainer(unittest.TestCase):
    """Test PPO trainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock components
        self.env_factory = lambda: MockEnvironment()
        self.policy_network = MockActorCritic()
        self.curriculum_scheduler = Mock(spec=CurriculumScheduler)
        self.curriculum_scheduler.update_curriculum.return_value = {'level': 1}
        self.curriculum_scheduler.get_state.return_value = {'level': 1}
        self.curriculum_scheduler.load_state.return_value = None

        # Create config with minimal parameters for fast testing
        self.config = PPOConfig(
            n_envs=2,
            n_steps=10,
            n_epochs=1,
            batch_size=5,
            total_timesteps=100,
            eval_freq=50,
            checkpoint_freq=50,
            log_freq=10
        )

        # Mock logger
        self.logger = Mock()
        self.logger.log_hyperparameters.return_value = None
        self.logger.log_training_metrics.return_value = None
        self.logger.log_performance_metrics.return_value = None
        self.logger.log_curriculum_metrics.return_value = None

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = PPOTrainer(
            env_factory=self.env_factory,
            policy_network=self.policy_network,
            curriculum_scheduler=self.curriculum_scheduler,
            config=self.config,
            logger=self.logger
        )

        self.assertIsInstance(trainer.config, PPOConfig)
        self.assertIsInstance(trainer.ppo_algorithm, PPOAlgorithm)
        self.assertEqual(len(trainer.envs), self.config.n_envs)
        self.assertEqual(len(trainer.current_obs), self.config.n_envs)
        self.assertEqual(trainer.global_step, 0)
        self.assertEqual(trainer.episode_count, 0)

    def test_collect_trajectories(self):
        """Test trajectory collection."""
        trainer = PPOTrainer(
            env_factory=self.env_factory,
            policy_network=self.policy_network,
            curriculum_scheduler=self.curriculum_scheduler,
            config=self.config,
            logger=self.logger
        )

        buffer = trainer.collect_trajectories(n_steps=5)

        self.assertIsInstance(buffer, PPOBuffer)
        expected_size = 5 * self.config.n_envs
        self.assertEqual(len(buffer.observations), expected_size)
        self.assertEqual(len(buffer.actions), expected_size)
        self.assertEqual(len(buffer.rewards), expected_size)

    @patch('src.training.ppo.trainer.PPOAlgorithm')
    def test_update_policy(self, mock_ppo_algorithm):
        """Test policy update."""
        # Mock the algorithm's update method
        mock_algorithm_instance = Mock()
        mock_algorithm_instance.update.return_value = {
            'policy_loss': 0.1,
            'value_loss': 0.05,
            'entropy_loss': 0.02,
            'total_loss': 0.17
        }
        mock_ppo_algorithm.return_value = mock_algorithm_instance

        trainer = PPOTrainer(
            env_factory=self.env_factory,
            policy_network=self.policy_network,
            curriculum_scheduler=self.curriculum_scheduler,
            config=self.config,
            logger=self.logger
        )

        # Create mock buffer
        buffer = Mock(spec=PPOBuffer)
        buffer.observations = np.random.randn(20, 10)
        buffer.actions = np.random.randint(0, 3, (20,))
        buffer.log_probs = np.random.randn(20)
        buffer.values = np.random.randn(20)
        buffer.advantages = np.random.randn(20)
        buffer.returns = np.random.randn(20)

        metrics = trainer.update_policy(buffer)

        self.assertIsInstance(metrics, dict)
        self.assertIn('policy_loss', metrics)
        self.assertIn('episode_count', metrics)
        mock_algorithm_instance.update.assert_called_once()

    def test_evaluate_performance(self):
        """Test performance evaluation."""
        trainer = PPOTrainer(
            env_factory=self.env_factory,
            policy_network=self.policy_network,
            curriculum_scheduler=self.curriculum_scheduler,
            config=self.config,
            logger=self.logger
        )

        metrics = trainer.evaluate_performance(n_episodes=2)

        self.assertIsInstance(metrics, dict)
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('win_rate', metrics)

    def test_save_and_load_checkpoint(self):
        """Test checkpoint saving and loading."""
        trainer = PPOTrainer(
            env_factory=self.env_factory,
            policy_network=self.policy_network,
            curriculum_scheduler=self.curriculum_scheduler,
            config=self.config,
            logger=self.logger
        )

        # Test simple checkpoint save/load
        checkpoint_path = Path(self.temp_dir) / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        self.assertTrue(checkpoint_path.exists())

        # Test loading
        loaded = trainer.load_checkpoint(str(checkpoint_path))
        self.assertTrue(loaded)

    def test_training_loop(self):
        """Test complete training loop."""
        # Use very small parameters for fast test
        config = PPOConfig(
            n_envs=1,
            n_steps=5,
            n_epochs=1,
            batch_size=2,
            total_timesteps=20,
            eval_freq=10,
            checkpoint_freq=100,  # No checkpointing in this test
            log_freq=5
        )

        trainer = PPOTrainer(
            env_factory=self.env_factory,
            policy_network=self.policy_network,
            curriculum_scheduler=self.curriculum_scheduler,
            config=config,
            logger=self.logger
        )

        results = trainer.train()

        self.assertIsInstance(results, dict)
        self.assertIn('global_step', results)
        self.assertIn('training_time', results)
        self.assertIn('final_results', results)
        self.assertGreater(results['global_step'], 0)
        self.assertGreater(results['training_time'], 0)

        # Verify logger was called
        self.logger.log_hyperparameters.assert_called()

    def test_early_stopping(self):
        """Test early stopping functionality."""
        config = PPOConfig(
            n_envs=1,
            n_steps=5,
            n_epochs=1,
            batch_size=2,
            total_timesteps=1000000,  # Very large
            eval_freq=10,
            early_stopping_patience=10,  # Very small patience
            log_freq=5
        )

        trainer = PPOTrainer(
            env_factory=self.env_factory,
            policy_network=self.policy_network,
            curriculum_scheduler=self.curriculum_scheduler,
            config=config,
            logger=self.logger
        )

        # Mock evaluator to return constant performance
        trainer.evaluator = Mock()
        trainer.evaluator.evaluate_agent.return_value = Mock(
            sharpe_ratio=0.0,  # Constant performance
            total_return=0.0,
            max_drawdown=0.0,
            win_rate=0.5,
            average_episode_length=50.0
        )

        results = trainer.train()

        # Should stop early due to no improvement
        self.assertLess(results['global_step'], config.total_timesteps)

    def test_get_training_state(self):
        """Test training state retrieval."""
        trainer = PPOTrainer(
            env_factory=self.env_factory,
            policy_network=self.policy_network,
            curriculum_scheduler=self.curriculum_scheduler,
            config=self.config,
            logger=self.logger
        )

        state = trainer.get_training_state()

        self.assertIsInstance(state, dict)
        self.assertIn('global_step', state)
        self.assertIn('episode_count', state)
        self.assertIn('best_performance', state)
        self.assertIn('curriculum_level', state)

    def test_with_evaluator(self):
        """Test trainer with evaluator."""
        evaluator = Mock()
        evaluation_results = Mock()
        evaluation_results.total_return = 0.1
        evaluation_results.sharpe_ratio = 0.5
        evaluation_results.max_drawdown = 0.05
        evaluation_results.win_rate = 0.6
        evaluation_results.average_episode_length = 50.0

        evaluator.evaluate_agent.return_value = evaluation_results

        trainer = PPOTrainer(
            env_factory=self.env_factory,
            policy_network=self.policy_network,
            curriculum_scheduler=self.curriculum_scheduler,
            config=self.config,
            logger=self.logger,
            evaluator=evaluator
        )

        metrics = trainer.evaluate_performance()

        self.assertEqual(metrics['total_return'], 0.1)
        self.assertEqual(metrics['sharpe_ratio'], 0.5)
        evaluator.evaluate_agent.assert_called_once()

    def test_curriculum_integration(self):
        """Test curriculum learning integration."""
        trainer = PPOTrainer(
            env_factory=self.env_factory,
            policy_network=self.policy_network,
            curriculum_scheduler=self.curriculum_scheduler,
            config=self.config,
            logger=self.logger
        )

        # Mock evaluation results
        eval_results = {
            'sharpe_ratio': 0.5,
            'total_return': 0.1,
            'max_drawdown': 0.05,
            'win_rate': 0.6
        }

        with patch.object(trainer, 'evaluate_performance', return_value=eval_results):
            # Run a few training steps to trigger curriculum update
            config = PPOConfig(
                n_envs=1,
                n_steps=5,
                n_epochs=1,
                batch_size=2,
                total_timesteps=30,
                eval_freq=10,
                log_freq=5
            )

            trainer.config = config
            results = trainer.train()

            # Verify curriculum scheduler was called
            self.curriculum_scheduler.update_curriculum.assert_called()

            # Verify PerformanceMetrics was created correctly
            call_args = self.curriculum_scheduler.update_curriculum.call_args[0][0]
            self.assertIsInstance(call_args, PerformanceMetrics)


class TestTrainerIntegration(unittest.TestCase):
    """Integration tests for the PPO trainer."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_training(self):
        """Test complete end-to-end training pipeline."""
        # Create real components for integration test
        from src.training.ppo.networks import ActorCritic
        from src.training.curriculum.scheduler import CurriculumScheduler
        from src.training.curriculum.levels import DifficultyLevel, CurriculumLevel

        # Simple environment factory
        env_factory = lambda: MockEnvironment(obs_dim=8, action_dim=4)

        # Real actor-critic network
        policy_network = ActorCritic(
            obs_dim=8,
            action_dim=4,
            hidden_dim=16,
            device='cpu'
        )

        # Real curriculum scheduler
        levels = [
            CurriculumLevel(
                level_id=0,
                difficulty_level=DifficultyLevel.BEGINNER,
                performance_threshold=0.0,
                parameters={}
            )
        ]
        curriculum_scheduler = CurriculumScheduler(levels)

        # Training config
        config = PPOConfig(
            n_envs=2,
            n_steps=8,
            n_epochs=1,
            batch_size=4,
            total_timesteps=50,
            eval_freq=20,
            checkpoint_freq=100,
            log_freq=10,
            device='cpu'
        )

        # Create trainer
        trainer = PPOTrainer(
            env_factory=env_factory,
            policy_network=policy_network,
            curriculum_scheduler=curriculum_scheduler,
            config=config
        )

        # Run training
        results = trainer.train()

        # Verify results
        self.assertIsInstance(results, dict)
        self.assertGreater(results['global_step'], 0)
        self.assertIn('final_results', results)

    def test_training_with_all_components(self):
        """Test training with all components (logger, checkpoint manager, evaluator)."""
        from src.training.ppo.logging import PPOLogger
        from src.training.ppo.checkpoints import CheckpointManager
        from src.training.ppo.evaluation import AgentEvaluator

        # Mock all components to avoid external dependencies
        logger = Mock(spec=PPOLogger)
        checkpoint_manager = Mock(spec=CheckpointManager)
        evaluator = Mock(spec=AgentEvaluator)

        # Mock evaluator returns
        eval_results = Mock()
        eval_results.total_return = 0.05
        eval_results.sharpe_ratio = 0.3
        eval_results.max_drawdown = 0.02
        eval_results.win_rate = 0.55
        eval_results.average_episode_length = 45.0
        evaluator.evaluate_agent.return_value = eval_results

        # Simple components
        env_factory = lambda: MockEnvironment()
        policy_network = MockActorCritic()
        curriculum_scheduler = Mock()
        curriculum_scheduler.update_curriculum.return_value = {'level': 1}
        curriculum_scheduler.get_state.return_value = {'level': 1}

        config = PPOConfig(
            n_envs=1,
            n_steps=5,
            n_epochs=1,
            batch_size=2,
            total_timesteps=20,
            eval_freq=10,
            log_freq=5
        )

        # Create trainer with all components
        trainer = PPOTrainer(
            env_factory=env_factory,
            policy_network=policy_network,
            curriculum_scheduler=curriculum_scheduler,
            config=config,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            evaluator=evaluator
        )

        # Run training
        results = trainer.train()

        # Verify all components were used
        logger.log_hyperparameters.assert_called()
        evaluator.evaluate_agent.assert_called()
        self.assertGreater(results['global_step'], 0)


if __name__ == '__main__':
    unittest.main()