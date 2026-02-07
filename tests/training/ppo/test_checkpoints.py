"""
Test PPO Checkpoint System.

Tests for comprehensive checkpoint management including saving, loading,
state recovery, and checkpoint cleanup functionality.
"""

import unittest
import tempfile
import shutil
import torch
import json
import time
import os
from pathlib import Path
from unittest.mock import Mock, patch

from src.training.ppo.checkpoints import (
    CheckpointManager,
    TrainerState,
    CheckpointInfo
)
from src.training.ppo.trainer import PPOConfig


class TestTrainerState(unittest.TestCase):
    """Test TrainerState dataclass."""

    def test_trainer_state_creation(self):
        """Test creating a TrainerState."""
        config = PPOConfig()
        model_state = {'layer1.weight': torch.randn(10, 10)}
        optimizer_state = {'state': {}, 'param_groups': []}

        state = TrainerState(
            step=1000,
            episode=50,
            model_state_dict=model_state,
            optimizer_state_dict=optimizer_state,
            curriculum_state={'level': 1},
            performance_history=[0.1, 0.2, 0.3],
            config=config,
            best_performance=0.5,
            steps_without_improvement=100
        )

        self.assertEqual(state.step, 1000)
        self.assertEqual(state.episode, 50)
        self.assertEqual(state.best_performance, 0.5)
        self.assertIsInstance(state.timestamp, float)

    def test_trainer_state_to_dict(self):
        """Test converting TrainerState to dictionary."""
        config = PPOConfig()
        model_state = {'param': torch.randn(5, 5)}

        state = TrainerState(
            step=100,
            episode=10,
            model_state_dict=model_state,
            optimizer_state_dict={},
            curriculum_state={},
            performance_history=[],
            config=config
        )

        state_dict = state.to_dict()

        self.assertIsInstance(state_dict, dict)
        self.assertEqual(state_dict['step'], 100)
        self.assertEqual(state_dict['episode'], 10)
        self.assertIn('config', state_dict)


class TestCheckpointInfo(unittest.TestCase):
    """Test CheckpointInfo functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = Path(self.temp_dir) / 'test_checkpoint.pt'

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_checkpoint_info_creation(self):
        """Test creating CheckpointInfo."""
        info = CheckpointInfo(
            filepath='/path/to/checkpoint.pt',
            step=1000,
            episode=50,
            performance_metric=0.75,
            timestamp=time.time(),
            file_size=1024,
            checksum='abcd1234',
            is_best=True
        )

        self.assertEqual(info.step, 1000)
        self.assertEqual(info.episode, 50)
        self.assertEqual(info.performance_metric, 0.75)
        self.assertTrue(info.is_best)

    def test_checkpoint_info_from_file(self):
        """Test creating CheckpointInfo from file."""
        # Create a test checkpoint file
        checkpoint_data = {
            'step': 500,
            'episode': 25,
            'best_performance': 0.6,
            'timestamp': time.time(),
            'model_state_dict': {'param': torch.randn(3, 3)}
        }
        torch.save(checkpoint_data, self.temp_file)

        info = CheckpointInfo.from_file(str(self.temp_file))

        self.assertIsNotNone(info)
        self.assertEqual(info.step, 500)
        self.assertEqual(info.episode, 25)
        self.assertEqual(info.performance_metric, 0.6)
        self.assertGreater(info.file_size, 0)
        self.assertIsInstance(info.checksum, str)

    def test_checkpoint_info_from_nonexistent_file(self):
        """Test creating CheckpointInfo from nonexistent file."""
        info = CheckpointInfo.from_file('/nonexistent/file.pt')
        self.assertIsNone(info)

    def test_checkpoint_info_from_corrupted_file(self):
        """Test creating CheckpointInfo from corrupted file."""
        # Create a corrupted file
        with open(self.temp_file, 'wb') as f:
            f.write(b'corrupted data')

        info = CheckpointInfo.from_file(str(self.temp_file))

        # Should still create info but with default values
        self.assertIsNotNone(info)
        self.assertEqual(info.step, 0)
        self.assertEqual(info.episode, 0)


class TestCheckpointManager(unittest.TestCase):
    """Test CheckpointManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / 'checkpoints'

        self.manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            max_checkpoints=3,
            save_best_only=False,
            auto_cleanup=True
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_manager_initialization(self):
        """Test checkpoint manager initialization."""
        self.assertEqual(self.manager.max_checkpoints, 3)
        self.assertFalse(self.manager.save_best_only)
        self.assertTrue(self.manager.auto_cleanup)

        # Check directory structure
        self.assertTrue(self.manager.checkpoint_dir.exists())
        self.assertTrue(self.manager.models_dir.exists())
        self.assertTrue(self.manager.best_dir.exists())
        self.assertTrue(self.manager.exports_dir.exists())

    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        trainer_state = {
            'step': 1000,
            'episode': 50,
            'model_state_dict': {'param': torch.randn(5, 5)},
            'optimizer_state_dict': {'state': {}, 'param_groups': []},
            'curriculum_state': {'level': 1},
            'performance_history': [0.1, 0.2, 0.3],
            'config': PPOConfig(),
            'best_performance': 0.5
        }

        filepath = self.manager.save_checkpoint(
            trainer_state=trainer_state,
            step=1000,
            performance_metric=0.5
        )

        self.assertTrue(Path(filepath).exists())
        self.assertEqual(len(self.manager.checkpoint_info), 1)

        # Verify checkpoint content
        checkpoint = torch.load(filepath)
        self.assertEqual(checkpoint['step'], 1000)
        self.assertIn('checkpoint_metadata', checkpoint)

    def test_save_best_checkpoint(self):
        """Test saving best checkpoint."""
        trainer_state = {
            'step': 2000,
            'episode': 100,
            'model_state_dict': {'param': torch.randn(5, 5)},
            'optimizer_state_dict': {},
            'curriculum_state': {},
            'performance_history': [],
            'config': PPOConfig()
        }

        filepath = self.manager.save_checkpoint(
            trainer_state=trainer_state,
            step=2000,
            performance_metric=0.8,
            is_best=True
        )

        self.assertTrue(Path(filepath).exists())
        self.assertIsNotNone(self.manager.best_checkpoint_info)
        self.assertTrue('best' in Path(filepath).name)

    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        # First save a checkpoint
        trainer_state = {
            'step': 1500,
            'episode': 75,
            'model_state_dict': {'param': torch.randn(3, 3)},
            'optimizer_state_dict': {},
            'curriculum_state': {},
            'performance_history': [],
            'config': PPOConfig()
        }

        saved_path = self.manager.save_checkpoint(
            trainer_state=trainer_state,
            step=1500,
            performance_metric=0.6
        )

        # Load the checkpoint
        loaded_state = self.manager.load_checkpoint(saved_path)

        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state['step'], 1500)
        self.assertEqual(loaded_state['episode'], 75)

    def test_load_nonexistent_checkpoint(self):
        """Test loading nonexistent checkpoint."""
        result = self.manager.load_checkpoint('/nonexistent/checkpoint.pt')
        self.assertIsNone(result)

    def test_load_latest_checkpoint(self):
        """Test loading the latest checkpoint."""
        # Save multiple checkpoints
        for i in range(3):
            trainer_state = {
                'step': (i + 1) * 1000,
                'episode': (i + 1) * 50,
                'model_state_dict': {'param': torch.randn(2, 2)},
                'optimizer_state_dict': {},
                'curriculum_state': {},
                'performance_history': [],
                'config': PPOConfig()
            }

            self.manager.save_checkpoint(
                trainer_state=trainer_state,
                step=(i + 1) * 1000,
                performance_metric=0.1 * (i + 1)
            )

        # Load latest
        latest = self.manager.load_latest_checkpoint()

        self.assertIsNotNone(latest)
        self.assertEqual(latest['step'], 3000)  # Should be the last one

    def test_load_latest_no_checkpoints(self):
        """Test loading latest when no checkpoints exist."""
        result = self.manager.load_latest_checkpoint()
        self.assertIsNone(result)

    def test_load_best_checkpoint(self):
        """Test loading the best checkpoint."""
        # Save a regular checkpoint
        trainer_state = {
            'step': 1000,
            'model_state_dict': {'param': torch.randn(2, 2)},
            'optimizer_state_dict': {},
            'curriculum_state': {},
            'performance_history': [],
            'config': PPOConfig()
        }

        self.manager.save_checkpoint(
            trainer_state=trainer_state,
            step=1000,
            performance_metric=0.5
        )

        # Save a best checkpoint
        self.manager.save_checkpoint(
            trainer_state=trainer_state,
            step=2000,
            performance_metric=0.8,
            is_best=True
        )

        # Load best
        best = self.manager.load_best_checkpoint()

        self.assertIsNotNone(best)
        self.assertEqual(best['step'], 2000)

    def test_load_best_no_best_checkpoint(self):
        """Test loading best when no best checkpoint exists."""
        result = self.manager.load_best_checkpoint()
        self.assertIsNone(result)

    def test_cleanup_old_checkpoints(self):
        """Test automatic cleanup of old checkpoints."""
        # Save more checkpoints than max_checkpoints
        for i in range(5):
            trainer_state = {
                'step': (i + 1) * 1000,
                'episode': (i + 1) * 50,
                'model_state_dict': {'param': torch.randn(2, 2)},
                'optimizer_state_dict': {},
                'curriculum_state': {},
                'performance_history': [],
                'config': PPOConfig()
            }

            self.manager.save_checkpoint(
                trainer_state=trainer_state,
                step=(i + 1) * 1000,
                performance_metric=0.1 * (i + 1)
            )

        # Should have cleaned up to max_checkpoints
        self.assertLessEqual(len(self.manager.checkpoint_info), self.manager.max_checkpoints)

        # Should keep the most recent checkpoints
        remaining_steps = [info.step for info in self.manager.checkpoint_info]
        self.assertIn(5000, remaining_steps)  # Most recent should be kept

    def test_export_model_for_inference(self):
        """Test exporting model for inference."""
        # Create a mock model
        mock_model = Mock()
        mock_model.state_dict.return_value = {'param': torch.randn(3, 3)}

        export_path = self.checkpoint_dir / 'exported_model.pt'
        config = {'learning_rate': 0.001, 'hidden_dim': 64}

        result_path = self.manager.export_model_for_inference(
            model=mock_model,
            filepath=str(export_path),
            config=config,
            include_optimizer=False
        )

        self.assertTrue(Path(result_path).exists())

        # Check export content
        exported = torch.load(result_path)
        self.assertIn('model_state_dict', exported)
        self.assertIn('config', exported)
        self.assertIn('inference_ready', exported)
        self.assertTrue(exported['inference_ready'])

        # Check info file
        info_file = export_path.with_suffix('.json')
        self.assertTrue(info_file.exists())

        with open(info_file) as f:
            info = json.load(f)
        self.assertIn('model_file', info)
        self.assertIn('checksum', info)

    def test_get_checkpoint_history(self):
        """Test getting checkpoint history."""
        # Save some checkpoints
        for i in range(3):
            trainer_state = {
                'step': (i + 1) * 1000,
                'model_state_dict': {'param': torch.randn(2, 2)},
                'optimizer_state_dict': {},
                'curriculum_state': {},
                'performance_history': [],
                'config': PPOConfig()
            }

            self.manager.save_checkpoint(
                trainer_state=trainer_state,
                step=(i + 1) * 1000,
                performance_metric=0.1 * (i + 1),
                is_best=(i == 2)  # Last one is best
            )

        history = self.manager.get_checkpoint_history()

        self.assertEqual(len(history), 4)  # 3 regular + 1 best
        # Should be sorted by step
        steps = [info.step for info in history]
        self.assertEqual(steps, sorted(steps))

    def test_get_training_summary(self):
        """Test getting training summary."""
        # Save some checkpoints
        for i in range(3):
            trainer_state = {
                'step': (i + 1) * 1000,
                'model_state_dict': {'param': torch.randn(2, 2)},
                'optimizer_state_dict': {},
                'curriculum_state': {},
                'performance_history': [],
                'config': PPOConfig()
            }

            self.manager.save_checkpoint(
                trainer_state=trainer_state,
                step=(i + 1) * 1000,
                performance_metric=0.2 + 0.1 * i
            )

        summary = self.manager.get_training_summary()

        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['total_checkpoints'], 3)
        self.assertEqual(summary['training_steps'], 3000)
        self.assertGreater(summary['best_performance'], 0)
        self.assertIn('performance_trend', summary)

    def test_save_and_load_metadata(self):
        """Test saving and loading checkpoint metadata."""
        # Save a checkpoint to trigger metadata save
        trainer_state = {
            'step': 1000,
            'model_state_dict': {'param': torch.randn(2, 2)},
            'optimizer_state_dict': {},
            'curriculum_state': {},
            'performance_history': [],
            'config': PPOConfig()
        }

        self.manager.save_checkpoint(
            trainer_state=trainer_state,
            step=1000,
            performance_metric=0.5
        )

        # Check metadata file exists
        metadata_file = self.manager.checkpoint_dir / 'checkpoint_metadata.json'
        self.assertTrue(metadata_file.exists())

        # Create new manager and check it loads metadata
        new_manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            max_checkpoints=3
        )

        self.assertEqual(len(new_manager.checkpoint_info), 1)

    def test_checkpoint_integrity_verification(self):
        """Test checkpoint integrity verification."""
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            verify_integrity=True
        )

        # Create a valid checkpoint
        trainer_state = {
            'step': 1000,
            'model_state_dict': {'param': torch.randn(2, 2)},
            'optimizer_state_dict': {},
            'curriculum_state': {},
            'performance_history': [],
            'config': PPOConfig()
        }

        saved_path = manager.save_checkpoint(
            trainer_state=trainer_state,
            step=1000,
            performance_metric=0.5
        )

        # Should load successfully
        loaded = manager.load_checkpoint(saved_path)
        self.assertIsNotNone(loaded)

        # Corrupt the file
        with open(saved_path, 'wb') as f:
            f.write(b'corrupted data')

        # Should fail to load corrupted file
        loaded = manager.load_checkpoint(saved_path)
        self.assertIsNone(loaded)

    def test_checkpoint_with_metadata(self):
        """Test checkpoint saving with additional metadata."""
        trainer_state = {
            'step': 1000,
            'model_state_dict': {'param': torch.randn(2, 2)},
            'optimizer_state_dict': {},
            'curriculum_state': {},
            'performance_history': [],
            'config': PPOConfig()
        }

        metadata = {
            'experiment_name': 'test_experiment',
            'git_commit': 'abc123',
            'notes': 'Test checkpoint with metadata'
        }

        saved_path = self.manager.save_checkpoint(
            trainer_state=trainer_state,
            step=1000,
            performance_metric=0.5,
            metadata=metadata
        )

        # Load and check metadata
        loaded = self.manager.load_checkpoint(saved_path)
        checkpoint_metadata = loaded['checkpoint_metadata']

        self.assertIn('experiment_name', checkpoint_metadata)
        self.assertEqual(checkpoint_metadata['experiment_name'], 'test_experiment')

    def test_manager_string_representation(self):
        """Test string representation of checkpoint manager."""
        string_repr = str(self.manager)

        self.assertIn('CheckpointManager', string_repr)
        self.assertIn('checkpoints=0', string_repr)
        self.assertIn('has_best=False', string_repr)


class TestCheckpointIntegration(unittest.TestCase):
    """Integration tests for checkpoint system."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_full_checkpoint_lifecycle(self):
        """Test complete checkpoint lifecycle."""
        checkpoint_dir = Path(self.temp_dir) / 'checkpoints'

        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            max_checkpoints=5,
            auto_cleanup=True
        )

        # Simulate training with multiple checkpoints
        for epoch in range(10):
            trainer_state = {
                'step': (epoch + 1) * 1000,
                'episode': (epoch + 1) * 100,
                'model_state_dict': {'param': torch.randn(5, 5)},
                'optimizer_state_dict': {'state': {}, 'param_groups': []},
                'curriculum_state': {'level': epoch // 3 + 1},
                'performance_history': [i * 0.1 for i in range(epoch + 1)],
                'config': PPOConfig(),
                'best_performance': epoch * 0.1,
                'training_time': epoch * 60
            }

            performance = 0.5 + epoch * 0.05  # Increasing performance
            is_best = (epoch == 9)  # Last one is best

            manager.save_checkpoint(
                trainer_state=trainer_state,
                step=(epoch + 1) * 1000,
                performance_metric=performance,
                is_best=is_best
            )

        # Verify cleanup happened
        self.assertLessEqual(len(manager.checkpoint_info), 5)

        # Verify best checkpoint exists
        self.assertIsNotNone(manager.best_checkpoint_info)

        # Test loading latest and best
        latest = manager.load_latest_checkpoint()
        best = manager.load_best_checkpoint()

        self.assertIsNotNone(latest)
        self.assertIsNotNone(best)
        self.assertEqual(best['step'], 10000)  # Last epoch

        # Test training summary
        summary = manager.get_training_summary()
        self.assertEqual(summary['training_steps'], 10000)
        self.assertTrue(summary['has_best_model'])

        # Test export
        mock_model = Mock()
        mock_model.state_dict.return_value = {'final_param': torch.randn(10, 10)}

        export_path = manager.export_model_for_inference(
            model=mock_model,
            filepath=str(checkpoint_dir / 'final_model.pt')
        )

        self.assertTrue(Path(export_path).exists())

    def test_checkpoint_recovery_after_crash(self):
        """Test checkpoint recovery after simulated crash."""
        checkpoint_dir = Path(self.temp_dir) / 'recovery_test'

        # First training session
        manager1 = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            max_checkpoints=3
        )

        # Save some checkpoints
        for i in range(3):
            trainer_state = {
                'step': (i + 1) * 500,
                'episode': (i + 1) * 25,
                'model_state_dict': {'param': torch.randn(3, 3)},
                'optimizer_state_dict': {},
                'curriculum_state': {'level': i},
                'performance_history': [],
                'config': PPOConfig()
            }

            manager1.save_checkpoint(
                trainer_state=trainer_state,
                step=(i + 1) * 500,
                performance_metric=0.3 + i * 0.1
            )

        # Simulate crash - create new manager
        manager2 = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            max_checkpoints=3
        )

        # Should recover existing checkpoints
        self.assertEqual(len(manager2.checkpoint_info), 3)

        # Should be able to load latest
        latest = manager2.load_latest_checkpoint()
        self.assertIsNotNone(latest)
        self.assertEqual(latest['step'], 1500)


if __name__ == '__main__':
    unittest.main()