"""
Comprehensive Checkpoint System for PPO Training.

This module implements robust checkpointing for training continuity,
including model state persistence, training state recovery, best model tracking,
and automatic checkpoint management.
"""

import torch
import numpy as np
import json
import pickle
import shutil
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import time
from datetime import datetime
import glob
import hashlib

# Avoid circular import - PPOConfig will be passed as needed

logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    """
    Complete training state for checkpointing.
    """
    step: int
    episode: int
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Dict[str, Any]
    curriculum_state: Dict[str, Any]
    performance_history: List[float]
    config: Any  # PPO config object
    best_performance: float = -float('inf')
    steps_without_improvement: int = 0
    timestamp: float = field(default_factory=time.time)
    training_time: float = 0.0
    additional_state: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        state_dict = asdict(self)
        # Handle non-serializable objects
        if 'config' in state_dict and hasattr(state_dict['config'], '__dict__'):
            state_dict['config'] = asdict(state_dict['config'])
        return state_dict


@dataclass
class CheckpointInfo:
    """
    Information about a checkpoint file.
    """
    filepath: str
    step: int
    episode: int
    performance_metric: float
    timestamp: float
    file_size: int
    checksum: str
    is_best: bool = False

    @classmethod
    def from_file(cls, filepath: str) -> Optional['CheckpointInfo']:
        """
        Create CheckpointInfo from checkpoint file.

        Args:
            filepath: Path to checkpoint file

        Returns:
            CheckpointInfo instance or None if file cannot be read
        """
        try:
            path = Path(filepath)
            if not path.exists():
                return None

            # Extract info from filename if possible
            filename = path.stem
            parts = filename.split('_')

            # Try to load checkpoint to get detailed info
            try:
                checkpoint = torch.load(filepath, map_location='cpu')
                step = checkpoint.get('step', 0)
                episode = checkpoint.get('episode', 0)
                performance = checkpoint.get('best_performance', 0.0)
                timestamp = checkpoint.get('timestamp', path.stat().st_mtime)
            except:
                # Fallback to filename parsing
                step = 0
                episode = 0
                performance = 0.0
                timestamp = path.stat().st_mtime

            # Calculate checksum
            checksum = cls._calculate_checksum(filepath)

            return cls(
                filepath=str(path),
                step=step,
                episode=episode,
                performance_metric=performance,
                timestamp=timestamp,
                file_size=path.stat().st_size,
                checksum=checksum,
                is_best='best' in filename.lower()
            )

        except Exception as e:
            logger.error(f"Failed to create CheckpointInfo from {filepath}: {e}")
            return None

    @staticmethod
    def _calculate_checksum(filepath: str) -> str:
        """Calculate MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return ""


class CheckpointManager:
    """
    Comprehensive checkpoint management system.

    Handles saving, loading, and managing training checkpoints with
    automatic cleanup, best model tracking, and integrity verification.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        metric_for_best: str = 'validation_sharpe_ratio',
        auto_cleanup: bool = True,
        verify_integrity: bool = True
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for storing checkpoints
            max_checkpoints: Maximum number of regular checkpoints to keep
            save_best_only: Whether to only save best performing models
            metric_for_best: Metric to use for determining best model
            auto_cleanup: Whether to automatically cleanup old checkpoints
            verify_integrity: Whether to verify checkpoint integrity
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.metric_for_best = metric_for_best
        self.auto_cleanup = auto_cleanup
        self.verify_integrity = verify_integrity

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.models_dir = self.checkpoint_dir / 'models'
        self.best_dir = self.checkpoint_dir / 'best'
        self.exports_dir = self.checkpoint_dir / 'exports'

        for dir_path in [self.models_dir, self.best_dir, self.exports_dir]:
            dir_path.mkdir(exist_ok=True)

        # Tracking
        self.checkpoint_info = []
        self.best_checkpoint_info = None
        self.load_existing_checkpoints()

        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
        logger.info(f"Max checkpoints: {self.max_checkpoints}, Best metric: {self.metric_for_best}")

    def save_checkpoint(
        self,
        trainer_state: Dict[str, Any],
        step: int,
        performance_metric: float,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save training checkpoint.

        Args:
            trainer_state: Complete training state
            step: Current training step
            performance_metric: Performance metric value
            is_best: Whether this is the best performing model
            metadata: Additional metadata to save

        Returns:
            Path to saved checkpoint file
        """
        try:
            # Create checkpoint data
            timestamp = time.time()
            checkpoint_data = {
                **trainer_state,
                'checkpoint_metadata': {
                    'step': step,
                    'performance_metric': performance_metric,
                    'timestamp': timestamp,
                    'is_best': is_best,
                    'metric_name': self.metric_for_best,
                    'save_time': datetime.now().isoformat(),
                    **(metadata or {})
                }
            }

            # Generate filename
            if is_best:
                filename = f"best_model_step_{step}_perf_{performance_metric:.4f}.pt"
                filepath = self.best_dir / filename
            else:
                filename = f"checkpoint_step_{step}_episode_{trainer_state.get('episode', 0)}.pt"
                filepath = self.models_dir / filename

            # Save checkpoint
            torch.save(checkpoint_data, filepath)

            # Create checkpoint info
            checkpoint_info = CheckpointInfo(
                filepath=str(filepath),
                step=step,
                episode=trainer_state.get('episode', 0),
                performance_metric=performance_metric,
                timestamp=timestamp,
                file_size=filepath.stat().st_size,
                checksum=CheckpointInfo._calculate_checksum(str(filepath)),
                is_best=is_best
            )

            # Update tracking
            if is_best:
                self.best_checkpoint_info = checkpoint_info
            else:
                self.checkpoint_info.append(checkpoint_info)

            # Save checkpoint metadata
            self._save_checkpoint_metadata()

            # Cleanup old checkpoints if needed
            if self.auto_cleanup and not is_best:
                self.cleanup_old_checkpoints()

            logger.info(f"Checkpoint saved: {filepath}")
            logger.info(f"Step: {step}, Performance: {performance_metric:.4f}, Is Best: {is_best}")

            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load specific checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Checkpoint data or None if loading failed
        """
        try:
            path = Path(filepath)
            if not path.exists():
                logger.error(f"Checkpoint file not found: {filepath}")
                return None

            # Verify integrity if enabled
            if self.verify_integrity:
                if not self._verify_checkpoint_integrity(filepath):
                    logger.error(f"Checkpoint integrity check failed: {filepath}")
                    return None

            # Load checkpoint
            checkpoint = torch.load(filepath, map_location='cpu')

            logger.info(f"Checkpoint loaded: {filepath}")
            if 'checkpoint_metadata' in checkpoint:
                metadata = checkpoint['checkpoint_metadata']
                logger.info(f"Step: {metadata.get('step', 0)}, "
                           f"Performance: {metadata.get('performance_metric', 0.0):.4f}")

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint from {filepath}: {e}")
            return None

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.

        Returns:
            Latest checkpoint data or None if no checkpoints available
        """
        if not self.checkpoint_info:
            logger.warning("No checkpoints available to load")
            return None

        # Sort by step (descending) to get latest
        latest_info = sorted(self.checkpoint_info, key=lambda x: x.step, reverse=True)[0]
        return self.load_checkpoint(latest_info.filepath)

    def load_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the best performing checkpoint.

        Returns:
            Best checkpoint data or None if no best checkpoint available
        """
        if not self.best_checkpoint_info:
            logger.warning("No best checkpoint available to load")
            return None

        return self.load_checkpoint(self.best_checkpoint_info.filepath)

    def cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond the maximum limit."""
        if len(self.checkpoint_info) <= self.max_checkpoints:
            return

        # Sort checkpoints by step (ascending) to remove oldest
        sorted_checkpoints = sorted(self.checkpoint_info, key=lambda x: x.step)
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]

        for checkpoint_info in checkpoints_to_remove:
            try:
                path = Path(checkpoint_info.filepath)
                if path.exists():
                    path.unlink()
                    logger.info(f"Removed old checkpoint: {path}")

                # Remove from tracking
                self.checkpoint_info.remove(checkpoint_info)

            except Exception as e:
                logger.error(f"Failed to remove checkpoint {checkpoint_info.filepath}: {e}")

        logger.info(f"Cleanup complete. {len(self.checkpoint_info)} checkpoints remaining")

    def export_model_for_inference(
        self,
        model: torch.nn.Module,
        filepath: str,
        config: Optional[Dict[str, Any]] = None,
        include_optimizer: bool = False
    ):
        """
        Export model for inference/deployment.

        Args:
            model: Trained model to export
            filepath: Export filepath
            config: Model configuration to include
            include_optimizer: Whether to include optimizer state
        """
        try:
            export_path = Path(filepath)
            if not export_path.parent.exists():
                export_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare export data
            export_data = {
                'model_state_dict': model.state_dict(),
                'model_architecture': str(model),
                'export_timestamp': datetime.now().isoformat(),
                'inference_ready': True
            }

            if config:
                export_data['config'] = config

            if include_optimizer and hasattr(model, 'optimizer'):
                export_data['optimizer_state_dict'] = model.optimizer.state_dict()

            # Save export
            torch.save(export_data, export_path)

            # Create deployment info file
            info_path = export_path.with_suffix('.json')
            deployment_info = {
                'model_file': export_path.name,
                'export_time': datetime.now().isoformat(),
                'model_size_bytes': export_path.stat().st_size,
                'checksum': CheckpointInfo._calculate_checksum(str(export_path)),
                'inference_ready': True,
                'deployment_notes': 'Exported model ready for inference'
            }

            with open(info_path, 'w') as f:
                json.dump(deployment_info, f, indent=2)

            logger.info(f"Model exported for inference: {export_path}")
            return str(export_path)

        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise

    def get_checkpoint_history(self) -> List[CheckpointInfo]:
        """
        Get list of all checkpoint information.

        Returns:
            List of CheckpointInfo objects
        """
        all_checkpoints = list(self.checkpoint_info)
        if self.best_checkpoint_info:
            all_checkpoints.append(self.best_checkpoint_info)

        return sorted(all_checkpoints, key=lambda x: x.step)

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training progress from checkpoints.

        Returns:
            Training summary statistics
        """
        if not self.checkpoint_info:
            return {'message': 'No training checkpoints available'}

        checkpoints = sorted(self.checkpoint_info, key=lambda x: x.step)
        performance_history = [c.performance_metric for c in checkpoints]

        summary = {
            'total_checkpoints': len(self.checkpoint_info),
            'training_steps': checkpoints[-1].step if checkpoints else 0,
            'best_performance': max(performance_history) if performance_history else 0.0,
            'latest_performance': performance_history[-1] if performance_history else 0.0,
            'performance_trend': self._calculate_trend(performance_history),
            'has_best_model': self.best_checkpoint_info is not None,
            'total_training_time': self._estimate_total_training_time(),
            'checkpoint_directory': str(self.checkpoint_dir)
        }

        if self.best_checkpoint_info:
            summary['best_model_step'] = self.best_checkpoint_info.step
            summary['best_model_performance'] = self.best_checkpoint_info.performance_metric

        return summary

    def _save_checkpoint_metadata(self):
        """Save checkpoint metadata to file."""
        metadata_file = self.checkpoint_dir / 'checkpoint_metadata.json'

        metadata = {
            'checkpoints': [asdict(info) for info in self.checkpoint_info],
            'best_checkpoint': asdict(self.best_checkpoint_info) if self.best_checkpoint_info else None,
            'manager_config': {
                'max_checkpoints': self.max_checkpoints,
                'save_best_only': self.save_best_only,
                'metric_for_best': self.metric_for_best
            },
            'last_updated': datetime.now().isoformat()
        }

        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata: {e}")

    def load_existing_checkpoints(self):
        """Load information about existing checkpoints."""
        metadata_file = self.checkpoint_dir / 'checkpoint_metadata.json'

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Load checkpoint info
                self.checkpoint_info = [
                    CheckpointInfo(**info) for info in metadata.get('checkpoints', [])
                ]

                # Load best checkpoint info
                best_info = metadata.get('best_checkpoint')
                if best_info:
                    self.best_checkpoint_info = CheckpointInfo(**best_info)

                logger.info(f"Loaded metadata for {len(self.checkpoint_info)} checkpoints")

            except Exception as e:
                logger.error(f"Failed to load checkpoint metadata: {e}")
                self._scan_checkpoint_directory()
        else:
            self._scan_checkpoint_directory()

    def _scan_checkpoint_directory(self):
        """Scan directory for existing checkpoint files."""
        try:
            # Scan regular checkpoints
            checkpoint_patterns = ['*.pt', '*.pth', '*.checkpoint']
            for pattern in checkpoint_patterns:
                for filepath in self.models_dir.glob(pattern):
                    info = CheckpointInfo.from_file(str(filepath))
                    if info:
                        self.checkpoint_info.append(info)

            # Scan best checkpoints
            for filepath in self.best_dir.glob('*.pt'):
                info = CheckpointInfo.from_file(str(filepath))
                if info and info.is_best:
                    self.best_checkpoint_info = info

            logger.info(f"Scanned directory: found {len(self.checkpoint_info)} checkpoints")

        except Exception as e:
            logger.error(f"Failed to scan checkpoint directory: {e}")

    def _verify_checkpoint_integrity(self, filepath: str) -> bool:
        """Verify checkpoint file integrity."""
        try:
            # Try to load checkpoint
            torch.load(filepath, map_location='cpu')
            return True
        except Exception:
            return False

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate performance trend."""
        if len(values) < 2:
            return 'stable'

        recent_values = values[-min(5, len(values)):]
        if len(recent_values) < 2:
            return 'stable'

        # Simple trend calculation
        slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]

        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'

    def _estimate_total_training_time(self) -> float:
        """Estimate total training time from checkpoints."""
        if len(self.checkpoint_info) < 2:
            return 0.0

        sorted_checkpoints = sorted(self.checkpoint_info, key=lambda x: x.timestamp)
        return sorted_checkpoints[-1].timestamp - sorted_checkpoints[0].timestamp

    def __str__(self) -> str:
        """String representation of checkpoint manager."""
        return (f"CheckpointManager(dir={self.checkpoint_dir}, "
                f"checkpoints={len(self.checkpoint_info)}, "
                f"has_best={self.best_checkpoint_info is not None})")