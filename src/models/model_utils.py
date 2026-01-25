"""
Model utilities for state management, checkpointing, and deployment.

Provides utilities for saving/loading model states, verifying compatibility,
and preparing models for inference deployment.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union, Tuple
from datetime import datetime
import json
import shutil

from .regime_detector import RegimeDetector
from ..config import TrainingConfig, config

logger = logging.getLogger(__name__)


def save_model_state(model: nn.Module, filepath: Union[str, Path],
                    metadata: Optional[Dict[str, Any]] = None,
                    include_optimizer: bool = False,
                    optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    """
    Save model state with metadata for deployment or checkpointing.

    Args:
        model: PyTorch model to save
        filepath: Path to save the model
        metadata: Optional metadata dictionary
        include_optimizer: Whether to include optimizer state
        optimizer: Optimizer to save (required if include_optimizer=True)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Basic model information
    model_info = {
        'model_class': model.__class__.__name__,
        'state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }

    # Add model-specific configuration for RegimeDetector
    if isinstance(model, RegimeDetector):
        model_info['config'] = {
            'input_dim': model.input_dim,
            'hidden_dims': model.hidden_dims,
            'num_regimes': model.num_regimes,
            'dropout_rate': model.dropout_rate
        }

    # Add optimizer state if requested
    if include_optimizer and optimizer is not None:
        model_info['optimizer_state_dict'] = optimizer.state_dict()
        model_info['optimizer_class'] = optimizer.__class__.__name__

    # Add custom metadata
    if metadata:
        model_info['metadata'] = metadata

    # Add model summary statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_info['model_stats'] = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }

    # Save the complete state
    torch.save(model_info, filepath)
    logger.info(f"Model state saved to {filepath}")


def load_model_state(filepath: Union[str, Path], device: Optional[torch.device] = None,
                    strict_loading: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load model state and metadata from saved file.

    Args:
        filepath: Path to the saved model
        device: Device to load the model on
        strict_loading: Whether to strictly enforce state dict loading

    Returns:
        Tuple of (model_state_dict, full_model_info)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Load the complete model information
    model_info = torch.load(filepath, map_location=device)

    if not isinstance(model_info, dict) or 'state_dict' not in model_info:
        raise ValueError(f"Invalid model file format: {filepath}")

    logger.info(f"Model state loaded from {filepath}")
    return model_info['state_dict'], model_info


def create_regime_detector_from_checkpoint(checkpoint_path: Union[str, Path],
                                         device: Optional[torch.device] = None) -> RegimeDetector:
    """
    Create and load a RegimeDetector model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Loaded RegimeDetector model
    """
    state_dict, model_info = load_model_state(checkpoint_path, device)

    # Extract model configuration
    if 'config' in model_info:
        config_dict = model_info['config']
        model = RegimeDetector(
            input_dim=config_dict.get('input_dim', 48),
            hidden_dims=tuple(config_dict.get('hidden_dims', (128, 64, 32))),
            num_regimes=config_dict.get('num_regimes', 8),
            dropout_rate=config_dict.get('dropout_rate', 0.2)
        )
    else:
        # Use default configuration
        logger.warning("No model configuration found in checkpoint, using defaults")
        model = RegimeDetector()

    # Load the state dict
    model.load_state_dict(state_dict)

    # Move to specified device
    if device is not None:
        model = model.to(device)

    logger.info(f"RegimeDetector model created from checkpoint: {checkpoint_path}")
    return model


def verify_model_compatibility(model1: nn.Module, model2: nn.Module) -> Dict[str, bool]:
    """
    Verify compatibility between two models for state transfer.

    Args:
        model1: First model
        model2: Second model

    Returns:
        Dictionary with compatibility results
    """
    results = {}

    # Check if same class
    results['same_class'] = model1.__class__ == model2.__class__

    # Check architecture compatibility for RegimeDetector
    if isinstance(model1, RegimeDetector) and isinstance(model2, RegimeDetector):
        results['same_input_dim'] = model1.input_dim == model2.input_dim
        results['same_hidden_dims'] = model1.hidden_dims == model2.hidden_dims
        results['same_num_regimes'] = model1.num_regimes == model2.num_regimes
        results['same_dropout_rate'] = model1.dropout_rate == model2.dropout_rate
    else:
        # For other models, check parameter shapes
        state1 = model1.state_dict()
        state2 = model2.state_dict()

        results['same_parameter_names'] = set(state1.keys()) == set(state2.keys())

        if results['same_parameter_names']:
            shape_matches = []
            for name in state1.keys():
                shape_matches.append(state1[name].shape == state2[name].shape)
            results['same_parameter_shapes'] = all(shape_matches)
        else:
            results['same_parameter_shapes'] = False

    # Overall compatibility
    if isinstance(model1, RegimeDetector) and isinstance(model2, RegimeDetector):
        results['compatible'] = all([
            results['same_class'],
            results['same_input_dim'],
            results['same_hidden_dims'],
            results['same_num_regimes']
        ])
    else:
        results['compatible'] = all([
            results['same_class'],
            results.get('same_parameter_names', True),
            results.get('same_parameter_shapes', True)
        ])

    return results


def export_model_for_inference(model: nn.Module, export_path: Union[str, Path],
                              sample_input: Optional[torch.Tensor] = None,
                              include_metadata: bool = True) -> None:
    """
    Export model for inference-only deployment (without training components).

    Args:
        model: Model to export
        export_path: Path to export the model
        sample_input: Sample input for tracing (if using TorchScript)
        include_metadata: Whether to include metadata
    """
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    # Create inference-only state
    inference_state = {
        'model_class': model.__class__.__name__,
        'state_dict': model.state_dict(),
        'inference_mode': True,
        'export_timestamp': datetime.now().isoformat()
    }

    # Add model configuration
    if isinstance(model, RegimeDetector):
        inference_state['config'] = {
            'input_dim': model.input_dim,
            'hidden_dims': model.hidden_dims,
            'num_regimes': model.num_regimes,
            'dropout_rate': model.dropout_rate
        }

    # Add metadata if requested
    if include_metadata:
        total_params = sum(p.numel() for p in model.parameters())
        inference_state['metadata'] = {
            'total_parameters': total_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'input_shape': f"(batch_size, {getattr(model, 'input_dim', 'unknown')})",
            'output_shape': f"(batch_size, {getattr(model, 'num_regimes', 8) + 1})"
        }

    # Save inference model
    torch.save(inference_state, export_path)
    logger.info(f"Inference model exported to {export_path}")


def load_inference_model(model_path: Union[str, Path],
                        device: Optional[torch.device] = None) -> nn.Module:
    """
    Load a model exported for inference.

    Args:
        model_path: Path to the inference model
        device: Device to load the model on

    Returns:
        Loaded model in evaluation mode
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Inference model not found: {model_path}")

    model_data = torch.load(model_path, map_location=device)

    if model_data.get('model_class') == 'RegimeDetector':
        # Create RegimeDetector from configuration
        config_dict = model_data.get('config', {})
        model = RegimeDetector(
            input_dim=config_dict.get('input_dim', 48),
            hidden_dims=tuple(config_dict.get('hidden_dims', (128, 64, 32))),
            num_regimes=config_dict.get('num_regimes', 8),
            dropout_rate=config_dict.get('dropout_rate', 0.2)
        )
    else:
        raise ValueError(f"Unknown model class: {model_data.get('model_class')}")

    # Load state and set to evaluation mode
    model.load_state_dict(model_data['state_dict'])
    model.eval()

    # Move to device
    if device is not None:
        model = model.to(device)

    logger.info(f"Inference model loaded from {model_path}")
    return model


def cleanup_old_checkpoints(checkpoint_dir: Union[str, Path], max_keep: int = 5,
                          pattern: str = "checkpoint_epoch_*.pth") -> None:
    """
    Clean up old checkpoint files, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        max_keep: Maximum number of checkpoints to keep
        pattern: File pattern for checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    # Find matching checkpoint files
    checkpoint_files = list(checkpoint_dir.glob(pattern))

    if len(checkpoint_files) <= max_keep:
        return  # Nothing to clean up

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Remove old checkpoints
    for old_checkpoint in checkpoint_files[max_keep:]:
        old_checkpoint.unlink()
        logger.debug(f"Removed old checkpoint: {old_checkpoint}")

    logger.info(f"Cleaned up {len(checkpoint_files) - max_keep} old checkpoints")


def backup_model(model_path: Union[str, Path], backup_dir: Union[str, Path]) -> Path:
    """
    Create a backup copy of a model file.

    Args:
        model_path: Path to the model to backup
        backup_dir: Directory to store backup

    Returns:
        Path to the backup file
    """
    model_path = Path(model_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{model_path.stem}_backup_{timestamp}{model_path.suffix}"
    backup_path = backup_dir / backup_name

    # Copy the model file
    shutil.copy2(model_path, backup_path)
    logger.info(f"Model backed up to {backup_path}")

    return backup_path


def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a saved model without loading it completely.

    Args:
        model_path: Path to the model file

    Returns:
        Dictionary with model information
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load only the metadata (not the full state dict)
    model_data = torch.load(model_path, map_location='cpu')

    info = {
        'file_path': str(model_path),
        'file_size_mb': model_path.stat().st_size / (1024 * 1024),
        'model_class': model_data.get('model_class', 'Unknown'),
        'timestamp': model_data.get('timestamp', 'Unknown'),
        'inference_mode': model_data.get('inference_mode', False)
    }

    # Add model-specific info
    if 'config' in model_data:
        info['config'] = model_data['config']

    if 'model_stats' in model_data:
        info['stats'] = model_data['model_stats']

    if 'metadata' in model_data:
        info['metadata'] = model_data['metadata']

    # Training info if available
    if 'epoch' in model_data:
        info['training'] = {
            'epoch': model_data['epoch'],
            'best_val_accuracy': model_data.get('best_val_accuracy'),
            'best_val_loss': model_data.get('best_val_loss')
        }

    return info


def convert_checkpoint_to_inference(checkpoint_path: Union[str, Path],
                                  output_path: Union[str, Path]) -> None:
    """
    Convert a training checkpoint to an inference-only model.

    Args:
        checkpoint_path: Path to training checkpoint
        output_path: Path for inference model output
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load the checkpoint
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

    # Create inference model
    model = create_regime_detector_from_checkpoint(checkpoint_path)

    # Export for inference
    export_model_for_inference(model, output_path, include_metadata=True)

    logger.info(f"Converted checkpoint {checkpoint_path} to inference model {output_path}")


# Validation functions for model integrity
def validate_model_file(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a model file for integrity and completeness.

    Args:
        model_path: Path to model file

    Returns:
        Dictionary with validation results
    """
    model_path = Path(model_path)
    results = {
        'file_exists': model_path.exists(),
        'valid_format': False,
        'has_state_dict': False,
        'has_config': False,
        'loadable': False,
        'errors': []
    }

    if not results['file_exists']:
        results['errors'].append("File does not exist")
        return results

    try:
        # Try to load the model data
        model_data = torch.load(model_path, map_location='cpu')
        results['valid_format'] = True

        # Check for required components
        if 'state_dict' in model_data:
            results['has_state_dict'] = True
        else:
            results['errors'].append("Missing state_dict")

        if 'config' in model_data:
            results['has_config'] = True

        # Try to actually load the model if it's a RegimeDetector
        if model_data.get('model_class') == 'RegimeDetector':
            try:
                model = create_regime_detector_from_checkpoint(model_path)
                results['loadable'] = True
            except Exception as e:
                results['errors'].append(f"Model loading failed: {str(e)}")

    except Exception as e:
        results['errors'].append(f"File loading failed: {str(e)}")

    results['valid'] = (results['valid_format'] and results['has_state_dict'] and
                       results['loadable'])

    return results