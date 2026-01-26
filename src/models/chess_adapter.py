"""
Chess model weight adapter for transfer learning to options domain.

Implements ChessWeightAdapter class for loading and adapting pre-trained chess
model weights from 8x8 board format to 7x6 options position format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
import warnings


class ChessWeightAdapter:
    """
    Adapter for loading and adapting chess model weights to options domain.

    Handles loading common chess AI model formats and adapting convolutional
    weights from 8x8 chess board to 7x6 options board using interpolation
    and cropping techniques.
    """

    def __init__(self, target_spatial_dims: Tuple[int, int] = (7, 6)):
        """
        Initialize chess weight adapter.

        Args:
            target_spatial_dims: Target spatial dimensions (height, width) for adaptation
        """
        self.target_height, self.target_width = target_spatial_dims
        self.supported_formats = ['.pth', '.pt', '.onnx', '.pkl']

    def load_chess_weights(self,
                          model_path: Union[str, Path],
                          model_type: str = 'pytorch') -> Dict[str, torch.Tensor]:
        """
        Load chess model weights from various formats.

        Args:
            model_path: Path to chess model file
            model_type: Type of model ('pytorch', 'onnx', 'checkpoint')

        Returns:
            Dictionary of loaded weights

        Raises:
            ValueError: If model format not supported
            FileNotFoundError: If model file not found
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Chess model file not found: {model_path}")

        if model_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")

        try:
            if model_type == 'pytorch' or model_path.suffix in ['.pth', '.pt']:
                return self._load_pytorch_weights(model_path)
            elif model_type == 'onnx' or model_path.suffix == '.onnx':
                return self._load_onnx_weights(model_path)
            elif model_type == 'checkpoint':
                return self._load_checkpoint_weights(model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        except Exception as e:
            raise RuntimeError(f"Failed to load chess weights from {model_path}: {e}")

    def _load_pytorch_weights(self, model_path: Path) -> Dict[str, torch.Tensor]:
        """Load PyTorch model weights."""
        checkpoint = torch.load(model_path, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                return checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict']
            elif 'weights' in checkpoint:
                return checkpoint['weights']
            else:
                # Assume the dict itself contains the weights
                return checkpoint
        else:
            # Assume it's a direct state dict
            return checkpoint

    def _load_onnx_weights(self, model_path: Path) -> Dict[str, torch.Tensor]:
        """Load ONNX model weights (simplified implementation)."""
        # For ONNX models, we would need onnx library
        # This is a placeholder for the interface
        warnings.warn("ONNX weight loading not fully implemented. Use PyTorch format.")
        return {}

    def _load_checkpoint_weights(self, model_path: Path) -> Dict[str, torch.Tensor]:
        """Load checkpoint format weights."""
        # Similar to PyTorch but with different key structure
        return self._load_pytorch_weights(model_path)

    def adapt_conv_layers(self,
                         chess_weights: Dict[str, torch.Tensor],
                         target_model_keys: Optional[Dict[str, str]] = None) -> Dict[str, torch.Tensor]:
        """
        Adapt convolutional layer weights from 8x8 to 7x6 format.

        Args:
            chess_weights: Dictionary of chess model weights
            target_model_keys: Mapping from chess weight keys to target model keys

        Returns:
            Dictionary of adapted weights
        """
        adapted_weights = {}
        target_model_keys = target_model_keys or {}

        for key, weight in chess_weights.items():
            # Skip non-convolutional layers
            if not self._is_conv_weight(key, weight):
                # Use mapped key if available
                target_key = target_model_keys.get(key, key)
                adapted_weights[target_key] = weight
                continue

            # Adapt convolutional weights
            if weight.dim() == 4:  # Conv2d weights: [out_channels, in_channels, height, width]
                adapted_weight = self._adapt_conv2d_weight(weight)
            elif weight.dim() == 2:  # Linear weights that might be spatial
                adapted_weight = self._adapt_linear_weight(weight)
            else:
                adapted_weight = weight

            # Use mapped key if available
            target_key = target_model_keys.get(key, key)
            adapted_weights[target_key] = adapted_weight

        return adapted_weights

    def _is_conv_weight(self, key: str, weight: torch.Tensor) -> bool:
        """Check if weight belongs to a convolutional layer."""
        conv_indicators = ['conv', 'Conv', 'spatial', 'feature']
        return any(indicator in key for indicator in conv_indicators) and weight.dim() >= 2

    def _adapt_conv2d_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Adapt Conv2d weights from 8x8 to 7x6 spatial dimensions.

        Args:
            weight: Conv2d weight tensor [out_channels, in_channels, height, width]

        Returns:
            Adapted weight tensor [out_channels, in_channels, 7, 6] or original if not 8x8
        """
        out_channels, in_channels, height, width = weight.shape

        # If already correct size, return as is
        if height == self.target_height and width == self.target_width:
            return weight

        # Only adapt if it's actually an 8x8 chess board spatial weight
        # Keep other conv weights (like 3x3, 1x1) unchanged
        if height != 8 or width != 8:
            return weight

        # Use interpolation to adapt spatial dimensions
        # Reshape to [out_channels * in_channels, height, width]
        reshaped = weight.view(out_channels * in_channels, 1, height, width)

        # Interpolate to target size
        adapted = F.interpolate(
            reshaped,
            size=(self.target_height, self.target_width),
            mode='bilinear',
            align_corners=False
        )

        # Reshape back to original format
        adapted = adapted.view(out_channels, in_channels, self.target_height, self.target_width)

        return adapted

    def _adapt_linear_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Adapt linear weights that might encode spatial relationships.

        Args:
            weight: Linear weight tensor

        Returns:
            Adapted weight tensor
        """
        # For linear weights, we might need to truncate or interpolate
        # if they encode spatial information (e.g., positional encodings)

        # Simple approach: if weight dimensions suggest spatial encoding,
        # truncate or pad appropriately
        if weight.shape[-1] == 64:  # 8x8 = 64 positions
            # Reshape to spatial format, adapt, then reshape back
            if weight.dim() == 2:
                out_features, _ = weight.shape
                spatial_weight = weight.view(out_features, 8, 8)
                adapted_spatial = F.interpolate(
                    spatial_weight.unsqueeze(1),
                    size=(self.target_height, self.target_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                return adapted_spatial.view(out_features, -1)

        return weight

    def adapt_residual_blocks(self,
                             chess_weights: Dict[str, torch.Tensor],
                             target_block_mapping: Optional[Dict[str, str]] = None) -> Dict[str, torch.Tensor]:
        """
        Adapt residual block weights from chess patterns to options domain.

        Args:
            chess_weights: Dictionary of chess model weights
            target_block_mapping: Mapping from chess block names to target block names

        Returns:
            Dictionary of adapted residual block weights
        """
        adapted_weights = {}
        target_block_mapping = target_block_mapping or {}

        # Group weights by residual block
        block_groups = self._group_residual_weights(chess_weights)

        for block_name, block_weights in block_groups.items():
            # Adapt each weight in the block
            adapted_block_weights = {}
            for weight_name, weight in block_weights.items():
                if weight.dim() == 4:  # Conv2d weights
                    adapted_weight = self._adapt_conv2d_weight(weight)
                else:
                    adapted_weight = weight
                adapted_block_weights[weight_name] = adapted_weight

            # Map to target block name if provided
            target_block = target_block_mapping.get(block_name, block_name)

            # Add adapted weights with proper keys
            for weight_name, adapted_weight in adapted_block_weights.items():
                full_key = f"{target_block}.{weight_name}"
                adapted_weights[full_key] = adapted_weight

        return adapted_weights

    def _group_residual_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Group weights by residual block."""
        block_groups = {}

        for key, weight in weights.items():
            # Extract block name from key (e.g., "residual_blocks.0.conv1.weight" -> "residual_blocks.0")
            if 'residual' in key.lower() or 'block' in key.lower():
                parts = key.split('.')
                if len(parts) >= 2:
                    block_name = '.'.join(parts[:2])  # e.g., "residual_blocks.0"
                    weight_name = '.'.join(parts[2:])  # e.g., "conv1.weight"

                    if block_name not in block_groups:
                        block_groups[block_name] = {}
                    block_groups[block_name][weight_name] = weight

        return block_groups

    def weight_compatibility_check(self,
                                 chess_weights: Dict[str, torch.Tensor],
                                 target_model: nn.Module) -> Dict[str, Any]:
        """
        Check compatibility between chess weights and target model architecture.

        Args:
            chess_weights: Dictionary of chess model weights
            target_model: Target PyTorch model

        Returns:
            Dictionary with compatibility report
        """
        target_state_dict = target_model.state_dict()
        report = {
            'compatible_layers': [],
            'incompatible_layers': [],
            'missing_in_chess': [],
            'extra_in_chess': [],
            'size_mismatches': [],
            'total_chess_params': sum(w.numel() for w in chess_weights.values()),
            'total_target_params': sum(w.numel() for w in target_state_dict.values())
        }

        # Check each target layer
        for target_key, target_weight in target_state_dict.items():
            if target_key in chess_weights:
                chess_weight = chess_weights[target_key]
                if target_weight.shape == chess_weight.shape:
                    report['compatible_layers'].append(target_key)
                else:
                    report['size_mismatches'].append({
                        'layer': target_key,
                        'target_shape': target_weight.shape,
                        'chess_shape': chess_weight.shape
                    })
            else:
                report['missing_in_chess'].append(target_key)

        # Check for extra layers in chess weights
        for chess_key in chess_weights.keys():
            if chess_key not in target_state_dict:
                report['extra_in_chess'].append(chess_key)

        # Categorize incompatible layers
        for mismatch in report['size_mismatches']:
            report['incompatible_layers'].append(mismatch['layer'])

        return report

    def create_weight_mapping(self,
                            chess_weights: Dict[str, torch.Tensor],
                            target_model: nn.Module,
                            mapping_strategy: str = 'auto') -> Dict[str, str]:
        """
        Create mapping between chess weight keys and target model keys.

        Args:
            chess_weights: Dictionary of chess model weights
            target_model: Target PyTorch model
            mapping_strategy: Strategy for mapping ('auto', 'exact', 'fuzzy')

        Returns:
            Dictionary mapping chess keys to target keys
        """
        target_keys = set(target_model.state_dict().keys())
        chess_keys = set(chess_weights.keys())
        mapping = {}

        if mapping_strategy == 'exact':
            # Only map exactly matching keys
            for key in chess_keys.intersection(target_keys):
                mapping[key] = key

        elif mapping_strategy == 'fuzzy':
            # Try to match similar layer names
            for chess_key in chess_keys:
                best_match = self._find_fuzzy_match(chess_key, target_keys)
                if best_match:
                    mapping[chess_key] = best_match

        else:  # auto strategy
            # Combine exact and fuzzy matching
            for key in chess_keys.intersection(target_keys):
                mapping[key] = key

            # For unmatched keys, try fuzzy matching
            unmatched_chess = chess_keys - target_keys
            unmatched_target = target_keys - chess_keys

            for chess_key in unmatched_chess:
                best_match = self._find_fuzzy_match(chess_key, unmatched_target)
                if best_match:
                    mapping[chess_key] = best_match
                    unmatched_target.remove(best_match)

        return mapping

    def _find_fuzzy_match(self, chess_key: str, target_keys: set) -> Optional[str]:
        """Find the best fuzzy match for a chess key among target keys."""
        # Simple fuzzy matching based on common substrings
        chess_parts = chess_key.split('.')
        best_match = None
        best_score = 0

        for target_key in target_keys:
            target_parts = target_key.split('.')

            # Count matching parts
            score = 0
            for chess_part in chess_parts:
                for target_part in target_parts:
                    if chess_part in target_part or target_part in chess_part:
                        score += 1

            if score > best_score:
                best_score = score
                best_match = target_key

        # Only return match if it has reasonable similarity
        return best_match if best_score >= len(chess_parts) // 2 else None


def load_and_adapt_chess_weights(chess_model_path: Union[str, Path],
                               target_model: nn.Module,
                               target_spatial_dims: Tuple[int, int] = (7, 6)) -> Dict[str, torch.Tensor]:
    """
    Convenience function to load and adapt chess weights in one step.

    Args:
        chess_model_path: Path to chess model file
        target_model: Target PyTorch model for options trading
        target_spatial_dims: Target spatial dimensions (height, width)

    Returns:
        Dictionary of adapted weights ready for loading into target model
    """
    adapter = ChessWeightAdapter(target_spatial_dims)

    # Load chess weights
    chess_weights = adapter.load_chess_weights(chess_model_path)

    # Create weight mapping
    weight_mapping = adapter.create_weight_mapping(chess_weights, target_model)

    # Adapt weights
    adapted_weights = adapter.adapt_conv_layers(chess_weights, weight_mapping)

    # Check compatibility
    compatibility = adapter.weight_compatibility_check(adapted_weights, target_model)

    if compatibility['incompatible_layers']:
        warnings.warn(f"Found {len(compatibility['incompatible_layers'])} incompatible layers")

    return adapted_weights