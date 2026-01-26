"""
Complete spatial neural network integrating spatial encoder, residual blocks, and attention.

Implements SpatialNet class that combines all components for chess-inspired
options strategy analysis with deep learning architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from .spatial_encoder import SpatialEncoder, SpatialConfig
from .market_encoder import MarketEncoder
from .residual_blocks import ResidualStack, ChessInspiredFeatureExtractor
from .attention import SpatialAttention, SpatialAttentionBlock, MultiScaleSpatialAttention
from .regime_detector import RegimeDetector
from ..features.position_models import Position


class SpatialNetConfig:
    """Configuration class for SpatialNet architecture."""

    def __init__(self,
                 # Spatial encoding config
                 spatial_config: Optional[SpatialConfig] = None,

                 # Feature extraction config
                 base_channels: int = 64,
                 num_stages: int = 3,
                 blocks_per_stage: int = 2,

                 # Attention config
                 attention_heads: int = 8,
                 attention_layers: int = 2,
                 use_multi_scale_attention: bool = True,
                 attention_scales: Tuple[int, ...] = (1, 2),

                 # Output config
                 output_features: int = 512,
                 strategy_classes: int = 16,  # Number of strategy types

                 # Regularization
                 dropout_rate: float = 0.1):
        """
        Initialize SpatialNet configuration.

        Args:
            spatial_config: Configuration for spatial encoding
            base_channels: Base number of channels for feature extraction
            num_stages: Number of feature extraction stages
            blocks_per_stage: Number of residual blocks per stage
            attention_heads: Number of attention heads
            attention_layers: Number of attention layers
            use_multi_scale_attention: Whether to use multi-scale attention
            attention_scales: Scales for multi-scale attention
            output_features: Final feature dimension
            strategy_classes: Number of strategy classification classes
            dropout_rate: Dropout rate for regularization
        """
        self.spatial_config = spatial_config or SpatialConfig()
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.blocks_per_stage = blocks_per_stage
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.use_multi_scale_attention = use_multi_scale_attention
        self.attention_scales = attention_scales
        self.output_features = output_features
        self.strategy_classes = strategy_classes
        self.dropout_rate = dropout_rate


class SpatialNet(nn.Module):
    """
    Complete spatial neural network for chess-inspired options strategy analysis.

    Architecture:
    1. SpatialEncoder: Position -> 7x6 spatial tensor
    2. MarketEncoder: Spatial + regime context -> multi-channel tensor
    3. ResidualBlocks: Deep feature extraction with chess-inspired architecture
    4. SpatialAttention: Self-attention for spatial relationship modeling
    5. Output layers: Strategy evaluation, classification, and feature extraction
    """

    def __init__(self,
                 regime_detector: RegimeDetector,
                 config: Optional[SpatialNetConfig] = None):
        """
        Initialize SpatialNet.

        Args:
            regime_detector: Pre-trained regime detector for market context
            config: Network configuration (uses defaults if None)
        """
        super(SpatialNet, self).__init__()

        self.config = config or SpatialNetConfig()
        self.regime_detector = regime_detector

        # Freeze regime detector (pre-trained)
        for param in self.regime_detector.parameters():
            param.requires_grad = False

        # Core components
        self.spatial_encoder = SpatialEncoder(self.config.spatial_config)
        self.market_encoder = MarketEncoder(regime_detector=regime_detector)

        # Calculate feature extractor input channels
        # MarketEncoder outputs 4 channels: [Position, Regime, Confidence, Uncertainty]
        market_encoder_channels = 4

        # Feature extraction with residual blocks
        self.feature_extractor = ChessInspiredFeatureExtractor(
            input_channels=market_encoder_channels,
            base_channels=self.config.base_channels,
            num_stages=self.config.num_stages,
            blocks_per_stage=self.config.blocks_per_stage,
            output_features=None  # Keep spatial features for attention
        )

        # Calculate channels after feature extraction
        # Final stage channels = base_channels * (2 ** (num_stages - 1))
        final_feature_channels = self.config.base_channels * (2 ** (self.config.num_stages - 1))

        # Attention layers
        self.attention_layers = nn.ModuleList()

        if self.config.use_multi_scale_attention:
            # Multi-scale attention for first layer
            self.attention_layers.append(
                MultiScaleSpatialAttention(
                    in_channels=final_feature_channels,
                    scales=self.config.attention_scales,
                    num_heads=self.config.attention_heads,
                    dropout_rate=self.config.dropout_rate
                )
            )

            # Regular attention blocks for remaining layers
            for _ in range(self.config.attention_layers - 1):
                self.attention_layers.append(
                    SpatialAttentionBlock(
                        in_channels=final_feature_channels,
                        num_heads=self.config.attention_heads,
                        dropout_rate=self.config.dropout_rate
                    )
                )
        else:
            # Use regular attention blocks for all layers
            for _ in range(self.config.attention_layers):
                self.attention_layers.append(
                    SpatialAttentionBlock(
                        in_channels=final_feature_channels,
                        num_heads=self.config.attention_heads,
                        dropout_rate=self.config.dropout_rate
                    )
                )

        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Output heads
        self.feature_head = nn.Sequential(
            nn.Linear(final_feature_channels, self.config.output_features),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.LayerNorm(self.config.output_features)
        )

        # Strategy classification head
        self.classification_head = nn.Linear(self.config.output_features, self.config.strategy_classes)

        # Strategy evaluation head (regression for P/L estimation)
        self.evaluation_head = nn.Sequential(
            nn.Linear(self.config.output_features, 256),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(256, 1)  # Single value: expected return
        )

        # Risk assessment head
        self.risk_head = nn.Sequential(
            nn.Linear(self.config.output_features, 256),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(256, 3)  # [VaR, CVaR, Max_Drawdown]
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self,
                positions: torch.Tensor,
                market_features: torch.Tensor,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SpatialNet.

        Args:
            positions: Batch of Position objects or pre-encoded spatial tensors
            market_features: Market feature vectors for regime detection
            return_features: Whether to return intermediate features

        Returns:
            Dictionary with model outputs:
            - 'features': Global feature representations
            - 'classification': Strategy type probabilities
            - 'evaluation': Expected return estimation
            - 'risk': Risk metrics [VaR, CVaR, Max_Drawdown]
            - 'attention_weights': Attention weights (if return_features=True)
            - 'spatial_features': Spatial feature maps (if return_features=True)
        """
        batch_size = positions.shape[0] if torch.is_tensor(positions) else len(positions)

        # 1. Spatial encoding (if needed)
        if not torch.is_tensor(positions):
            # Positions is a list of Position objects
            spatial_tensors = []
            for position in positions:
                spatial_tensor = self.spatial_encoder(position)
                spatial_tensors.append(spatial_tensor)
            spatial_input = torch.stack(spatial_tensors, dim=0)
        else:
            # Positions is already a tensor
            spatial_input = positions

        # 2. Market encoding with regime context
        market_encoded = self.market_encoder(spatial_input, market_features)

        # 3. Deep feature extraction with residual blocks
        spatial_features = self.feature_extractor(market_encoded)

        # 4. Attention processing
        attention_weights = None
        for i, attention_layer in enumerate(self.attention_layers):
            if i == 0 and return_features and hasattr(attention_layer, 'get_attention_weights'):
                # Store attention weights from first layer for visualization
                attention_weights = attention_layer.get_attention_weights(spatial_features)

            spatial_features = attention_layer(spatial_features)

        # 5. Global feature extraction
        global_features = self.global_pool(spatial_features)  # (batch, channels, 1, 1)
        global_features = global_features.view(batch_size, -1)  # (batch, channels)

        # 6. Feature projection
        features = self.feature_head(global_features)

        # 7. Output heads
        classification_logits = self.classification_head(features)
        classification_probs = F.softmax(classification_logits, dim=1)

        evaluation = self.evaluation_head(features)
        risk_metrics = self.risk_head(features)

        # Prepare outputs
        outputs = {
            'features': features,
            'classification': classification_probs,
            'evaluation': evaluation,
            'risk': risk_metrics
        }

        if return_features:
            outputs['spatial_features'] = spatial_features
            if attention_weights is not None:
                outputs['attention_weights'] = attention_weights

        return outputs

    def predict_strategy(self,
                        positions: torch.Tensor,
                        market_features: torch.Tensor) -> Dict[str, Any]:
        """
        Predict strategy properties with interpretable outputs.

        Args:
            positions: Position data
            market_features: Market feature vectors

        Returns:
            Dictionary with predictions and confidence scores
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(positions, market_features, return_features=True)

            # Get predicted strategy class
            class_probs = outputs['classification']
            predicted_class = torch.argmax(class_probs, dim=1)
            class_confidence = torch.max(class_probs, dim=1)[0]

            # Extract risk metrics
            risk_metrics = outputs['risk']
            var_estimate = risk_metrics[:, 0]
            cvar_estimate = risk_metrics[:, 1]
            max_drawdown = risk_metrics[:, 2]

            predictions = {
                'strategy_class': predicted_class.cpu().numpy(),
                'class_confidence': class_confidence.cpu().numpy(),
                'expected_return': outputs['evaluation'].squeeze().cpu().numpy(),
                'value_at_risk': var_estimate.cpu().numpy(),
                'conditional_var': cvar_estimate.cpu().numpy(),
                'max_drawdown': max_drawdown.cpu().numpy(),
                'features': outputs['features'].cpu().numpy()
            }

            if 'attention_weights' in outputs:
                predictions['attention_weights'] = outputs['attention_weights'].cpu().numpy()

        return predictions

    def get_spatial_attention_maps(self,
                                  positions: torch.Tensor,
                                  market_features: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial attention maps for visualization.

        Args:
            positions: Position data
            market_features: Market features

        Returns:
            Attention maps showing which spatial regions are important
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(positions, market_features, return_features=True)

            if 'attention_weights' in outputs:
                # Average attention across heads and reshape to spatial format
                attention_weights = outputs['attention_weights']  # (batch, heads, seq_len, seq_len)

                # Average across heads
                avg_attention = attention_weights.mean(dim=1)  # (batch, seq_len, seq_len)

                # Get self-attention (diagonal attention to each position)
                batch_size = avg_attention.shape[0]
                spatial_attention = torch.zeros(batch_size, 7, 6)

                for b in range(batch_size):
                    for i in range(42):  # 7*6 = 42 positions
                        row, col = divmod(i, 6)
                        # Sum attention from this position to all others
                        spatial_attention[b, row, col] = avg_attention[b, i, :].sum()

                return spatial_attention
            else:
                # Return dummy attention if not available
                return torch.zeros(positions.shape[0], 7, 6)

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in different components."""
        param_counts = {}

        # Count parameters in each component
        param_counts['spatial_encoder'] = sum(p.numel() for p in self.spatial_encoder.parameters())
        param_counts['market_encoder'] = sum(p.numel() for p in self.market_encoder.parameters())
        param_counts['feature_extractor'] = sum(p.numel() for p in self.feature_extractor.parameters())

        attention_params = 0
        for layer in self.attention_layers:
            attention_params += sum(p.numel() for p in layer.parameters())
        param_counts['attention_layers'] = attention_params

        param_counts['feature_head'] = sum(p.numel() for p in self.feature_head.parameters())
        param_counts['classification_head'] = sum(p.numel() for p in self.classification_head.parameters())
        param_counts['evaluation_head'] = sum(p.numel() for p in self.evaluation_head.parameters())
        param_counts['risk_head'] = sum(p.numel() for p in self.risk_head.parameters())

        param_counts['total'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        param_counts['regime_detector_frozen'] = sum(p.numel() for p in self.regime_detector.parameters())

        return param_counts

    def __repr__(self) -> str:
        """String representation of SpatialNet."""
        param_counts = self.count_parameters()

        return (f"SpatialNet(\n"
                f"  Architecture: Spatial -> Residual -> Attention -> Output\n"
                f"  Input: Positions + Market Features -> 7x6 spatial tensors\n"
                f"  Feature extraction: {self.config.num_stages} stages, {self.config.base_channels} base channels\n"
                f"  Attention: {self.config.attention_layers} layers, {self.config.attention_heads} heads\n"
                f"  Output features: {self.config.output_features}\n"
                f"  Strategy classes: {self.config.strategy_classes}\n"
                f"  Total trainable parameters: {param_counts['total']:,}\n"
                f"  Regime detector parameters (frozen): {param_counts['regime_detector_frozen']:,}\n"
                f")")