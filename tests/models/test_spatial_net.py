"""
Tests for SpatialNet integration and complete architecture.

Comprehensive test suite covering SpatialNet, SpatialNetConfig,
and integration with all components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.models.spatial_net import SpatialNet, SpatialNetConfig
from src.models.spatial_encoder import SpatialEncoder, SpatialConfig
from src.models.regime_detector import RegimeDetector
from src.features.position_models import Position, StrategyType, OptionType, PositionZones


class TestSpatialNetConfig:
    """Test cases for SpatialNetConfig class."""

    def test_spatial_net_config_defaults(self):
        """Test SpatialNetConfig default initialization."""
        config = SpatialNetConfig()

        assert config.base_channels == 64
        assert config.num_stages == 3
        assert config.blocks_per_stage == 2
        assert config.attention_heads == 8
        assert config.attention_layers == 2
        assert config.use_multi_scale_attention == True
        assert config.attention_scales == (1, 2)
        assert config.output_features == 512
        assert config.strategy_classes == 16
        assert config.dropout_rate == 0.1

    def test_spatial_net_config_custom(self):
        """Test SpatialNetConfig with custom parameters."""
        custom_spatial_config = SpatialConfig(normalize_output=False)

        config = SpatialNetConfig(
            spatial_config=custom_spatial_config,
            base_channels=32,
            num_stages=4,
            attention_heads=4,
            output_features=256,
            dropout_rate=0.2
        )

        assert config.spatial_config == custom_spatial_config
        assert config.base_channels == 32
        assert config.num_stages == 4
        assert config.attention_heads == 4
        assert config.output_features == 256
        assert config.dropout_rate == 0.2


class TestSpatialNet:
    """Test cases for SpatialNet class."""

    @pytest.fixture
    def mock_regime_detector(self):
        """Create a mock RegimeDetector for testing."""
        regime_detector = Mock(spec=RegimeDetector)
        regime_detector.parameters.return_value = [torch.randn(10, requires_grad=False)]
        return regime_detector

    @pytest.fixture
    def sample_spatial_net(self, mock_regime_detector):
        """Create a sample SpatialNet for testing."""
        config = SpatialNetConfig(
            base_channels=32,  # Smaller for faster testing
            num_stages=2,
            blocks_per_stage=1,
            attention_layers=1,
            output_features=128
        )
        return SpatialNet(regime_detector=mock_regime_detector, config=config)

    @pytest.fixture
    def sample_positions(self):
        """Create sample Position objects for testing."""
        position = Position(
            strategy_type=StrategyType.LONG_CALL,
            strikes=[42000],  # $420 strike
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[500],  # $5.00 entry
            current_prices=[600],  # $6.00 current
            current_underlying_price=42500,  # $425 current underlying
            expiry_date="2024-03-15",
            entry_date="2024-02-01"
        )
        return [position]

    def test_spatial_net_initialization(self, mock_regime_detector):
        """Test SpatialNet initialization."""
        config = SpatialNetConfig(base_channels=32, num_stages=2)
        spatial_net = SpatialNet(regime_detector=mock_regime_detector, config=config)

        assert spatial_net.config == config
        assert spatial_net.regime_detector == mock_regime_detector

        # Check components exist
        assert hasattr(spatial_net, 'spatial_encoder')
        assert hasattr(spatial_net, 'market_encoder')
        assert hasattr(spatial_net, 'feature_extractor')
        assert hasattr(spatial_net, 'attention_layers')
        assert hasattr(spatial_net, 'feature_head')
        assert hasattr(spatial_net, 'classification_head')
        assert hasattr(spatial_net, 'evaluation_head')
        assert hasattr(spatial_net, 'risk_head')

    def test_regime_detector_frozen(self, sample_spatial_net):
        """Test that regime detector parameters are frozen."""
        for param in sample_spatial_net.regime_detector.parameters():
            assert not param.requires_grad

    @patch('src.models.spatial_net.MarketEncoder')
    def test_spatial_net_forward_with_positions(self, mock_market_encoder, sample_spatial_net, sample_positions):
        """Test SpatialNet forward pass with Position objects."""
        # Mock MarketEncoder
        mock_market_encoder_instance = Mock()
        mock_market_encoder_instance.return_value = torch.randn(1, 4, 7, 6)  # Multi-channel output
        mock_market_encoder.return_value = mock_market_encoder_instance

        # Replace market encoder in spatial net
        sample_spatial_net.market_encoder = mock_market_encoder_instance

        # Market features for regime detection
        market_features = torch.randn(1, 48)  # RegimeDetector input size

        # Forward pass
        outputs = sample_spatial_net.forward(sample_positions, market_features)

        # Check output structure
        assert isinstance(outputs, dict)
        assert 'features' in outputs
        assert 'classification' in outputs
        assert 'evaluation' in outputs
        assert 'risk' in outputs

        # Check output shapes
        assert outputs['features'].shape == (1, 128)  # output_features=128
        assert outputs['classification'].shape == (1, 16)  # strategy_classes=16
        assert outputs['evaluation'].shape == (1, 1)  # Single value
        assert outputs['risk'].shape == (1, 3)  # [VaR, CVaR, Max_Drawdown]

    def test_spatial_net_forward_with_tensors(self, sample_spatial_net):
        """Test SpatialNet forward pass with pre-encoded tensors."""
        # Mock MarketEncoder
        mock_market_encoder = Mock()
        mock_market_encoder.return_value = torch.randn(2, 4, 7, 6)
        sample_spatial_net.market_encoder = mock_market_encoder

        # Pre-encoded spatial tensors
        spatial_tensors = torch.randn(2, 1, 7, 6)  # Single channel spatial input
        market_features = torch.randn(2, 48)

        # Forward pass
        outputs = sample_spatial_net.forward(spatial_tensors, market_features)

        # Check batch processing
        assert outputs['features'].shape[0] == 2
        assert outputs['classification'].shape[0] == 2
        assert outputs['evaluation'].shape[0] == 2
        assert outputs['risk'].shape[0] == 2

    def test_spatial_net_forward_with_features(self, sample_spatial_net):
        """Test SpatialNet forward pass with return_features=True."""
        # Mock MarketEncoder
        mock_market_encoder = Mock()
        mock_market_encoder.return_value = torch.randn(1, 4, 7, 6)
        sample_spatial_net.market_encoder = mock_market_encoder

        spatial_tensors = torch.randn(1, 1, 7, 6)
        market_features = torch.randn(1, 48)

        outputs = sample_spatial_net.forward(
            spatial_tensors, market_features, return_features=True
        )

        # Should include additional features
        assert 'spatial_features' in outputs
        assert outputs['spatial_features'].shape[2:] == (7, 6)  # Spatial dimensions preserved

    def test_spatial_net_predict_strategy(self, sample_spatial_net):
        """Test SpatialNet predict_strategy method."""
        # Mock MarketEncoder
        mock_market_encoder = Mock()
        mock_market_encoder.return_value = torch.randn(1, 4, 7, 6)
        sample_spatial_net.market_encoder = mock_market_encoder

        spatial_tensors = torch.randn(1, 1, 7, 6)
        market_features = torch.randn(1, 48)

        predictions = sample_spatial_net.predict_strategy(spatial_tensors, market_features)

        # Check prediction structure
        assert isinstance(predictions, dict)
        assert 'strategy_class' in predictions
        assert 'class_confidence' in predictions
        assert 'expected_return' in predictions
        assert 'value_at_risk' in predictions
        assert 'conditional_var' in predictions
        assert 'max_drawdown' in predictions
        assert 'features' in predictions

        # Check that outputs are numpy arrays
        assert isinstance(predictions['strategy_class'], np.ndarray)
        assert isinstance(predictions['class_confidence'], np.ndarray)
        assert isinstance(predictions['expected_return'], np.ndarray)

    def test_spatial_net_attention_maps(self, sample_spatial_net):
        """Test getting spatial attention maps."""
        # Mock MarketEncoder
        mock_market_encoder = Mock()
        mock_market_encoder.return_value = torch.randn(1, 4, 7, 6)
        sample_spatial_net.market_encoder = mock_market_encoder

        spatial_tensors = torch.randn(1, 1, 7, 6)
        market_features = torch.randn(1, 48)

        attention_maps = sample_spatial_net.get_spatial_attention_maps(
            spatial_tensors, market_features
        )

        # Should return spatial attention maps
        assert attention_maps.shape == (1, 7, 6)  # Batch x Height x Width

    def test_spatial_net_parameter_counting(self, sample_spatial_net):
        """Test parameter counting functionality."""
        param_counts = sample_spatial_net.count_parameters()

        # Check that all components are counted
        expected_components = [
            'spatial_encoder', 'market_encoder', 'feature_extractor',
            'attention_layers', 'feature_head', 'classification_head',
            'evaluation_head', 'risk_head', 'total', 'regime_detector_frozen'
        ]

        for component in expected_components:
            assert component in param_counts
            assert isinstance(param_counts[component], int)
            assert param_counts[component] >= 0

        # Total should be sum of trainable parameters
        assert param_counts['total'] > 0

    def test_spatial_net_gradient_flow(self, sample_spatial_net):
        """Test gradient flow through complete SpatialNet."""
        # Mock MarketEncoder
        mock_market_encoder = Mock()
        mock_market_encoder.return_value = torch.randn(1, 4, 7, 6)
        sample_spatial_net.market_encoder = mock_market_encoder

        spatial_tensors = torch.randn(1, 1, 7, 6, requires_grad=True)
        market_features = torch.randn(1, 48)

        # Forward pass
        outputs = sample_spatial_net.forward(spatial_tensors, market_features)

        # Compute loss
        loss = outputs['features'].sum()
        loss.backward()

        # Check gradients exist
        assert spatial_tensors.grad is not None

        # Check that trainable parameters have gradients
        for param in sample_spatial_net.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_spatial_net_device_handling(self, sample_spatial_net):
        """Test SpatialNet device compatibility."""
        # Test on CPU
        spatial_tensors = torch.randn(1, 1, 7, 6)
        market_features = torch.randn(1, 48)

        # Mock MarketEncoder
        mock_market_encoder = Mock()
        mock_market_encoder.return_value = torch.randn(1, 4, 7, 6)
        sample_spatial_net.market_encoder = mock_market_encoder

        outputs = sample_spatial_net.forward(spatial_tensors, market_features)

        # All outputs should be on CPU
        for key, value in outputs.items():
            if torch.is_tensor(value):
                assert value.device.type == 'cpu'

        # Test GPU if available
        if torch.cuda.is_available():
            sample_spatial_net_gpu = sample_spatial_net.cuda()
            spatial_tensors_gpu = spatial_tensors.cuda()
            market_features_gpu = market_features.cuda()

            # Mock GPU MarketEncoder
            mock_market_encoder_gpu = Mock()
            mock_market_encoder_gpu.return_value = torch.randn(1, 4, 7, 6).cuda()
            sample_spatial_net_gpu.market_encoder = mock_market_encoder_gpu

            outputs_gpu = sample_spatial_net_gpu.forward(spatial_tensors_gpu, market_features_gpu)

            for key, value in outputs_gpu.items():
                if torch.is_tensor(value):
                    assert value.device.type == 'cuda'

    def test_spatial_net_batch_processing(self, sample_spatial_net):
        """Test SpatialNet with different batch sizes."""
        # Mock MarketEncoder
        mock_market_encoder = Mock()

        for batch_size in [1, 4, 8, 16]:
            mock_market_encoder.return_value = torch.randn(batch_size, 4, 7, 6)
            sample_spatial_net.market_encoder = mock_market_encoder

            spatial_tensors = torch.randn(batch_size, 1, 7, 6)
            market_features = torch.randn(batch_size, 48)

            outputs = sample_spatial_net.forward(spatial_tensors, market_features)

            # Check batch dimensions
            assert outputs['features'].shape[0] == batch_size
            assert outputs['classification'].shape[0] == batch_size
            assert outputs['evaluation'].shape[0] == batch_size
            assert outputs['risk'].shape[0] == batch_size

    def test_spatial_net_eval_mode(self, sample_spatial_net):
        """Test SpatialNet behavior in evaluation mode."""
        # Mock MarketEncoder
        mock_market_encoder = Mock()
        mock_market_encoder.return_value = torch.randn(1, 4, 7, 6)
        sample_spatial_net.market_encoder = mock_market_encoder

        spatial_tensors = torch.randn(1, 1, 7, 6)
        market_features = torch.randn(1, 48)

        # Set to eval mode
        sample_spatial_net.eval()

        # Forward passes should be deterministic
        outputs1 = sample_spatial_net.forward(spatial_tensors, market_features)
        outputs2 = sample_spatial_net.forward(spatial_tensors, market_features)

        # Due to dropout, outputs might differ slightly, but should be close
        assert torch.allclose(outputs1['features'], outputs2['features'], atol=1e-3)

    def test_spatial_net_different_configs(self, mock_regime_detector):
        """Test SpatialNet with different configurations."""
        configs = [
            SpatialNetConfig(base_channels=16, num_stages=1, attention_layers=1),
            SpatialNetConfig(base_channels=64, num_stages=4, attention_layers=3),
            SpatialNetConfig(use_multi_scale_attention=False, attention_heads=4),
        ]

        for config in configs:
            spatial_net = SpatialNet(regime_detector=mock_regime_detector, config=config)

            # Mock MarketEncoder
            mock_market_encoder = Mock()
            mock_market_encoder.return_value = torch.randn(1, 4, 7, 6)
            spatial_net.market_encoder = mock_market_encoder

            spatial_tensors = torch.randn(1, 1, 7, 6)
            market_features = torch.randn(1, 48)

            # Should work with all configurations
            outputs = spatial_net.forward(spatial_tensors, market_features)
            assert 'features' in outputs
            assert outputs['features'].shape == (1, config.output_features)


class TestSpatialNetIntegration:
    """Integration tests for SpatialNet with real components."""

    def test_integration_with_real_regime_detector(self):
        """Test SpatialNet integration with real RegimeDetector."""
        regime_detector = RegimeDetector(input_dim=48, hidden_dims=(64, 32, 16))

        config = SpatialNetConfig(
            base_channels=32,
            num_stages=2,
            blocks_per_stage=1,
            attention_layers=1,
            output_features=128
        )

        spatial_net = SpatialNet(regime_detector=regime_detector, config=config)

        # Test forward pass (with mock MarketEncoder)
        mock_market_encoder = Mock()
        mock_market_encoder.return_value = torch.randn(1, 4, 7, 6)
        spatial_net.market_encoder = mock_market_encoder

        spatial_tensors = torch.randn(1, 1, 7, 6)
        market_features = torch.randn(1, 48)

        outputs = spatial_net.forward(spatial_tensors, market_features)

        assert 'features' in outputs
        assert outputs['features'].shape == (1, 128)

    def test_memory_usage_large_batch(self):
        """Test SpatialNet memory usage with large batch."""
        regime_detector = RegimeDetector(input_dim=48)

        config = SpatialNetConfig(
            base_channels=32,  # Keep small for memory efficiency
            num_stages=2,
            blocks_per_stage=1,
            attention_layers=1,
            output_features=64
        )

        spatial_net = SpatialNet(regime_detector=regime_detector, config=config)

        # Mock MarketEncoder
        mock_market_encoder = Mock()
        batch_size = 32
        mock_market_encoder.return_value = torch.randn(batch_size, 4, 7, 6)
        spatial_net.market_encoder = mock_market_encoder

        spatial_tensors = torch.randn(batch_size, 1, 7, 6)
        market_features = torch.randn(batch_size, 48)

        # Should handle large batch without memory errors
        outputs = spatial_net.forward(spatial_tensors, market_features)
        assert outputs['features'].shape == (batch_size, 64)

    def test_end_to_end_strategy_evaluation(self):
        """Test complete end-to-end strategy evaluation pipeline."""
        regime_detector = RegimeDetector()

        config = SpatialNetConfig(
            base_channels=64,
            num_stages=3,
            blocks_per_stage=2,
            attention_layers=2,
            output_features=256
        )

        spatial_net = SpatialNet(regime_detector=regime_detector, config=config)

        # Mock MarketEncoder
        mock_market_encoder = Mock()
        mock_market_encoder.return_value = torch.randn(2, 4, 7, 6)
        spatial_net.market_encoder = mock_market_encoder

        # Create diverse inputs
        spatial_tensors = torch.randn(2, 1, 7, 6)
        market_features = torch.randn(2, 48)

        # Full prediction pipeline
        predictions = spatial_net.predict_strategy(spatial_tensors, market_features)

        # Check all expected outputs
        assert len(predictions['strategy_class']) == 2
        assert len(predictions['expected_return']) == 2
        assert len(predictions['value_at_risk']) == 2
        assert predictions['features'].shape == (2, 256)

        # Values should be reasonable
        assert np.all(predictions['class_confidence'] >= 0)
        assert np.all(predictions['class_confidence'] <= 1)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])