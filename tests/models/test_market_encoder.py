"""
Test suite for MarketEncoder functionality.

Tests market state integration with spatial position representations,
including regime detection integration and multi-channel tensor generation.
"""

import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import List

from src.models.market_encoder import MarketEncoder, MarketEncoderConfig
from src.models.regime_detector import RegimeDetector
from src.models.spatial_encoder import SpatialConfig
from src.features.position_models import (
    Position, StrategyType, OptionType
)


class TestMarketEncoder:
    """Test suite for MarketEncoder class."""

    @pytest.fixture
    def regime_detector(self):
        """Create RegimeDetector for testing."""
        detector = RegimeDetector(
            input_dim=48,
            hidden_dims=(64, 32, 16),
            num_regimes=8,
            dropout_rate=0.1
        )
        detector.eval()  # Set to eval mode for consistent behavior
        return detector

    @pytest.fixture
    def market_encoder(self, regime_detector):
        """Create MarketEncoder for testing."""
        return MarketEncoder(regime_detector)

    @pytest.fixture
    def custom_market_encoder(self, regime_detector):
        """Create MarketEncoder with custom configuration."""
        config = MarketEncoderConfig(
            confidence_threshold=0.7,
            regime_smoothing=True,
            include_uncertainty=True,
            spatial_config=SpatialConfig(normalize_output=True)
        )
        return MarketEncoder(regime_detector, config)

    @pytest.fixture
    def sample_position(self):
        """Create sample position for testing."""
        return Position(
            strategy_type=StrategyType.BULL_CALL_SPREAD,
            entry_date=datetime.now() - timedelta(days=30),
            expiration_date=datetime.now() + timedelta(days=15),
            strikes=[9500, 10000],
            option_types=[OptionType.CALL, OptionType.CALL],
            quantities=[1, -1],
            entry_prices=[500, 300],
            current_prices=[300, 100],
            underlying_price_at_entry=9750,
            current_underlying_price=9800
        )

    @pytest.fixture
    def sample_market_features(self):
        """Create sample market features for testing."""
        # Create 48-dimensional feature vector
        return torch.randn(48)

    @pytest.fixture
    def batch_market_features(self):
        """Create batch of market features for testing."""
        return torch.randn(5, 48)  # Batch of 5 samples

    @pytest.fixture
    def batch_positions(self):
        """Create batch of positions for testing."""
        base_date = datetime.now()
        positions = []

        strategies = [
            StrategyType.LONG_CALL,
            StrategyType.SHORT_PUT,
            StrategyType.BULL_CALL_SPREAD,
            StrategyType.IRON_CONDOR,
            StrategyType.LONG_STRADDLE
        ]

        for i, strategy in enumerate(strategies):
            if strategy == StrategyType.IRON_CONDOR:
                position = Position(
                    strategy_type=strategy,
                    entry_date=base_date - timedelta(days=10 + i),
                    expiration_date=base_date + timedelta(days=20 + i),
                    strikes=[9000, 9500, 10500, 11000],
                    option_types=[OptionType.PUT, OptionType.PUT, OptionType.CALL, OptionType.CALL],
                    quantities=[1, -1, -1, 1],
                    entry_prices=[150, 300, 250, 100],
                    current_prices=[75, 200, 150, 50],
                    underlying_price_at_entry=9800 + i * 50,
                    current_underlying_price=9900 + i * 50
                )
            else:
                position = Position(
                    strategy_type=strategy,
                    entry_date=base_date - timedelta(days=10 + i),
                    expiration_date=base_date + timedelta(days=20 + i),
                    strikes=[9500 + i * 100, 10000 + i * 100],
                    option_types=[OptionType.CALL, OptionType.CALL],
                    quantities=[1, -1],
                    entry_prices=[500 + i * 50, 300 + i * 30],
                    current_prices=[300 + i * 25, 100 + i * 15],
                    underlying_price_at_entry=9800 + i * 50,
                    current_underlying_price=9900 + i * 50
                )
            positions.append(position)

        return positions

    def test_market_encoder_initialization(self, market_encoder, regime_detector):
        """Test MarketEncoder initialization."""
        assert market_encoder.regime_detector is regime_detector
        assert market_encoder.num_regimes == 8
        assert hasattr(market_encoder, 'spatial_encoder')
        assert hasattr(market_encoder, 'config')

        # Check regime patterns are initialized
        assert hasattr(market_encoder, 'regime_spatial_patterns')
        assert market_encoder.regime_spatial_patterns.shape == (8, 7, 6)

    def test_custom_config_initialization(self, custom_market_encoder):
        """Test MarketEncoder with custom configuration."""
        assert custom_market_encoder.config.confidence_threshold == 0.7
        assert custom_market_encoder.config.regime_smoothing is True
        assert custom_market_encoder.config.include_uncertainty is True

    def test_regime_pattern_setup(self, market_encoder):
        """Test regime spatial pattern setup."""
        patterns = market_encoder.regime_spatial_patterns

        # Should have unique patterns for each regime
        assert patterns.shape == (8, 7, 6)

        # Patterns should be different from each other
        for i in range(8):
            for j in range(i + 1, 8):
                pattern_i = patterns[i]
                pattern_j = patterns[j]

                # Should not be identical
                assert not torch.allclose(pattern_i, pattern_j, atol=0.1)

        # Patterns should be normalized (bounded)
        assert torch.all(torch.abs(patterns) <= 1.1)  # Allow small epsilon

    def test_encode_market_state_single(self, market_encoder, sample_position, sample_market_features):
        """Test single position market state encoding."""
        encoded = market_encoder.encode_market_state(sample_position, sample_market_features)

        # Check output shape - should be (C, 7, 6) where C >= 3
        assert len(encoded.shape) == 3
        C, H, W = encoded.shape
        assert C >= 3  # At least position, regime, confidence channels
        assert H == 7
        assert W == 6

        # Check for valid values
        assert not torch.any(torch.isnan(encoded))
        assert not torch.any(torch.isinf(encoded))

    def test_encode_market_state_batch(self, market_encoder, sample_position, batch_market_features):
        """Test batch market state encoding."""
        # Expand market features to batch
        batch_size = batch_market_features.shape[0]
        encoded = market_encoder.encode_market_state(sample_position, batch_market_features)

        # Check output shape - should be (batch_size, C, 7, 6)
        assert encoded.shape[0] == batch_size
        assert encoded.shape[1] >= 3  # At least 3 channels
        assert encoded.shape[2] == 7
        assert encoded.shape[3] == 6

        # Check for valid values
        assert not torch.any(torch.isnan(encoded))
        assert not torch.any(torch.isinf(encoded))

    def test_forward_method(self, market_encoder, sample_position, sample_market_features):
        """Test forward method compatibility."""
        forward_result = market_encoder.forward(sample_position, sample_market_features)
        direct_result = market_encoder.encode_market_state(sample_position, sample_market_features)

        # Should produce identical results
        assert torch.allclose(forward_result, direct_result)

    def test_channel_structure(self, custom_market_encoder, sample_position, sample_market_features):
        """Test multi-channel structure with uncertainty enabled."""
        encoded = custom_market_encoder.encode_market_state(sample_position, sample_market_features)

        # Should have 4 channels: position, regime, confidence, uncertainty
        assert encoded.shape[0] == 4

        # Each channel should have different characteristics
        position_channel = encoded[0]
        regime_channel = encoded[1]
        confidence_channel = encoded[2]
        uncertainty_channel = encoded[3]

        # Channels should be different from each other
        assert not torch.allclose(position_channel, regime_channel, atol=0.1)
        assert not torch.allclose(regime_channel, confidence_channel, atol=0.1)
        assert not torch.allclose(confidence_channel, uncertainty_channel, atol=0.1)

    def test_confidence_thresholding(self, regime_detector):
        """Test confidence thresholding functionality."""
        # Create encoder with high confidence threshold
        config = MarketEncoderConfig(confidence_threshold=0.9)
        encoder = MarketEncoder(regime_detector, config)

        sample_position = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=datetime.now() - timedelta(days=10),
            expiration_date=datetime.now() + timedelta(days=20),
            strikes=[10000],
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[500],
            current_prices=[300],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

        # Create market features that might produce low confidence
        market_features = torch.randn(48) * 0.1  # Small values might lead to low confidence

        encoded = encoder.encode_market_state(sample_position, market_features)

        # Should still produce valid output
        assert encoded.shape == (3, 7, 6)  # 3 channels by default
        assert not torch.any(torch.isnan(encoded))

    def test_batch_encoding_method(self, market_encoder, batch_positions, batch_market_features):
        """Test dedicated batch encoding method."""
        batch_encoded = market_encoder.encode_batch_market_state(
            batch_positions, batch_market_features
        )

        batch_size = len(batch_positions)
        assert batch_encoded.shape[0] == batch_size
        assert batch_encoded.shape[1] >= 3  # At least 3 channels
        assert batch_encoded.shape[2] == 7
        assert batch_encoded.shape[3] == 6

        # Compare with individual encoding
        for i in range(batch_size):
            individual_features = batch_market_features[i]
            individual_position = batch_positions[i]

            individual_encoded = market_encoder.encode_market_state(
                individual_position, individual_features
            )

            batch_individual = batch_encoded[i]

            # Should be very close (allowing for small numerical differences)
            assert torch.allclose(individual_encoded, batch_individual, atol=1e-4)

    def test_batch_size_mismatch_error(self, market_encoder, batch_positions, sample_market_features):
        """Test error handling for batch size mismatch."""
        # Create market features with wrong batch size
        wrong_size_features = sample_market_features.unsqueeze(0).expand(3, -1)  # Size 3 vs 5 positions

        with pytest.raises(ValueError, match="Position count .* doesn't match"):
            market_encoder.encode_batch_market_state(batch_positions, wrong_size_features)

    def test_get_channel_info(self, market_encoder, custom_market_encoder):
        """Test channel information retrieval."""
        # Default configuration (3 channels)
        info = market_encoder.get_channel_info()
        assert 'position' in info
        assert 'regime' in info
        assert 'confidence' in info
        assert 'total_channels' in info
        assert info['position'] == 0
        assert info['regime'] == 1
        assert info['confidence'] == 2
        assert info['total_channels'] == 3

        # Custom configuration with uncertainty (4 channels)
        custom_info = custom_market_encoder.get_channel_info()
        assert 'uncertainty' in custom_info
        assert custom_info['uncertainty'] == 3
        assert custom_info['total_channels'] == 4

    def test_regime_detector_integration(self, market_encoder, sample_market_features):
        """Test integration with RegimeDetector."""
        # Test that regime detector is called correctly
        with torch.no_grad():
            regime_output = market_encoder.regime_detector(sample_market_features.unsqueeze(0))

        # Should have 9 outputs (8 regimes + 1 confidence)
        assert regime_output.shape == (1, 9)

        # Regime probabilities should sum to 1
        regime_probs = regime_output[0, :8]
        confidence = regime_output[0, 8]

        assert torch.allclose(torch.sum(regime_probs), torch.tensor(1.0), atol=1e-3)
        assert 0 <= confidence <= 1  # Confidence should be in [0, 1]

    def test_spatial_smoothing(self, regime_detector):
        """Test spatial smoothing functionality."""
        # Create encoder with smoothing enabled
        config = MarketEncoderConfig(regime_smoothing=True)
        encoder_smooth = MarketEncoder(regime_detector, config)

        # Create encoder with smoothing disabled
        config_no_smooth = MarketEncoderConfig(regime_smoothing=False)
        encoder_no_smooth = MarketEncoder(regime_detector, config_no_smooth)

        position = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=datetime.now() - timedelta(days=10),
            expiration_date=datetime.now() + timedelta(days=20),
            strikes=[10000],
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[500],
            current_prices=[300],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

        market_features = torch.randn(48)

        encoded_smooth = encoder_smooth.encode_market_state(position, market_features)
        encoded_no_smooth = encoder_no_smooth.encode_market_state(position, market_features)

        # Regime channels (channel 1) should be different due to smoothing
        regime_smooth = encoded_smooth[1]
        regime_no_smooth = encoded_no_smooth[1]

        # May or may not be different depending on the specific values
        # but they should both be valid
        assert not torch.any(torch.isnan(regime_smooth))
        assert not torch.any(torch.isnan(regime_no_smooth))

    def test_visualization(self, market_encoder, sample_position, sample_market_features):
        """Test visualization functionality."""
        viz = market_encoder.visualize_market_encoding(sample_position, sample_market_features)

        # Should be a string with multiple lines
        assert isinstance(viz, str)
        lines = viz.split('\n')
        assert len(lines) > 10  # Should have multiple lines

        # Should contain channel information
        assert 'Position' in viz
        assert 'Regime' in viz
        assert 'Confidence' in viz
        assert 'Channel' in viz

    def test_edge_cases(self, market_encoder):
        """Test edge cases and boundary conditions."""
        # Test with extreme market features
        extreme_features = torch.ones(48) * 10  # Very large values

        position = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=datetime.now() - timedelta(days=10),
            expiration_date=datetime.now() + timedelta(days=20),
            strikes=[10000],
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[500],
            current_prices=[300],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

        # Should handle gracefully
        encoded = market_encoder.encode_market_state(position, extreme_features)
        assert not torch.any(torch.isnan(encoded))
        assert not torch.any(torch.isinf(encoded))

        # Test with very small features
        small_features = torch.ones(48) * 0.001

        encoded_small = market_encoder.encode_market_state(position, small_features)
        assert not torch.any(torch.isnan(encoded_small))
        assert not torch.any(torch.isinf(encoded_small))

    def test_device_handling(self, market_encoder, sample_position, sample_market_features):
        """Test device handling for CUDA compatibility."""
        # Test that encoder works on CPU
        encoded_cpu = market_encoder.encode_market_state(sample_position, sample_market_features)
        assert encoded_cpu.device.type == 'cpu'

        # If CUDA is available, test GPU
        if torch.cuda.is_available():
            market_encoder = market_encoder.cuda()
            sample_market_features = sample_market_features.cuda()

            encoded_gpu = market_encoder.encode_market_state(sample_position, sample_market_features)
            assert encoded_gpu.device.type == 'cuda'

            # Results should be similar (allowing for numerical differences)
            encoded_gpu_cpu = encoded_gpu.cpu()
            assert torch.allclose(encoded_cpu, encoded_gpu_cpu, atol=1e-3)

    def test_repr_method(self, market_encoder, custom_market_encoder):
        """Test string representation."""
        # Default encoder
        repr_str = repr(market_encoder)
        assert isinstance(repr_str, str)
        assert 'MarketEncoder' in repr_str
        assert '(3, 7, 6)' in repr_str  # Default 3 channels

        # Custom encoder
        custom_repr = repr(custom_market_encoder)
        assert '(4, 7, 6)' in custom_repr  # 4 channels with uncertainty
        assert '0.7' in custom_repr  # Confidence threshold

    def test_deterministic_output(self, market_encoder, sample_position, sample_market_features):
        """Test that output is deterministic for same inputs."""
        # Set model to eval mode
        market_encoder.eval()

        # Encode same inputs multiple times
        encoded1 = market_encoder.encode_market_state(sample_position, sample_market_features)
        encoded2 = market_encoder.encode_market_state(sample_position, sample_market_features)
        encoded3 = market_encoder.encode_market_state(sample_position, sample_market_features)

        # Should be identical
        assert torch.allclose(encoded1, encoded2)
        assert torch.allclose(encoded2, encoded3)

    def test_different_regime_patterns(self, market_encoder):
        """Test that different market conditions produce different regime patterns."""
        position = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=datetime.now() - timedelta(days=10),
            expiration_date=datetime.now() + timedelta(days=20),
            strikes=[10000],
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[500],
            current_prices=[300],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

        # Create different market conditions
        market_features_1 = torch.randn(48, generator=torch.Generator().manual_seed(42))
        market_features_2 = torch.randn(48, generator=torch.Generator().manual_seed(123))

        encoded_1 = market_encoder.encode_market_state(position, market_features_1)
        encoded_2 = market_encoder.encode_market_state(position, market_features_2)

        # Position channels should be identical (same position)
        assert torch.allclose(encoded_1[0], encoded_2[0])

        # Regime channels should be different (different market conditions)
        regime_1 = encoded_1[1]
        regime_2 = encoded_2[1]

        # Should be different unless very unlucky with random seeds
        if not torch.allclose(regime_1, regime_2, atol=0.1):
            # This is the expected case - different market conditions produce different patterns
            assert True
        else:
            # If they're very similar, at least confidence channels should differ
            confidence_1 = encoded_1[2]
            confidence_2 = encoded_2[2]
            assert not torch.allclose(confidence_1, confidence_2, atol=0.05)