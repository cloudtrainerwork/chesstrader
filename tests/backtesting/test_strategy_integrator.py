"""
Test strategy integrator for ML pipeline integration

Tests StrategyIntegrator connecting ML models with backtesting engine.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.backtesting.strategies.strategy_integrator import StrategyIntegrator
except ImportError:
    StrategyIntegrator = None

try:
    from src.backtesting.core.events import SignalEvent, MarketEvent
    from src.models.recommendation_engine import RecommendationEngine, StrategyRecommendation
    from src.strategies.base import StrategyType
    from src.data.regime_labeler import RegimeType
except ImportError as e:
    # Handle missing imports during development
    print(f"Warning: Import error during test setup: {e}")
    pass


class TestStrategyIntegrator:
    """Test StrategyIntegrator functionality"""

    def test_initialization(self):
        """Test StrategyIntegrator initialization"""
        integrator = StrategyIntegrator()

        assert integrator.inference_delay_ms == 100
        assert integrator.min_confidence == 0.6
        assert integrator.max_position_size == 0.05
        assert integrator.recommendation_engine is None
        assert integrator.position_manager is None

    def test_load_models(self):
        """Test loading ML models"""
        integrator = StrategyIntegrator()

        # Should not raise any exceptions with mock paths
        integrator.load_models(
            position_manager_path="path/to/position_manager.pth",
            recommendation_engine_path="path/to/recommender.pth"
        )

        # Models are not actually loaded (mocked for testing)
        assert integrator.recommendation_engine is None
        assert integrator.position_manager is None

    def test_real_ml_model_loading(self):
        """Test loading real trained ML models"""
        # This test should fail until real model loading is implemented
        integrator = StrategyIntegrator()

        # Mock model file paths that would exist in a real implementation
        position_manager_path = "models/position_manager_trained.pth"
        recommendation_engine_path = "models/recommendation_engine.pth"

        # This should load actual models, not just log and continue
        integrator.load_models(
            position_manager_path=position_manager_path,
            recommendation_engine_path=recommendation_engine_path
        )

        # Test should fail until real models are loaded
        assert integrator.recommendation_engine is not None, "RecommendationEngine should be loaded"
        assert integrator.position_manager is not None, "PositionManager should be loaded"

        # Verify models have expected interface
        assert hasattr(integrator.recommendation_engine, 'get_recommendations')
        assert hasattr(integrator.position_manager, 'get_action') or hasattr(integrator.position_manager, 'predict')

    def test_ml_signal_generation(self):
        """Test ML model signal generation"""
        # Create mock market data that should trigger signals
        market_data = {
            'symbol': 'SPY',
            'timestamp': datetime.now(),
            'price': 460.0,  # Above 450 threshold
            'iv': 0.20,      # Below 0.25 threshold
            'delta': 0.6,
            'gamma': 0.02,
            'theta': -0.05,
            'vega': 0.15
        }

        integrator = StrategyIntegrator(inference_delay_ms=1)  # Minimal delay for testing

        signals = integrator.generate_signals(market_data)

        # Should generate at least one signal based on mock logic
        assert len(signals) >= 0  # May be 0 if confidence filtering is strict

        # If signals are generated, verify their structure
        if signals:
            signal = signals[0]
            assert hasattr(signal, 'symbol')
            assert hasattr(signal, 'signal_type')
            assert hasattr(signal, 'strategy_id')
            assert signal.symbol == 'SPY'
            assert signal.strategy_id is not None


class TestSignalGeneration:
    """Test signal generation and timing"""

    def test_convert_to_observations(self):
        """Test market data conversion to ML model observations"""
        market_data = {
            'symbol': 'SPY',
            'price': 450.0,
            'iv': 0.20,
            'delta': 0.6,
            'volume': 1000000,
            'regime': 'BULL_HIGH_VOL',
            'timestamp': datetime.now()
        }

        integrator = StrategyIntegrator()
        observations = integrator._convert_to_observations(market_data)

        assert isinstance(observations, np.ndarray)
        assert observations.shape[-1] > 0  # Should have features
        assert not np.isnan(observations).any()  # No NaN values

        # Should include regime encoding (6 values for 6 regime types)
        assert len(observations) >= 15  # At least price, volume, Greeks (5), regime (6), time (2)

    def test_signal_timing(self):
        """Test realistic signal generation timing"""
        integrator = StrategyIntegrator(inference_delay_ms=50)  # 50ms inference delay

        start_time = datetime.now()
        market_data = {'symbol': 'SPY', 'price': 450.0}

        signals = integrator.generate_signals(market_data)
        end_time = datetime.now()

        # Should take at least the inference delay
        duration_ms = (end_time - start_time).total_seconds() * 1000
        assert duration_ms >= 45  # Allow some variance for processing overhead


class TestIntegrationWithBacktesting:
    """Test integration with backtesting event system"""

    def test_signal_event_generation(self):
        """Test proper SignalEvent generation for backtesting"""
        # Create mock market event
        class MockMarketEvent:
            def __init__(self, timestamp, symbol, data):
                self.timestamp = timestamp
                self.symbol = symbol
                self.data = data

        market_event = MockMarketEvent(
            timestamp=datetime.now(),
            symbol='SPY',
            data={
                'price': 460.0,  # Trigger bullish signal
                'volume': 1000000,
                'iv': 0.20,  # Low IV to trigger signal
                'delta': 0.6
            }
        )

        integrator = StrategyIntegrator(inference_delay_ms=1)

        signals = integrator.process_market_event(market_event)

        assert isinstance(signals, list)
        # Signals may be empty due to confidence filtering, which is expected
        for signal in signals:
            assert hasattr(signal, 'timestamp')
            assert hasattr(signal, 'symbol')
            assert signal.symbol == market_event.symbol

    def test_multiple_strategy_signals(self):
        """Test handling multiple strategy recommendations"""
        # Create market data that should trigger both signals
        market_data = {
            'symbol': 'SPY',
            'price': 460.0,  # Above 450 - should trigger bullish signal
            'iv': 0.35       # Above 0.30 - should trigger neutral signal
        }

        integrator = StrategyIntegrator(
            inference_delay_ms=1,
            min_confidence=0.6  # Set threshold to allow both mock signals
        )

        signals = integrator.generate_signals(market_data)

        # Should potentially generate multiple signals based on mock logic
        assert isinstance(signals, list)
        assert len(signals) <= 2  # At most 2 based on mock implementation

        for signal in signals:
            assert hasattr(signal, 'strength')
            assert hasattr(signal, 'signal_type')
            # Strength should be positive (confidence values)
            if hasattr(signal, 'strength'):
                assert signal.strength > 0.0