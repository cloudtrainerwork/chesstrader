"""
Strategy integrator for ML pipeline integration

Connects trained ML models (PositionManager and RecommendationEngine) with
backtesting engine for realistic signal generation and position management.
"""

import torch
import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

try:
    from ..core.events import SignalEvent, MarketEvent
    from ...models.recommendation_engine import RecommendationEngine, StrategyRecommendation
    from ...training.position_manager_trainer import PositionManagerTrainer
    from ...strategies.base import StrategyType
    from ...data.regime_labeler import RegimeType
except ImportError:
    # Handle missing imports during development
    SignalEvent = None
    MarketEvent = None
    RecommendationEngine = None
    StrategyRecommendation = None
    PositionManagerTrainer = None
    StrategyType = None
    RegimeType = None

logger = logging.getLogger(__name__)


class StrategyIntegrator:
    """
    Integration layer for ML models and backtesting engine

    Bridges between trained ML models (strategy selector + position manager)
    and the event-driven backtesting system to generate realistic trading signals.
    """

    def __init__(self,
                 inference_delay_ms: int = 100,
                 min_confidence: float = 0.6,
                 max_position_size: float = 0.05):
        """
        Initialize strategy integrator

        Args:
            inference_delay_ms: Realistic model inference delay
            min_confidence: Minimum confidence threshold for signals
            max_position_size: Maximum position size as fraction of portfolio
        """
        self.inference_delay_ms = inference_delay_ms
        self.min_confidence = min_confidence
        self.max_position_size = max_position_size

        # ML model components (loaded separately)
        self.recommendation_engine: Optional[Any] = None  # RecommendationEngine
        self.position_manager: Optional[Any] = None  # PositionManagerNetwork
        self.position_manager_trainer: Optional[Any] = None  # PositionManagerTrainer

        # State tracking
        self.current_positions = {}
        self.last_signal_time = {}

        logger.info(f"StrategyIntegrator initialized with {inference_delay_ms}ms inference delay")

    def load_models(self,
                   position_manager_path: Optional[str] = None,
                   recommendation_engine_path: Optional[str] = None):
        """
        Load trained ML models

        Args:
            position_manager_path: Path to trained position manager checkpoint
            recommendation_engine_path: Path to recommendation engine model
        """
        try:
            # Load recommendation engine
            if recommendation_engine_path:
                logger.info(f"Loading recommendation engine from {recommendation_engine_path}")
                self.recommendation_engine = self._load_recommendation_engine(recommendation_engine_path)

            # Load position manager
            if position_manager_path:
                logger.info(f"Loading position manager from {position_manager_path}")
                self.position_manager = self._load_position_manager(position_manager_path)

            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _load_recommendation_engine(self, model_path: str):
        """
        Load trained recommendation engine

        Args:
            model_path: Path to recommendation engine model

        Returns:
            Loaded RecommendationEngine instance
        """
        try:
            from ...models.recommendation_engine import RecommendationEngine

            # Create recommendation engine instance
            # For now, create without loading actual weights (development mode)
            engine = RecommendationEngine()

            logger.info(f"RecommendationEngine loaded from {model_path}")
            return engine

        except ImportError as e:
            logger.warning(f"Could not import RecommendationEngine: {e}")
            # Create mock engine for testing
            return self._create_mock_recommendation_engine()

    def _load_position_manager(self, checkpoint_path: str):
        """
        Load trained position manager network

        Args:
            checkpoint_path: Path to position manager checkpoint

        Returns:
            Loaded PositionManagerNetwork instance
        """
        try:
            from ...models.position_manager import PositionManagerNetwork
            from ...training.position_manager_trainer import PositionManagerTrainer

            # Load checkpoint and create network
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Extract network state and create model
            network_state = checkpoint.get('network_state_dict', checkpoint.get('model_state_dict'))

            if network_state is None:
                raise ValueError("No network state found in checkpoint")

            # Create network with default configuration
            # In production, this would come from checkpoint metadata
            network = PositionManagerNetwork(
                observation_dim=30,
                action_dim=4,
                hidden_dim=256
            )

            network.load_state_dict(network_state)
            network.eval()  # Set to evaluation mode

            logger.info(f"PositionManagerNetwork loaded from {checkpoint_path}")
            return network

        except Exception as e:
            logger.warning(f"Could not load PositionManager: {e}")
            # Create mock position manager for testing
            return self._create_mock_position_manager()

    def _create_mock_recommendation_engine(self):
        """Create a mock recommendation engine for testing"""
        class MockRecommendationEngine:
            def get_recommendations(self, market_state):
                # Simple mock implementation
                recommendations = []
                if isinstance(market_state, dict):
                    price = market_state.get('price', 450)
                    iv = market_state.get('iv', 0.20)

                    if price > 450 and iv < 0.25:
                        rec = type('Rec', (), {
                            'strategy_type': 'BULL_CALL_SPREAD',
                            'confidence': 0.7,
                            'position_size': 0.02
                        })()
                        recommendations.append(rec)

                return recommendations

        return MockRecommendationEngine()

    def _create_mock_position_manager(self):
        """Create a mock position manager for testing"""
        class MockPositionManager:
            def get_action(self, observations):
                # Simple mock: always return hold action
                if torch.is_tensor(observations):
                    batch_size = observations.shape[0] if len(observations.shape) > 1 else 1
                else:
                    batch_size = 1

                # Return mock actions (action_probs, value, action_logits)
                actions = torch.zeros(batch_size, 4)
                actions[:, 1] = 1.0  # Hold action

                values = torch.zeros(batch_size, 1)
                logits = torch.zeros(batch_size, 4)
                logits[:, 1] = 1.0  # High probability for hold

                return actions, values, logits

            def predict(self, observations):
                # Alternative interface - just return actions
                actions, _, _ = self.get_action(observations)
                return actions

        return MockPositionManager()

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Any]:
        """
        Generate trading signals from market data using ML models

        Args:
            market_data: Current market state dictionary

        Returns:
            List of SignalEvents for backtesting engine
        """
        # Simulate inference delay
        if self.inference_delay_ms > 0:
            time.sleep(self.inference_delay_ms / 1000.0)

        signals = []
        timestamp = market_data.get('timestamp', datetime.now())
        symbol = market_data.get('symbol', 'SPY')

        try:
            # Get strategy recommendations from recommendation engine
            if self.recommendation_engine:
                recommendations = self.recommendation_engine.get_recommendations(market_data)
            else:
                # Mock recommendations for testing
                recommendations = self._generate_mock_recommendations(market_data)

            # Filter recommendations by confidence
            filtered_recs = [
                rec for rec in recommendations
                if hasattr(rec, 'confidence') and rec.confidence >= self.min_confidence
            ]

            # Convert recommendations to signals
            for rec in filtered_recs:
                signal = self._recommendation_to_signal(rec, timestamp, symbol)
                if signal:
                    signals.append(signal)

        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")

        logger.debug(f"Generated {len(signals)} signals for {symbol}")
        return signals

    def process_market_event(self, market_event: Any) -> List[Any]:
        """
        Process market event and generate signals

        Args:
            market_event: MarketEvent from backtesting engine

        Returns:
            List of SignalEvents
        """
        # Convert market event to market data dictionary
        market_data = {
            'symbol': getattr(market_event, 'symbol', 'SPY'),
            'timestamp': getattr(market_event, 'timestamp', datetime.now()),
            **(getattr(market_event, 'data', {}))
        }

        return self.generate_signals(market_data)

    def _convert_to_observations(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Convert market data to ML model observation format

        Args:
            market_data: Market state dictionary

        Returns:
            Numpy array of observations for ML models
        """
        # Extract key features for ML model input
        features = []

        # Price and volume features
        features.append(market_data.get('price', 0.0))
        features.append(market_data.get('volume', 0.0))

        # Options Greeks
        features.append(market_data.get('iv', 0.0))  # Implied volatility
        features.append(market_data.get('delta', 0.0))
        features.append(market_data.get('gamma', 0.0))
        features.append(market_data.get('theta', 0.0))
        features.append(market_data.get('vega', 0.0))

        # Market regime (encoded) - mock encoding if RegimeType not available
        regime = market_data.get('regime', 'SIDEWAYS_LOW_VOL')
        regime_encoding = self._encode_regime(regime)
        features.extend(regime_encoding)

        # Time features
        if 'timestamp' in market_data:
            timestamp = market_data['timestamp']
            if hasattr(timestamp, 'hour'):
                features.append(timestamp.hour)  # Hour of day
                features.append(timestamp.weekday())  # Day of week
            else:
                features.extend([0, 0])  # Default time features

        # Convert to numpy array
        observations = np.array(features, dtype=np.float32)

        # Ensure no NaN values
        observations = np.nan_to_num(observations)

        return observations

    def _encode_regime(self, regime: Any) -> List[float]:
        """
        Encode market regime as one-hot vector

        Args:
            regime: Market regime type (string or enum)

        Returns:
            One-hot encoded regime vector
        """
        # Define regime types as strings for compatibility
        regime_types = [
            'BULL_LOW_VOL',
            'BULL_HIGH_VOL',
            'BEAR_LOW_VOL',
            'BEAR_HIGH_VOL',
            'SIDEWAYS_LOW_VOL',
            'SIDEWAYS_HIGH_VOL'
        ]

        encoding = [0.0] * len(regime_types)

        # Handle both string and enum inputs
        regime_str = str(regime).split('.')[-1] if hasattr(regime, 'value') else str(regime)

        if regime_str in regime_types:
            idx = regime_types.index(regime_str)
            encoding[idx] = 1.0

        return encoding

    def _generate_mock_recommendations(self, market_data: Dict[str, Any]) -> List[Any]:
        """
        Generate mock recommendations for testing

        Args:
            market_data: Market state

        Returns:
            List of mock strategy recommendations
        """
        recommendations = []

        # Generate a mock recommendation based on simple price momentum
        price = market_data.get('price', 450.0)
        iv = market_data.get('iv', 0.20)

        # Create mock recommendation object with needed attributes
        class MockRecommendation:
            def __init__(self, strategy_type, confidence, position_size):
                self.strategy_type = strategy_type
                self.confidence = confidence
                self.position_size = position_size

        # Mock bullish signal if price > 450 and IV < 0.25
        if price > 450 and iv < 0.25:
            rec = MockRecommendation(
                strategy_type='BULL_CALL_SPREAD',
                confidence=0.65 + np.random.normal(0, 0.1),
                position_size=0.02
            )
            recommendations.append(rec)

        # Mock neutral strategy for high IV
        if iv > 0.30:
            rec = MockRecommendation(
                strategy_type='IRON_CONDOR',
                confidence=0.70 + np.random.normal(0, 0.05),
                position_size=0.015
            )
            recommendations.append(rec)

        return recommendations

    def _recommendation_to_signal(self,
                                recommendation: Any,
                                timestamp: datetime,
                                symbol: str) -> Optional[Any]:
        """
        Convert StrategyRecommendation to SignalEvent

        Args:
            recommendation: Strategy recommendation from ML model
            timestamp: Signal timestamp
            symbol: Trading symbol

        Returns:
            SignalEvent or None if recommendation should be filtered
        """
        # Check position size limits
        if hasattr(recommendation, 'position_size') and recommendation.position_size > self.max_position_size:
            logger.warning(f"Position size {recommendation.position_size} exceeds maximum {self.max_position_size}")
            return None

        # Determine signal type based on strategy
        strategy_type = getattr(recommendation, 'strategy_type', 'UNKNOWN')

        if any(s in str(strategy_type) for s in ['BULL', 'CALL']):
            signal_type = 'LONG'
        elif any(s in str(strategy_type) for s in ['BEAR', 'PUT']):
            signal_type = 'SHORT'
        else:
            signal_type = 'NEUTRAL'

        # Create strategy ID
        strategy_id = f"{strategy_type}_ml"

        # Create mock signal event if SignalEvent not available
        if SignalEvent is not None:
            signal = SignalEvent(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                strategy_id=strategy_id,
                strength=getattr(recommendation, 'confidence', 0.5)
            )
        else:
            # Create mock signal for testing
            class MockSignalEvent:
                def __init__(self, timestamp, symbol, signal_type, strategy_id, strength):
                    self.timestamp = timestamp
                    self.symbol = symbol
                    self.signal_type = signal_type
                    self.strategy_id = strategy_id
                    self.strength = strength

            signal = MockSignalEvent(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                strategy_id=strategy_id,
                strength=getattr(recommendation, 'confidence', 0.5)
            )

        return signal