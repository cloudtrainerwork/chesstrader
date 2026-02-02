"""
Clean API interface for strategy recommendations.

Provides simplified access to the recommendation engine for external consumers.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

import numpy as np
import pandas as pd

from ..models.recommendation_engine import RecommendationEngine, StrategyRecommendation
from ..models.integrated_selector import IntegratedStrategySelector
from ..models.scoring_engine import ScoringEngine
from ..models.spatial_net import SpatialNet
from ..models.strategy_selector import StrategySelector
from ..features.regime_detector import RegimeDetector
from ..strategies.base import StrategyType

logger = logging.getLogger(__name__)


class StrategyRecommender:
    """
    High-level API for strategy recommendations.

    Provides clean interface for getting strategy recommendations without
    needing to understand the internal architecture.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.4,
                 max_recommendations: int = 3,
                 use_historical_data: bool = True):
        """
        Initialize strategy recommender API.

        Args:
            model_path: Path to saved model weights (optional)
            confidence_threshold: Minimum confidence for recommendations
            max_recommendations: Maximum recommendations to return
            use_historical_data: Whether to use historical performance data
        """
        self.confidence_threshold = confidence_threshold
        self.max_recommendations = max_recommendations
        self.use_historical_data = use_historical_data

        # Initialize components
        self._initialize_models(model_path)

        logger.info("StrategyRecommender initialized")

    def _initialize_models(self, model_path: Optional[str] = None):
        """Initialize all required models and components."""
        # Initialize spatial encoder
        self.spatial_net = SpatialNet(
            board_height=7,
            board_width=6,
            piece_channels=16,  # Number of features per position
            spatial_hidden_dim=256,
            output_dim=512
        )

        # Initialize strategy selector
        self.strategy_selector = StrategySelector(
            spatial_net=self.spatial_net,
            freeze_spatial_net=False,
            ranking_hidden_dim=256
        )

        # Initialize regime detector
        self.regime_detector = RegimeDetector(
            n_regimes=5,
            lookback_period=20,
            hidden_dim=128
        )

        # Initialize integrated selector
        self.integrated_selector = IntegratedStrategySelector(
            strategy_selector=self.strategy_selector,
            regime_detector=self.regime_detector
        )

        # Initialize scoring engine
        self.scoring_engine = ScoringEngine(
            kelly_fraction=0.25,
            min_confidence=0.3,
            max_position_size=0.2
        )

        # Initialize recommendation engine
        self.recommendation_engine = RecommendationEngine(
            integrated_selector=self.integrated_selector,
            scoring_engine=self.scoring_engine,
            confidence_threshold=self.confidence_threshold,
            max_recommendations=self.max_recommendations
        )

        # Load model weights if provided
        if model_path:
            self._load_model_weights(model_path)

    def _load_model_weights(self, model_path: str):
        """Load pre-trained model weights."""
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')

            if 'spatial_net' in checkpoint:
                self.spatial_net.load_state_dict(checkpoint['spatial_net'])
            if 'strategy_selector' in checkpoint:
                self.strategy_selector.load_state_dict(checkpoint['strategy_selector'])
            if 'regime_detector' in checkpoint:
                self.regime_detector.load_state_dict(checkpoint['regime_detector'])

            logger.info(f"Loaded model weights from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load model weights: {e}")

    def recommend(self,
                 symbol: str,
                 current_price: float,
                 price_history: Union[List[float], np.ndarray],
                 volume_history: Optional[Union[List[float], np.ndarray]] = None,
                 implied_volatility: Optional[float] = None,
                 option_chain: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """
        Get strategy recommendations for a symbol.

        Args:
            symbol: Stock symbol
            current_price: Current stock price
            price_history: Historical prices (at least 20 data points)
            volume_history: Historical volumes (optional)
            implied_volatility: Current implied volatility (optional)
            option_chain: Option chain data (optional)

        Returns:
            List of recommendation dictionaries
        """
        # Prepare market data
        market_data = self._prepare_market_data(
            symbol,
            current_price,
            price_history,
            volume_history,
            implied_volatility,
            option_chain
        )

        # Extract regime features
        regime_features = self._extract_regime_features(
            price_history,
            volume_history
        )

        # Get recommendations
        recommendations = self.recommendation_engine.get_top_recommendations(
            market_data,
            regime_features
        )

        # Convert to API format
        return self._format_recommendations(recommendations)

    def batch_recommend(self,
                       symbols: List[str],
                       prices: List[float],
                       price_histories: List[Union[List[float], np.ndarray]],
                       **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get recommendations for multiple symbols.

        Args:
            symbols: List of stock symbols
            prices: Current prices for each symbol
            price_histories: Price histories for each symbol
            **kwargs: Additional data (volumes, IV, etc.)

        Returns:
            Dictionary mapping symbols to recommendations
        """
        batch_results = {}

        for i, symbol in enumerate(symbols):
            try:
                recommendations = self.recommend(
                    symbol,
                    prices[i],
                    price_histories[i],
                    volume_history=kwargs.get('volume_histories', [None]*len(symbols))[i],
                    implied_volatility=kwargs.get('implied_volatilities', [None]*len(symbols))[i]
                )
                batch_results[symbol] = recommendations
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                batch_results[symbol] = []

        return batch_results

    def _prepare_market_data(self,
                           symbol: str,
                           current_price: float,
                           price_history: Union[List[float], np.ndarray],
                           volume_history: Optional[Union[List[float], np.ndarray]],
                           implied_volatility: Optional[float],
                           option_chain: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Prepare market data dictionary."""
        # Convert to numpy arrays
        prices = np.array(price_history)

        # Calculate basic metrics
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.2

        market_data = {
            'symbol': symbol,
            'current_price': current_price,
            'price_history': prices,
            'returns': returns,
            'historical_volatility': volatility,
            'implied_volatility': implied_volatility or volatility,
            'volume': np.mean(volume_history) if volume_history is not None else 0,
            'price_change': (current_price - prices[0]) / prices[0] if len(prices) > 0 else 0
        }

        # Add option chain data if available
        if option_chain is not None:
            market_data['option_chain'] = option_chain
            market_data['has_options'] = True
        else:
            market_data['has_options'] = False

        return market_data

    def _extract_regime_features(self,
                                price_history: Union[List[float], np.ndarray],
                                volume_history: Optional[Union[List[float], np.ndarray]]) -> np.ndarray:
        """Extract features for regime detection."""
        prices = np.array(price_history)

        # Calculate returns
        returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0])

        # Calculate features
        features = []

        # Price-based features
        features.append(np.mean(returns))  # Mean return
        features.append(np.std(returns))   # Volatility
        features.append(np.min(returns))   # Min return
        features.append(np.max(returns))   # Max return

        # Trend features
        if len(prices) >= 5:
            ma5 = np.mean(prices[-5:])
            features.append((prices[-1] - ma5) / ma5)  # Price vs MA5
        else:
            features.append(0)

        if len(prices) >= 10:
            ma10 = np.mean(prices[-10:])
            features.append((prices[-1] - ma10) / ma10)  # Price vs MA10
        else:
            features.append(0)

        if len(prices) >= 20:
            ma20 = np.mean(prices[-20:])
            features.append((prices[-1] - ma20) / ma20)  # Price vs MA20
        else:
            features.append(0)

        # Volume features if available
        if volume_history is not None:
            volumes = np.array(volume_history)
            if len(volumes) > 0:
                features.append(np.mean(volumes))  # Mean volume
                features.append(np.std(volumes))   # Volume volatility
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])

        # Momentum features
        if len(returns) >= 5:
            features.append(np.sum(returns[-5:]))  # 5-day momentum
        else:
            features.append(0)

        if len(returns) >= 10:
            features.append(np.sum(returns[-10:]))  # 10-day momentum
        else:
            features.append(0)

        # Pad or trim to expected size (20 features)
        while len(features) < 20:
            features.append(0)

        return np.array(features[:20])

    def _format_recommendations(self,
                              recommendations: List[StrategyRecommendation]) -> List[Dict[str, Any]]:
        """Format recommendations for API response."""
        formatted = []

        for rec in recommendations:
            formatted.append({
                'strategy': rec.strategy_type.value,
                'score': rec.score,
                'confidence': round(rec.confidence, 3),
                'position_size': round(rec.position_size, 3),
                'expected_return': round(rec.expected_return, 3),
                'max_risk': round(rec.max_risk, 3),
                'regime': rec.regime.value,
                'explanation': rec.explanation,
                'timestamp': rec.timestamp.isoformat(),
                'metadata': rec.metadata
            })

        return formatted

    def get_strategy_details(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with strategy details
        """
        try:
            strategy_type = StrategyType(strategy_name)
        except ValueError:
            return {'error': f'Unknown strategy: {strategy_name}'}

        details = {
            'name': strategy_type.value,
            'description': self._get_strategy_description(strategy_type),
            'risk_profile': self._get_risk_profile(strategy_type),
            'optimal_conditions': self._get_optimal_conditions(strategy_type),
            'typical_returns': self._get_typical_returns(strategy_type)
        }

        return details

    def _get_strategy_description(self, strategy: StrategyType) -> str:
        """Get strategy description."""
        descriptions = {
            StrategyType.IRON_CONDOR: "Neutral strategy selling OTM call and put spreads",
            StrategyType.IRON_BUTTERFLY: "Neutral strategy selling ATM straddle with protection",
            StrategyType.BULL_CALL_SPREAD: "Bullish strategy buying call and selling higher strike call",
            StrategyType.BEAR_PUT_SPREAD: "Bearish strategy buying put and selling lower strike put",
            StrategyType.LONG_STRADDLE: "Volatility strategy buying ATM call and put",
            StrategyType.SHORT_STRADDLE: "Neutral strategy selling ATM call and put",
            StrategyType.LONG_STRANGLE: "Volatility strategy buying OTM call and put",
            StrategyType.SHORT_STRANGLE: "Neutral strategy selling OTM call and put",
            StrategyType.CALENDAR_SPREAD: "Time decay strategy with different expirations",
            StrategyType.DIAGONAL_SPREAD: "Directional calendar spread with different strikes",
            StrategyType.COVERED_CALL: "Income strategy selling calls against stock",
            StrategyType.PROTECTIVE_COLLAR: "Protective strategy with put protection and call cap",
            StrategyType.BULL_PUT_SPREAD: "Bullish strategy selling put spread",
            StrategyType.BEAR_CALL_SPREAD: "Bearish strategy selling call spread"
        }
        return descriptions.get(strategy, "Options strategy")

    def _get_risk_profile(self, strategy: StrategyType) -> str:
        """Get risk profile description."""
        profiles = {
            StrategyType.IRON_CONDOR: "Limited risk, limited profit",
            StrategyType.IRON_BUTTERFLY: "Limited risk, limited profit",
            StrategyType.BULL_CALL_SPREAD: "Limited risk, limited profit",
            StrategyType.BEAR_PUT_SPREAD: "Limited risk, limited profit",
            StrategyType.LONG_STRADDLE: "Limited risk, unlimited profit",
            StrategyType.SHORT_STRADDLE: "Unlimited risk, limited profit",
            StrategyType.LONG_STRANGLE: "Limited risk, unlimited profit",
            StrategyType.SHORT_STRANGLE: "Unlimited risk, limited profit",
            StrategyType.CALENDAR_SPREAD: "Limited risk, limited profit",
            StrategyType.DIAGONAL_SPREAD: "Limited risk, limited profit",
            StrategyType.COVERED_CALL: "Limited upside, downside risk",
            StrategyType.PROTECTIVE_COLLAR: "Limited risk, limited profit",
            StrategyType.BULL_PUT_SPREAD: "Limited risk, limited profit",
            StrategyType.BEAR_CALL_SPREAD: "Limited risk, limited profit"
        }
        return profiles.get(strategy, "Variable risk profile")

    def _get_optimal_conditions(self, strategy: StrategyType) -> str:
        """Get optimal market conditions."""
        conditions = {
            StrategyType.IRON_CONDOR: "Low volatility, range-bound markets",
            StrategyType.IRON_BUTTERFLY: "Very low volatility, tight range",
            StrategyType.BULL_CALL_SPREAD: "Moderately bullish outlook",
            StrategyType.BEAR_PUT_SPREAD: "Moderately bearish outlook",
            StrategyType.LONG_STRADDLE: "High volatility expected",
            StrategyType.SHORT_STRADDLE: "Low volatility expected",
            StrategyType.LONG_STRANGLE: "Very high volatility expected",
            StrategyType.SHORT_STRANGLE: "Moderately low volatility",
            StrategyType.CALENDAR_SPREAD: "Stable prices, IV changes",
            StrategyType.DIAGONAL_SPREAD: "Gradual directional move",
            StrategyType.COVERED_CALL: "Neutral to slightly bullish",
            StrategyType.PROTECTIVE_COLLAR: "Concerned about downside",
            StrategyType.BULL_PUT_SPREAD: "Bullish with support level",
            StrategyType.BEAR_CALL_SPREAD: "Bearish with resistance level"
        }
        return conditions.get(strategy, "Various market conditions")

    def _get_typical_returns(self, strategy: StrategyType) -> str:
        """Get typical return range."""
        returns = {
            StrategyType.IRON_CONDOR: "5-10% per month",
            StrategyType.IRON_BUTTERFLY: "8-12% per month",
            StrategyType.BULL_CALL_SPREAD: "20-50% on risk",
            StrategyType.BEAR_PUT_SPREAD: "20-50% on risk",
            StrategyType.LONG_STRADDLE: "50-200% on large moves",
            StrategyType.SHORT_STRADDLE: "10-15% per month",
            StrategyType.LONG_STRANGLE: "100-500% on extreme moves",
            StrategyType.SHORT_STRANGLE: "8-12% per month",
            StrategyType.CALENDAR_SPREAD: "5-8% per month",
            StrategyType.DIAGONAL_SPREAD: "10-15% per month",
            StrategyType.COVERED_CALL: "2-5% per month",
            StrategyType.PROTECTIVE_COLLAR: "Protected downside",
            StrategyType.BULL_PUT_SPREAD: "10-15% per month",
            StrategyType.BEAR_CALL_SPREAD: "10-15% per month"
        }
        return returns.get(strategy, "Variable returns")

    def update_historical_performance(self,
                                    strategy_name: str,
                                    performance_data: Dict[str, float]):
        """
        Update historical performance data for a strategy.

        Args:
            strategy_name: Name of the strategy
            performance_data: Performance metrics
        """
        try:
            strategy_type = StrategyType(strategy_name)
            self.scoring_engine.update_performance_cache(strategy_type, performance_data)
            logger.info(f"Updated performance data for {strategy_name}")
        except ValueError:
            logger.error(f"Unknown strategy: {strategy_name}")