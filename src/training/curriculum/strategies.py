"""
Strategy-specific curriculum implementations for progressive options training.

Provides specialized curriculum paths for different option strategies with
strategy-appropriate progression through strike selection, expiration timing,
market regime exposure, and risk management complexity.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .levels import DifficultyLevel, CurriculumParameters
from ...strategies.base import StrategyType
from ...environments.market_sim import MarketRegime


@dataclass
class StrategyProgressionRules:
    """
    Rules for strategy-specific progression through difficulty levels.

    Attributes:
        strike_selection_rules: Rules for strike selection complexity
        expiration_rules: Rules for expiration timing
        position_sizing_rules: Rules for position sizing progression
        market_exposure_rules: Rules for market regime exposure
    """
    strike_selection_rules: Dict[str, Any]
    expiration_rules: Dict[str, Any]
    position_sizing_rules: Dict[str, Any]
    market_exposure_rules: Dict[str, Any]


class StrategyCurriculum(ABC):
    """
    Abstract base class for strategy-specific curriculum implementations.

    Each strategy type has different learning challenges and progression paths.
    This class provides the framework for implementing strategy-specific
    curricula that gradually increase complexity appropriately.
    """

    def __init__(self, strategy_type: StrategyType):
        """
        Initialize strategy curriculum.

        Args:
            strategy_type: The option strategy type this curriculum supports
        """
        self.strategy_type = strategy_type

    @abstractmethod
    def get_strike_selection_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """
        Get strike selection rules for the given difficulty level.

        Args:
            level: Current difficulty level

        Returns:
            Dictionary with strike selection parameters
        """
        pass

    @abstractmethod
    def get_expiration_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """
        Get expiration timing rules for the given difficulty level.

        Args:
            level: Current difficulty level

        Returns:
            Dictionary with expiration timing parameters
        """
        pass

    @abstractmethod
    def get_position_sizing_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """
        Get position sizing rules for the given difficulty level.

        Args:
            level: Current difficulty level

        Returns:
            Dictionary with position sizing parameters
        """
        pass

    def get_market_exposure_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """
        Get market exposure rules for the given difficulty level.

        Default implementation - can be overridden by specific strategies.

        Args:
            level: Current difficulty level

        Returns:
            Dictionary with market exposure parameters
        """
        if level == DifficultyLevel.BEGINNER:
            return {
                'preferred_regimes': [MarketRegime.LOW_VOLATILITY],
                'avoid_regimes': [MarketRegime.HIGH_VOLATILITY],
                'regime_change_frequency': 'low'
            }
        elif level == DifficultyLevel.INTERMEDIATE:
            return {
                'preferred_regimes': [MarketRegime.LOW_VOLATILITY, MarketRegime.TRENDING_UP],
                'avoid_regimes': [MarketRegime.HIGH_VOLATILITY],
                'regime_change_frequency': 'medium'
            }
        elif level == DifficultyLevel.ADVANCED:
            return {
                'preferred_regimes': list(MarketRegime),
                'avoid_regimes': [],
                'regime_change_frequency': 'high'
            }
        else:  # EXPERT
            return {
                'preferred_regimes': list(MarketRegime),
                'avoid_regimes': [],
                'regime_change_frequency': 'very_high'
            }

    def get_progression_rules(self, level: DifficultyLevel) -> StrategyProgressionRules:
        """
        Get comprehensive progression rules for the strategy at given level.

        Args:
            level: Current difficulty level

        Returns:
            StrategyProgressionRules with all rule sets
        """
        return StrategyProgressionRules(
            strike_selection_rules=self.get_strike_selection_rules(level),
            expiration_rules=self.get_expiration_rules(level),
            position_sizing_rules=self.get_position_sizing_rules(level),
            market_exposure_rules=self.get_market_exposure_rules(level)
        )


class IronCondorCurriculum(StrategyCurriculum):
    """Curriculum for Iron Condor strategy - progression from wide to narrow spreads."""

    def __init__(self):
        super().__init__(StrategyType.IRON_CONDOR)

    def get_strike_selection_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Iron Condor strike selection progression."""
        if level == DifficultyLevel.BEGINNER:
            return {
                'spread_width': 50,  # Very wide spreads
                'otm_distance': 30,  # Far OTM for safety
                'strike_spacing': 5,  # Standard spacing
                'symmetry_required': True  # Require symmetric wings
            }
        elif level == DifficultyLevel.INTERMEDIATE:
            return {
                'spread_width': 35,  # Moderate spreads
                'otm_distance': 20,  # Moderate OTM
                'strike_spacing': 5,
                'symmetry_required': True
            }
        elif level == DifficultyLevel.ADVANCED:
            return {
                'spread_width': 25,  # Narrower spreads
                'otm_distance': 15,  # Closer to ATM
                'strike_spacing': 5,
                'symmetry_required': False  # Allow asymmetric
            }
        else:  # EXPERT
            return {
                'spread_width': 15,  # Very narrow spreads
                'otm_distance': 10,  # Close to ATM
                'strike_spacing': 5,
                'symmetry_required': False,
                'dynamic_adjustment': True  # Allow mid-trade adjustments
            }

    def get_expiration_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Iron Condor expiration timing progression."""
        if level == DifficultyLevel.BEGINNER:
            return {
                'min_days_to_expiration': 30,
                'max_days_to_expiration': 60,
                'preferred_days': 45,  # Sweet spot for beginners
                'avoid_weekly': True
            }
        elif level == DifficultyLevel.INTERMEDIATE:
            return {
                'min_days_to_expiration': 21,
                'max_days_to_expiration': 45,
                'preferred_days': 30,
                'avoid_weekly': True
            }
        elif level == DifficultyLevel.ADVANCED:
            return {
                'min_days_to_expiration': 14,
                'max_days_to_expiration': 45,
                'preferred_days': 21,
                'avoid_weekly': False
            }
        else:  # EXPERT
            return {
                'min_days_to_expiration': 7,
                'max_days_to_expiration': 60,
                'preferred_days': None,  # No preference
                'avoid_weekly': False,
                'include_earnings': True  # Allow earnings plays
            }

    def get_position_sizing_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Iron Condor position sizing progression."""
        if level == DifficultyLevel.BEGINNER:
            return {
                'max_contracts': 1,
                'risk_per_trade': 0.01,  # 1% of capital
                'max_portfolio_exposure': 0.05  # 5% total exposure
            }
        elif level == DifficultyLevel.INTERMEDIATE:
            return {
                'max_contracts': 3,
                'risk_per_trade': 0.02,  # 2% of capital
                'max_portfolio_exposure': 0.10  # 10% total exposure
            }
        elif level == DifficultyLevel.ADVANCED:
            return {
                'max_contracts': 5,
                'risk_per_trade': 0.03,  # 3% of capital
                'max_portfolio_exposure': 0.15  # 15% total exposure
            }
        else:  # EXPERT
            return {
                'max_contracts': 10,
                'risk_per_trade': 0.05,  # 5% of capital
                'max_portfolio_exposure': 0.25,  # 25% total exposure
                'dynamic_sizing': True  # Size based on market conditions
            }


class IronButterflyStrategy(StrategyCurriculum):
    """Curriculum for Iron Butterfly strategy - progression from ATM to OTM variations."""

    def __init__(self):
        super().__init__(StrategyType.BUTTERFLY)

    def get_strike_selection_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Iron Butterfly strike selection progression."""
        if level == DifficultyLevel.BEGINNER:
            return {
                'center_strike_offset': 0,  # ATM center
                'wing_width': 20,  # Wide wings for safety
                'strike_spacing': 5,
                'require_atm_center': True
            }
        elif level == DifficultyLevel.INTERMEDIATE:
            return {
                'center_strike_offset': 5,  # Slightly OTM
                'wing_width': 15,
                'strike_spacing': 5,
                'require_atm_center': False
            }
        elif level == DifficultyLevel.ADVANCED:
            return {
                'center_strike_offset': 10,  # More OTM
                'wing_width': 10,
                'strike_spacing': 5,
                'require_atm_center': False
            }
        else:  # EXPERT
            return {
                'center_strike_offset': 15,  # Far OTM
                'wing_width': 5,  # Narrow wings
                'strike_spacing': 5,
                'require_atm_center': False,
                'directional_bias_allowed': True
            }

    def get_expiration_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Iron Butterfly expiration timing progression."""
        if level == DifficultyLevel.BEGINNER:
            return {
                'min_days_to_expiration': 21,
                'max_days_to_expiration': 45,
                'preferred_days': 30,
                'avoid_weekly': True
            }
        elif level == DifficultyLevel.INTERMEDIATE:
            return {
                'min_days_to_expiration': 14,
                'max_days_to_expiration': 30,
                'preferred_days': 21,
                'avoid_weekly': True
            }
        elif level == DifficultyLevel.ADVANCED:
            return {
                'min_days_to_expiration': 7,
                'max_days_to_expiration': 30,
                'preferred_days': 14,
                'avoid_weekly': False
            }
        else:  # EXPERT
            return {
                'min_days_to_expiration': 3,
                'max_days_to_expiration': 45,
                'preferred_days': None,
                'avoid_weekly': False,
                'include_0dte': True  # Allow 0DTE trades
            }

    def get_position_sizing_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Iron Butterfly position sizing progression."""
        if level == DifficultyLevel.BEGINNER:
            return {
                'max_contracts': 1,
                'risk_per_trade': 0.015,  # 1.5% of capital
                'max_portfolio_exposure': 0.06  # 6% total exposure
            }
        elif level == DifficultyLevel.INTERMEDIATE:
            return {
                'max_contracts': 2,
                'risk_per_trade': 0.025,  # 2.5% of capital
                'max_portfolio_exposure': 0.10  # 10% total exposure
            }
        elif level == DifficultyLevel.ADVANCED:
            return {
                'max_contracts': 4,
                'risk_per_trade': 0.04,  # 4% of capital
                'max_portfolio_exposure': 0.15  # 15% total exposure
            }
        else:  # EXPERT
            return {
                'max_contracts': 8,
                'risk_per_trade': 0.06,  # 6% of capital
                'max_portfolio_exposure': 0.25,  # 25% total exposure
                'kelly_sizing': True  # Use Kelly criterion
            }


class StraddleStrangleCurriculum(StrategyCurriculum):
    """Curriculum for Straddle and Strangle strategies - volatility play progression."""

    def __init__(self, strategy_type: StrategyType):
        if strategy_type not in [StrategyType.LONG_STRADDLE, StrategyType.SHORT_STRADDLE,
                               StrategyType.LONG_STRANGLE, StrategyType.SHORT_STRANGLE]:
            raise ValueError(f"Invalid strategy type for StraddleStrangleCurriculum: {strategy_type}")
        super().__init__(strategy_type)

    def get_strike_selection_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Straddle/Strangle strike selection progression."""
        is_straddle = 'STRADDLE' in self.strategy_type.value
        is_long = 'LONG' in self.strategy_type.value

        if level == DifficultyLevel.BEGINNER:
            if is_straddle:
                return {
                    'strike_offset': 0,  # ATM only
                    'require_atm': True,
                    'max_iv_rank': 50 if is_long else None,  # Long straddle needs high IV
                    'min_iv_rank': None if is_long else 70   # Short straddle needs very high IV
                }
            else:  # Strangle
                return {
                    'strike_spread': 20,  # Wide strangle for safety
                    'center_offset': 0,  # Centered around ATM
                    'max_iv_rank': 50 if is_long else None,
                    'min_iv_rank': None if is_long else 70
                }

        elif level == DifficultyLevel.INTERMEDIATE:
            if is_straddle:
                return {
                    'strike_offset': 5,  # Slightly OTM allowed
                    'require_atm': False,
                    'max_iv_rank': 60 if is_long else None,
                    'min_iv_rank': None if is_long else 60
                }
            else:  # Strangle
                return {
                    'strike_spread': 15,  # Moderate spread
                    'center_offset': 5,   # Slight directional bias allowed
                    'max_iv_rank': 60 if is_long else None,
                    'min_iv_rank': None if is_long else 60
                }

        elif level == DifficultyLevel.ADVANCED:
            if is_straddle:
                return {
                    'strike_offset': 10,  # More OTM
                    'require_atm': False,
                    'max_iv_rank': 70 if is_long else None,
                    'min_iv_rank': None if is_long else 50
                }
            else:  # Strangle
                return {
                    'strike_spread': 10,  # Narrow spread
                    'center_offset': 10,  # More directional bias
                    'max_iv_rank': 70 if is_long else None,
                    'min_iv_rank': None if is_long else 50
                }

        else:  # EXPERT
            if is_straddle:
                return {
                    'strike_offset': 15,  # Far OTM allowed
                    'require_atm': False,
                    'max_iv_rank': None,  # No IV restrictions
                    'min_iv_rank': None,
                    'earnings_plays': True
                }
            else:  # Strangle
                return {
                    'strike_spread': 5,   # Very narrow
                    'center_offset': 15,  # Strong directional bias
                    'max_iv_rank': None,
                    'min_iv_rank': None,
                    'dynamic_adjustment': True
                }

    def get_expiration_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Straddle/Strangle expiration timing progression."""
        is_long = 'LONG' in self.strategy_type.value

        if level == DifficultyLevel.BEGINNER:
            return {
                'min_days_to_expiration': 30 if is_long else 21,
                'max_days_to_expiration': 60 if is_long else 45,
                'preferred_days': 45 if is_long else 30,
                'avoid_weekly': True
            }
        elif level == DifficultyLevel.INTERMEDIATE:
            return {
                'min_days_to_expiration': 21 if is_long else 14,
                'max_days_to_expiration': 45 if is_long else 30,
                'preferred_days': 30 if is_long else 21,
                'avoid_weekly': True
            }
        elif level == DifficultyLevel.ADVANCED:
            return {
                'min_days_to_expiration': 14 if is_long else 7,
                'max_days_to_expiration': 45 if is_long else 30,
                'preferred_days': 21 if is_long else 14,
                'avoid_weekly': False
            }
        else:  # EXPERT
            return {
                'min_days_to_expiration': 7 if is_long else 3,
                'max_days_to_expiration': 60 if is_long else 45,
                'preferred_days': None,
                'avoid_weekly': False,
                'include_earnings': True
            }

    def get_position_sizing_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Straddle/Strangle position sizing progression."""
        is_long = 'LONG' in self.strategy_type.value

        base_risk = 0.02 if is_long else 0.03  # Long positions generally safer

        if level == DifficultyLevel.BEGINNER:
            return {
                'max_contracts': 1,
                'risk_per_trade': base_risk * 0.5,
                'max_portfolio_exposure': 0.05
            }
        elif level == DifficultyLevel.INTERMEDIATE:
            return {
                'max_contracts': 2,
                'risk_per_trade': base_risk * 0.75,
                'max_portfolio_exposure': 0.08
            }
        elif level == DifficultyLevel.ADVANCED:
            return {
                'max_contracts': 4,
                'risk_per_trade': base_risk,
                'max_portfolio_exposure': 0.12
            }
        else:  # EXPERT
            return {
                'max_contracts': 8,
                'risk_per_trade': base_risk * 1.5,
                'max_portfolio_exposure': 0.20,
                'volatility_based_sizing': True
            }


class VerticalSpreadCurriculum(StrategyCurriculum):
    """Curriculum for vertical spread strategies - directional progression."""

    def __init__(self, strategy_type: StrategyType):
        valid_types = [
            StrategyType.BULL_CALL_SPREAD, StrategyType.BEAR_CALL_SPREAD,
            StrategyType.BULL_PUT_SPREAD, StrategyType.BEAR_PUT_SPREAD
        ]
        if strategy_type not in valid_types:
            raise ValueError(f"Invalid strategy type for VerticalSpreadCurriculum: {strategy_type}")
        super().__init__(strategy_type)

    def get_strike_selection_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Vertical spread strike selection progression."""
        is_call_spread = 'CALL' in self.strategy_type.value
        is_bull_spread = 'BULL' in self.strategy_type.value

        if level == DifficultyLevel.BEGINNER:
            return {
                'spread_width': 10,  # Wide spreads
                'otm_offset': 5,     # Slightly OTM
                'max_delta': 0.7,    # Conservative delta
                'min_premium': 2.0   # Minimum premium for safety
            }
        elif level == DifficultyLevel.INTERMEDIATE:
            return {
                'spread_width': 7,   # Moderate spreads
                'otm_offset': 3,     # Closer to ATM
                'max_delta': 0.8,
                'min_premium': 1.5
            }
        elif level == DifficultyLevel.ADVANCED:
            return {
                'spread_width': 5,   # Narrow spreads
                'otm_offset': 1,     # Near ATM
                'max_delta': 0.9,
                'min_premium': 1.0
            }
        else:  # EXPERT
            return {
                'spread_width': 3,   # Very narrow
                'otm_offset': 0,     # ATM allowed
                'max_delta': 1.0,    # No delta limit
                'min_premium': 0.5,
                'dynamic_width': True
            }

    def get_expiration_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Vertical spread expiration timing progression."""
        if level == DifficultyLevel.BEGINNER:
            return {
                'min_days_to_expiration': 21,
                'max_days_to_expiration': 45,
                'preferred_days': 30,
                'avoid_weekly': True
            }
        elif level == DifficultyLevel.INTERMEDIATE:
            return {
                'min_days_to_expiration': 14,
                'max_days_to_expiration': 30,
                'preferred_days': 21,
                'avoid_weekly': True
            }
        elif level == DifficultyLevel.ADVANCED:
            return {
                'min_days_to_expiration': 7,
                'max_days_to_expiration': 30,
                'preferred_days': 14,
                'avoid_weekly': False
            }
        else:  # EXPERT
            return {
                'min_days_to_expiration': 3,
                'max_days_to_expiration': 60,
                'preferred_days': None,
                'avoid_weekly': False,
                'include_0dte': True
            }

    def get_position_sizing_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Vertical spread position sizing progression."""
        if level == DifficultyLevel.BEGINNER:
            return {
                'max_contracts': 2,
                'risk_per_trade': 0.015,
                'max_portfolio_exposure': 0.06
            }
        elif level == DifficultyLevel.INTERMEDIATE:
            return {
                'max_contracts': 5,
                'risk_per_trade': 0.025,
                'max_portfolio_exposure': 0.10
            }
        elif level == DifficultyLevel.ADVANCED:
            return {
                'max_contracts': 10,
                'risk_per_trade': 0.04,
                'max_portfolio_exposure': 0.15
            }
        else:  # EXPERT
            return {
                'max_contracts': 20,
                'risk_per_trade': 0.06,
                'max_portfolio_exposure': 0.25,
                'trend_based_sizing': True
            }


class CurriculumFactory:
    """Factory for creating strategy-specific curricula."""

    @staticmethod
    def create_curriculum(strategy_type: StrategyType) -> StrategyCurriculum:
        """
        Create appropriate curriculum for the given strategy type.

        Args:
            strategy_type: The strategy type to create curriculum for

        Returns:
            StrategyCurriculum instance for the strategy

        Raises:
            ValueError: If strategy type is not supported
        """
        if strategy_type == StrategyType.IRON_CONDOR:
            return IronCondorCurriculum()

        elif strategy_type == StrategyType.BUTTERFLY:
            return IronButterflyStrategy()

        elif strategy_type in [StrategyType.LONG_STRADDLE, StrategyType.SHORT_STRADDLE,
                              StrategyType.LONG_STRANGLE, StrategyType.SHORT_STRANGLE]:
            return StraddleStrangleCurriculum(strategy_type)

        elif strategy_type in [StrategyType.BULL_CALL_SPREAD, StrategyType.BEAR_CALL_SPREAD,
                              StrategyType.BULL_PUT_SPREAD, StrategyType.BEAR_PUT_SPREAD]:
            return VerticalSpreadCurriculum(strategy_type)

        else:
            # For unsupported strategies, return a generic curriculum
            return _GenericCurriculum(strategy_type)

    @staticmethod
    def get_supported_strategies() -> List[StrategyType]:
        """
        Get list of strategies with dedicated curriculum support.

        Returns:
            List of supported StrategyType values
        """
        return [
            StrategyType.IRON_CONDOR,
            StrategyType.BUTTERFLY,
            StrategyType.LONG_STRADDLE,
            StrategyType.SHORT_STRADDLE,
            StrategyType.LONG_STRANGLE,
            StrategyType.SHORT_STRANGLE,
            StrategyType.BULL_CALL_SPREAD,
            StrategyType.BEAR_CALL_SPREAD,
            StrategyType.BULL_PUT_SPREAD,
            StrategyType.BEAR_PUT_SPREAD
        ]


class _GenericCurriculum(StrategyCurriculum):
    """Generic curriculum for strategies without specific implementations."""

    def get_strike_selection_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Generic strike selection rules."""
        complexity_map = {
            DifficultyLevel.BEGINNER: {'otm_range': 20, 'max_strikes': 2},
            DifficultyLevel.INTERMEDIATE: {'otm_range': 15, 'max_strikes': 3},
            DifficultyLevel.ADVANCED: {'otm_range': 10, 'max_strikes': 4},
            DifficultyLevel.EXPERT: {'otm_range': 5, 'max_strikes': 6}
        }
        return complexity_map[level]

    def get_expiration_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Generic expiration rules."""
        if level == DifficultyLevel.BEGINNER:
            return {'min_days': 21, 'max_days': 45, 'avoid_weekly': True}
        elif level == DifficultyLevel.INTERMEDIATE:
            return {'min_days': 14, 'max_days': 30, 'avoid_weekly': True}
        elif level == DifficultyLevel.ADVANCED:
            return {'min_days': 7, 'max_days': 30, 'avoid_weekly': False}
        else:  # EXPERT
            return {'min_days': 3, 'max_days': 60, 'avoid_weekly': False}

    def get_position_sizing_rules(self, level: DifficultyLevel) -> Dict[str, Any]:
        """Generic position sizing rules."""
        risk_map = {
            DifficultyLevel.BEGINNER: {'max_contracts': 1, 'risk_per_trade': 0.01},
            DifficultyLevel.INTERMEDIATE: {'max_contracts': 2, 'risk_per_trade': 0.02},
            DifficultyLevel.ADVANCED: {'max_contracts': 4, 'risk_per_trade': 0.03},
            DifficultyLevel.EXPERT: {'max_contracts': 8, 'risk_per_trade': 0.05}
        }
        return risk_map[level]