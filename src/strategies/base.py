"""
Base strategy class providing standardized interface for options strategies.

Defines abstract base class that all strategy implementations must inherit from,
ensuring consistent interface for entry/exit criteria, risk metrics, and position construction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from ..features.position_models import Position

# Define needed enums locally to avoid import issues
class OptionType(Enum):
    """Option type enumeration."""
    CALL = "CALL"
    PUT = "PUT"


class StrategyType(Enum):
    """Enumeration of the 16 core options strategies supported by the system."""

    # Single leg strategies
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"

    # Vertical spreads
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"

    # Horizontal spreads
    CALENDAR_CALL = "calendar_call"
    CALENDAR_PUT = "calendar_put"

    # Volatility strategies
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"

    # Complex strategies
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    IRON_BUTTERFLY = "butterfly"


class StrategyCategory(Enum):
    """Strategy category classification."""
    NEUTRAL = "neutral"
    DIRECTIONAL = "directional"
    VOLATILITY = "volatility"
    ADVANCED = "advanced"


class RiskLevel(Enum):
    """Risk level classification for strategies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class StrategyMetadata:
    """
    Strategy metadata containing descriptive information.

    Attributes:
        name: Human-readable strategy name
        category: Strategy category classification
        risk_level: Risk assessment for the strategy
        capital_requirement: Relative capital requirement (1.0 = base)
        description: Brief strategy description
        typical_market_conditions: Ideal market conditions for strategy
    """
    name: str
    category: StrategyCategory
    risk_level: RiskLevel
    capital_requirement: float
    description: str
    typical_market_conditions: List[str]


@dataclass
class MarketConditions:
    """
    Market condition data for strategy evaluation.

    Attributes:
        regime: Market regime (0-7 from regime detection system)
        volatility_rank: Volatility percentile rank
        trend_strength: Trend strength indicator
        time_to_expiration: Days to expiration for options
        underlying_price: Current underlying price in cents
        risk_free_rate: Current risk-free rate
    """
    regime: int
    volatility_rank: float
    trend_strength: float
    time_to_expiration: int
    underlying_price: int
    risk_free_rate: float = 0.05


@dataclass
class PositionLeg:
    """
    Individual option leg for position construction.

    Attributes:
        option_type: CALL or PUT
        strike: Strike price in cents
        quantity: Number of contracts (negative for short)
        expiration_date: Option expiration date
    """
    option_type: OptionType
    strike: int
    quantity: int
    expiration_date: datetime


@dataclass
class RiskMetrics:
    """
    Comprehensive risk metrics for a strategy.

    Attributes:
        max_profit: Maximum possible profit in cents
        max_loss: Maximum possible loss in cents
        breakeven_points: List of breakeven prices in cents
        profit_probability: Estimated probability of profit
        risk_reward_ratio: Reward to risk ratio
        capital_requirement: Required capital in cents
        margin_requirement: Margin requirement in cents
    """
    max_profit: int
    max_loss: int
    breakeven_points: List[int]
    profit_probability: float
    risk_reward_ratio: float
    capital_requirement: int
    margin_requirement: int


@dataclass
class EntrySignal:
    """
    Strategy entry signal with scoring.

    Attributes:
        should_enter: Boolean entry decision
        confidence: Confidence score (0.0 to 1.0)
        reasons: List of reasons supporting the decision
        recommended_size: Recommended position size multiplier
    """
    should_enter: bool
    confidence: float
    reasons: List[str]
    recommended_size: float = 1.0


@dataclass
class ExitSignal:
    """
    Strategy exit signal with reasoning.

    Attributes:
        should_exit: Boolean exit decision
        urgency: Exit urgency (0.0 to 1.0)
        exit_type: Type of exit (profit, loss, time, adjustment)
        reasons: List of reasons for exit
    """
    should_exit: bool
    urgency: float
    exit_type: str
    reasons: List[str]


class BaseStrategy(ABC):
    """
    Abstract base class for all options trading strategies.

    Provides standardized interface for strategy implementation including
    market condition validation, entry/exit criteria, risk metrics calculation,
    and position leg construction.

    All concrete strategy implementations must inherit from this class and
    implement all abstract methods to ensure consistent behavior.
    """

    def __init__(self):
        """Initialize base strategy."""
        self._metadata = self._create_metadata()
        self._validate_metadata()

    @property
    def metadata(self) -> StrategyMetadata:
        """Get strategy metadata."""
        return self._metadata

    @property
    def name(self) -> str:
        """Get strategy name."""
        return self._metadata.name

    @property
    def category(self) -> StrategyCategory:
        """Get strategy category."""
        return self._metadata.category

    @property
    def risk_level(self) -> RiskLevel:
        """Get risk level."""
        return self._metadata.risk_level

    @abstractmethod
    def _create_metadata(self) -> StrategyMetadata:
        """
        Create strategy metadata.

        Must be implemented by each strategy to provide descriptive information.

        Returns:
            StrategyMetadata with strategy-specific information
        """
        pass

    @abstractmethod
    def validate_market_conditions(self, conditions: MarketConditions) -> bool:
        """
        Validate if current market conditions are suitable for this strategy.

        Args:
            conditions: Current market condition data

        Returns:
            True if conditions are suitable, False otherwise
        """
        pass

    @abstractmethod
    def calculate_entry_criteria(self, conditions: MarketConditions) -> EntrySignal:
        """
        Calculate entry signal based on market conditions.

        Args:
            conditions: Current market condition data

        Returns:
            EntrySignal with entry decision and confidence
        """
        pass

    @abstractmethod
    def calculate_exit_criteria(self, position: 'Position', conditions: MarketConditions) -> ExitSignal:
        """
        Calculate exit signal for existing position.

        Args:
            position: Current position
            conditions: Current market conditions

        Returns:
            ExitSignal with exit decision and reasoning
        """
        pass

    @abstractmethod
    def get_risk_metrics(self, strikes: List[int], underlying_price: int,
                        time_to_expiration: int, volatility: float) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for the strategy.

        Args:
            strikes: Strike prices for the strategy
            underlying_price: Current underlying price in cents
            time_to_expiration: Days to expiration
            volatility: Implied volatility estimate

        Returns:
            RiskMetrics with comprehensive risk analysis
        """
        pass

    @abstractmethod
    def get_position_legs(self, strikes: List[int], expiration_date: datetime) -> List[PositionLeg]:
        """
        Get position legs for strategy construction.

        Args:
            strikes: Strike prices for the strategy
            expiration_date: Option expiration date

        Returns:
            List of PositionLeg objects defining the complete position
        """
        pass

    def validate_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Validate strike price configuration for the strategy.

        Args:
            strikes: Proposed strike prices in cents
            underlying_price: Current underlying price in cents

        Returns:
            True if strike configuration is valid, False otherwise
        """
        if not strikes:
            return False

        # Basic validations
        if any(strike <= 0 for strike in strikes):
            return False

        if any(strike > underlying_price * 3 for strike in strikes):
            return False  # Strikes too far OTM

        # Strategy-specific validation will be implemented in subclasses
        return self._validate_strategy_strikes(strikes, underlying_price)

    @abstractmethod
    def _validate_strategy_strikes(self, strikes: List[int], underlying_price: int) -> bool:
        """
        Strategy-specific strike validation.

        Args:
            strikes: Strike prices to validate
            underlying_price: Current underlying price

        Returns:
            True if strikes are valid for this strategy
        """
        pass

    def validate_expiration_date(self, expiration_date: datetime) -> bool:
        """
        Validate expiration date for strategy requirements.

        Args:
            expiration_date: Proposed expiration date

        Returns:
            True if expiration date is suitable
        """
        if expiration_date <= datetime.now():
            return False

        days_to_exp = (expiration_date - datetime.now()).days

        # Most strategies work best with 15-60 days to expiration
        if days_to_exp < 7 or days_to_exp > 365:
            return False

        return True

    def _validate_metadata(self) -> None:
        """Validate that metadata is properly configured."""
        if not self._metadata.name:
            raise ValueError("Strategy must have a name")
        if not self._metadata.description:
            raise ValueError("Strategy must have a description")
        if self._metadata.capital_requirement <= 0:
            raise ValueError("Capital requirement must be positive")

    def get_strategy_type(self) -> StrategyType:
        """
        Get the StrategyType enum value for this strategy.

        Must be overridden by concrete implementations to return
        the appropriate enum value from position_models.StrategyType.

        Returns:
            StrategyType enum value
        """
        raise NotImplementedError("Subclasses must implement get_strategy_type()")

    def calculate_margin_requirement(self, legs: List[PositionLeg], underlying_price: int) -> int:
        """
        Calculate margin requirement for the strategy.

        Basic implementation using standard margin rules.
        Can be overridden for strategy-specific requirements.

        Args:
            legs: Position legs
            underlying_price: Current underlying price in cents

        Returns:
            Margin requirement in cents
        """
        total_margin = 0

        for leg in legs:
            if leg.quantity < 0:  # Short positions require margin
                if leg.option_type == OptionType.CALL:
                    # Short call margin
                    margin = max(
                        underlying_price * 20 // 100,  # 20% of underlying
                        (underlying_price - leg.strike + underlying_price * 10 // 100)  # 10% minimum
                    )
                else:  # Short put
                    # Short put margin
                    margin = max(
                        underlying_price * 20 // 100,  # 20% of underlying
                        (leg.strike - underlying_price + underlying_price * 10 // 100)  # 10% minimum
                    )

                total_margin += margin * abs(leg.quantity)

        return total_margin

    def __str__(self) -> str:
        """String representation of strategy."""
        return f"{self.name} ({self.category.value}, {self.risk_level.value} risk)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"<{self.__class__.__name__}: {self.name}>"
