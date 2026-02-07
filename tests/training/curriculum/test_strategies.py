"""
Tests for strategy-specific curriculum implementations.
"""

import pytest
from unittest.mock import Mock

from src.training.curriculum.strategies import (
    StrategyCurriculum,
    StrategyProgressionRules,
    CurriculumFactory,
    IronCondorCurriculum,
    IronButterflyStrategy,
    StraddleStrangleCurriculum,
    VerticalSpreadCurriculum
)
from src.training.curriculum.levels import DifficultyLevel
from src.strategies.base import StrategyType


class TestCurriculumFactory:
    """Test CurriculumFactory."""

    def test_iron_condor_creation(self):
        """Test creating Iron Condor curriculum."""
        curriculum = CurriculumFactory.create_curriculum(StrategyType.IRON_CONDOR)
        assert isinstance(curriculum, IronCondorCurriculum)
        assert curriculum.strategy_type == StrategyType.IRON_CONDOR

    def test_butterfly_creation(self):
        """Test creating Butterfly curriculum."""
        curriculum = CurriculumFactory.create_curriculum(StrategyType.BUTTERFLY)
        assert isinstance(curriculum, IronButterflyStrategy)
        assert curriculum.strategy_type == StrategyType.BUTTERFLY

    def test_straddle_creation(self):
        """Test creating Straddle curriculum."""
        for strategy_type in [StrategyType.LONG_STRADDLE, StrategyType.SHORT_STRADDLE]:
            curriculum = CurriculumFactory.create_curriculum(strategy_type)
            assert isinstance(curriculum, StraddleStrangleCurriculum)
            assert curriculum.strategy_type == strategy_type

    def test_strangle_creation(self):
        """Test creating Strangle curriculum."""
        for strategy_type in [StrategyType.LONG_STRANGLE, StrategyType.SHORT_STRANGLE]:
            curriculum = CurriculumFactory.create_curriculum(strategy_type)
            assert isinstance(curriculum, StraddleStrangleCurriculum)
            assert curriculum.strategy_type == strategy_type

    def test_vertical_spread_creation(self):
        """Test creating vertical spread curriculum."""
        spread_types = [
            StrategyType.BULL_CALL_SPREAD,
            StrategyType.BEAR_CALL_SPREAD,
            StrategyType.BULL_PUT_SPREAD,
            StrategyType.BEAR_PUT_SPREAD
        ]

        for strategy_type in spread_types:
            curriculum = CurriculumFactory.create_curriculum(strategy_type)
            assert isinstance(curriculum, VerticalSpreadCurriculum)
            assert curriculum.strategy_type == strategy_type

    def test_unsupported_strategy_generic(self):
        """Test that unsupported strategies get generic curriculum."""
        # Use a single leg strategy not specifically supported
        curriculum = CurriculumFactory.create_curriculum(StrategyType.LONG_CALL)
        assert curriculum.strategy_type == StrategyType.LONG_CALL
        # Should be generic curriculum (not one of the specialized ones)
        assert not isinstance(curriculum, (IronCondorCurriculum, IronButterflyStrategy,
                                         StraddleStrangleCurriculum, VerticalSpreadCurriculum))

    def test_supported_strategies_list(self):
        """Test getting list of supported strategies."""
        supported = CurriculumFactory.get_supported_strategies()

        assert StrategyType.IRON_CONDOR in supported
        assert StrategyType.BUTTERFLY in supported
        assert StrategyType.LONG_STRADDLE in supported
        assert StrategyType.BULL_CALL_SPREAD in supported

        # Should have all major complex strategies
        assert len(supported) >= 10


class TestIronCondorCurriculum:
    """Test Iron Condor specific curriculum."""

    def setUp(self):
        self.curriculum = IronCondorCurriculum()

    def test_initialization(self):
        """Test Iron Condor curriculum initialization."""
        curriculum = IronCondorCurriculum()
        assert curriculum.strategy_type == StrategyType.IRON_CONDOR

    def test_strike_selection_progression(self):
        """Test strike selection becomes more aggressive with higher levels."""
        curriculum = IronCondorCurriculum()

        beginner_rules = curriculum.get_strike_selection_rules(DifficultyLevel.BEGINNER)
        expert_rules = curriculum.get_strike_selection_rules(DifficultyLevel.EXPERT)

        # Spread width should decrease (more aggressive)
        assert beginner_rules['spread_width'] > expert_rules['spread_width']

        # OTM distance should decrease (closer to ATM)
        assert beginner_rules['otm_distance'] > expert_rules['otm_distance']

        # Symmetry requirement should be relaxed
        assert beginner_rules['symmetry_required'] == True
        assert expert_rules['symmetry_required'] == False

    def test_expiration_progression(self):
        """Test expiration rules progression."""
        curriculum = IronCondorCurriculum()

        beginner_rules = curriculum.get_expiration_rules(DifficultyLevel.BEGINNER)
        expert_rules = curriculum.get_expiration_rules(DifficultyLevel.EXPERT)

        # Beginners should avoid weekly options
        assert beginner_rules['avoid_weekly'] == True
        assert expert_rules['avoid_weekly'] == False

        # Min days should decrease (allow shorter expiration)
        assert beginner_rules['min_days_to_expiration'] > expert_rules['min_days_to_expiration']

    def test_position_sizing_progression(self):
        """Test position sizing progression."""
        curriculum = IronCondorCurriculum()

        beginner_rules = curriculum.get_position_sizing_rules(DifficultyLevel.BEGINNER)
        expert_rules = curriculum.get_position_sizing_rules(DifficultyLevel.EXPERT)

        # Contract size should increase
        assert beginner_rules['max_contracts'] < expert_rules['max_contracts']

        # Risk per trade should increase
        assert beginner_rules['risk_per_trade'] < expert_rules['risk_per_trade']

        # Portfolio exposure should increase
        assert beginner_rules['max_portfolio_exposure'] < expert_rules['max_portfolio_exposure']

    def test_progression_rules(self):
        """Test getting complete progression rules."""
        curriculum = IronCondorCurriculum()

        for level in DifficultyLevel:
            rules = curriculum.get_progression_rules(level)

            assert isinstance(rules, StrategyProgressionRules)
            assert 'spread_width' in rules.strike_selection_rules
            assert 'min_days_to_expiration' in rules.expiration_rules
            assert 'max_contracts' in rules.position_sizing_rules
            assert 'preferred_regimes' in rules.market_exposure_rules


class TestIronButterflyStrategy:
    """Test Iron Butterfly curriculum."""

    def test_initialization(self):
        """Test Iron Butterfly curriculum initialization."""
        curriculum = IronButterflyStrategy()
        assert curriculum.strategy_type == StrategyType.BUTTERFLY

    def test_strike_selection_progression(self):
        """Test butterfly strike selection progression."""
        curriculum = IronButterflyStrategy()

        beginner_rules = curriculum.get_strike_selection_rules(DifficultyLevel.BEGINNER)
        expert_rules = curriculum.get_strike_selection_rules(DifficultyLevel.EXPERT)

        # Center should move away from ATM
        assert beginner_rules['center_strike_offset'] < expert_rules['center_strike_offset']

        # Wing width should decrease
        assert beginner_rules['wing_width'] > expert_rules['wing_width']

        # ATM requirement should be relaxed
        assert beginner_rules['require_atm_center'] == True
        assert expert_rules['require_atm_center'] == False

    def test_expert_features(self):
        """Test expert level has advanced features."""
        curriculum = IronButterflyStrategy()

        expert_rules = curriculum.get_strike_selection_rules(DifficultyLevel.EXPERT)

        assert 'directional_bias_allowed' in expert_rules
        assert expert_rules['directional_bias_allowed'] == True


class TestStraddleStrangleCurriculum:
    """Test Straddle/Strangle curriculum."""

    def test_straddle_initialization(self):
        """Test straddle curriculum initialization."""
        long_straddle = StraddleStrangleCurriculum(StrategyType.LONG_STRADDLE)
        assert long_straddle.strategy_type == StrategyType.LONG_STRADDLE

        short_straddle = StraddleStrangleCurriculum(StrategyType.SHORT_STRADDLE)
        assert short_straddle.strategy_type == StrategyType.SHORT_STRADDLE

    def test_strangle_initialization(self):
        """Test strangle curriculum initialization."""
        long_strangle = StraddleStrangleCurriculum(StrategyType.LONG_STRANGLE)
        assert long_strangle.strategy_type == StrategyType.LONG_STRANGLE

        short_strangle = StraddleStrangleCurriculum(StrategyType.SHORT_STRANGLE)
        assert short_strangle.strategy_type == StrategyType.SHORT_STRANGLE

    def test_invalid_strategy_type(self):
        """Test that invalid strategy types raise error."""
        with pytest.raises(ValueError):
            StraddleStrangleCurriculum(StrategyType.IRON_CONDOR)

    def test_long_straddle_progression(self):
        """Test long straddle specific progression."""
        curriculum = StraddleStrangleCurriculum(StrategyType.LONG_STRADDLE)

        beginner_rules = curriculum.get_strike_selection_rules(DifficultyLevel.BEGINNER)
        expert_rules = curriculum.get_strike_selection_rules(DifficultyLevel.EXPERT)

        # Straddle should start ATM
        assert beginner_rules['strike_offset'] == 0
        assert beginner_rules['require_atm'] == True

        # Expert level should allow more flexibility
        assert expert_rules['require_atm'] == False

    def test_strangle_progression(self):
        """Test strangle specific progression."""
        curriculum = StraddleStrangleCurriculum(StrategyType.LONG_STRANGLE)

        beginner_rules = curriculum.get_strike_selection_rules(DifficultyLevel.BEGINNER)
        expert_rules = curriculum.get_strike_selection_rules(DifficultyLevel.EXPERT)

        # Strike spread should decrease (tighter strangle)
        assert beginner_rules['strike_spread'] > expert_rules['strike_spread']

        # Center offset should increase (more directional bias)
        assert beginner_rules['center_offset'] < expert_rules['center_offset']

    def test_long_vs_short_differences(self):
        """Test differences between long and short straddles."""
        long_curriculum = StraddleStrangleCurriculum(StrategyType.LONG_STRADDLE)
        short_curriculum = StraddleStrangleCurriculum(StrategyType.SHORT_STRADDLE)

        long_exp_rules = long_curriculum.get_expiration_rules(DifficultyLevel.BEGINNER)
        short_exp_rules = short_curriculum.get_expiration_rules(DifficultyLevel.BEGINNER)

        # Long positions typically need more time
        assert long_exp_rules['min_days_to_expiration'] >= short_exp_rules['min_days_to_expiration']


class TestVerticalSpreadCurriculum:
    """Test Vertical Spread curriculum."""

    def test_initialization(self):
        """Test vertical spread curriculum initialization."""
        curriculum = VerticalSpreadCurriculum(StrategyType.BULL_CALL_SPREAD)
        assert curriculum.strategy_type == StrategyType.BULL_CALL_SPREAD

    def test_invalid_strategy_type(self):
        """Test that invalid strategy types raise error."""
        with pytest.raises(ValueError):
            VerticalSpreadCurriculum(StrategyType.IRON_CONDOR)

    def test_spread_width_progression(self):
        """Test spread width progression."""
        curriculum = VerticalSpreadCurriculum(StrategyType.BULL_CALL_SPREAD)

        beginner_rules = curriculum.get_strike_selection_rules(DifficultyLevel.BEGINNER)
        expert_rules = curriculum.get_strike_selection_rules(DifficultyLevel.EXPERT)

        # Spread width should decrease (more aggressive)
        assert beginner_rules['spread_width'] > expert_rules['spread_width']

        # OTM offset should decrease (closer to ATM)
        assert beginner_rules['otm_offset'] > expert_rules['otm_offset']

    def test_risk_progression(self):
        """Test risk parameter progression."""
        curriculum = VerticalSpreadCurriculum(StrategyType.BULL_CALL_SPREAD)

        beginner_rules = curriculum.get_strike_selection_rules(DifficultyLevel.BEGINNER)
        expert_rules = curriculum.get_strike_selection_rules(DifficultyLevel.EXPERT)

        # Delta limits should increase (more aggressive)
        assert beginner_rules['max_delta'] < expert_rules['max_delta']

        # Minimum premium should decrease (allow riskier trades)
        assert beginner_rules['min_premium'] > expert_rules['min_premium']

    def test_position_sizing_scaling(self):
        """Test position sizing scales appropriately."""
        curriculum = VerticalSpreadCurriculum(StrategyType.BULL_CALL_SPREAD)

        beginner_sizing = curriculum.get_position_sizing_rules(DifficultyLevel.BEGINNER)
        expert_sizing = curriculum.get_position_sizing_rules(DifficultyLevel.EXPERT)

        # Expert should allow larger positions
        assert expert_sizing['max_contracts'] > beginner_sizing['max_contracts']
        assert expert_sizing['risk_per_trade'] > beginner_sizing['risk_per_trade']


class TestGenericCurriculum:
    """Test generic curriculum for unsupported strategies."""

    def test_generic_curriculum_creation(self):
        """Test generic curriculum for unsupported strategy."""
        curriculum = CurriculumFactory.create_curriculum(StrategyType.LONG_CALL)

        # Should have basic rules for all levels
        for level in DifficultyLevel:
            strike_rules = curriculum.get_strike_selection_rules(level)
            exp_rules = curriculum.get_expiration_rules(level)
            sizing_rules = curriculum.get_position_sizing_rules(level)

            assert 'otm_range' in strike_rules
            assert 'max_strikes' in strike_rules
            assert 'min_days' in exp_rules
            assert 'max_days' in exp_rules
            assert 'max_contracts' in sizing_rules
            assert 'risk_per_trade' in sizing_rules

    def test_generic_progression(self):
        """Test that generic curriculum has logical progression."""
        curriculum = CurriculumFactory.create_curriculum(StrategyType.LONG_PUT)

        beginner_strike = curriculum.get_strike_selection_rules(DifficultyLevel.BEGINNER)
        expert_strike = curriculum.get_strike_selection_rules(DifficultyLevel.EXPERT)

        # Should allow more strikes at higher levels
        assert expert_strike['max_strikes'] >= beginner_strike['max_strikes']

        # Should allow closer to ATM at higher levels
        assert expert_strike['otm_range'] <= beginner_strike['otm_range']


class TestIntegration:
    """Integration tests for strategy curricula."""

    def test_all_supported_strategies(self):
        """Test that all supported strategies can be created and have valid rules."""
        supported_strategies = CurriculumFactory.get_supported_strategies()

        for strategy_type in supported_strategies:
            curriculum = CurriculumFactory.create_curriculum(strategy_type)

            # Test all difficulty levels have complete rules
            for level in DifficultyLevel:
                rules = curriculum.get_progression_rules(level)

                assert isinstance(rules, StrategyProgressionRules)
                assert rules.strike_selection_rules is not None
                assert rules.expiration_rules is not None
                assert rules.position_sizing_rules is not None
                assert rules.market_exposure_rules is not None

    def test_progressive_difficulty(self):
        """Test that difficulty progresses logically across strategies."""
        test_strategies = [
            StrategyType.IRON_CONDOR,
            StrategyType.BUTTERFLY,
            StrategyType.LONG_STRADDLE,
            StrategyType.BULL_CALL_SPREAD
        ]

        for strategy_type in test_strategies:
            curriculum = CurriculumFactory.create_curriculum(strategy_type)

            prev_risk = 0
            for level in DifficultyLevel:
                sizing_rules = curriculum.get_position_sizing_rules(level)

                # Risk should generally increase or stay same
                current_risk = sizing_rules['risk_per_trade']
                assert current_risk >= prev_risk
                prev_risk = current_risk

    def test_market_exposure_consistency(self):
        """Test that market exposure rules are consistent."""
        curriculum = IronCondorCurriculum()

        for level in DifficultyLevel:
            exposure_rules = curriculum.get_market_exposure_rules(level)

            assert 'preferred_regimes' in exposure_rules
            assert 'avoid_regimes' in exposure_rules
            assert 'regime_change_frequency' in exposure_rules

            # Should have some preferred regimes
            assert len(exposure_rules['preferred_regimes']) > 0

    def test_curriculum_parameter_bounds(self):
        """Test that curriculum parameters stay within reasonable bounds."""
        test_strategies = [StrategyType.IRON_CONDOR, StrategyType.LONG_STRADDLE]

        for strategy_type in test_strategies:
            curriculum = CurriculumFactory.create_curriculum(strategy_type)

            for level in DifficultyLevel:
                sizing_rules = curriculum.get_position_sizing_rules(level)

                # Risk per trade should be reasonable
                assert 0 < sizing_rules['risk_per_trade'] <= 0.1  # Max 10%

                # Max contracts should be reasonable
                assert 1 <= sizing_rules['max_contracts'] <= 100

                # Portfolio exposure should be reasonable
                if 'max_portfolio_exposure' in sizing_rules:
                    assert 0 < sizing_rules['max_portfolio_exposure'] <= 0.5  # Max 50%