"""
Curriculum learning framework for progressive options trading training.

Provides structured difficulty progression through market complexity,
strategy sophistication, and risk parameters to ensure stable and
effective reinforcement learning.
"""

from .levels import (
    DifficultyLevel,
    CurriculumParameters,
    CurriculumLevel,
    get_level_for_performance,
    create_custom_level
)
from .scheduler import (
    CurriculumScheduler,
    PerformanceMetrics
)
from .strategies import (
    StrategyCurriculum,
    StrategyProgressionRules,
    CurriculumFactory,
    IronCondorCurriculum,
    IronButterflyStrategy,
    StraddleStrangleCurriculum,
    VerticalSpreadCurriculum
)
from .adaptation import (
    AdaptiveCurriculum,
    PerformanceAnalyzer,
    PerformanceTrend,
    PlateauDetection,
    DifficultyRecommendation
)

__all__ = [
    # Levels
    'DifficultyLevel',
    'CurriculumParameters',
    'CurriculumLevel',
    'get_level_for_performance',
    'create_custom_level',

    # Scheduler
    'CurriculumScheduler',
    'PerformanceMetrics',

    # Strategies
    'StrategyCurriculum',
    'StrategyProgressionRules',
    'CurriculumFactory',
    'IronCondorCurriculum',
    'IronButterflyStrategy',
    'StraddleStrangleCurriculum',
    'VerticalSpreadCurriculum',

    # Adaptation
    'AdaptiveCurriculum',
    'PerformanceAnalyzer',
    'PerformanceTrend',
    'PlateauDetection',
    'DifficultyRecommendation'
]