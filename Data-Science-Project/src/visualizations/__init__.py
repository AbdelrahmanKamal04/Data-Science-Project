"""
Data analysis module.
Provides functionality for univariate, bivariate, multivariate, temporal, and interaction analysis.
"""

from .univariate import univariate_analysis, UnivariateAnalyzer
from .bivariate_categorical import bivariate_categorical_analysis, BivariateCategoricalAnalyzer
from .bivariate_continuous import bivariate_continuous_analysis, BivariateContinuousAnalyzer
from .multivariate import multivariate_analysis, MultivariateAnalyzer
from .temporal import temporal_analysis, TemporalAnalyzer
from .interaction import interaction_analysis, InteractionAnalyzer
from .data_loader import load_analysis_data

__all__ = [
    'univariate_analysis',
    'UnivariateAnalyzer',
    'bivariate_categorical_analysis',
    'BivariateCategoricalAnalyzer',
    'bivariate_continuous_analysis',
    'BivariateContinuousAnalyzer',
    'multivariate_analysis',
    'MultivariateAnalyzer',
    'temporal_analysis',
    'TemporalAnalyzer',
    'interaction_analysis',
    'InteractionAnalyzer',
    'load_analysis_data'
]