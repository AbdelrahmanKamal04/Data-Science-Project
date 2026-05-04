"""
Data analysis module.
Provides functionality for univariate, bivariate, multivariate, temporal, and interaction analysis.
"""

from .univariate import univariate_analysis
from .bivariate_categorical import bivariate_categorical_analysis
from .bivariate_continuous import bivariate_continuous_analysis
from .multivariate import multivariate_analysis
from .temporal import temporal_analysis
from .interaction import interaction_analysis
from .data_loader import load_datasets

__all__ = [
    'univariate_analysis',
    'bivariate_categorical_analysis',
    'bivariate_continuous_analysis',
    'multivariate_analysis',
    'temporal_analysis',
    'interaction_analysis',
    'load_datasets'
]