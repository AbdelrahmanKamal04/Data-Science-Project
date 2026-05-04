"""
Feature engineering module.
Provides functionality for feature encoding, scaling, engineering, and data splitting.
"""

from .encoding import encode_features, FeatureEncoder
from .engineering import engineer_features, FeatureEngineer
from .scaling import scale_features, FeatureScaler
from .split import split_data, DataSplitter
from .save_data import save_data

__all__ = [
    'encode_features',
    'FeatureEncoder',
    'engineer_features',
    'FeatureEngineer',
    'scale_features',
    'FeatureScaler',
    'split_data',
    'DataSplitter',
    'save_data'
]