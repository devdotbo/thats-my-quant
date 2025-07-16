"""
Validation Framework for Quantitative Trading Strategies

This module provides various validation techniques to ensure
strategy robustness and detect overfitting.
"""

from .walk_forward import (
    WalkForwardValidator,
    WalkForwardWindow,
    WalkForwardResult,
    WindowType,
    OptimizationMetric
)

__all__ = [
    'WalkForwardValidator',
    'WalkForwardWindow', 
    'WalkForwardResult',
    'WindowType',
    'OptimizationMetric'
]