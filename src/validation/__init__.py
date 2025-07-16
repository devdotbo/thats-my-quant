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

from .monte_carlo import (
    MonteCarloValidator,
    MonteCarloResult,
    ResamplingMethod,
    ConfidenceLevel
)

__all__ = [
    # Walk-Forward
    'WalkForwardValidator',
    'WalkForwardWindow', 
    'WalkForwardResult',
    'WindowType',
    'OptimizationMetric',
    # Monte Carlo
    'MonteCarloValidator',
    'MonteCarloResult',
    'ResamplingMethod',
    'ConfidenceLevel'
]