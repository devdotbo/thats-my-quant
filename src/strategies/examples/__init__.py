"""
Example trading strategies
"""

from .moving_average import MovingAverageCrossover
from .orb import OpeningRangeBreakout, RangeType, StopType

__all__ = ['MovingAverageCrossover', 'OpeningRangeBreakout', 'RangeType', 'StopType']