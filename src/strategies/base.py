"""
Base Strategy Interface
All trading strategies must inherit from this abstract base class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Trading signal types"""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0
    
    @classmethod
    def from_value(cls, value: Union[int, float]) -> 'SignalType':
        """Convert numeric value to SignalType"""
        if value > 0:
            return cls.LONG
        elif value < 0:
            return cls.SHORT
        else:
            return cls.NEUTRAL


@dataclass
class StrategyMetadata:
    """Metadata about a strategy"""
    name: str
    version: str
    author: str
    description: str
    parameters: Dict[str, Any]
    required_history: int
    required_features: List[str]
    supports_short: bool = True
    supports_intraday: bool = True


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy with parameters
        
        Args:
            parameters: Strategy-specific parameters
        """
        self.parameters = parameters or {}
        self._validate_parameters()
        self._metadata = self._create_metadata()
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on market data
        
        Args:
            data: DataFrame with OHLCV data and any additional features
                 Must have columns: ['open', 'high', 'low', 'close', 'volume']
                 Index should be DatetimeIndex
        
        Returns:
            pd.Series: Signal values (-1: short, 0: neutral, 1: long)
                      Index must match input data index
        
        Example:
            >>> data = pd.DataFrame({
            ...     'open': [100, 101, 102],
            ...     'high': [101, 102, 103],
            ...     'low': [99, 100, 101],
            ...     'close': [100.5, 101.5, 102.5],
            ...     'volume': [1000, 1100, 1200]
            ... }, index=pd.date_range('2023-01-01', periods=3))
            >>> signals = strategy.generate_signals(data)
            >>> assert len(signals) == len(data)
        """
        pass
    
    @abstractmethod
    def calculate_positions(self, 
                          signals: pd.Series,
                          capital: float,
                          current_positions: Optional[pd.Series] = None,
                          risk_params: Optional[Dict[str, Any]] = None) -> pd.Series:
        """
        Convert signals to actual position sizes
        
        Args:
            signals: Trading signals from generate_signals()
            capital: Available capital for trading
            current_positions: Current position sizes (optional)
            risk_params: Risk management parameters (optional)
                - max_position_size: Maximum position as fraction of capital
                - stop_loss: Stop loss percentage
                - position_sizing_method: 'fixed', 'kelly', 'volatility'
        
        Returns:
            pd.Series: Position sizes in units (shares/contracts)
                      Positive for long, negative for short
        """
        pass
    
    @property
    @abstractmethod
    def required_history(self) -> int:
        """
        Minimum number of historical bars required before generating signals
        
        This is used to ensure enough data is available for indicators
        
        Example:
            If strategy uses 200-day moving average, return 200
        """
        pass
    
    @property
    @abstractmethod
    def required_features(self) -> List[str]:
        """
        List of required data columns beyond basic OHLCV
        
        Example:
            ['atr', 'rsi', 'volume_ma20']
        """
        pass
    
    @abstractmethod
    def _create_metadata(self) -> StrategyMetadata:
        """
        Create strategy metadata
        
        Must be implemented by each strategy to provide information
        """
        pass
    
    def _validate_parameters(self) -> None:
        """
        Validate strategy parameters
        
        Override this method to add custom parameter validation
        Default implementation does nothing
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that input data meets strategy requirements
        
        Args:
            data: Input market data
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValueError: If data is invalid
        """
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check additional required features
        missing_features = [feat for feat in self.required_features 
                          if feat not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Check minimum history
        if len(data) < self.required_history:
            raise ValueError(
                f"Insufficient data: {len(data)} bars provided, "
                f"{self.required_history} required"
            )
        
        # Check data types
        numeric_cols = required_cols + self.required_features
        for col in numeric_cols:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column '{col}' must be numeric")
        
        # Check for NaN values in critical columns
        critical_cols = ['close', 'volume']
        for col in critical_cols:
            if data[col].isna().any():
                raise ValueError(f"NaN values found in '{col}' column")
        
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get strategy information and current parameters
        
        Returns:
            Dict containing strategy metadata and parameters
        """
        return {
            'name': self._metadata.name,
            'version': self._metadata.version,
            'author': self._metadata.author,
            'description': self._metadata.description,
            'parameters': self.parameters,
            'required_history': self.required_history,
            'required_features': self.required_features,
            'supports_short': self._metadata.supports_short,
            'supports_intraday': self._metadata.supports_intraday
        }
    
    def calculate_position_size_fixed(self, 
                                    signal: float,
                                    capital: float,
                                    price: float,
                                    risk_fraction: float = 0.02) -> int:
        """
        Calculate position size using fixed fractional method
        
        Args:
            signal: Trading signal strength (-1 to 1)
            capital: Available capital
            price: Current asset price
            risk_fraction: Fraction of capital to risk per trade
            
        Returns:
            int: Number of shares/contracts (negative for short)
        """
        if signal == 0:
            return 0
        
        position_value = capital * risk_fraction
        shares = int(position_value / price)
        
        return shares if signal > 0 else -shares
    
    def calculate_position_size_volatility(self,
                                         signal: float,
                                         capital: float,
                                         price: float,
                                         volatility: float,
                                         target_risk: float = 0.02) -> int:
        """
        Calculate position size based on volatility
        
        Args:
            signal: Trading signal strength (-1 to 1)
            capital: Available capital
            price: Current asset price
            volatility: Asset volatility (standard deviation of returns)
            target_risk: Target portfolio risk per position
            
        Returns:
            int: Number of shares/contracts (negative for short)
        """
        if signal == 0 or volatility <= 0:
            return 0
        
        # Calculate position size to achieve target risk
        dollar_risk = capital * target_risk
        price_risk = price * volatility
        
        if price_risk > 0:
            shares = int(dollar_risk / price_risk)
            return shares if signal > 0 else -shares
        
        return 0
    
    def __repr__(self) -> str:
        """String representation of strategy"""
        return (f"{self.__class__.__name__}("
                f"parameters={self.parameters})")
    
    def __str__(self) -> str:
        """Human-readable string representation"""
        return f"{self._metadata.name} v{self._metadata.version}"