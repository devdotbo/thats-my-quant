"""
Moving Average Crossover Strategy
A classic trend-following strategy using MA crossovers
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from src.strategies.base import BaseStrategy, SignalType, StrategyMetadata
from src.utils.logging import get_logger


class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    Generates trading signals based on the crossover of fast and slow moving averages.
    Supports SMA, EMA, and WMA with optional volume and volatility filters.
    
    Parameters:
        fast_period (int): Period for fast moving average (default: 10)
        slow_period (int): Period for slow moving average (default: 30)
        ma_type (str): Type of MA - 'sma', 'ema', or 'wma' (default: 'sma')
        use_volume_filter (bool): Filter signals by volume (default: False)
        volume_threshold (float): Volume must be this multiple of average (default: 1.2)
        position_sizing (str): 'fixed' or 'volatility' (default: 'fixed')
        risk_per_trade (float): Risk per trade as fraction of capital (default: 0.02)
        stop_loss (float): Stop loss percentage (default: 0.02)
        take_profit (float): Take profit percentage (default: 0.05)
        use_stops (bool): Whether to use stop loss/take profit (default: False)
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """Initialize with default parameters"""
        default_params = {
            'fast_period': 10,
            'slow_period': 30,
            'ma_type': 'sma',
            'use_volume_filter': False,
            'volume_threshold': 1.2,
            'position_sizing': 'fixed',
            'risk_per_trade': 0.02,
            'stop_loss': 0.02,
            'take_profit': 0.05,
            'use_stops': False
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(default_params)
        self.logger = get_logger("ma_crossover_strategy")
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters"""
        # Check periods
        if self.parameters.get('fast_period', 0) <= 0 or self.parameters.get('slow_period', 0) <= 0:
            raise ValueError("Periods must be positive")
        
        if self.parameters['fast_period'] >= self.parameters['slow_period']:
            raise ValueError("Fast period must be less than slow period")
        
        # Check MA type
        valid_ma_types = ['sma', 'ema', 'wma']
        if self.parameters['ma_type'] not in valid_ma_types:
            raise ValueError(f"ma_type must be one of {valid_ma_types}")
        
        # Check risk parameters
        if self.parameters.get('risk_per_trade', 0) <= 0 or self.parameters.get('risk_per_trade', 1) > 0.1:
            self.logger.warning("Risk per trade should be between 0 and 0.1 (10%)")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MA crossover
        
        Args:
            data: Market data with OHLCV
            
        Returns:
            Series of signals (-1, 0, 1)
        """
        # Validate data
        self.validate_data(data)
        
        # Calculate moving averages
        fast_ma = self._calculate_ma(data['close'], self.parameters['fast_period'], self.parameters['ma_type'])
        slow_ma = self._calculate_ma(data['close'], self.parameters['slow_period'], self.parameters['ma_type'])
        
        # Generate base signals
        signals = pd.Series(0, index=data.index, dtype=float)
        
        # Create position based on MA relationship
        # 1 when fast > slow, -1 when fast < slow
        ma_position = pd.Series(0, index=data.index, dtype=float)
        ma_position[fast_ma > slow_ma] = 1
        ma_position[fast_ma < slow_ma] = -1
        
        # Only start after we have valid MAs
        valid_idx = ~(fast_ma.isna() | slow_ma.isna())
        if valid_idx.any():
            first_valid = valid_idx.idxmax()
            signals.loc[first_valid:] = ma_position.loc[first_valid:]
        
        # Apply volume filter if enabled
        if self.parameters['use_volume_filter'] and 'volume_ma' in data.columns:
            volume_filter = data['volume'] > (data['volume_ma'] * self.parameters['volume_threshold'])
            # Only allow new signals when volume is high
            signal_changes = signals.diff() != 0
            signals[signal_changes & ~volume_filter] = signals.shift(1)[signal_changes & ~volume_filter]
        
        # Apply risk management if enabled
        if self.parameters['use_stops']:
            signals = self.apply_risk_management(signals, data['close'])
        
        return signals
    
    def calculate_positions(self, 
                          signals: pd.Series,
                          capital: float,
                          current_positions: Optional[pd.Series] = None,
                          risk_params: Optional[Dict[str, Any]] = None) -> pd.Series:
        """
        Calculate position sizes based on signals and risk management
        
        Args:
            signals: Trading signals
            capital: Available capital
            current_positions: Current positions (not used)
            risk_params: Additional risk parameters (e.g., 'atr' for volatility sizing)
            
        Returns:
            Position sizes in shares
        """
        positions = pd.Series(0, index=signals.index)
        
        if self.parameters['position_sizing'] == 'fixed':
            # Fixed position sizing
            for i, signal in enumerate(signals):
                if signal != 0:
                    # Assume we can get price from index or use fixed estimate
                    price = 100  # Simplified - in real implementation would get actual price
                    position_size = self.calculate_position_size_fixed(
                        signal, capital, price, self.parameters['risk_per_trade']
                    )
                    positions.iloc[i] = position_size
        
        elif self.parameters['position_sizing'] == 'volatility':
            # Volatility-based sizing
            if risk_params and 'atr' in risk_params:
                atr = risk_params['atr']
                for i, signal in enumerate(signals):
                    if signal != 0 and not pd.isna(atr.iloc[i]):
                        price = 100  # Simplified
                        volatility = atr.iloc[i] / price  # ATR as % of price
                        position_size = self.calculate_position_size_volatility(
                            signal, capital, price, volatility, self.parameters['risk_per_trade']
                        )
                        positions.iloc[i] = position_size
            else:
                self.logger.warning("Volatility sizing requested but no ATR provided")
                # Fall back to fixed sizing
                return self.calculate_positions(
                    signals, capital, current_positions, 
                    {'position_sizing': 'fixed'}
                )
        
        return positions
    
    def apply_risk_management(self, signals: pd.Series, prices: pd.Series) -> pd.Series:
        """
        Apply stop loss and take profit to signals
        
        Args:
            signals: Original signals
            prices: Price series
            
        Returns:
            Modified signals with stops applied
        """
        modified_signals = signals.copy()
        
        position_start_price = None
        position_signal = 0
        
        for i in range(len(signals)):
            current_signal = signals.iloc[i]
            current_price = prices.iloc[i]
            
            # New position
            if current_signal != position_signal and current_signal != 0:
                position_start_price = current_price
                position_signal = current_signal
            
            # Check stops if in position
            elif position_signal != 0 and position_start_price is not None:
                price_change = (current_price - position_start_price) / position_start_price
                
                if position_signal > 0:  # Long position
                    # Stop loss
                    if price_change <= -self.parameters['stop_loss']:
                        modified_signals.iloc[i] = 0
                        position_signal = 0
                    # Take profit
                    elif price_change >= self.parameters['take_profit']:
                        modified_signals.iloc[i] = 0
                        position_signal = 0
                
                else:  # Short position
                    # Stop loss (price went up)
                    if price_change >= self.parameters['stop_loss']:
                        modified_signals.iloc[i] = 0
                        position_signal = 0
                    # Take profit (price went down)
                    elif price_change <= -self.parameters['take_profit']:
                        modified_signals.iloc[i] = 0
                        position_signal = 0
        
        return modified_signals
    
    @property
    def required_history(self) -> int:
        """Minimum history required"""
        return self.parameters['slow_period']
    
    @property
    def required_features(self) -> List[str]:
        """Required additional features"""
        features = []
        
        if self.parameters['use_volume_filter']:
            features.append('volume_ma')
        
        if self.parameters['position_sizing'] == 'volatility':
            features.append('atr')
        
        return features
    
    def _create_metadata(self) -> StrategyMetadata:
        """Create strategy metadata"""
        return StrategyMetadata(
            name="Moving Average Crossover",
            version="1.0",
            author="That's My Quant",
            description="Classic trend-following strategy using MA crossovers",
            parameters=self.parameters,
            required_history=self.required_history,
            required_features=self.required_features,
            supports_short=True,
            supports_intraday=True
        )
    
    def _calculate_ma(self, series: pd.Series, period: int, ma_type: str) -> pd.Series:
        """
        Calculate moving average of specified type
        
        Args:
            series: Price series
            period: MA period
            ma_type: 'sma', 'ema', or 'wma'
            
        Returns:
            Moving average series
        """
        if ma_type == 'sma':
            return series.rolling(window=period).mean()
        
        elif ma_type == 'ema':
            return series.ewm(span=period, adjust=False).mean()
        
        elif ma_type == 'wma':
            # Weighted moving average
            weights = np.arange(1, period + 1)
            
            def weighted_mean(x):
                if len(x) == period:
                    return np.average(x, weights=weights)
                return np.nan
            
            return series.rolling(window=period).apply(weighted_mean, raw=True)
        
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")