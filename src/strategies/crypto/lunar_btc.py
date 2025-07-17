"""
Lunar Bitcoin Trading Strategy
Trades Bitcoin based on moon phases and lunar cycles
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.strategies.base import BaseStrategy, StrategyMetadata
from src.features.lunar_features import LunarCalculator


class LunarBitcoinStrategy(BaseStrategy):
    """
    Trade Bitcoin based on lunar cycles
    
    Multiple trading approaches:
    1. Classic: Buy new moon, sell full moon
    2. Momentum: Buy after full moon for 3-4 days
    3. Distance: Trade based on moon distance (apogee/perigee)
    4. Combined: Use moon phase + technical indicators
    
    Parameters:
        strategy_type: Type of lunar strategy to use
        entry_phase_min: Minimum moon phase to enter position
        entry_phase_max: Maximum moon phase to enter position
        exit_phase_min: Minimum moon phase to exit position
        exit_phase_max: Maximum moon phase to exit position
        hold_days: Days to hold position after signal
        use_distance: Whether to consider moon distance
        distance_threshold: Distance ratio threshold for signals
        stop_loss: Stop loss percentage (0.02 = 2%)
        use_trailing_stop: Whether to use trailing stop
        min_volume_percentile: Minimum volume percentile to trade
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """Initialize lunar strategy with parameters"""
        # Default parameters
        default_params = {
            'strategy_type': 'momentum',  # classic, momentum, distance, combined
            'entry_phase_min': 0.48,  # Full moon range start
            'entry_phase_max': 0.52,  # Full moon range end
            'exit_phase_min': 0.0,    # Exit conditions
            'exit_phase_max': 0.0,
            'hold_days': 3,           # Days to hold after signal
            'use_distance': False,    # Consider moon distance
            'distance_threshold': 1.03,  # Distance ratio for apogee
            'stop_loss': 0.02,        # 2% stop loss
            'use_trailing_stop': True,
            'min_volume_percentile': 20  # Minimum volume filter
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
            
        super().__init__(default_params)
        
        # Initialize lunar calculator
        self.lunar_calc = LunarCalculator()
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on lunar cycles
        
        Args:
            data: DataFrame with OHLCV data and DatetimeIndex
            
        Returns:
            Series of trading signals (-1, 0, 1)
        """
        # Validate data
        self.validate_data(data)
        
        # Add lunar features if not present
        if 'phase' not in data.columns:
            data = self.lunar_calc.add_lunar_features_to_data(data)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Apply volume filter
        if self.parameters['min_volume_percentile'] > 0:
            volume_threshold = data['volume'].quantile(
                self.parameters['min_volume_percentile'] / 100
            )
            low_volume_mask = data['volume'] < volume_threshold
        else:
            low_volume_mask = pd.Series(False, index=data.index)
        
        # Generate signals based on strategy type
        strategy_type = self.parameters['strategy_type']
        
        if strategy_type == 'classic':
            signals = self._generate_classic_signals(data)
            
        elif strategy_type == 'momentum':
            signals = self._generate_momentum_signals(data)
            
        elif strategy_type == 'distance':
            signals = self._generate_distance_signals(data)
            
        elif strategy_type == 'combined':
            signals = self._generate_combined_signals(data)
            
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        # Apply volume filter
        signals[low_volume_mask] = 0
        
        # Apply stop loss if configured
        if self.parameters['stop_loss'] > 0:
            signals = self._apply_stop_loss(signals, data)
        
        return signals
    
    def _generate_classic_signals(self, data: pd.DataFrame) -> pd.Series:
        """Classic lunar trading: buy new moon, sell full moon"""
        signals = pd.Series(0, index=data.index)
        
        # Buy on new moon
        new_moon_mask = (data['phase'] < 0.05) | (data['phase'] > 0.95)
        signals[new_moon_mask] = 1
        
        # Sell on full moon
        full_moon_mask = (data['phase'] > 0.45) & (data['phase'] < 0.55)
        signals[full_moon_mask] = -1
        
        return signals
    
    def _generate_momentum_signals(self, data: pd.DataFrame) -> pd.Series:
        """Momentum trading: buy after full moon"""
        signals = pd.Series(0, index=data.index)
        hold_days = self.parameters['hold_days']
        
        # Find full moon days
        phase_min = self.parameters['entry_phase_min']
        phase_max = self.parameters['entry_phase_max']
        full_moon_mask = (data['phase'] >= phase_min) & (data['phase'] <= phase_max)
        full_moon_days = data[full_moon_mask].index
        
        # Generate buy signals for N days after each full moon
        for fm_date in full_moon_days:
            # Find next N trading days
            fm_idx = data.index.get_loc(fm_date)
            end_idx = min(fm_idx + hold_days + 1, len(data))
            
            # Set buy signals
            signals.iloc[fm_idx + 1:end_idx] = 1
        
        return signals
    
    def _generate_distance_signals(self, data: pd.DataFrame) -> pd.Series:
        """Trade based on moon distance (apogee/perigee)"""
        signals = pd.Series(0, index=data.index)
        threshold = self.parameters['distance_threshold']
        
        # Buy when moon is far (apogee)
        apogee_mask = data['distance_ratio'] > threshold
        signals[apogee_mask] = 1
        
        # Sell when moon is close (perigee)
        perigee_mask = data['distance_ratio'] < (2 - threshold)
        signals[perigee_mask] = -1
        
        return signals
    
    def _generate_combined_signals(self, data: pd.DataFrame) -> pd.Series:
        """Combine lunar signals with technical indicators"""
        signals = pd.Series(0, index=data.index)
        
        # Start with momentum signals
        lunar_signals = self._generate_momentum_signals(data)
        
        # Add RSI confirmation if available
        if 'rsi' in data.columns:
            # Only buy if RSI < 70 (not overbought)
            buy_mask = (lunar_signals == 1) & (data['rsi'] < 70)
            # Only sell if RSI > 30 (not oversold)
            sell_mask = (lunar_signals == -1) & (data['rsi'] > 30)
            
            signals[buy_mask] = 1
            signals[sell_mask] = -1
        else:
            signals = lunar_signals
            
        # Add distance filter if enabled
        if self.parameters['use_distance']:
            # Strengthen signals when at distance extremes
            distance_signals = self._generate_distance_signals(data)
            
            # Only take signals that agree
            signals = signals * (signals == distance_signals)
            
        return signals
    
    def _apply_stop_loss(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Apply stop loss to signals"""
        stop_loss_pct = self.parameters['stop_loss']
        use_trailing = self.parameters['use_trailing_stop']
        
        # Track positions and stops
        position = 0
        entry_price = 0
        highest_price = 0
        stop_price = 0
        
        modified_signals = signals.copy()
        
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            
            # Update position based on signal
            if signals.iloc[i] != 0:
                position = signals.iloc[i]
                entry_price = current_price
                highest_price = current_price
                stop_price = entry_price * (1 - stop_loss_pct) if position > 0 else 0
                
            # Check stop loss if in position
            elif position > 0:
                # Update trailing stop
                if use_trailing and current_price > highest_price:
                    highest_price = current_price
                    stop_price = highest_price * (1 - stop_loss_pct)
                
                # Check if stopped out
                if current_price < stop_price:
                    modified_signals.iloc[i] = -1  # Exit signal
                    position = 0
                    
            elif position < 0:
                # For short positions
                if use_trailing and current_price < highest_price:
                    highest_price = current_price
                    stop_price = highest_price * (1 + stop_loss_pct)
                    
                if current_price > stop_price:
                    modified_signals.iloc[i] = 1  # Cover signal
                    position = 0
        
        return modified_signals
    
    def calculate_positions(self, 
                          signals: pd.Series,
                          capital: float,
                          current_positions: Optional[pd.Series] = None,
                          risk_params: Optional[Dict[str, Any]] = None) -> pd.Series:
        """
        Convert signals to position sizes
        
        For crypto, we'll use fixed fractional position sizing
        """
        # Default risk parameters for crypto
        if risk_params is None:
            risk_params = {
                'max_position_size': 0.95,  # Can use most capital (no margin)
                'position_sizing_method': 'fixed'
            }
        
        positions = pd.Series(0, index=signals.index, dtype=float)
        
        # Simple position sizing for crypto
        position_size = capital * risk_params.get('max_position_size', 0.95)
        
        # Convert signals to positions
        for i in range(len(signals)):
            if signals.iloc[i] != 0:
                # For crypto, position size is in USD value
                # Will need to be converted to BTC amount by backtester
                positions.iloc[i] = position_size * signals.iloc[i]
        
        return positions
    
    @property
    def required_history(self) -> int:
        """Minimum history needed"""
        # Need at least one full lunar cycle (29.5 days)
        return 30
    
    @property 
    def required_features(self) -> List[str]:
        """Required data features"""
        # Basic lunar features will be calculated if not present
        features = []
        
        # Add technical indicators for combined strategy
        if self.parameters.get('strategy_type') == 'combined':
            features.extend(['rsi'])
            
        return features
    
    def _create_metadata(self) -> StrategyMetadata:
        """Create strategy metadata"""
        return StrategyMetadata(
            name="Lunar Bitcoin Trading Strategy",
            version="1.0.0",
            author="That's My Quant",
            description="Trades Bitcoin based on moon phases and lunar cycles",
            parameters=self.parameters,
            required_history=self.required_history,
            required_features=self.required_features,
            supports_short=True,
            supports_intraday=True
        )
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters"""
        params = self.parameters
        
        # Validate strategy type
        valid_types = ['classic', 'momentum', 'distance', 'combined']
        if params['strategy_type'] not in valid_types:
            raise ValueError(f"strategy_type must be one of {valid_types}")
        
        # Validate phase ranges
        for key in ['entry_phase_min', 'entry_phase_max', 'exit_phase_min', 'exit_phase_max']:
            if not 0 <= params[key] <= 1:
                raise ValueError(f"{key} must be between 0 and 1")
                
        if params['entry_phase_min'] > params['entry_phase_max']:
            raise ValueError("entry_phase_min must be <= entry_phase_max")
            
        # Validate other parameters
        if params['hold_days'] < 0:
            raise ValueError("hold_days must be non-negative")
            
        if params['stop_loss'] < 0 or params['stop_loss'] > 1:
            raise ValueError("stop_loss must be between 0 and 1")
            
        if params['distance_threshold'] <= 0:
            raise ValueError("distance_threshold must be positive")
            
        if params['min_volume_percentile'] < 0 or params['min_volume_percentile'] > 100:
            raise ValueError("min_volume_percentile must be between 0 and 100")