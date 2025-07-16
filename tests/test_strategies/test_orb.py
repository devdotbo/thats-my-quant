"""
Tests for Opening Range Breakout (ORB) Strategy
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime, time

from src.strategies.examples.orb import OpeningRangeBreakout, RangeType


class TestOpeningRangeBreakout:
    """Test Opening Range Breakout strategy"""
    
    @pytest.fixture
    def intraday_data(self):
        """Create sample intraday data with opening range"""
        # Create 390 minutes of trading data (9:30 AM to 4:00 PM)
        dates = pd.date_range('2024-01-02 09:30:00', '2024-01-02 16:00:00', freq='1min')
        
        # First 5 minutes: establish range (100-101)
        range_high = 101.0
        range_low = 100.0
        
        # Create price action
        prices = []
        for i, dt in enumerate(dates):
            if i < 5:  # First 5 minutes
                price = 100.5 + 0.3 * np.sin(i)
            elif i < 60:  # Breakout upward
                price = range_high + (i - 5) * 0.02
            elif i < 120:  # Pull back
                price = range_high + 1.0 - (i - 60) * 0.01
            else:  # Trend up again
                price = range_high + 0.4 + (i - 120) * 0.005
            prices.append(price)
        
        prices = np.array(prices)
        
        df = pd.DataFrame({
            'open': prices + np.random.RandomState(42).randn(len(prices)) * 0.01,
            'high': prices + np.abs(np.random.RandomState(43).randn(len(prices)) * 0.02),
            'low': prices - np.abs(np.random.RandomState(44).randn(len(prices)) * 0.02),
            'close': prices,
            'volume': np.random.RandomState(45).randint(10000, 50000, len(prices))
        }, index=dates)
        
        # Ensure first 5 bars stay within range
        df.loc[df.index[:5], 'high'] = np.clip(df.loc[df.index[:5], 'high'], None, range_high)
        df.loc[df.index[:5], 'low'] = np.clip(df.loc[df.index[:5], 'low'], range_low, None)
        
        return df
    
    @pytest.fixture
    def multi_day_data(self):
        """Create multi-day intraday data"""
        dfs = []
        
        for day in range(5):
            date = pd.Timestamp(f'2024-01-0{day+2}')
            times = pd.date_range(f'{date.date()} 09:30:00', f'{date.date()} 16:00:00', freq='1min')
            
            # Different patterns each day
            if day == 0:  # Upward breakout
                base_price = 100
                trend = 0.01
            elif day == 1:  # Downward breakout
                base_price = 102
                trend = -0.01
            elif day == 2:  # No breakout (choppy)
                base_price = 101
                trend = 0
            elif day == 3:  # Strong upward
                base_price = 100.5
                trend = 0.02
            else:  # Strong downward
                base_price = 103
                trend = -0.015
            
            prices = base_price + np.cumsum(np.random.RandomState(42 + day).randn(len(times)) * 0.1 + trend)
            
            df = pd.DataFrame({
                'open': prices + np.random.RandomState(43 + day).randn(len(times)) * 0.05,
                'high': prices + np.abs(np.random.RandomState(44 + day).randn(len(times)) * 0.1),
                'low': prices - np.abs(np.random.RandomState(45 + day).randn(len(times)) * 0.1),
                'close': prices,
                'volume': np.random.RandomState(46 + day).randint(10000, 50000, len(times))
            }, index=times)
            
            dfs.append(df)
        
        return pd.concat(dfs)
    
    def test_strategy_initialization(self):
        """Test strategy initialization with different parameters"""
        # Default initialization
        strategy = OpeningRangeBreakout()
        assert strategy.parameters['range_minutes'] == 5
        assert strategy.parameters['range_type'] == 'high_low'
        assert strategy.parameters['stop_type'] == 'range'
        assert strategy.parameters['profit_target_r'] == 10.0
        assert strategy.parameters['exit_at_close'] == True
        
        # Custom parameters
        custom_params = {
            'range_minutes': 15,
            'range_type': 'close',
            'stop_type': 'atr',
            'atr_stop_multiplier': 0.05,
            'profit_target_r': 5.0,
            'position_sizing': 'volatility',
            'risk_per_trade': 0.01
        }
        strategy = OpeningRangeBreakout(parameters=custom_params)
        assert strategy.parameters['range_minutes'] == 15
        assert strategy.parameters['stop_type'] == 'atr'
        assert strategy.parameters['atr_stop_multiplier'] == 0.05
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameters should work
        valid_params = {'range_minutes': 10, 'profit_target_r': 8.0}
        strategy = OpeningRangeBreakout(parameters=valid_params)
        assert strategy is not None
        
        # Invalid range_minutes
        with pytest.raises(ValueError, match="range_minutes must be positive"):
            OpeningRangeBreakout(parameters={'range_minutes': 0})
        
        with pytest.raises(ValueError, match="range_minutes must be positive"):
            OpeningRangeBreakout(parameters={'range_minutes': -5})
        
        # Invalid profit target
        with pytest.raises(ValueError, match="profit_target_r must be positive"):
            OpeningRangeBreakout(parameters={'profit_target_r': 0})
        
        # Invalid stop type
        with pytest.raises(ValueError, match="stop_type must be"):
            OpeningRangeBreakout(parameters={'stop_type': 'invalid'})
        
        # Invalid range type
        with pytest.raises(ValueError, match="range_type must be"):
            OpeningRangeBreakout(parameters={'range_type': 'invalid'})
    
    def test_range_detection(self, intraday_data):
        """Test opening range detection"""
        strategy = OpeningRangeBreakout()
        
        # Test range detection for first 5 minutes
        range_data = intraday_data.iloc[:5]
        range_high, range_low = strategy._calculate_opening_range(range_data)
        
        # Range should encompass all first 5 bars
        assert range_high >= range_data['high'].max()
        assert range_low <= range_data['low'].min()
        assert range_high > range_low
    
    def test_signal_generation_long(self, intraday_data):
        """Test long signal generation on upward breakout"""
        strategy = OpeningRangeBreakout()
        signals = strategy.generate_signals(intraday_data)
        
        # Should have no signals during opening range
        assert all(signals.iloc[:5] == 0)
        
        # Should generate long signal after breakout
        long_signals = signals[signals > 0]
        assert len(long_signals) > 0
        
        # First long signal should be after range period
        first_long_idx = long_signals.index[0]
        assert first_long_idx > intraday_data.index[4]
    
    def test_signal_generation_short(self):
        """Test short signal generation on downward breakout"""
        # Create data with downward breakout
        dates = pd.date_range('2024-01-02 09:30:00', '2024-01-02 16:00:00', freq='1min')
        
        # First 5 minutes: establish range (100-101)
        range_high = 101.0
        range_low = 100.0
        
        prices = []
        for i, dt in enumerate(dates):
            if i < 5:  # First 5 minutes
                price = 100.5
            elif i < 60:  # Breakout downward
                price = range_low - (i - 5) * 0.02
            else:  # Continue down
                price = range_low - 1.0 - (i - 60) * 0.005
            prices.append(price)
        
        prices = np.array(prices)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + 0.1,
            'low': prices - 0.1,
            'close': prices,
            'volume': 10000
        }, index=dates)
        
        # Ensure first 5 bars stay within range
        df.loc[df.index[:5], 'high'] = np.clip(df.loc[df.index[:5], 'high'], None, range_high)
        df.loc[df.index[:5], 'low'] = np.clip(df.loc[df.index[:5], 'low'], range_low, None)
        
        strategy = OpeningRangeBreakout()
        signals = strategy.generate_signals(df)
        
        # Should generate short signal after downward breakout
        short_signals = signals[signals < 0]
        assert len(short_signals) > 0
    
    def test_no_signal_in_range(self):
        """Test no signals when price stays within range"""
        dates = pd.date_range('2024-01-02 09:30:00', '2024-01-02 16:00:00', freq='1min')
        
        # Price oscillates within range all day
        prices = 100.5 + 0.4 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + 0.1,
            'low': prices - 0.1,
            'close': prices,
            'volume': 10000
        }, index=dates)
        
        strategy = OpeningRangeBreakout()
        signals = strategy.generate_signals(df)
        
        # Should have few or no signals
        assert signals.abs().sum() <= 2  # Allow for possible noise
    
    def test_position_sizing(self, intraday_data):
        """Test position sizing calculation"""
        strategy = OpeningRangeBreakout(parameters={'risk_per_trade': 0.01})
        
        signals = strategy.generate_signals(intraday_data)
        capital = 100000
        
        positions = strategy.calculate_positions(
            signals=signals,
            capital=capital,
            current_positions=None,
            risk_params={'max_position_size': 0.1}
        )
        
        # Positions should be non-negative
        assert all(positions >= 0)
        
        # Position size should respect risk limits
        max_position_value = positions.max() * intraday_data['close'].max()
        assert max_position_value <= capital * 0.1
    
    def test_multi_day_signals(self, multi_day_data):
        """Test signal generation across multiple days"""
        strategy = OpeningRangeBreakout()
        signals = strategy.generate_signals(multi_day_data)
        
        # Should generate signals on multiple days
        signal_days = signals[signals != 0].index.date
        unique_days = np.unique(signal_days)
        assert len(unique_days) >= 3  # At least 3 days should have signals
        
        # Each day should respect its own opening range
        for date in pd.date_range('2024-01-02', '2024-01-06'):
            day_data = multi_day_data[multi_day_data.index.date == date.date()]
            if len(day_data) > 0:
                day_signals = signals[signals.index.date == date.date()]
                # No signals in first range_minutes
                first_n = min(5, len(day_signals))
                assert all(day_signals.iloc[:first_n] == 0)
    
    def test_exit_at_close(self, intraday_data):
        """Test exit at market close functionality"""
        strategy = OpeningRangeBreakout(parameters={'exit_at_close': True})
        
        # Add exit time information to strategy
        signals = strategy.generate_signals(intraday_data)
        
        # Last bar of the day should not have an active signal
        # (implementation will handle exit logic in backtesting)
        assert signals.iloc[-1] == 0 or signals.iloc[-1] == -999  # -999 might indicate exit
    
    def test_stop_loss_calculation(self, intraday_data):
        """Test stop loss calculation for different stop types"""
        # Test range-based stop
        strategy_range = OpeningRangeBreakout(parameters={'stop_type': 'range'})
        range_high, range_low = strategy_range._calculate_opening_range(intraday_data.iloc[:5])
        
        # For long position, stop should be at range low
        # For short position, stop should be at range high
        assert range_high > range_low
        
        # Test ATR-based stop
        strategy_atr = OpeningRangeBreakout(parameters={
            'stop_type': 'atr',
            'atr_stop_multiplier': 0.05,
            'atr_period': 14
        })
        
        # ATR calculation should work
        atr = strategy_atr._calculate_atr(intraday_data)
        assert len(atr) == len(intraday_data)
        assert all(atr > 0)
    
    def test_metadata(self):
        """Test strategy metadata"""
        strategy = OpeningRangeBreakout()
        metadata = strategy.get_metadata()
        
        assert metadata.name == "Opening Range Breakout"
        assert metadata.supports_short == True
        assert metadata.supports_intraday == True
        assert metadata.required_history >= 14  # For ATR calculation
        assert 'range_minutes' in metadata.parameters
    
    def test_performance(self, multi_day_data):
        """Test strategy performance"""
        strategy = OpeningRangeBreakout()
        
        import time
        start = time.time()
        signals = strategy.generate_signals(multi_day_data)
        elapsed = time.time() - start
        
        # Should process signals quickly
        bars_per_second = len(multi_day_data) / elapsed
        assert bars_per_second > 10000  # Should process >10k bars/second
        
        # Signals should be valid
        assert len(signals) == len(multi_day_data)
        assert signals.isnull().sum() == 0