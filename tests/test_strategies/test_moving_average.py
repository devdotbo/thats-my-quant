"""
Tests for Moving Average Crossover Strategy
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.strategies.examples.moving_average import MovingAverageCrossover


class TestMovingAverageCrossover:
    """Test Moving Average Crossover strategy"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data with trend"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # Create trending data
        trend = np.linspace(100, 110, 100)
        noise = np.random.RandomState(42).randn(100) * 0.5
        prices = trend + noise
        
        df = pd.DataFrame({
            'open': prices + np.random.RandomState(43).randn(100) * 0.1,
            'high': prices + np.abs(np.random.RandomState(44).randn(100) * 0.2),
            'low': prices - np.abs(np.random.RandomState(45).randn(100) * 0.2),
            'close': prices,
            'volume': np.random.RandomState(46).randint(50000, 150000, 100)
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def choppy_data(self):
        """Create choppy/sideways market data"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # Oscillating prices
        prices = 100 + 5 * np.sin(np.linspace(0, 4*np.pi, 100))
        prices += np.random.RandomState(42).randn(100) * 0.5
        
        df = pd.DataFrame({
            'open': prices + np.random.RandomState(43).randn(100) * 0.1,
            'high': prices + np.abs(np.random.RandomState(44).randn(100) * 0.2),
            'low': prices - np.abs(np.random.RandomState(45).randn(100) * 0.2),
            'close': prices,
            'volume': np.random.RandomState(46).randint(50000, 150000, 100)
        }, index=dates)
        
        return df
    
    def test_strategy_initialization(self):
        """Test strategy initialization with different parameters"""
        # Default initialization
        strategy = MovingAverageCrossover()
        assert strategy.parameters['fast_period'] == 10
        assert strategy.parameters['slow_period'] == 30
        assert strategy.parameters['ma_type'] == 'sma'
        
        # Custom parameters
        custom_params = {
            'fast_period': 20,
            'slow_period': 50,
            'ma_type': 'ema',
            'stop_loss': 0.03,
            'take_profit': 0.06
        }
        strategy = MovingAverageCrossover(parameters=custom_params)
        assert strategy.parameters['fast_period'] == 20
        assert strategy.parameters['slow_period'] == 50
        assert strategy.parameters['ma_type'] == 'ema'
        assert strategy.parameters['stop_loss'] == 0.03
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameters should work
        valid_params = {'fast_period': 5, 'slow_period': 20}
        strategy = MovingAverageCrossover(parameters=valid_params)
        assert strategy is not None
        
        # Fast period >= slow period should raise error
        with pytest.raises(ValueError, match="Fast period must be less than slow period"):
            MovingAverageCrossover(parameters={'fast_period': 20, 'slow_period': 10})
        
        # Invalid MA type should raise error
        with pytest.raises(ValueError, match="ma_type must be one of"):
            MovingAverageCrossover(parameters={'ma_type': 'invalid'})
        
        # Negative periods should raise error
        with pytest.raises(ValueError, match="Periods must be positive"):
            MovingAverageCrossover(parameters={'fast_period': -5})
    
    def test_generate_signals_sma(self, choppy_data):
        """Test signal generation with SMA"""
        strategy = MovingAverageCrossover(parameters={
            'fast_period': 10,
            'slow_period': 20,
            'ma_type': 'sma'
        })
        
        signals = strategy.generate_signals(choppy_data)
        
        # Check signal properties
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(choppy_data)
        assert signals.index.equals(choppy_data.index)
        
        # Check signal values
        assert set(signals.unique()).issubset({-1, 0, 1})
        
        # First slow_period-1 bars should be 0 (no signal)
        assert (signals.iloc[:19] == 0).all()
        
        # Should have signals starting from bar 20 (index 19)
        assert signals.iloc[19] != 0
        
        # Should have some long and short signals
        assert (signals == 1).any()
        assert (signals == -1).any()
    
    def test_generate_signals_ema(self, sample_data):
        """Test signal generation with EMA"""
        strategy_sma = MovingAverageCrossover(parameters={
            'fast_period': 10,
            'slow_period': 20,
            'ma_type': 'sma'
        })
        
        strategy_ema = MovingAverageCrossover(parameters={
            'fast_period': 10,
            'slow_period': 20,
            'ma_type': 'ema'
        })
        
        signals_sma = strategy_sma.generate_signals(sample_data)
        signals_ema = strategy_ema.generate_signals(sample_data)
        
        # EMA should react faster, so signals should be different
        assert not signals_sma.equals(signals_ema)
        
        # Both should have valid signals
        assert set(signals_ema.unique()).issubset({-1, 0, 1})
    
    def test_generate_signals_with_filters(self, sample_data):
        """Test signal generation with volume filter"""
        # Add volume moving average
        sample_data['volume_ma'] = sample_data['volume'].rolling(20).mean()
        
        strategy = MovingAverageCrossover(parameters={
            'fast_period': 10,
            'slow_period': 20,
            'use_volume_filter': True,
            'volume_threshold': 1.2
        })
        
        signals = strategy.generate_signals(sample_data)
        
        # Should have fewer signals with volume filter
        strategy_no_filter = MovingAverageCrossover(parameters={
            'fast_period': 10,
            'slow_period': 20,
            'use_volume_filter': False
        })
        
        signals_no_filter = strategy_no_filter.generate_signals(sample_data)
        
        # Volume filter should reduce number of signals
        assert abs(signals).sum() <= abs(signals_no_filter).sum()
    
    def test_calculate_positions_fixed(self, sample_data):
        """Test fixed position sizing"""
        strategy = MovingAverageCrossover(parameters={
            'position_sizing': 'fixed',
            'risk_per_trade': 0.02
        })
        
        signals = strategy.generate_signals(sample_data)
        positions = strategy.calculate_positions(signals, capital=100000)
        
        # Check position properties
        assert isinstance(positions, pd.Series)
        assert len(positions) == len(signals)
        
        # Positions should align with signals
        mask = signals != 0
        assert ((positions[mask] > 0) == (signals[mask] > 0)).all()
        
        # Check position sizes (assuming average price around 100)
        position_values = positions * 100  # Approximate position values
        max_risk = 100000 * 0.02  # 2% of capital
        
        # Position values should be reasonable
        assert abs(position_values[mask]).max() < max_risk * 1.5
    
    def test_calculate_positions_volatility(self, sample_data):
        """Test volatility-based position sizing"""
        # Add ATR for volatility sizing
        high_low = sample_data['high'] - sample_data['low']
        sample_data['atr'] = high_low.rolling(14).mean()
        
        strategy = MovingAverageCrossover(parameters={
            'position_sizing': 'volatility',
            'risk_per_trade': 0.02
        })
        
        signals = strategy.generate_signals(sample_data)
        positions = strategy.calculate_positions(
            signals, 
            capital=100000,
            risk_params={'atr': sample_data['atr']}
        )
        
        # Positions should vary based on volatility
        assert positions[signals != 0].std() > 0
    
    def test_stop_loss_take_profit(self, sample_data):
        """Test stop loss and take profit logic"""
        # Strategy without stops
        strategy_no_stops = MovingAverageCrossover(parameters={
            'stop_loss': 0.02,
            'take_profit': 0.05,
            'use_stops': False
        })
        
        # Strategy with stops
        strategy_with_stops = MovingAverageCrossover(parameters={
            'stop_loss': 0.02,
            'take_profit': 0.05,
            'use_stops': True
        })
        
        signals_no_stops = strategy_no_stops.generate_signals(sample_data)
        signals_with_stops = strategy_with_stops.generate_signals(sample_data)
        
        # Should exit some positions early with stops
        assert not signals_no_stops.equals(signals_with_stops)
    
    def test_required_history(self):
        """Test required history calculation"""
        strategy = MovingAverageCrossover(parameters={
            'fast_period': 10,
            'slow_period': 30
        })
        
        assert strategy.required_history == 30
        
        # Different periods
        strategy2 = MovingAverageCrossover(parameters={
            'fast_period': 20,
            'slow_period': 50
        })
        
        assert strategy2.required_history == 50
    
    def test_required_features(self):
        """Test required features"""
        # Basic strategy needs no extra features
        strategy = MovingAverageCrossover()
        assert strategy.required_features == []
        
        # With volume filter needs volume_ma
        strategy = MovingAverageCrossover(parameters={
            'use_volume_filter': True
        })
        assert 'volume_ma' in strategy.required_features
        
        # With volatility sizing needs atr
        strategy = MovingAverageCrossover(parameters={
            'position_sizing': 'volatility'
        })
        assert 'atr' in strategy.required_features
    
    def test_trending_vs_choppy_markets(self, sample_data, choppy_data):
        """Test strategy performance in different market conditions"""
        strategy = MovingAverageCrossover()
        
        # Trending market should have clear signals
        signals_trend = strategy.generate_signals(sample_data)
        
        # Choppy market should have more whipsaws
        signals_choppy = strategy.generate_signals(choppy_data)
        
        # Count signal changes (whipsaws)
        changes_trend = abs(signals_trend.diff()).sum()
        changes_choppy = abs(signals_choppy.diff()).sum()
        
        # Choppy market should have more signal changes
        assert changes_choppy > changes_trend
    
    def test_metadata(self):
        """Test strategy metadata"""
        strategy = MovingAverageCrossover(parameters={
            'fast_period': 5,
            'slow_period': 20
        })
        
        info = strategy.get_info()
        
        assert info['name'] == "Moving Average Crossover"
        assert info['version'] == "1.0"
        assert info['parameters']['fast_period'] == 5
        assert info['parameters']['slow_period'] == 20
        assert info['supports_short'] is True
        assert info['supports_intraday'] is True
    
    def test_wma_implementation(self, choppy_data):
        """Test weighted moving average"""
        strategy = MovingAverageCrossover(parameters={
            'fast_period': 10,
            'slow_period': 20,
            'ma_type': 'wma'
        })
        
        signals = strategy.generate_signals(choppy_data)
        
        # Should generate valid signals
        assert set(signals.unique()).issubset({-1, 0, 1})
        assert (signals != 0).any()
        
        # WMA should be different from SMA
        strategy_sma = MovingAverageCrossover(parameters={
            'fast_period': 10,
            'slow_period': 20,
            'ma_type': 'sma'
        })
        signals_sma = strategy_sma.generate_signals(choppy_data)
        
        assert not signals.equals(signals_sma)


class TestMovingAverageCrossoverIntegration:
    """Integration tests for MA strategy"""
    
    @pytest.fixture
    def full_year_data(self):
        """Create full year of realistic data"""
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        
        # Create realistic bull market with corrections
        np.random.seed(42)
        
        # Base trend with volatility clusters
        t = np.linspace(0, 1, 252)
        trend = 100 * (1 + 0.20 * t)  # 20% annual growth
        
        # Add volatility clustering
        volatility = 0.01 + 0.02 * np.abs(np.sin(2 * np.pi * t * 3))
        returns = np.array([np.random.normal(0, vol) for vol in volatility])
        
        # Add a correction
        returns[100:120] -= 0.01  # 20-day correction
        
        prices = trend * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(252) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(252) * 0.005)),
            'low': prices * (1 - np.abs(np.random.randn(252) * 0.005)),
            'close': prices,
            'volume': np.random.lognormal(11, 0.5, 252).astype(int)
        }, index=dates)
        
        # Add required indicators
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        
        return df
    
    def test_full_backtest_workflow(self, full_year_data):
        """Test complete backtest workflow"""
        strategy = MovingAverageCrossover(parameters={
            'fast_period': 20,
            'slow_period': 50,
            'ma_type': 'ema',
            'use_volume_filter': True,
            'position_sizing': 'volatility',
            'risk_per_trade': 0.02
        })
        
        # Validate data
        assert strategy.validate_data(full_year_data)
        
        # Generate signals
        signals = strategy.generate_signals(full_year_data)
        
        # Calculate positions
        positions = strategy.calculate_positions(
            signals,
            capital=100000,
            risk_params={'atr': full_year_data['atr']}
        )
        
        # Verify results
        assert len(signals) == len(full_year_data)
        assert len(positions) == len(full_year_data)
        
        # Should have taken some positions
        assert (positions != 0).any()
        
        # Check that positions change over time (strategy is active)
        position_changes = positions.diff().fillna(0)
        assert (position_changes != 0).sum() > 5  # At least 5 position changes
        
        # Should have both long and short positions
        long_days = (positions > 0).sum()
        short_days = (positions < 0).sum()
        assert long_days > 0
        assert short_days > 0
    
    def test_parameter_optimization_grid(self, full_year_data):
        """Test different parameter combinations"""
        param_combinations = [
            {'fast_period': 10, 'slow_period': 30},
            {'fast_period': 20, 'slow_period': 50},
            {'fast_period': 50, 'slow_period': 100}
        ]
        
        results = []
        
        for params in param_combinations:
            strategy = MovingAverageCrossover(parameters=params)
            signals = strategy.generate_signals(full_year_data)
            
            # Count trades
            trades = abs(signals.diff()).sum() / 2
            
            results.append({
                'params': params,
                'trades': trades,
                'signal_ratio': (signals == 1).sum() / (signals == -1).sum() if (signals == -1).sum() > 0 else float('inf')
            })
        
        # Shorter periods should generate more trades
        assert results[0]['trades'] > results[2]['trades']
        
        # Should have both long and short signals
        for result in results:
            assert result['signal_ratio'] > 0  # At least some long signals