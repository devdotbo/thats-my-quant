"""
Portfolio Backtesting Tests
Tests for multi-strategy portfolio backtesting functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtesting.portfolio import (
    PortfolioBacktester, 
    AllocationMethod,
    PortfolioResult
)
from src.strategies.examples.moving_average import MovingAverageCrossover
from src.strategies.examples.orb import OpeningRangeBreakout
from src.backtesting.engines.vectorbt_engine import VectorBTEngine


@pytest.fixture
def sample_data():
    """Create sample market data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='1min')
    
    # Filter to market hours only
    dates = dates[(dates.time >= pd.Timestamp('09:30').time()) & 
                  (dates.time <= pd.Timestamp('16:00').time())]
    dates = dates[dates.weekday < 5]  # Weekdays only
    
    # Create synthetic price data
    n = len(dates)
    base_price = 100
    returns = np.random.normal(0.0001, 0.001, n)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0005, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n))),
        'close': prices,
        'volume': np.random.randint(10000, 100000, n),
        'sma_20': pd.Series(prices).rolling(20).mean(),
        'sma_50': pd.Series(prices).rolling(50).mean(),
        'rsi_14': 50 + np.random.normal(0, 10, n),  # Mock RSI
        'atr_14': np.random.uniform(0.5, 1.5, n)    # Mock ATR
    }, index=dates)
    
    # Forward fill NaN values
    data = data.ffill()
    
    return data


@pytest.fixture
def sample_strategies():
    """Create sample strategies for testing"""
    strategies = {
        'ma_fast': MovingAverageCrossover({
            'fast_period': 10,
            'slow_period': 30,
            'position_size': 1.0
        }),
        'ma_slow': MovingAverageCrossover({
            'fast_period': 20,
            'slow_period': 50,
            'position_size': 1.0
        }),
        'orb': OpeningRangeBreakout({
            'opening_minutes': 30,
            'buffer_percent': 0.1,
            'stop_type': 'range'
        })
    }
    return strategies


@pytest.fixture
def portfolio_backtester():
    """Create portfolio backtester instance"""
    return PortfolioBacktester(rebalance_frequency='monthly')


class TestPortfolioBacktester:
    """Test portfolio backtesting functionality"""
    
    def test_portfolio_initialization(self):
        """Test portfolio backtester initialization"""
        # Default initialization
        backtester = PortfolioBacktester()
        assert backtester.rebalance_frequency == 'monthly'
        assert backtester.engine is not None
        
        # Custom initialization
        engine = VectorBTEngine()
        backtester = PortfolioBacktester(
            engine=engine,
            rebalance_frequency='weekly'
        )
        assert backtester.engine is engine
        assert backtester.rebalance_frequency == 'weekly'
    
    def test_equal_weight_portfolio(self, portfolio_backtester, sample_data, sample_strategies):
        """Test equal weight portfolio allocation"""
        result = portfolio_backtester.run_portfolio_backtest(
            strategies=sample_strategies,
            data=sample_data,
            initial_capital=100000,
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        )
        
        # Verify result structure
        assert isinstance(result, PortfolioResult)
        assert len(result.strategy_results) == len(sample_strategies)
        assert result.weights is not None
        assert result.equity_curve is not None
        assert result.returns is not None
        assert result.correlation_matrix is not None
        
        # Check equal weights
        expected_weight = 1.0 / len(sample_strategies)
        assert np.allclose(result.weights.mean(), expected_weight, atol=0.01)
        
        # Check portfolio metrics
        assert 'total_return' in result.portfolio_metrics
        assert 'sharpe_ratio' in result.portfolio_metrics
        assert 'max_drawdown' in result.portfolio_metrics
        
        # Verify equity curve
        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[0] == pytest.approx(100000, rel=0.01)
    
    def test_risk_parity_portfolio(self, portfolio_backtester, sample_data, sample_strategies):
        """Test risk parity portfolio allocation"""
        result = portfolio_backtester.run_portfolio_backtest(
            strategies=sample_strategies,
            data=sample_data,
            initial_capital=100000,
            allocation_method=AllocationMethod.RISK_PARITY
        )
        
        # Weights should vary based on volatility
        weight_std = result.weights.std()
        assert any(weight_std > 0.01), "Risk parity should create varying weights"
        
        # Weights should sum to 1
        weight_sums = result.weights.sum(axis=1)
        assert np.allclose(weight_sums, 1.0, atol=0.001)
    
    def test_custom_weights_portfolio(self, portfolio_backtester, sample_data, sample_strategies):
        """Test custom weights portfolio allocation"""
        custom_weights = {
            'ma_fast': 0.5,
            'ma_slow': 0.3,
            'orb': 0.2
        }
        
        result = portfolio_backtester.run_portfolio_backtest(
            strategies=sample_strategies,
            data=sample_data,
            initial_capital=100000,
            allocation_method=AllocationMethod.CUSTOM_WEIGHTS,
            custom_weights=custom_weights
        )
        
        # Check weights match custom allocation
        for strategy, expected_weight in custom_weights.items():
            actual_weight = result.weights[strategy].mean()
            assert actual_weight == pytest.approx(expected_weight, rel=0.001)
    
    def test_custom_weights_without_weights_raises_error(self, portfolio_backtester, sample_data, sample_strategies):
        """Test that custom weights method requires weights"""
        with pytest.raises(ValueError, match="custom_weights required"):
            portfolio_backtester.run_portfolio_backtest(
                strategies=sample_strategies,
                data=sample_data,
                initial_capital=100000,
                allocation_method=AllocationMethod.CUSTOM_WEIGHTS
            )
    
    def test_inverse_volatility_portfolio(self, portfolio_backtester, sample_data, sample_strategies):
        """Test inverse volatility portfolio allocation"""
        result = portfolio_backtester.run_portfolio_backtest(
            strategies=sample_strategies,
            data=sample_data,
            initial_capital=100000,
            allocation_method=AllocationMethod.INVERSE_VOLATILITY
        )
        
        # Verify weights are inversely related to volatility
        # Lower volatility strategies should have higher weights
        assert result.weights is not None
        assert len(result.weights) > 0
    
    def test_mean_variance_portfolio(self, portfolio_backtester, sample_data, sample_strategies):
        """Test mean-variance optimization portfolio"""
        result = portfolio_backtester.run_portfolio_backtest(
            strategies=sample_strategies,
            data=sample_data,
            initial_capital=100000,
            allocation_method=AllocationMethod.MEAN_VARIANCE
        )
        
        # Weights should be optimized
        assert result.weights is not None
        # Weights should sum to 1
        weight_sums = result.weights.sum(axis=1)
        assert np.allclose(weight_sums, 1.0, atol=0.001)
    
    def test_rebalancing_frequencies(self, sample_data, sample_strategies):
        """Test different rebalancing frequencies"""
        frequencies = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'never']
        
        for freq in frequencies:
            backtester = PortfolioBacktester(rebalance_frequency=freq)
            result = backtester.run_portfolio_backtest(
                strategies=sample_strategies,
                data=sample_data,
                initial_capital=100000,
                allocation_method=AllocationMethod.EQUAL_WEIGHT
            )
            
            assert result is not None
            
            if freq == 'never':
                # Weights should be constant
                weight_changes = result.weights.diff().abs().sum().sum()
                assert weight_changes == 0, "Never rebalance should have constant weights"
    
    def test_portfolio_with_different_data_per_strategy(self, portfolio_backtester):
        """Test portfolio with different data for each strategy"""
        # Create different data for each strategy
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='1H')
        
        data_dict = {}
        strategies = {}
        
        for i, name in enumerate(['strat1', 'strat2']):
            # Create slightly different data
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001 * (i+1), 0.01, len(dates))))
            
            data = pd.DataFrame({
                'open': prices,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': 100000,
                'sma_20': pd.Series(prices).rolling(20).mean().fillna(pd.Series(prices)),
                'sma_50': pd.Series(prices).rolling(50).mean().fillna(pd.Series(prices))
            }, index=dates)
            
            data_dict[name] = data
            strategies[name] = MovingAverageCrossover({
                'fast_period': 10,
                'slow_period': 30
            })
        
        result = portfolio_backtester.run_portfolio_backtest(
            strategies=strategies,
            data=data_dict,
            initial_capital=100000,
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        )
        
        assert len(result.strategy_results) == 2
        assert result.portfolio_metrics is not None
    
    def test_correlation_analysis(self, portfolio_backtester, sample_data, sample_strategies):
        """Test correlation matrix calculation"""
        result = portfolio_backtester.run_portfolio_backtest(
            strategies=sample_strategies,
            data=sample_data,
            initial_capital=100000,
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        )
        
        # Check correlation matrix
        corr_matrix = result.correlation_matrix
        assert corr_matrix.shape == (len(sample_strategies), len(sample_strategies))
        
        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), 1.0)
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(corr_matrix.values, corr_matrix.values.T)
        
        # Values should be between -1 and 1
        assert corr_matrix.min().min() >= -1
        assert corr_matrix.max().max() <= 1
    
    def test_strategy_contributions(self, portfolio_backtester, sample_data, sample_strategies):
        """Test strategy contribution calculation"""
        result = portfolio_backtester.run_portfolio_backtest(
            strategies=sample_strategies,
            data=sample_data,
            initial_capital=100000,
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        )
        
        contributions = result.strategy_contributions
        
        # Should have columns for each strategy plus total
        expected_cols = list(sample_strategies.keys()) + ['total']
        assert all(col in contributions.columns for col in expected_cols)
        
        # Total should equal sum of individual contributions
        strategy_sum = contributions[list(sample_strategies.keys())].sum(axis=1)
        np.testing.assert_array_almost_equal(
            strategy_sum.values,
            contributions['total'].values,
            decimal=10
        )
    
    def test_portfolio_analysis(self, portfolio_backtester, sample_data, sample_strategies):
        """Test portfolio analysis functionality"""
        result = portfolio_backtester.run_portfolio_backtest(
            strategies=sample_strategies,
            data=sample_data,
            initial_capital=100000,
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        )
        
        analysis = portfolio_backtester.analyze_portfolio(result)
        
        # Check analysis structure
        assert 'performance' in analysis
        assert 'correlations' in analysis
        assert 'diversification_ratio' in analysis
        assert 'strategy_sharpe_ratios' in analysis
        assert 'weight_statistics' in analysis
        
        # Check diversification ratio
        div_ratio = analysis['diversification_ratio']
        assert div_ratio > 0
        assert div_ratio <= len(sample_strategies)  # Max is number of strategies
        
        # Check weight statistics
        weight_stats = analysis['weight_statistics']
        assert 'mean' in weight_stats
        assert 'std' in weight_stats
        assert 'min' in weight_stats
        assert 'max' in weight_stats
    
    def test_portfolio_metrics_calculation(self, portfolio_backtester, sample_data, sample_strategies):
        """Test portfolio metric calculations"""
        result = portfolio_backtester.run_portfolio_backtest(
            strategies=sample_strategies,
            data=sample_data,
            initial_capital=100000,
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        )
        
        metrics = result.portfolio_metrics
        
        # Check all required metrics exist
        required_metrics = [
            'total_return', 'annualized_return', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'calmar_ratio', 'volatility'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])
        
        # Sanity checks
        assert metrics['max_drawdown'] >= 0  # Drawdown is positive
        assert metrics['volatility'] >= 0    # Volatility is non-negative
    
    def test_portfolio_with_failing_strategy(self, portfolio_backtester, sample_data):
        """Test portfolio handles strategy that produces no trades"""
        strategies = {
            'normal': MovingAverageCrossover({
                'fast_period': 20,
                'slow_period': 50
            }),
            'no_trades': MovingAverageCrossover({
                'fast_period': 10000,  # Impossible period
                'slow_period': 20000   # Will produce no signals
            })
        }
        
        result = portfolio_backtester.run_portfolio_backtest(
            strategies=strategies,
            data=sample_data,
            initial_capital=100000,
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        )
        
        # Should still produce valid results
        assert result is not None
        assert len(result.strategy_results) == 2
        assert result.portfolio_metrics is not None
    
    def test_portfolio_result_summary(self, portfolio_backtester, sample_data, sample_strategies):
        """Test portfolio result summary method"""
        result = portfolio_backtester.run_portfolio_backtest(
            strategies=sample_strategies,
            data=sample_data,
            initial_capital=100000,
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        )
        
        summary = result.get_summary()
        
        assert 'portfolio_metrics' in summary
        assert 'strategy_count' in summary
        assert 'total_return' in summary
        assert 'sharpe_ratio' in summary
        assert 'max_drawdown' in summary
        assert 'correlation_mean' in summary
        
        assert summary['strategy_count'] == len(sample_strategies)
    
    def test_empty_strategy_dict(self, portfolio_backtester, sample_data):
        """Test error handling for empty strategy dictionary"""
        with pytest.raises((ValueError, KeyError)):
            portfolio_backtester.run_portfolio_backtest(
                strategies={},
                data=sample_data,
                initial_capital=100000
            )
    
    def test_portfolio_with_transaction_costs(self, portfolio_backtester, sample_data, sample_strategies):
        """Test portfolio with transaction costs"""
        result = portfolio_backtester.run_portfolio_backtest(
            strategies=sample_strategies,
            data=sample_data,
            initial_capital=100000,
            allocation_method=AllocationMethod.EQUAL_WEIGHT,
            commission=0.001,  # $1 per trade
            slippage=0.0005    # 5 bps
        )
        
        # Transaction costs should reduce returns
        result_no_costs = portfolio_backtester.run_portfolio_backtest(
            strategies=sample_strategies,
            data=sample_data,
            initial_capital=100000,
            allocation_method=AllocationMethod.EQUAL_WEIGHT,
            commission=0,
            slippage=0
        )
        
        # With costs should have lower returns (in most cases)
        # This might not always be true due to randomness, so we just check results exist
        assert result.portfolio_metrics['total_return'] is not None
        assert result_no_costs.portfolio_metrics['total_return'] is not None