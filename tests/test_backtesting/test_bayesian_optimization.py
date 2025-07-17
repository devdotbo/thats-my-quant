"""
Tests for Bayesian optimization functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtesting.engines.vectorbt_engine import VectorBTEngine
from src.strategies.examples.moving_average import MovingAverageCrossover
from src.strategies.examples.orb import OpeningRangeBreakout


@pytest.fixture
def sample_data():
    """Generate sample market data for testing"""
    dates = pd.date_range(start='2024-01-01 09:30', end='2024-01-31 16:00', freq='1min')
    
    # Filter to market hours only
    dates = dates[(dates.time >= pd.Timestamp('09:30').time()) & 
                  (dates.time < pd.Timestamp('16:00').time())]
    
    # Generate realistic price data
    np.random.seed(42)
    initial_price = 100
    returns = np.random.normal(0.0001, 0.001, len(dates))
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Add some volatility
    volatility = np.random.uniform(0.5, 1.5, len(dates))
    prices = prices * volatility
    
    data = pd.DataFrame({
        'open': prices * np.random.uniform(0.999, 1.001, len(prices)),
        'high': prices * np.random.uniform(1.001, 1.01, len(prices)),
        'low': prices * np.random.uniform(0.99, 0.999, len(prices)),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, len(prices))
    }, index=dates)
    
    return data


def test_bayesian_optimization_basic(sample_data):
    """Test basic Bayesian optimization functionality"""
    engine = VectorBTEngine()
    
    # Define parameter space for MA strategy
    param_space = {
        'fast_period': {'type': 'int', 'low': 5, 'high': 20},
        'slow_period': {'type': 'int', 'low': 20, 'high': 50},
        'ma_type': {'type': 'categorical', 'choices': ['sma', 'ema']}
    }
    
    # Run optimization with small number of trials
    result = engine.optimize_parameters_bayesian(
        strategy_class=MovingAverageCrossover,
        data=sample_data,
        param_space=param_space,
        metric='sharpe_ratio',
        n_trials=10,
        n_jobs=1
    )
    
    # Check results
    assert isinstance(result.best_params, dict)
    assert 'fast_period' in result.best_params
    assert 'slow_period' in result.best_params
    assert 'ma_type' in result.best_params
    assert result.best_params['fast_period'] < result.best_params['slow_period']
    assert isinstance(result.best_metric, float)
    assert len(result.results_df) > 0
    # Check that we have the objective column
    assert 'objective_sharpe_ratio' in result.results_df.columns
    # Check that trials actually ran
    assert 'trial_number' in result.results_df.columns


def test_bayesian_optimization_with_pruning(sample_data):
    """Test Bayesian optimization with pruning"""
    import optuna
    
    engine = VectorBTEngine()
    
    # Define parameter space
    param_space = {
        'fast_period': {'type': 'int', 'low': 5, 'high': 30},
        'slow_period': {'type': 'int', 'low': 20, 'high': 100},
        'stop_loss': {'type': 'float', 'low': 0.01, 'high': 0.05, 'step': 0.01}
    }
    
    # Use median pruner to stop bad trials early
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    
    result = engine.optimize_parameters_bayesian(
        strategy_class=MovingAverageCrossover,
        data=sample_data,
        param_space=param_space,
        metric='sharpe_ratio',
        n_trials=20,
        pruner=pruner,
        n_jobs=1
    )
    
    assert len(result.results_df) <= 20  # Some trials may be pruned


def test_bayesian_optimization_orb_strategy(sample_data):
    """Test Bayesian optimization with ORB strategy"""
    engine = VectorBTEngine()
    
    # Define parameter space for ORB
    param_space = {
        'opening_minutes': {'type': 'int', 'low': 5, 'high': 30, 'step': 5},
        'stop_type': {'type': 'categorical', 'choices': ['range', 'atr', 'fixed']},
        'profit_target_r': {'type': 'float', 'low': 1.0, 'high': 10.0, 'step': 0.5},
        'atr_multiplier': {'type': 'float', 'low': 0.5, 'high': 3.0, 'step': 0.5}
    }
    
    result = engine.optimize_parameters_bayesian(
        strategy_class=OpeningRangeBreakout,
        data=sample_data,
        param_space=param_space,
        metric='total_return',
        direction='maximize',
        n_trials=15,
        n_jobs=1
    )
    
    assert 'opening_minutes' in result.best_params
    assert 'stop_type' in result.best_params
    assert result.best_params['opening_minutes'] in [5, 10, 15, 20, 25, 30]


def test_bayesian_optimization_parallel(sample_data):
    """Test parallel Bayesian optimization"""
    engine = VectorBTEngine()
    
    param_space = {
        'fast_period': {'type': 'int', 'low': 5, 'high': 20},
        'slow_period': {'type': 'int', 'low': 20, 'high': 50}
    }
    
    # Run with multiple jobs (if available)
    result = engine.optimize_parameters_bayesian(
        strategy_class=MovingAverageCrossover,
        data=sample_data,
        param_space=param_space,
        metric='sharpe_ratio',
        n_trials=10,
        n_jobs=-1  # Use all available CPUs
    )
    
    assert len(result.results_df) > 0


def test_bayesian_optimization_timeout(sample_data):
    """Test Bayesian optimization with timeout"""
    engine = VectorBTEngine()
    
    param_space = {
        'fast_period': {'type': 'int', 'low': 5, 'high': 50},
        'slow_period': {'type': 'int', 'low': 20, 'high': 200}
    }
    
    # Run with 5 second timeout
    result = engine.optimize_parameters_bayesian(
        strategy_class=MovingAverageCrossover,
        data=sample_data,
        param_space=param_space,
        metric='sharpe_ratio',
        n_trials=1000,  # High number of trials
        timeout=5.0,  # But limited by timeout
        n_jobs=1
    )
    
    # Should have completed some trials but not all
    assert len(result.results_df) > 0
    assert len(result.results_df) < 1000


def test_bayesian_vs_grid_search(sample_data):
    """Compare Bayesian optimization with grid search"""
    engine = VectorBTEngine()
    
    # Limited parameter space for fair comparison
    param_grid = {
        'fast_period': [5, 10, 15],
        'slow_period': [20, 30, 40]
    }
    
    param_space = {
        'fast_period': {'type': 'int', 'low': 5, 'high': 15, 'step': 5},
        'slow_period': {'type': 'int', 'low': 20, 'high': 40, 'step': 10}
    }
    
    # Run grid search
    grid_result = engine.optimize_parameters(
        strategy_class=MovingAverageCrossover,
        data=sample_data,
        param_grid=param_grid,
        metric='sharpe_ratio'
    )
    
    # Run Bayesian optimization with same number of trials
    n_combinations = len(param_grid['fast_period']) * len(param_grid['slow_period'])
    bayes_result = engine.optimize_parameters_bayesian(
        strategy_class=MovingAverageCrossover,
        data=sample_data,
        param_space=param_space,
        metric='sharpe_ratio',
        n_trials=n_combinations,
        n_jobs=1
    )
    
    # Bayesian should find at least as good results
    assert bayes_result.best_metric >= grid_result.best_metric - 0.1  # Small tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])