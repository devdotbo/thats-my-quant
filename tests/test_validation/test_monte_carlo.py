"""
Tests for Monte Carlo Validation Framework

This module tests the Monte Carlo simulation functionality for assessing
strategy robustness through statistical resampling techniques.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.validation.monte_carlo import (
    MonteCarloValidator,
    MonteCarloResult,
    ResamplingMethod,
    ConfidenceLevel
)
from src.backtesting.engines.vectorbt_engine import BacktestResult, VectorBTEngine
from src.strategies.examples.moving_average import MovingAverageCrossover
import vectorbt as vbt


class TestMonteCarloValidator:
    """Test the Monte Carlo validation functionality"""
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data for testing"""
        np.random.seed(42)
        n_trades = 50
        
        # Generate realistic trade data
        trades = pd.DataFrame({
            'entry_time': pd.date_range('2024-01-01', periods=n_trades, freq='D'),
            'exit_time': pd.date_range('2024-01-02', periods=n_trades, freq='D'),
            'pnl': np.random.normal(100, 500, n_trades),  # Mean $100, std $500
            'return_pct': np.random.normal(0.002, 0.02, n_trades),  # 0.2% mean, 2% std
            'size': np.abs(np.random.normal(100, 20, n_trades)),
            'side': np.random.choice(['long', 'short'], n_trades),
            'commission': np.abs(np.random.normal(5, 1, n_trades))
        })
        
        # Make sure we have both winning and losing trades
        trades.loc[:10, 'pnl'] = np.abs(trades.loc[:10, 'pnl'])  # Winners
        trades.loc[40:, 'pnl'] = -np.abs(trades.loc[40:, 'pnl'])  # Losers
        
        return trades
    
    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve"""
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        initial_capital = 100000
        
        # Generate realistic equity curve with trend and volatility
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, len(dates))  # Daily returns
        equity = initial_capital * (1 + returns).cumprod()
        
        return pd.Series(equity, index=dates)
    
    @pytest.fixture
    def sample_backtest_result(self, sample_trades, sample_equity_curve):
        """Create a mock BacktestResult"""
        # Create mock portfolio object
        mock_portfolio = type('MockPortfolio', (), {
            'init_cash': 100000,
            'final_value': lambda: sample_equity_curve.iloc[-1],
            'total_return': lambda: (sample_equity_curve.iloc[-1] / 100000) - 1,
            'returns': lambda: sample_equity_curve.pct_change().dropna(),
            'value': lambda: sample_equity_curve,
            'trades': type('Trades', (), {'records_readable': sample_trades})
        })()
        
        metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.12,
            'win_rate': 0.55,
            'profit_factor': 1.8,
            'trades_count': len(sample_trades)
        }
        
        return BacktestResult(
            portfolio=mock_portfolio,
            trades=sample_trades,
            metrics=metrics,
            equity_curve=sample_equity_curve,
            signals=pd.Series(),
            positions=pd.Series()
        )
    
    @pytest.fixture
    def validator(self):
        """Create MonteCarloValidator instance"""
        return MonteCarloValidator(
            n_simulations=1000,
            confidence_levels=[0.95, 0.99],
            random_seed=42
        )
    
    def test_validator_initialization(self, validator):
        """Test validator initialization"""
        assert validator.n_simulations == 1000
        assert validator.confidence_levels == [0.95, 0.99]
        assert validator.random_seed == 42
        
    def test_resample_trades_basic(self, validator, sample_trades):
        """Test basic trade resampling"""
        resampled = validator.resample_trades(
            trades=sample_trades,
            method=ResamplingMethod.BOOTSTRAP
        )
        
        # Should have same number of trades
        assert len(resampled) == len(sample_trades)
        
        # Should have same columns
        assert set(resampled.columns) == set(sample_trades.columns)
        
        # Should have different order (with high probability)
        assert not resampled.equals(sample_trades)
        
    def test_resample_trades_block(self, validator, sample_trades):
        """Test block resampling to preserve trade sequences"""
        resampled = validator.resample_trades(
            trades=sample_trades,
            method=ResamplingMethod.BLOCK,
            block_size=5
        )
        
        assert len(resampled) == len(sample_trades)
        
        # Check that blocks are preserved
        # This is probabilistic, but with seed should be consistent
        first_block = resampled.iloc[:5]
        assert len(first_block['entry_time'].diff().dropna().unique()) <= 3  # Some consistency in timing
        
    def test_calculate_metrics_distribution(self, validator, sample_backtest_result):
        """Test calculating metric distributions from simulations"""
        # Create multiple simulation results
        simulation_results = []
        for i in range(100):
            metrics = {
                'total_return': np.random.normal(0.15, 0.05),
                'sharpe_ratio': np.random.normal(1.5, 0.3),
                'max_drawdown': np.random.uniform(0.05, 0.25),
                'win_rate': np.random.uniform(0.45, 0.65)
            }
            simulation_results.append(metrics)
        
        distributions = validator.calculate_metric_distributions(simulation_results)
        
        # Check that we have distributions for all metrics
        assert 'total_return' in distributions
        assert 'sharpe_ratio' in distributions
        assert 'max_drawdown' in distributions
        assert 'win_rate' in distributions
        
        # Each distribution should have n_simulations values
        assert len(distributions['total_return']) == 100
        
    def test_calculate_confidence_intervals(self, validator):
        """Test confidence interval calculation"""
        # Create sample distribution
        np.random.seed(42)
        values = np.random.normal(1.5, 0.3, 1000)
        
        ci_95 = validator.calculate_confidence_interval(values, 0.95)
        ci_99 = validator.calculate_confidence_interval(values, 0.99)
        
        # Check structure
        assert 'lower' in ci_95
        assert 'upper' in ci_95
        assert 'mean' in ci_95
        assert 'median' in ci_95
        
        # 99% CI should be wider than 95% CI
        assert ci_99['upper'] - ci_99['lower'] > ci_95['upper'] - ci_95['lower']
        
        # Mean should be between bounds
        assert ci_95['lower'] < ci_95['mean'] < ci_95['upper']
        
    def test_run_single_simulation(self, validator, sample_backtest_result):
        """Test running a single Monte Carlo simulation"""
        result = validator.run_single_simulation(
            backtest_result=sample_backtest_result,
            simulation_id=1
        )
        
        assert isinstance(result, dict)
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert 'equity_curve' in result
        
        # Equity curve should start at initial capital
        assert result['equity_curve'].iloc[0] == 100000
        
    def test_full_monte_carlo_validation(self, validator, sample_backtest_result):
        """Test full Monte Carlo validation process"""
        mc_result = validator.run_validation(
            backtest_result=sample_backtest_result,
            n_simulations=100  # Fewer for testing
        )
        
        assert isinstance(mc_result, MonteCarloResult)
        assert len(mc_result.simulation_results) == 100
        
        # Check confidence intervals
        assert 'total_return' in mc_result.confidence_intervals
        assert 0.95 in mc_result.confidence_intervals['total_return']
        assert 0.99 in mc_result.confidence_intervals['total_return']
        
        # Check risk metrics
        risk_metrics = mc_result.get_risk_metrics()
        assert 'risk_of_ruin' in risk_metrics
        assert 'max_drawdown_95th_percentile' in risk_metrics
        assert 'sharpe_below_zero_probability' in risk_metrics
        
    def test_risk_of_ruin_calculation(self, validator):
        """Test risk of ruin calculation"""
        # Create equity curves with some going to ruin
        equity_curves = []
        for i in range(100):
            if i < 10:  # 10% go to ruin
                # Declining equity curve
                dates = pd.date_range('2024-01-01', periods=252, freq='D')
                equity = 100000 * np.exp(-0.01 * np.arange(252))
                equity[-50:] = equity[-50] * 0.2  # Drop to 20% of capital
            else:
                # Normal equity curve
                dates = pd.date_range('2024-01-01', periods=252, freq='D')
                returns = np.random.normal(0.0005, 0.01, 252)
                equity = 100000 * (1 + returns).cumprod()
            
            equity_curves.append(pd.Series(equity, index=dates))
        
        risk_of_ruin = validator.calculate_risk_of_ruin(
            equity_curves,
            ruin_threshold=0.5  # 50% drawdown
        )
        
        assert risk_of_ruin == pytest.approx(0.1, rel=0.1)  # Should be around 10%
        
    def test_bootstrap_returns(self, validator, sample_equity_curve):
        """Test bootstrap resampling of returns"""
        returns = sample_equity_curve.pct_change().dropna()
        
        bootstrapped = validator.bootstrap_returns(
            returns=returns,
            n_days=len(returns)
        )
        
        # Should have same length
        assert len(bootstrapped) == len(returns)
        
        # Should have similar statistical properties
        assert abs(bootstrapped.mean() - returns.mean()) < 0.01
        assert abs(bootstrapped.std() - returns.std()) < 0.02
        
    def test_monte_carlo_with_different_methods(self, validator, sample_backtest_result):
        """Test different resampling methods"""
        methods = [
            ResamplingMethod.BOOTSTRAP,
            ResamplingMethod.BLOCK,
            ResamplingMethod.STATIONARY_BOOTSTRAP
        ]
        
        results = {}
        for method in methods:
            validator.resampling_method = method
            mc_result = validator.run_validation(
                backtest_result=sample_backtest_result,
                n_simulations=50
            )
            results[method] = mc_result
        
        # All methods should produce results
        assert all(len(r.simulation_results) == 50 for r in results.values())
        
        # Results should be different across methods
        bootstrap_sharpe = results[ResamplingMethod.BOOTSTRAP].confidence_intervals['sharpe_ratio'][0.95]['mean']
        block_sharpe = results[ResamplingMethod.BLOCK].confidence_intervals['sharpe_ratio'][0.95]['mean']
        
        # They should be similar but not identical
        assert abs(bootstrap_sharpe - block_sharpe) < 1.0  # Allow more difference due to different methods
        
    def test_parallel_simulation(self, validator, sample_backtest_result):
        """Test parallel execution of simulations"""
        import time
        
        # Time sequential execution
        start = time.time()
        mc_result_seq = validator.run_validation(
            backtest_result=sample_backtest_result,
            n_simulations=100,
            n_jobs=1
        )
        seq_time = time.time() - start
        
        # Time parallel execution
        start = time.time()
        mc_result_par = validator.run_validation(
            backtest_result=sample_backtest_result,
            n_simulations=100,
            n_jobs=-1  # Use all CPUs
        )
        par_time = time.time() - start
        
        # Both should produce same number of results
        assert len(mc_result_seq.simulation_results) == 100
        assert len(mc_result_par.simulation_results) == 100
        
        # Results should be similar (same seed)
        seq_ci = mc_result_seq.confidence_intervals['total_return'][0.95]
        par_ci = mc_result_par.confidence_intervals['total_return'][0.95]
        assert abs(seq_ci['mean'] - par_ci['mean']) < 0.01
        
    def test_statistical_significance(self, validator):
        """Test statistical significance testing"""
        # Create two sets of results - one clearly better
        results_strategy = [{'sharpe_ratio': np.random.normal(1.5, 0.2)} for _ in range(100)]
        results_baseline = [{'sharpe_ratio': np.random.normal(0.5, 0.2)} for _ in range(100)]
        
        p_value = validator.test_statistical_significance(
            results_strategy,
            results_baseline,
            metric='sharpe_ratio'
        )
        
        # Should be highly significant
        assert p_value < 0.001
        
        # Test with similar results
        results_similar = [{'sharpe_ratio': np.random.normal(1.45, 0.2)} for _ in range(100)]
        p_value_similar = validator.test_statistical_significance(
            results_strategy,
            results_similar,
            metric='sharpe_ratio'
        )
        
        # Should not be significant
        assert p_value_similar > 0.05
        
    def test_percentile_outcomes(self, validator, sample_backtest_result):
        """Test percentile outcome analysis"""
        mc_result = validator.run_validation(
            backtest_result=sample_backtest_result,
            n_simulations=100
        )
        
        percentiles = mc_result.get_percentile_outcomes([5, 25, 50, 75, 95])
        
        # Check structure
        assert 5 in percentiles['total_return']
        assert 95 in percentiles['total_return']
        
        # Check ordering
        assert percentiles['total_return'][5] < percentiles['total_return'][50]
        assert percentiles['total_return'][50] < percentiles['total_return'][95]
        
    def test_export_results(self, validator, sample_backtest_result, tmp_path):
        """Test exporting Monte Carlo results"""
        mc_result = validator.run_validation(
            backtest_result=sample_backtest_result,
            n_simulations=50
        )
        
        # Export to CSV
        csv_path = tmp_path / "mc_results.csv"
        mc_result.export_metrics_to_csv(csv_path)
        assert csv_path.exists()
        
        # Verify CSV content
        df = pd.read_csv(csv_path)
        assert len(df) == 50  # One row per simulation
        assert 'total_return' in df.columns
        assert 'sharpe_ratio' in df.columns
        
        # Export summary to JSON
        json_path = tmp_path / "mc_summary.json"
        mc_result.export_summary_to_json(json_path)
        assert json_path.exists()
        
    def test_reproducibility(self, sample_backtest_result):
        """Test that results are reproducible with same seed"""
        validator1 = MonteCarloValidator(n_simulations=50, random_seed=42)
        validator2 = MonteCarloValidator(n_simulations=50, random_seed=42)
        
        result1 = validator1.run_validation(sample_backtest_result)
        result2 = validator2.run_validation(sample_backtest_result)
        
        # Results should be identical
        ci1 = result1.confidence_intervals['total_return'][0.95]['mean']
        ci2 = result2.confidence_intervals['total_return'][0.95]['mean']
        assert ci1 == ci2
        
        # Different seed should give different results
        validator3 = MonteCarloValidator(n_simulations=50, random_seed=123)
        result3 = validator3.run_validation(sample_backtest_result)
        ci3 = result3.confidence_intervals['total_return'][0.95]['mean']
        assert ci1 != ci3
        
    def test_integration_with_real_backtest(self, validator):
        """Test integration with actual backtest engine"""
        # Create sample data
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)).cumsum(),
            'high': 101 + np.random.randn(len(dates)).cumsum(),
            'low': 99 + np.random.randn(len(dates)).cumsum(),
            'close': 100 + np.random.randn(len(dates)).cumsum(),
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Ensure price consistency
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)
        
        # Run actual backtest
        strategy = MovingAverageCrossover(parameters={
            'fast_period': 10,
            'slow_period': 30
        })
        
        engine = VectorBTEngine()
        backtest_result = engine.run_backtest(
            strategy=strategy,
            data=data,
            initial_capital=100000
        )
        
        # Run Monte Carlo validation
        mc_result = validator.run_validation(
            backtest_result=backtest_result,
            n_simulations=50
        )
        
        # Verify results
        assert isinstance(mc_result, MonteCarloResult)
        assert len(mc_result.simulation_results) == 50
        
        # Original metrics should be within confidence intervals
        original_sharpe = backtest_result.metrics['sharpe_ratio']
        ci = mc_result.confidence_intervals['sharpe_ratio'][0.95]
        
        # Original might not always be within CI, but should be close
        assert abs(original_sharpe - ci['mean']) < 2 * (ci['upper'] - ci['mean'])