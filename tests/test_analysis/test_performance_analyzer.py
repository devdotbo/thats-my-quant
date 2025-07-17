"""
Tests for Performance Analyzer
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.analysis.performance_analyzer import (
    PerformanceAnalyzer, 
    ComparisonResult,
    RankingMethod
)
from src.backtesting.engines.vectorbt_engine import BacktestResult


class TestPerformanceAnalyzer:
    """Test performance analysis functionality"""
    
    @pytest.fixture
    def sample_backtest_results(self):
        """Create sample backtest results for testing"""
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        
        # Strategy 1: Steady growth
        equity1 = 100000 * (1 + 0.0008) ** np.arange(252)  # ~20% annual
        equity1 += np.random.normal(0, 100, 252)  # Add noise
        
        # Strategy 2: Higher returns but more volatile
        returns2 = np.random.normal(0.001, 0.02, 252)
        equity2 = 100000 * (1 + returns2).cumprod()
        
        # Strategy 3: Lower returns, lower volatility
        equity3 = 100000 * (1 + 0.0004) ** np.arange(252)
        equity3 += np.random.normal(0, 50, 252)
        
        # Create BacktestResult objects
        results = {}
        
        for name, equity in [('Strategy1', equity1), ('Strategy2', equity2), ('Strategy3', equity3)]:
            equity_series = pd.Series(equity, index=dates)
            returns = equity_series.pct_change().dropna()
            
            # Calculate metrics
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
            total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
            
            # Calculate drawdowns
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # Create mock trades
            n_trades = 100
            trade_returns = np.random.normal(0.002, 0.01, n_trades)
            trades = pd.DataFrame({
                'entry_time': pd.date_range(dates[0], dates[-1], periods=n_trades),
                'exit_time': pd.date_range(dates[0], dates[-1], periods=n_trades) + timedelta(days=5),
                'pnl': trade_returns * 1000,
                'return': trade_returns
            })
            
            win_rate = (trades['pnl'] > 0).mean()
            
            result = BacktestResult(
                portfolio=None,  # Mock portfolio object
                equity_curve=equity_series,
                trades=trades,
                metrics={
                    'sharpe_ratio': sharpe,
                    'total_return': total_return,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'total_trades': len(trades),
                    'volatility': returns.std() * np.sqrt(252)
                },
                signals=pd.Series(0, index=dates),  # Mock signals
                positions=pd.Series(0, index=dates)  # Mock positions
            )
            
            results[name] = result
        
        return results
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return PerformanceAnalyzer(risk_free_rate=0.02)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        analyzer = PerformanceAnalyzer(risk_free_rate=0.03)
        assert analyzer.risk_free_rate == 0.03
        assert analyzer.logger is not None
    
    def test_compare_strategies_basic(self, analyzer, sample_backtest_results):
        """Test basic strategy comparison"""
        result = analyzer.compare_strategies(sample_backtest_results)
        
        # Check result structure
        assert isinstance(result, ComparisonResult)
        assert isinstance(result.strategy_metrics, pd.DataFrame)
        assert isinstance(result.rankings, pd.DataFrame)
        assert isinstance(result.statistical_tests, dict)
        assert isinstance(result.relative_performance, pd.DataFrame)
        assert isinstance(result.correlation_matrix, pd.DataFrame)
        assert isinstance(result.summary_stats, dict)
        
        # Check metrics DataFrame
        assert len(result.strategy_metrics) == 3
        assert 'sharpe_ratio' in result.strategy_metrics.columns
        assert 'total_return' in result.strategy_metrics.columns
        
        # Check rankings
        assert len(result.rankings) == 3
        assert 'composite_rank' in result.rankings.columns
    
    def test_extract_all_metrics(self, analyzer, sample_backtest_results):
        """Test metric extraction"""
        metrics_df = analyzer._extract_all_metrics(sample_backtest_results)
        
        # Check all strategies included
        assert len(metrics_df) == 3
        assert set(metrics_df.index) == {'Strategy1', 'Strategy2', 'Strategy3'}
        
        # Check required metrics
        required_metrics = [
            'sharpe_ratio', 'total_return', 'max_drawdown',
            'volatility', 'skewness', 'kurtosis', 'calmar_ratio'
        ]
        for metric in required_metrics:
            assert metric in metrics_df.columns
        
        # Check calculations
        assert all(metrics_df['volatility'] > 0)
        assert all(metrics_df['max_drawdown'] >= 0)
    
    def test_calculate_rankings(self, analyzer, sample_backtest_results):
        """Test strategy ranking calculation"""
        metrics_df = analyzer._extract_all_metrics(sample_backtest_results)
        
        ranking_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
        rankings = analyzer._calculate_rankings(metrics_df, ranking_metrics)
        
        # Check structure
        assert 'strategy' in rankings.columns
        assert 'composite_rank' in rankings.columns
        
        # Check individual rankings
        for metric in ranking_metrics:
            assert f'{metric}_rank' in rankings.columns
            assert metric in rankings.columns
        
        # Verify ranking logic
        # Higher sharpe should have lower rank number (better)
        sharpe_sorted = rankings.sort_values('sharpe_ratio', ascending=False)
        assert all(sharpe_sorted['sharpe_ratio_rank'].diff()[1:] >= 0)
        
        # Lower drawdown should have lower rank number (better)
        dd_sorted = rankings.sort_values('max_drawdown', ascending=True)
        assert all(dd_sorted['max_drawdown_rank'].diff()[1:] >= 0)
    
    def test_statistical_tests(self, analyzer, sample_backtest_results):
        """Test statistical significance tests"""
        test_results = analyzer._perform_statistical_tests(
            sample_backtest_results,
            ['sharpe_difference', 't_test']
        )
        
        # Check structure
        assert 'sharpe_difference' in test_results
        assert 't_test' in test_results
        
        # Check Sharpe difference test
        sharpe_df = test_results['sharpe_difference']
        assert len(sharpe_df) == 3  # 3 pairwise comparisons for 3 strategies
        assert all(col in sharpe_df.columns for col in 
                  ['strategy_1', 'strategy_2', 'p_value', 'significant_5%'])
        
        # Check p-values are valid
        assert all(0 <= p <= 1 for p in sharpe_df['p_value'])
        
        # Check t-test
        t_test_df = test_results['t_test']
        assert len(t_test_df) == 3
        assert 't_statistic' in t_test_df.columns
    
    def test_relative_performance_calculation(self, analyzer, sample_backtest_results):
        """Test relative performance calculation"""
        # Test with no benchmark (equal weight)
        rel_perf = analyzer._calculate_relative_performance(sample_backtest_results)
        
        assert len(rel_perf) == 3
        assert all(col in rel_perf.columns for col in 
                  ['alpha', 'beta', 'information_ratio', 'tracking_error'])
        
        # Test with specific benchmark
        rel_perf_bench = analyzer._calculate_relative_performance(
            sample_backtest_results,
            benchmark='Strategy1'
        )
        
        # Should have 2 rows (excluding benchmark itself)
        assert len(rel_perf_bench) == 2
        assert 'Strategy1' not in rel_perf_bench['strategy'].values
    
    def test_correlation_matrix(self, analyzer, sample_backtest_results):
        """Test correlation matrix calculation"""
        corr_matrix = analyzer._calculate_correlation_matrix(sample_backtest_results)
        
        # Check structure
        assert corr_matrix.shape == (3, 3)
        assert set(corr_matrix.index) == set(corr_matrix.columns)
        
        # Check properties
        # Diagonal should be 1
        assert all(corr_matrix.values[i, i] == 1 for i in range(3))
        
        # Should be symmetric
        assert np.allclose(corr_matrix.values, corr_matrix.values.T)
        
        # Values should be between -1 and 1
        assert corr_matrix.min().min() >= -1
        assert corr_matrix.max().max() <= 1
    
    def test_rank_strategies_single_metric(self, analyzer, sample_backtest_results):
        """Test ranking strategies by single metric"""
        # Rank by Sharpe ratio
        rankings = analyzer.rank_strategies(
            sample_backtest_results,
            method=RankingMethod.SHARPE_RATIO
        )
        
        assert 'rank' in rankings.columns
        assert 'strategy' in rankings.columns
        assert 'sharpe_ratio' in rankings.columns
        
        # Verify ranking order
        assert rankings['rank'].tolist() == [1, 2, 3]
        
        # Test other ranking methods
        for method in [RankingMethod.TOTAL_RETURN, RankingMethod.MAX_DRAWDOWN]:
            rankings = analyzer.rank_strategies(sample_backtest_results, method=method)
            assert len(rankings) == 3
    
    def test_rank_strategies_composite(self, analyzer, sample_backtest_results):
        """Test composite ranking with weights"""
        weights = {
            'sharpe_ratio': 0.4,
            'total_return': 0.3,
            'max_drawdown': 0.3
        }
        
        rankings = analyzer.rank_strategies(
            sample_backtest_results,
            method=RankingMethod.SHARPE_RATIO,
            weights=weights
        )
        
        assert 'score' in rankings.columns
        assert len(rankings) == 3
        
        # Scores should be normalized
        assert all(0 <= score <= sum(weights.values()) for score in rankings['score'])
    
    def test_rank_strategies_custom_scorer(self, analyzer, sample_backtest_results):
        """Test ranking with custom scoring function"""
        def custom_scorer(result: BacktestResult) -> float:
            # Custom score: Sharpe * (1 - max_drawdown)
            return (result.metrics['sharpe_ratio'] * 
                   (1 - result.metrics['max_drawdown']))
        
        rankings = analyzer.rank_strategies(
            sample_backtest_results,
            method=RankingMethod.CUSTOM,
            custom_scorer=custom_scorer
        )
        
        assert 'score' in rankings.columns
        assert len(rankings) == 3
        
        # Test error without custom scorer
        with pytest.raises(ValueError):
            analyzer.rank_strategies(
                sample_backtest_results,
                method=RankingMethod.CUSTOM
            )
    
    def test_confidence_intervals(self, analyzer, sample_backtest_results):
        """Test bootstrap confidence interval calculation"""
        ci_df = analyzer.calculate_confidence_intervals(
            sample_backtest_results,
            metrics=['sharpe_ratio', 'total_return'],
            confidence_level=0.95,
            n_bootstrap=100  # Small number for testing
        )
        
        # Check structure
        assert len(ci_df) == 3
        assert 'strategy' in ci_df.columns
        
        # Check CI columns
        for metric in ['sharpe_ratio', 'total_return']:
            assert f'{metric}_mean' in ci_df.columns
            assert f'{metric}_ci_lower' in ci_df.columns
            assert f'{metric}_ci_upper' in ci_df.columns
            assert f'{metric}_std' in ci_df.columns
        
        # Verify CI properties
        for _, row in ci_df.iterrows():
            # Mean should be between CI bounds
            assert row['sharpe_ratio_ci_lower'] <= row['sharpe_ratio_mean'] <= row['sharpe_ratio_ci_upper']
            assert row['total_return_ci_lower'] <= row['total_return_mean'] <= row['total_return_ci_upper']
            
            # CI width should be positive
            assert row['sharpe_ratio_ci_upper'] > row['sharpe_ratio_ci_lower']
    
    def test_get_top_strategies(self, analyzer, sample_backtest_results):
        """Test getting top N strategies"""
        result = analyzer.compare_strategies(sample_backtest_results)
        
        # Get top 2 by Sharpe
        top_2 = result.get_top_strategies(metric='sharpe_ratio', n=2)
        assert len(top_2) == 2
        assert 'strategy' in top_2.columns
        assert 'sharpe_ratio' in top_2.columns
        
        # Verify ordering
        assert top_2['sharpe_ratio'].iloc[0] >= top_2['sharpe_ratio'].iloc[1]
        
        # Test invalid metric
        with pytest.raises(ValueError):
            result.get_top_strategies(metric='invalid_metric')
    
    def test_get_statistical_summary(self, analyzer, sample_backtest_results):
        """Test statistical summary generation"""
        result = analyzer.compare_strategies(sample_backtest_results)
        summary = result.get_statistical_summary()
        
        assert isinstance(summary, dict)
        
        # Check structure for each test type
        for test_name in result.statistical_tests.keys():
            assert test_name in summary
            assert 'total_comparisons' in summary[test_name]
            assert 'significant_at_5%' in summary[test_name]
            assert 'significant_pairs' in summary[test_name]
    
    def test_empty_results_handling(self, analyzer):
        """Test handling of empty results"""
        with pytest.raises(ValueError):
            analyzer.compare_strategies({})
    
    def test_single_strategy_handling(self, analyzer, sample_backtest_results):
        """Test handling of single strategy"""
        single_result = {'Strategy1': sample_backtest_results['Strategy1']}
        
        result = analyzer.compare_strategies(single_result)
        
        # Should still work but with limited comparisons
        assert len(result.strategy_metrics) == 1
        assert len(result.statistical_tests['sharpe_difference']) == 0  # No pairs to compare
    
    def test_nan_handling(self, analyzer):
        """Test handling of NaN values"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # Create result with some NaN values
        equity = pd.Series([100000] * 50 + [np.nan] * 50, index=dates)
        
        result = BacktestResult(
            portfolio=None,
            equity_curve=equity,
            trades=pd.DataFrame(),
            metrics={
                'sharpe_ratio': np.nan,
                'total_return': 0.1,
                'max_drawdown': 0.05,
                'win_rate': 0.5,
                'total_trades': 10
            },
            signals=pd.Series(0, index=equity.index),
            positions=pd.Series(0, index=equity.index)
        )
        
        results = {'TestStrategy': result}
        
        # Should handle NaN gracefully
        comparison = analyzer.compare_strategies(results)
        assert len(comparison.strategy_metrics) == 1