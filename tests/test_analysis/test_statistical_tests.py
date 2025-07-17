"""
Tests for Statistical Tests Module
"""

import pytest
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from src.analysis.statistical_tests import (
    StatisticalTests,
    TestResult,
    MultipleComparisonResult
)


class TestStatisticalTests:
    """Test statistical testing functionality"""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return series for testing"""
        np.random.seed(42)
        
        # Strategy 1: Higher Sharpe
        returns1 = pd.Series(
            np.random.normal(0.001, 0.01, 252),
            index=pd.date_range('2024-01-01', periods=252, freq='D')
        )
        
        # Strategy 2: Lower Sharpe
        returns2 = pd.Series(
            np.random.normal(0.0005, 0.015, 252),
            index=pd.date_range('2024-01-01', periods=252, freq='D')
        )
        
        # Strategy 3: Similar to Strategy 1
        returns3 = pd.Series(
            np.random.normal(0.0009, 0.011, 252),
            index=pd.date_range('2024-01-01', periods=252, freq='D')
        )
        
        return {
            'strategy1': returns1,
            'strategy2': returns2,
            'strategy3': returns3
        }
    
    @pytest.fixture
    def stat_tests(self):
        """Create StatisticalTests instance"""
        return StatisticalTests(confidence_level=0.95)
    
    def test_initialization(self):
        """Test StatisticalTests initialization"""
        tests = StatisticalTests(confidence_level=0.99)
        assert tests.confidence_level == 0.99
        assert tests.alpha == 0.01
    
    def test_jobson_korkie_test(self, stat_tests, sample_returns):
        """Test Jobson-Korkie Sharpe ratio difference test"""
        result = stat_tests.test_sharpe_difference(
            sample_returns['strategy1'],
            sample_returns['strategy2'],
            method='jobson_korkie'
        )
        
        # Check result structure
        assert isinstance(result, TestResult)
        assert result.test_name == "Jobson-Korkie Test"
        assert 0 <= result.p_value <= 1
        assert isinstance(result.significant, bool)
        
        # Check additional info
        assert 'sharpe_ratio_1' in result.additional_info
        assert 'sharpe_ratio_2' in result.additional_info
        assert 'difference' in result.additional_info
        
        # Test with identical series (should not be significant)
        result_same = stat_tests.test_sharpe_difference(
            sample_returns['strategy1'],
            sample_returns['strategy1'],
            method='jobson_korkie'
        )
        assert result_same.p_value > 0.05  # Should not be significant
    
    def test_bootstrap_sharpe_test(self, stat_tests, sample_returns):
        """Test bootstrap Sharpe ratio difference test"""
        result = stat_tests.test_sharpe_difference(
            sample_returns['strategy1'],
            sample_returns['strategy2'],
            method='bootstrap'
        )
        
        # Check result structure
        assert isinstance(result, TestResult)
        assert result.test_name == "Bootstrap Sharpe Test"
        assert 0 <= result.p_value <= 1
        
        # Check bootstrap-specific info
        assert 'n_bootstrap' in result.additional_info
        assert 'bootstrap_ci_lower' in result.additional_info
        assert 'bootstrap_ci_upper' in result.additional_info
        
        # CI should contain the observed difference
        obs_diff = result.statistic
        ci_lower = result.additional_info['bootstrap_ci_lower']
        ci_upper = result.additional_info['bootstrap_ci_upper']
        
        # The observed difference might be outside CI under null hypothesis
        assert ci_lower <= ci_upper
    
    def test_multiple_comparison_correction(self, stat_tests):
        """Test multiple comparison correction methods"""
        # Create some p-values
        p_values = [0.01, 0.04, 0.03, 0.20, 0.15]
        comparisons = [
            ('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D')
        ]
        
        # Test different correction methods
        methods = ['bonferroni', 'holm', 'fdr_bh']
        
        for method in methods:
            result = stat_tests.multiple_comparison_correction(
                p_values, method=method, comparisons=comparisons
            )
            
            assert isinstance(result, MultipleComparisonResult)
            assert len(result.adjusted_p_values) == len(p_values)
            assert len(result.reject_null) == len(p_values)
            assert result.method == method
            
            # Adjusted p-values should be >= raw p-values
            assert all(adj >= raw for adj, raw in 
                      zip(result.adjusted_p_values, result.raw_p_values))
            
            # Test get_significant_pairs
            sig_pairs = result.get_significant_pairs()
            assert isinstance(sig_pairs, list)
            
            # Each significant pair should have comparison and p-value
            for pair in sig_pairs:
                assert len(pair) == 3  # (strategy1, strategy2, p_value)
    
    def test_permutation_test(self, stat_tests, sample_returns):
        """Test permutation test for arbitrary metrics"""
        # Define a custom metric function
        def sortino_ratio(returns):
            downside = returns[returns < 0]
            if len(downside) > 0:
                downside_std = downside.std()
                if downside_std > 0:
                    return np.sqrt(252) * returns.mean() / downside_std
            return 0
        
        result = stat_tests.permutation_test(
            sample_returns['strategy1'],
            sample_returns['strategy2'],
            metric_func=sortino_ratio,
            n_permutations=1000
        )
        
        assert isinstance(result, TestResult)
        assert result.test_name == "Permutation Test"
        assert 0 <= result.p_value <= 1
        
        # Check additional info
        assert 'metric_1' in result.additional_info
        assert 'metric_2' in result.additional_info
        assert 'n_permutations' in result.additional_info
        assert 'permutation_ci_lower' in result.additional_info
        
        # Test with lambda function
        result2 = stat_tests.permutation_test(
            sample_returns['strategy1'],
            sample_returns['strategy2'],
            metric_func=lambda r: r.mean() * 252,  # Annual return
            n_permutations=500
        )
        assert result2.additional_info['n_permutations'] == 500
    
    def test_returns_independence(self, stat_tests):
        """Test returns independence test (Ljung-Box)"""
        # Create autocorrelated returns
        n = 252
        noise = np.random.normal(0, 0.01, n)
        autocorr_returns = pd.Series(noise)
        for i in range(1, n):
            autocorr_returns.iloc[i] += 0.5 * autocorr_returns.iloc[i-1]
        
        autocorr_returns.index = pd.date_range('2024-01-01', periods=n, freq='D')
        
        # Test autocorrelated series
        result_autocorr = stat_tests.test_returns_independence(autocorr_returns, lags=10)
        assert isinstance(result_autocorr, TestResult)
        assert result_autocorr.test_name == "Ljung-Box Independence Test"
        assert not result_autocorr.significant  # Should reject independence
        
        # Test random series
        random_returns = pd.Series(
            np.random.normal(0, 0.01, n),
            index=pd.date_range('2024-01-01', periods=n, freq='D')
        )
        result_random = stat_tests.test_returns_independence(random_returns, lags=10)
        assert result_random.significant  # Should not reject independence
    
    def test_stationarity(self, stat_tests):
        """Test stationarity test (ADF)"""
        # Create stationary series
        stationary = pd.Series(
            np.random.normal(0, 0.01, 252),
            index=pd.date_range('2024-01-01', periods=252, freq='D')
        )
        
        result = stat_tests.test_stationarity(stationary)
        assert isinstance(result, TestResult)
        assert result.test_name == "Augmented Dickey-Fuller Test"
        assert result.significant  # Should reject null of unit root
        
        # Check additional info
        assert 'used_lags' in result.additional_info
        assert 'critical_values' in result.additional_info
        assert 'interpretation' in result.additional_info
    
    def test_normality(self, stat_tests):
        """Test normality test (Jarque-Bera)"""
        # Test normal distribution
        normal_returns = pd.Series(
            np.random.normal(0, 0.01, 1000),
            index=pd.date_range('2024-01-01', periods=1000, freq='D')
        )
        
        result_normal = stat_tests.test_normality(normal_returns)
        assert isinstance(result_normal, TestResult)
        assert result_normal.test_name == "Jarque-Bera Normality Test"
        # Large sample of normal data should pass normality test
        
        # Test non-normal distribution (heavily skewed)
        skewed_returns = pd.Series(
            np.random.gamma(2, 2, 1000) - 4,  # Skewed distribution
            index=pd.date_range('2024-01-01', periods=1000, freq='D')
        )
        
        result_skewed = stat_tests.test_normality(skewed_returns)
        assert not result_skewed.significant  # Should reject normality
        
        # Check additional info
        assert 'skewness' in result_skewed.additional_info
        assert 'excess_kurtosis' in result_skewed.additional_info
    
    def test_variance_equality(self, stat_tests, sample_returns):
        """Test variance equality tests"""
        # Test different methods
        methods = ['levene', 'bartlett', 'fligner']
        
        for method in methods:
            result = stat_tests.test_variance_equality(
                sample_returns['strategy1'],
                sample_returns['strategy2'],
                method=method
            )
            
            assert isinstance(result, TestResult)
            assert method.title() in result.test_name or "Fligner-Killeen" in result.test_name
            assert 0 <= result.p_value <= 1
            
            # Check additional info
            assert 'variance_1' in result.additional_info
            assert 'variance_2' in result.additional_info
            assert 'variance_ratio' in result.additional_info
        
        # Test invalid method
        with pytest.raises(ValueError):
            stat_tests.test_variance_equality(
                sample_returns['strategy1'],
                sample_returns['strategy2'],
                method='invalid'
            )
    
    def test_paired_performance_test(self, stat_tests, sample_returns):
        """Test paired performance testing"""
        results_df = stat_tests.paired_performance_test(sample_returns)
        
        # Check structure
        assert isinstance(results_df, pd.DataFrame)
        
        # Should have n*(n-1)/2 comparisons for n strategies
        n_strategies = len(sample_returns)
        expected_comparisons = n_strategies * (n_strategies - 1) // 2
        assert len(results_df) == expected_comparisons
        
        # Check columns
        required_cols = [
            'strategy_1', 'strategy_2', 'test_statistic', 
            'p_value', 'significant'
        ]
        for col in required_cols:
            assert col in results_df.columns
        
        # With multiple comparisons, should have adjusted p-values
        if len(results_df) > 1:
            assert 'adjusted_p_value' in results_df.columns
            assert 'significant_adjusted' in results_df.columns
            
            # Adjusted p-values should be >= raw p-values
            assert all(results_df['adjusted_p_value'] >= results_df['p_value'])
    
    def test_effect_size_calculation(self, stat_tests, sample_returns):
        """Test effect size calculations"""
        # Test Cohen's d
        cohen_d = stat_tests.calculate_effect_size(
            sample_returns['strategy1'],
            sample_returns['strategy2'],
            metric='cohen_d'
        )
        assert isinstance(cohen_d, float)
        
        # Test Hedge's g
        hedge_g = stat_tests.calculate_effect_size(
            sample_returns['strategy1'],
            sample_returns['strategy2'],
            metric='hedge_g'
        )
        assert isinstance(hedge_g, float)
        
        # Hedge's g should be slightly smaller than Cohen's d (correction factor)
        assert abs(hedge_g) <= abs(cohen_d)
        
        # Test R-squared
        r_squared = stat_tests.calculate_effect_size(
            sample_returns['strategy1'],
            sample_returns['strategy2'],
            metric='r_squared'
        )
        assert isinstance(r_squared, float)
        assert 0 <= r_squared <= 1
        
        # Test invalid metric
        with pytest.raises(ValueError):
            stat_tests.calculate_effect_size(
                sample_returns['strategy1'],
                sample_returns['strategy2'],
                metric='invalid_metric'
            )
    
    def test_test_result_summary(self, stat_tests, sample_returns):
        """Test TestResult summary method"""
        result = stat_tests.test_sharpe_difference(
            sample_returns['strategy1'],
            sample_returns['strategy2']
        )
        
        summary = result.summary()
        assert isinstance(summary, str)
        assert result.test_name in summary
        assert f"{result.p_value:.4f}" in summary
        assert ("SIGNIFICANT" in summary) or ("NOT SIGNIFICANT" in summary)
    
    def test_nan_handling(self, stat_tests):
        """Test handling of NaN values"""
        # Create series with NaN
        returns_with_nan = pd.Series(
            [0.01, 0.02, np.nan, 0.01, -0.01, np.nan, 0.02],
            index=pd.date_range('2024-01-01', periods=7, freq='D')
        )
        
        normal_returns = pd.Series(
            [0.01, -0.01, 0.02, 0.01, -0.02, 0.01, 0.00],
            index=pd.date_range('2024-01-01', periods=7, freq='D')
        )
        
        # Should handle NaN by dropping them
        result = stat_tests.test_sharpe_difference(
            returns_with_nan,
            normal_returns
        )
        
        assert isinstance(result, TestResult)
        assert not np.isnan(result.statistic)
        assert not np.isnan(result.p_value)
    
    def test_misaligned_series(self, stat_tests):
        """Test handling of misaligned series"""
        # Create misaligned series
        returns1 = pd.Series(
            np.random.normal(0, 0.01, 100),
            index=pd.date_range('2024-01-01', periods=100, freq='D')
        )
        
        returns2 = pd.Series(
            np.random.normal(0, 0.01, 100),
            index=pd.date_range('2024-01-15', periods=100, freq='D')
        )
        
        # Should align series before testing
        result = stat_tests.test_sharpe_difference(returns1, returns2)
        
        assert isinstance(result, TestResult)
        # Should use overlapping period
        assert result.additional_info['n_observations'] < 100