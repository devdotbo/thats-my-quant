"""
Statistical Tests Module
Advanced statistical testing for strategy comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import warnings

from src.utils.logging import get_logger


@dataclass
class TestResult:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    additional_info: Dict[str, any] = None
    
    def summary(self) -> str:
        """Get test summary string"""
        sig_text = "SIGNIFICANT" if self.significant else "NOT SIGNIFICANT"
        return (f"{self.test_name}: statistic={self.statistic:.4f}, "
                f"p-value={self.p_value:.4f} ({sig_text} at {self.confidence_level:.0%})")


@dataclass
class MultipleComparisonResult:
    """Results from multiple comparison tests"""
    raw_p_values: np.ndarray
    adjusted_p_values: np.ndarray
    reject_null: np.ndarray
    method: str
    alpha: float
    comparisons: List[Tuple[str, str]]
    
    def get_significant_pairs(self) -> List[Tuple[str, str, float]]:
        """Get pairs with significant differences"""
        significant = []
        for i, (pair, reject) in enumerate(zip(self.comparisons, self.reject_null)):
            if reject:
                significant.append((*pair, self.adjusted_p_values[i]))
        return significant


class StatisticalTests:
    """
    Advanced statistical tests for trading strategy comparison
    
    Includes:
    - Sharpe ratio difference tests (Jobson-Korkie)
    - Multiple comparison corrections
    - Bootstrap and permutation tests
    - Time series specific tests
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical tests
        
        Args:
            confidence_level: Default confidence level for tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.logger = get_logger('statistical_tests')
    
    def test_sharpe_difference(self,
                             returns1: pd.Series,
                             returns2: pd.Series,
                             method: str = 'jobson_korkie') -> TestResult:
        """
        Test if Sharpe ratios are significantly different
        
        Args:
            returns1: Returns series for strategy 1
            returns2: Returns series for strategy 2
            method: Test method ('jobson_korkie' or 'bootstrap')
            
        Returns:
            TestResult with test statistics
        """
        # Align returns
        aligned = pd.DataFrame({'r1': returns1, 'r2': returns2}).dropna()
        r1 = aligned['r1'].values
        r2 = aligned['r2'].values
        
        if method == 'jobson_korkie':
            return self._jobson_korkie_test(r1, r2)
        elif method == 'bootstrap':
            return self._bootstrap_sharpe_test(r1, r2)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _jobson_korkie_test(self, returns1: np.ndarray, returns2: np.ndarray) -> TestResult:
        """
        Jobson-Korkie test for Sharpe ratio differences
        
        Based on:
        Jobson, J.D. and Korkie, B.M. (1981), "Performance Hypothesis Testing with 
        the Sharpe and Treynor Measures", Journal of Finance, 36, 889-908.
        """
        n = len(returns1)
        
        # Calculate means and standard deviations
        mu1, mu2 = np.mean(returns1), np.mean(returns2)
        sigma1, sigma2 = np.std(returns1, ddof=1), np.std(returns2, ddof=1)
        
        # Calculate Sharpe ratios (assuming zero risk-free rate for simplicity)
        sr1 = mu1 / sigma1 if sigma1 > 0 else 0
        sr2 = mu2 / sigma2 if sigma2 > 0 else 0
        
        # Calculate variances and covariances
        v11 = np.var(returns1, ddof=1)
        v22 = np.var(returns2, ddof=1)
        v12 = np.cov(returns1, returns2)[0, 1]
        
        # Calculate test statistic variance
        theta = (2 * v11 * v22 - 2 * v12**2 + 
                0.5 * mu1**2 * v22 + 
                0.5 * mu2**2 * v11 - 
                (mu1 * mu2 * v12) / (sigma1 * sigma2))
        
        # Test statistic
        if theta > 0:
            z_stat = (sr1 - sr2) / np.sqrt(theta / n)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0
            p_value = 1.0
        
        return TestResult(
            test_name="Jobson-Korkie Test",
            statistic=z_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            additional_info={
                'sharpe_ratio_1': sr1,
                'sharpe_ratio_2': sr2,
                'difference': sr1 - sr2,
                'n_observations': n
            }
        )
    
    def _bootstrap_sharpe_test(self, 
                             returns1: np.ndarray,
                             returns2: np.ndarray,
                             n_bootstrap: int = 10000) -> TestResult:
        """Bootstrap test for Sharpe ratio differences"""
        n = len(returns1)
        
        # Calculate observed Sharpe difference
        sr1 = np.mean(returns1) / np.std(returns1) if np.std(returns1) > 0 else 0
        sr2 = np.mean(returns2) / np.std(returns2) if np.std(returns2) > 0 else 0
        observed_diff = sr1 - sr2
        
        # Bootstrap under null hypothesis (no difference)
        # Combine returns and bootstrap from combined sample
        combined_returns = np.concatenate([returns1, returns2])
        
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            sample1 = np.random.choice(combined_returns, size=n, replace=True)
            sample2 = np.random.choice(combined_returns, size=n, replace=True)
            
            # Calculate Sharpe ratios
            sr1_boot = np.mean(sample1) / np.std(sample1) if np.std(sample1) > 0 else 0
            sr2_boot = np.mean(sample2) / np.std(sample2) if np.std(sample2) > 0 else 0
            
            bootstrap_diffs.append(sr1_boot - sr2_boot)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate p-value
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        return TestResult(
            test_name="Bootstrap Sharpe Test",
            statistic=observed_diff,
            p_value=p_value,
            significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            additional_info={
                'sharpe_ratio_1': sr1,
                'sharpe_ratio_2': sr2,
                'n_bootstrap': n_bootstrap,
                'bootstrap_ci_lower': np.percentile(bootstrap_diffs, 2.5),
                'bootstrap_ci_upper': np.percentile(bootstrap_diffs, 97.5)
            }
        )
    
    def multiple_comparison_correction(self,
                                     p_values: List[float],
                                     method: str = 'bonferroni',
                                     comparisons: Optional[List[Tuple[str, str]]] = None) -> MultipleComparisonResult:
        """
        Apply multiple comparison correction to p-values
        
        Args:
            p_values: List of p-values from multiple tests
            method: Correction method ('bonferroni', 'holm', 'fdr_bh', 'fdr_by')
            comparisons: Optional list of comparison pairs (for labeling)
            
        Returns:
            MultipleComparisonResult with adjusted p-values
        """
        p_array = np.array(p_values)
        
        # Apply correction
        reject, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
            p_array, 
            alpha=self.alpha,
            method=method
        )
        
        # Create comparison labels if not provided
        if comparisons is None:
            comparisons = [(f"Test_{i}", f"Test_{j}") 
                          for i in range(len(p_values)) 
                          for j in range(i+1, len(p_values))][:len(p_values)]
        
        return MultipleComparisonResult(
            raw_p_values=p_array,
            adjusted_p_values=p_adjusted,
            reject_null=reject,
            method=method,
            alpha=self.alpha,
            comparisons=comparisons
        )
    
    def permutation_test(self,
                        returns1: pd.Series,
                        returns2: pd.Series,
                        metric_func: callable,
                        n_permutations: int = 10000) -> TestResult:
        """
        Permutation test for arbitrary metric differences
        
        Args:
            returns1: Returns series for strategy 1
            returns2: Returns series for strategy 2
            metric_func: Function to calculate metric from returns
            n_permutations: Number of permutations
            
        Returns:
            TestResult with permutation test results
        """
        # Align returns
        aligned = pd.DataFrame({'r1': returns1, 'r2': returns2}).dropna()
        r1 = aligned['r1'].values
        r2 = aligned['r2'].values
        
        # Calculate observed difference
        metric1 = metric_func(r1)
        metric2 = metric_func(r2)
        observed_diff = metric1 - metric2
        
        # Combine data
        combined = np.concatenate([r1, r2])
        n1 = len(r1)
        
        # Permutation test
        permuted_diffs = []
        for _ in range(n_permutations):
            # Shuffle combined data
            np.random.shuffle(combined)
            
            # Split into two groups
            perm_r1 = combined[:n1]
            perm_r2 = combined[n1:]
            
            # Calculate metric difference
            perm_metric1 = metric_func(perm_r1)
            perm_metric2 = metric_func(perm_r2)
            permuted_diffs.append(perm_metric1 - perm_metric2)
        
        permuted_diffs = np.array(permuted_diffs)
        
        # Calculate p-value
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        
        return TestResult(
            test_name="Permutation Test",
            statistic=observed_diff,
            p_value=p_value,
            significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            additional_info={
                'metric_1': metric1,
                'metric_2': metric2,
                'n_permutations': n_permutations,
                'permutation_ci_lower': np.percentile(permuted_diffs, 2.5),
                'permutation_ci_upper': np.percentile(permuted_diffs, 97.5)
            }
        )
    
    def test_returns_independence(self, returns: pd.Series, lags: int = 20) -> TestResult:
        """
        Test if returns are independently distributed (Ljung-Box test)
        
        Args:
            returns: Returns series to test
            lags: Number of lags to test
            
        Returns:
            TestResult for independence test
        """
        # Ljung-Box test
        lb_result = acorr_ljungbox(returns, lags=lags, return_df=True)
        
        # Get overall p-value (minimum across all lags)
        min_p_value = lb_result['lb_pvalue'].min()
        
        return TestResult(
            test_name="Ljung-Box Independence Test",
            statistic=lb_result['lb_stat'].iloc[-1],  # Last lag statistic
            p_value=min_p_value,
            significant=min_p_value >= self.alpha,  # Note: we want to NOT reject null of independence
            confidence_level=self.confidence_level,
            additional_info={
                'lags_tested': lags,
                'all_p_values': lb_result['lb_pvalue'].to_dict(),
                'interpretation': 'Returns are independent' if min_p_value >= self.alpha else 'Returns show autocorrelation'
            }
        )
    
    def test_stationarity(self, returns: pd.Series) -> TestResult:
        """
        Test if returns are stationary (Augmented Dickey-Fuller test)
        
        Args:
            returns: Returns series to test
            
        Returns:
            TestResult for stationarity test
        """
        # ADF test
        adf_stat, p_value, used_lags, n_obs, critical_values, icbest = adfuller(returns, autolag='AIC')
        
        return TestResult(
            test_name="Augmented Dickey-Fuller Test",
            statistic=adf_stat,
            p_value=p_value,
            significant=p_value < self.alpha,  # Reject null of unit root
            confidence_level=self.confidence_level,
            additional_info={
                'used_lags': used_lags,
                'n_observations': n_obs,
                'critical_values': critical_values,
                'interpretation': 'Returns are stationary' if p_value < self.alpha else 'Returns may be non-stationary'
            }
        )
    
    def test_normality(self, returns: pd.Series) -> TestResult:
        """
        Test if returns are normally distributed (Jarque-Bera test)
        
        Args:
            returns: Returns series to test
            
        Returns:
            TestResult for normality test
        """
        # Jarque-Bera test
        jb_stat, p_value = stats.jarque_bera(returns)
        
        # Also calculate skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
        
        return TestResult(
            test_name="Jarque-Bera Normality Test",
            statistic=jb_stat,
            p_value=p_value,
            significant=p_value >= self.alpha,  # Note: we want to NOT reject null of normality
            confidence_level=self.confidence_level,
            additional_info={
                'skewness': skewness,
                'excess_kurtosis': kurtosis,
                'interpretation': 'Returns are normally distributed' if p_value >= self.alpha else 'Returns are not normally distributed'
            }
        )
    
    def test_variance_equality(self,
                             returns1: pd.Series,
                             returns2: pd.Series,
                             method: str = 'levene') -> TestResult:
        """
        Test if two return series have equal variances
        
        Args:
            returns1: First returns series
            returns2: Second returns series
            method: Test method ('levene', 'bartlett', 'fligner')
            
        Returns:
            TestResult for variance equality test
        """
        # Align returns
        aligned = pd.DataFrame({'r1': returns1, 'r2': returns2}).dropna()
        r1 = aligned['r1'].values
        r2 = aligned['r2'].values
        
        # Select test
        if method == 'levene':
            stat, p_value = stats.levene(r1, r2)
            test_name = "Levene's Test"
        elif method == 'bartlett':
            stat, p_value = stats.bartlett(r1, r2)
            test_name = "Bartlett's Test"
        elif method == 'fligner':
            stat, p_value = stats.fligner(r1, r2)
            test_name = "Fligner-Killeen Test"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return TestResult(
            test_name=test_name,
            statistic=stat,
            p_value=p_value,
            significant=p_value >= self.alpha,  # Note: we want to NOT reject null of equal variances
            confidence_level=self.confidence_level,
            additional_info={
                'variance_1': np.var(r1, ddof=1),
                'variance_2': np.var(r2, ddof=1),
                'variance_ratio': np.var(r1, ddof=1) / np.var(r2, ddof=1),
                'interpretation': 'Equal variances' if p_value >= self.alpha else 'Unequal variances'
            }
        )
    
    def paired_performance_test(self,
                              strategy_returns: Dict[str, pd.Series],
                              test_func: Optional[callable] = None) -> pd.DataFrame:
        """
        Perform pairwise performance tests between all strategies
        
        Args:
            strategy_returns: Dict of strategy_name -> returns series
            test_func: Test function to use (defaults to Sharpe difference test)
            
        Returns:
            DataFrame with all pairwise test results
        """
        if test_func is None:
            test_func = self.test_sharpe_difference
        
        strategy_names = list(strategy_returns.keys())
        n_strategies = len(strategy_names)
        
        results = []
        p_values = []
        comparisons = []
        
        # Pairwise comparisons
        for i in range(n_strategies):
            for j in range(i + 1, n_strategies):
                strat1, strat2 = strategy_names[i], strategy_names[j]
                
                # Run test
                test_result = test_func(
                    strategy_returns[strat1],
                    strategy_returns[strat2]
                )
                
                results.append({
                    'strategy_1': strat1,
                    'strategy_2': strat2,
                    'test_statistic': test_result.statistic,
                    'p_value': test_result.p_value,
                    'significant': test_result.significant
                })
                
                p_values.append(test_result.p_value)
                comparisons.append((strat1, strat2))
        
        results_df = pd.DataFrame(results)
        
        # Apply multiple comparison correction
        if len(p_values) > 1:
            mc_result = self.multiple_comparison_correction(
                p_values,
                method='fdr_bh',
                comparisons=comparisons
            )
            
            results_df['adjusted_p_value'] = mc_result.adjusted_p_values
            results_df['significant_adjusted'] = mc_result.reject_null
        
        return results_df
    
    def calculate_effect_size(self,
                            returns1: pd.Series,
                            returns2: pd.Series,
                            metric: str = 'cohen_d') -> float:
        """
        Calculate effect size between two strategies
        
        Args:
            returns1: Returns series for strategy 1
            returns2: Returns series for strategy 2
            metric: Effect size metric ('cohen_d', 'hedge_g', 'r_squared')
            
        Returns:
            Effect size value
        """
        # Align returns
        aligned = pd.DataFrame({'r1': returns1, 'r2': returns2}).dropna()
        r1 = aligned['r1'].values
        r2 = aligned['r2'].values
        
        if metric == 'cohen_d':
            # Cohen's d
            pooled_std = np.sqrt((np.var(r1, ddof=1) + np.var(r2, ddof=1)) / 2)
            if pooled_std > 0:
                d = (np.mean(r1) - np.mean(r2)) / pooled_std
            else:
                d = 0
            return d
            
        elif metric == 'hedge_g':
            # Hedge's g (corrected Cohen's d)
            n1, n2 = len(r1), len(r2)
            pooled_std = np.sqrt(((n1 - 1) * np.var(r1, ddof=1) + 
                                 (n2 - 1) * np.var(r2, ddof=1)) / 
                                (n1 + n2 - 2))
            if pooled_std > 0:
                g = (np.mean(r1) - np.mean(r2)) / pooled_std
                # Correction factor
                cf = 1 - 3 / (4 * (n1 + n2) - 9)
                g *= cf
            else:
                g = 0
            return g
            
        elif metric == 'r_squared':
            # R-squared from correlation
            correlation = np.corrcoef(r1, r2)[0, 1]
            return correlation ** 2
            
        else:
            raise ValueError(f"Unknown effect size metric: {metric}")