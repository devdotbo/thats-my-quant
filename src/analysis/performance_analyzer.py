"""
Performance Analysis Module
Comprehensive strategy comparison and analysis tools
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.backtesting.engines.vectorbt_engine import BacktestResult
from src.utils.logging import get_logger


class RankingMethod(Enum):
    """Methods for ranking strategies"""
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    INFORMATION_RATIO = "information_ratio"
    CUSTOM = "custom"


@dataclass
class ComparisonResult:
    """Container for strategy comparison results"""
    strategy_metrics: pd.DataFrame
    rankings: pd.DataFrame
    statistical_tests: Dict[str, pd.DataFrame]
    relative_performance: pd.DataFrame
    correlation_matrix: pd.DataFrame
    summary_stats: Dict[str, Any]
    
    def get_top_strategies(self, 
                          metric: str = "sharpe_ratio", 
                          n: int = 5) -> pd.DataFrame:
        """Get top N strategies by specified metric"""
        if metric not in self.rankings.columns:
            raise ValueError(f"Metric {metric} not found in rankings")
        
        return self.rankings.nsmallest(n, metric)[['strategy', metric]]
    
    def get_statistical_summary(self) -> Dict[str, Any]:
        """Get summary of statistical test results"""
        summary = {}
        for test_name, results in self.statistical_tests.items():
            significant_pairs = results[results['p_value'] < 0.05]
            summary[test_name] = {
                'total_comparisons': len(results),
                'significant_at_5%': len(significant_pairs),
                'significant_pairs': significant_pairs[['strategy_1', 'strategy_2', 'p_value']].to_dict('records')
            }
        return summary


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for strategy comparison
    
    Features:
    - Multi-strategy comparison with statistical significance testing
    - Strategy ranking by various metrics
    - Relative performance calculation
    - Correlation analysis
    - Information ratio and alpha calculation
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
        """
        self.risk_free_rate = risk_free_rate
        self.logger = get_logger('performance_analyzer')
    
    def compare_strategies(self,
                         results: Dict[str, BacktestResult],
                         benchmark: Optional[Union[str, pd.Series]] = None,
                         ranking_metrics: Optional[List[str]] = None,
                         statistical_tests: Optional[List[str]] = None) -> ComparisonResult:
        """
        Compare multiple strategy backtest results
        
        Args:
            results: Dictionary of strategy_name -> BacktestResult
            benchmark: Benchmark strategy name or returns series
            ranking_metrics: Metrics to use for ranking (defaults to common set)
            statistical_tests: Statistical tests to perform (defaults to ['sharpe_difference', 't_test'])
            
        Returns:
            ComparisonResult with comprehensive comparison data
        """
        self.logger.info(f"Comparing {len(results)} strategies")
        
        if not results:
            raise ValueError("At least one strategy result required")
        
        # Default metrics for ranking
        if ranking_metrics is None:
            ranking_metrics = [
                'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                'total_return', 'max_drawdown', 'win_rate'
            ]
        
        # Default statistical tests
        if statistical_tests is None:
            statistical_tests = ['sharpe_difference', 't_test']
        
        # Extract metrics for all strategies
        strategy_metrics = self._extract_all_metrics(results)
        
        # Calculate rankings
        rankings = self._calculate_rankings(strategy_metrics, ranking_metrics)
        
        # Perform statistical tests
        test_results = self._perform_statistical_tests(results, statistical_tests)
        
        # Calculate relative performance
        relative_perf = self._calculate_relative_performance(results, benchmark)
        
        # Calculate correlation matrix
        correlation = self._calculate_correlation_matrix(results)
        
        # Summary statistics
        summary_stats = self._calculate_summary_stats(strategy_metrics, test_results)
        
        return ComparisonResult(
            strategy_metrics=strategy_metrics,
            rankings=rankings,
            statistical_tests=test_results,
            relative_performance=relative_perf,
            correlation_matrix=correlation,
            summary_stats=summary_stats
        )
    
    def _extract_all_metrics(self, results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """Extract performance metrics for all strategies"""
        metrics_list = []
        
        for name, result in results.items():
            metrics = result.metrics.copy()
            metrics['strategy'] = name
            
            # Add additional calculated metrics
            returns = result.equity_curve.pct_change().dropna()
            
            # Downside deviation for Sortino
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            
            # Calmar ratio (already in metrics usually)
            if 'calmar_ratio' not in metrics:
                annual_return = (result.equity_curve.iloc[-1] / result.equity_curve.iloc[0]) ** (252/len(returns)) - 1
                max_dd = abs(result.metrics.get('max_drawdown', 0))
                metrics['calmar_ratio'] = annual_return / max_dd if max_dd > 0 else 0
            
            # Profit factor
            if hasattr(result, 'trades') and result.trades is not None and len(result.trades) > 0:
                winning_trades = result.trades[result.trades['pnl'] > 0]['pnl'].sum()
                losing_trades = abs(result.trades[result.trades['pnl'] < 0]['pnl'].sum())
                metrics['profit_factor'] = winning_trades / losing_trades if losing_trades > 0 else np.inf
            else:
                metrics['profit_factor'] = np.nan
            
            # Volatility
            metrics['volatility'] = returns.std() * np.sqrt(252)
            
            # Skewness and kurtosis
            metrics['skewness'] = returns.skew()
            metrics['kurtosis'] = returns.kurtosis()
            
            # Maximum consecutive wins/losses
            if hasattr(result, 'trades') and result.trades is not None and len(result.trades) > 0:
                trade_results = (result.trades['pnl'] > 0).astype(int)
                # Count consecutive wins
                wins = (trade_results.groupby((trade_results != trade_results.shift()).cumsum()).cumsum())
                metrics['max_consecutive_wins'] = wins.max()
                # Count consecutive losses
                losses = ((1 - trade_results).groupby(((1 - trade_results) != (1 - trade_results).shift()).cumsum()).cumsum())
                metrics['max_consecutive_losses'] = losses.max()
            else:
                metrics['max_consecutive_wins'] = 0
                metrics['max_consecutive_losses'] = 0
            
            metrics_list.append(metrics)
        
        # Create DataFrame
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.set_index('strategy', inplace=True)
        
        return metrics_df
    
    def _calculate_rankings(self, 
                          metrics_df: pd.DataFrame,
                          ranking_metrics: List[str]) -> pd.DataFrame:
        """Calculate strategy rankings by various metrics"""
        rankings = pd.DataFrame(index=metrics_df.index)
        rankings['strategy'] = metrics_df.index
        
        for metric in ranking_metrics:
            if metric not in metrics_df.columns:
                self.logger.warning(f"Metric {metric} not found in results")
                continue
            
            # Determine if higher is better or lower is better
            lower_is_better = metric in ['max_drawdown', 'volatility']
            
            # Rank strategies
            if lower_is_better:
                rankings[f'{metric}_rank'] = metrics_df[metric].rank(method='min')
            else:
                rankings[f'{metric}_rank'] = metrics_df[metric].rank(method='min', ascending=False)
            
            # Add actual metric value
            rankings[metric] = metrics_df[metric]
        
        # Calculate composite rank (average of all individual ranks)
        rank_columns = [col for col in rankings.columns if col.endswith('_rank')]
        if rank_columns:
            rankings['composite_rank'] = rankings[rank_columns].mean(axis=1)
            rankings = rankings.sort_values('composite_rank')
        
        return rankings
    
    def _perform_statistical_tests(self,
                                 results: Dict[str, BacktestResult],
                                 test_types: List[str]) -> Dict[str, pd.DataFrame]:
        """Perform statistical significance tests between strategies"""
        test_results = {}
        
        if 'sharpe_difference' in test_types:
            test_results['sharpe_difference'] = self._test_sharpe_differences(results)
        
        if 't_test' in test_types:
            test_results['t_test'] = self._test_returns_difference(results, test_type='t')
        
        if 'mann_whitney' in test_types:
            test_results['mann_whitney'] = self._test_returns_difference(results, test_type='mann_whitney')
        
        return test_results
    
    def _test_sharpe_differences(self, results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """Test for significant differences in Sharpe ratios"""
        strategy_names = list(results.keys())
        n_strategies = len(strategy_names)
        
        # Create results matrix
        comparisons = []
        
        for i in range(n_strategies):
            for j in range(i + 1, n_strategies):
                strat1, strat2 = strategy_names[i], strategy_names[j]
                
                # Get returns
                returns1 = results[strat1].equity_curve.pct_change().dropna()
                returns2 = results[strat2].equity_curve.pct_change().dropna()
                
                # Align returns
                aligned_dates = returns1.index.intersection(returns2.index)
                returns1 = returns1.loc[aligned_dates]
                returns2 = returns2.loc[aligned_dates]
                
                # Calculate Sharpe ratios
                sharpe1 = results[strat1].metrics['sharpe_ratio']
                sharpe2 = results[strat2].metrics['sharpe_ratio']
                
                # Test for difference using Jobson-Korkie test
                # This is a simplified version - full implementation would use the JK statistic
                diff_returns = returns1 - returns2
                t_stat = np.sqrt(len(diff_returns)) * diff_returns.mean() / diff_returns.std()
                p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
                
                comparisons.append({
                    'strategy_1': strat1,
                    'strategy_2': strat2,
                    'sharpe_1': sharpe1,
                    'sharpe_2': sharpe2,
                    'difference': sharpe1 - sharpe2,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant_5%': p_value < 0.05
                })
        
        return pd.DataFrame(comparisons)
    
    def _test_returns_difference(self, 
                               results: Dict[str, BacktestResult],
                               test_type: str = 't') -> pd.DataFrame:
        """Test for differences in returns using t-test or Mann-Whitney U"""
        strategy_names = list(results.keys())
        n_strategies = len(strategy_names)
        
        comparisons = []
        
        for i in range(n_strategies):
            for j in range(i + 1, n_strategies):
                strat1, strat2 = strategy_names[i], strategy_names[j]
                
                # Get returns
                returns1 = results[strat1].equity_curve.pct_change().dropna()
                returns2 = results[strat2].equity_curve.pct_change().dropna()
                
                # Align returns
                aligned_dates = returns1.index.intersection(returns2.index)
                returns1 = returns1.loc[aligned_dates]
                returns2 = returns2.loc[aligned_dates]
                
                # Perform test
                if test_type == 't':
                    stat, p_value = stats.ttest_ind(returns1, returns2)
                    test_name = 't_statistic'
                else:  # mann_whitney
                    stat, p_value = stats.mannwhitneyu(returns1, returns2, alternative='two-sided')
                    test_name = 'u_statistic'
                
                comparisons.append({
                    'strategy_1': strat1,
                    'strategy_2': strat2,
                    'mean_return_1': returns1.mean() * 252,
                    'mean_return_2': returns2.mean() * 252,
                    'difference': (returns1.mean() - returns2.mean()) * 252,
                    test_name: stat,
                    'p_value': p_value,
                    'significant_5%': p_value < 0.05
                })
        
        return pd.DataFrame(comparisons)
    
    def _calculate_relative_performance(self,
                                      results: Dict[str, BacktestResult],
                                      benchmark: Optional[Union[str, pd.Series]] = None) -> pd.DataFrame:
        """Calculate relative performance metrics against benchmark"""
        if benchmark is None:
            # Use equal-weighted portfolio as benchmark
            self.logger.info("No benchmark specified, using equal-weighted portfolio")
            benchmark_returns = self._create_equal_weight_benchmark(results)
            benchmark_name = "Equal Weight"
        elif isinstance(benchmark, str):
            if benchmark not in results:
                raise ValueError(f"Benchmark strategy '{benchmark}' not found in results")
            benchmark_returns = results[benchmark].equity_curve.pct_change().dropna()
            benchmark_name = benchmark
        else:
            benchmark_returns = benchmark
            benchmark_name = "Custom Benchmark"
        
        relative_metrics = []
        
        for name, result in results.items():
            if name == benchmark_name:
                continue
            
            strategy_returns = result.equity_curve.pct_change().dropna()
            
            # Align returns
            aligned_dates = strategy_returns.index.intersection(benchmark_returns.index)
            strat_ret = strategy_returns.loc[aligned_dates]
            bench_ret = benchmark_returns.loc[aligned_dates]
            
            # Calculate alpha and beta
            if len(strat_ret) > 30:  # Need sufficient data points
                # Simple linear regression for alpha/beta
                X = bench_ret.values.reshape(-1, 1)
                y = strat_ret.values
                
                # Add constant for alpha
                X_with_const = np.column_stack([np.ones(len(X)), X])
                
                # OLS regression
                try:
                    beta_alpha = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                    alpha = beta_alpha[0] * 252  # Annualized
                    beta = beta_alpha[1]
                except:
                    alpha, beta = np.nan, np.nan
            else:
                alpha, beta = np.nan, np.nan
            
            # Information ratio
            excess_returns = strat_ret - bench_ret
            if excess_returns.std() > 0:
                information_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            else:
                information_ratio = 0
            
            # Tracking error
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            # Outperformance
            total_outperformance = (result.equity_curve.iloc[-1] / result.equity_curve.iloc[0]) / \
                                 ((1 + bench_ret).cumprod().iloc[-1] / (1 + bench_ret).cumprod().iloc[0]) - 1
            
            relative_metrics.append({
                'strategy': name,
                'benchmark': benchmark_name,
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error,
                'total_outperformance': total_outperformance,
                'correlation': strat_ret.corr(bench_ret)
            })
        
        return pd.DataFrame(relative_metrics)
    
    def _create_equal_weight_benchmark(self, results: Dict[str, BacktestResult]) -> pd.Series:
        """Create equal-weighted benchmark from all strategies"""
        returns_list = []
        for name, result in results.items():
            returns = result.equity_curve.pct_change().dropna()
            returns_list.append(returns)
        
        # Align all returns to common dates
        returns_df = pd.concat(returns_list, axis=1)
        returns_df.columns = list(results.keys())
        
        # Equal weight
        benchmark_returns = returns_df.mean(axis=1)
        
        return benchmark_returns
    
    def _calculate_correlation_matrix(self, results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """Calculate correlation matrix of strategy returns"""
        returns_dict = {}
        
        for name, result in results.items():
            returns = result.equity_curve.pct_change().dropna()
            returns_dict[name] = returns
        
        returns_df = pd.DataFrame(returns_dict)
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def _calculate_summary_stats(self,
                               metrics_df: pd.DataFrame,
                               test_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate summary statistics for the comparison"""
        summary = {
            'n_strategies': len(metrics_df),
            'best_sharpe': metrics_df['sharpe_ratio'].max(),
            'best_sharpe_strategy': metrics_df['sharpe_ratio'].idxmax(),
            'best_return': metrics_df['total_return'].max(),
            'best_return_strategy': metrics_df['total_return'].idxmax(),
            'avg_sharpe': metrics_df['sharpe_ratio'].mean(),
            'avg_return': metrics_df['total_return'].mean(),
            'avg_max_drawdown': metrics_df['max_drawdown'].mean()
        }
        
        # Add statistical test summary
        for test_name, results in test_results.items():
            if len(results) > 0:
                summary[f'{test_name}_significant_pairs'] = len(results[results['p_value'] < 0.05])
                summary[f'{test_name}_total_comparisons'] = len(results)
        
        return summary
    
    def rank_strategies(self,
                       results: Dict[str, BacktestResult],
                       method: RankingMethod = RankingMethod.SHARPE_RATIO,
                       custom_scorer: Optional[Callable] = None,
                       weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Rank strategies by specified method or custom scoring function
        
        Args:
            results: Strategy backtest results
            method: Ranking method to use
            custom_scorer: Custom scoring function (for CUSTOM method)
            weights: Weights for composite scoring (metric_name -> weight)
            
        Returns:
            DataFrame with strategy rankings
        """
        metrics_df = self._extract_all_metrics(results)
        
        if method == RankingMethod.CUSTOM:
            if custom_scorer is None:
                raise ValueError("custom_scorer required for CUSTOM ranking method")
            
            scores = {}
            for name, result in results.items():
                scores[name] = custom_scorer(result)
            
            rankings = pd.DataFrame({
                'strategy': list(scores.keys()),
                'score': list(scores.values())
            })
            rankings = rankings.sort_values('score', ascending=False)
            
        elif weights is not None:
            # Composite scoring with weights
            scores = pd.Series(0.0, index=metrics_df.index)
            
            for metric, weight in weights.items():
                if metric in metrics_df.columns:
                    # Normalize metric (0-1 range)
                    normalized = (metrics_df[metric] - metrics_df[metric].min()) / \
                               (metrics_df[metric].max() - metrics_df[metric].min())
                    
                    # Reverse if lower is better
                    if metric in ['max_drawdown', 'volatility']:
                        normalized = 1 - normalized
                    
                    scores += weight * normalized
            
            rankings = pd.DataFrame({
                'strategy': scores.index,
                'score': scores.values
            })
            rankings = rankings.sort_values('score', ascending=False)
            
        else:
            # Single metric ranking
            metric = method.value
            if metric not in metrics_df.columns:
                raise ValueError(f"Metric {metric} not found in results")
            
            ascending = metric in ['max_drawdown', 'volatility']
            
            rankings = pd.DataFrame({
                'strategy': metrics_df.index,
                metric: metrics_df[metric]
            })
            rankings = rankings.sort_values(metric, ascending=ascending)
        
        # Add rank column
        rankings['rank'] = range(1, len(rankings) + 1)
        
        return rankings
    
    def calculate_confidence_intervals(self,
                                     results: Dict[str, BacktestResult],
                                     metrics: List[str] = None,
                                     confidence_level: float = 0.95,
                                     n_bootstrap: int = 1000) -> pd.DataFrame:
        """
        Calculate bootstrap confidence intervals for strategy metrics
        
        Args:
            results: Strategy backtest results
            metrics: Metrics to calculate CIs for (defaults to key metrics)
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            DataFrame with confidence intervals
        """
        if metrics is None:
            metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
        
        ci_results = []
        
        for name, result in results.items():
            returns = result.equity_curve.pct_change().dropna()
            
            # Bootstrap metrics
            bootstrap_metrics = {metric: [] for metric in metrics}
            
            for _ in range(n_bootstrap):
                # Resample returns with replacement
                sample_returns = returns.sample(n=len(returns), replace=True)
                
                # Reconstruct equity curve
                sample_equity = 100000 * (1 + sample_returns).cumprod()
                
                # Calculate metrics
                if 'sharpe_ratio' in metrics:
                    sharpe = np.sqrt(252) * sample_returns.mean() / sample_returns.std() if sample_returns.std() > 0 else 0
                    bootstrap_metrics['sharpe_ratio'].append(sharpe)
                
                if 'total_return' in metrics:
                    total_ret = (sample_equity.iloc[-1] / sample_equity.iloc[0]) - 1
                    bootstrap_metrics['total_return'].append(total_ret)
                
                if 'max_drawdown' in metrics:
                    rolling_max = sample_equity.expanding().max()
                    drawdowns = (sample_equity - rolling_max) / rolling_max
                    max_dd = abs(drawdowns.min())
                    bootstrap_metrics['max_drawdown'].append(max_dd)
            
            # Calculate confidence intervals
            ci_data = {'strategy': name}
            
            for metric in metrics:
                values = np.array(bootstrap_metrics[metric])
                lower_percentile = (1 - confidence_level) / 2
                upper_percentile = 1 - lower_percentile
                
                ci_data[f'{metric}_mean'] = np.mean(values)
                ci_data[f'{metric}_ci_lower'] = np.percentile(values, lower_percentile * 100)
                ci_data[f'{metric}_ci_upper'] = np.percentile(values, upper_percentile * 100)
                ci_data[f'{metric}_std'] = np.std(values)
            
            ci_results.append(ci_data)
        
        return pd.DataFrame(ci_results)