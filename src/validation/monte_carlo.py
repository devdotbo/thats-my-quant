"""
Monte Carlo Validation Framework

This module provides Monte Carlo simulation capabilities for assessing
the statistical robustness of trading strategies through resampling
techniques and confidence interval estimation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
import warnings

from src.backtesting.engines.vectorbt_engine import BacktestResult
from src.utils.logging import get_logger


class ResamplingMethod(Enum):
    """Methods for resampling trade sequences"""
    BOOTSTRAP = "bootstrap"              # Standard bootstrap with replacement
    BLOCK = "block"                     # Block bootstrap for time series
    STATIONARY_BOOTSTRAP = "stationary" # Stationary bootstrap with random block sizes


class ConfidenceLevel(Enum):
    """Common confidence levels"""
    CL_90 = 0.90
    CL_95 = 0.95
    CL_99 = 0.99
    CL_99_9 = 0.999


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation results"""
    original_metrics: Dict[str, float]
    simulation_results: List[Dict[str, Any]]
    confidence_intervals: Dict[str, Dict[float, Dict[str, float]]]
    risk_metrics: Optional[Dict[str, float]] = None
    equity_curves: Optional[List[pd.Series]] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of Monte Carlo results"""
        n_sims = len(self.simulation_results)
        
        summary = {
            'n_simulations': n_sims,
            'original_metrics': self.original_metrics,
            'confidence_intervals': self.confidence_intervals
        }
        
        if self.risk_metrics:
            summary['risk_metrics'] = self.risk_metrics
            
        return summary
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        if self.risk_metrics is not None:
            return self.risk_metrics
            
        metrics = {}
        
        # Extract metric arrays
        sharpe_ratios = [r['sharpe_ratio'] for r in self.simulation_results]
        max_drawdowns = [r['max_drawdown'] for r in self.simulation_results]
        total_returns = [r['total_return'] for r in self.simulation_results]
        
        # Risk of negative Sharpe
        if len(sharpe_ratios) > 0:
            metrics['sharpe_below_zero_probability'] = sum(1 for s in sharpe_ratios if s < 0) / len(sharpe_ratios)
        else:
            metrics['sharpe_below_zero_probability'] = 0.0
        
        # Extreme drawdown risk
        if len(max_drawdowns) > 0:
            metrics['max_drawdown_95th_percentile'] = np.percentile(max_drawdowns, 95)
            metrics['max_drawdown_99th_percentile'] = np.percentile(max_drawdowns, 99)
        else:
            metrics['max_drawdown_95th_percentile'] = 0.0
            metrics['max_drawdown_99th_percentile'] = 0.0
        
        # Risk of loss
        if len(total_returns) > 0:
            metrics['probability_of_loss'] = sum(1 for r in total_returns if r < 0) / len(total_returns)
        else:
            metrics['probability_of_loss'] = 0.0
        
        # Risk of ruin (if equity curves available)
        if self.equity_curves:
            metrics['risk_of_ruin'] = self._calculate_risk_of_ruin_from_curves()
        
        self.risk_metrics = metrics
        return metrics
    
    def _calculate_risk_of_ruin_from_curves(self, threshold: float = 0.5) -> float:
        """Calculate risk of ruin from equity curves"""
        if not self.equity_curves:
            return 0.0
            
        ruin_count = 0
        for curve in self.equity_curves:
            initial = curve.iloc[0]
            min_value = curve.min()
            if min_value / initial <= threshold:
                ruin_count += 1
                
        return ruin_count / len(self.equity_curves)
    
    def get_percentile_outcomes(self, percentiles: List[float]) -> Dict[str, Dict[float, float]]:
        """Get outcomes at specific percentiles"""
        outcomes: Dict[str, Dict[float, float]] = {}
        
        # Get unique metrics
        if not self.simulation_results:
            return outcomes
            
        metrics = list(self.simulation_results[0].keys())
        if 'equity_curve' in metrics:
            metrics.remove('equity_curve')
            
        for metric in metrics:
            values = [r[metric] for r in self.simulation_results]
            outcomes[metric] = {}
            for p in percentiles:
                outcomes[metric][p] = np.percentile(values, p)
                
        return outcomes
    
    def export_metrics_to_csv(self, filepath: Union[str, Path]) -> None:
        """Export simulation metrics to CSV"""
        # Convert results to DataFrame
        rows = []
        for i, result in enumerate(self.simulation_results):
            row = {'simulation_id': i}
            for metric, value in result.items():
                if metric != 'equity_curve':  # Skip series data
                    row[metric] = value
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        
    def export_summary_to_json(self, filepath: Union[str, Path]) -> None:
        """Export summary statistics to JSON"""
        summary = self.get_summary()
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
            
        summary_converted = convert_types(summary)
        
        with open(filepath, 'w') as f:
            json.dump(summary_converted, f, indent=2)


class MonteCarloValidator:
    """
    Monte Carlo Validation for Trading Strategies
    
    Uses resampling techniques to assess the statistical significance
    and robustness of backtest results.
    """
    
    def __init__(self,
                 n_simulations: int = 1000,
                 confidence_levels: Optional[List[float]] = None,
                 resampling_method: ResamplingMethod = ResamplingMethod.BOOTSTRAP,
                 random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo validator
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            confidence_levels: Confidence levels for intervals
            resampling_method: Method for resampling trades
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.resampling_method = resampling_method
        self.random_seed = random_seed
        self.logger = get_logger(self.__class__.__name__)
        self._rng: Union[np.random.RandomState, Any] = np.random.RandomState(random_seed) if random_seed is not None else np.random
    
    def resample_trades(self,
                       trades: pd.DataFrame,
                       method: ResamplingMethod,
                       block_size: Optional[int] = None) -> pd.DataFrame:
        """
        Resample trades using specified method
        
        Args:
            trades: Original trades DataFrame
            method: Resampling method
            block_size: Size of blocks for block bootstrap
            
        Returns:
            Resampled trades DataFrame
        """
        n_trades = len(trades)
        
        if n_trades == 0:
            return trades.copy()
        
        if method == ResamplingMethod.BOOTSTRAP:
            # Standard bootstrap with replacement
            indices = self._rng.choice(n_trades, size=n_trades, replace=True)
            resampled = trades.iloc[indices].reset_index(drop=True)
            
        elif method == ResamplingMethod.BLOCK:
            # Block bootstrap
            if block_size is None:
                # Use rule of thumb: n^(1/3)
                block_size = max(1, int(n_trades ** (1/3)))
                
            resampled_indices: List[int] = []
            while len(resampled_indices) < n_trades:
                # Pick random starting point
                start = self._rng.randint(0, n_trades)
                # Take block_size trades (with wrap-around)
                for i in range(block_size):
                    if len(resampled_indices) >= n_trades:
                        break
                    resampled_indices.append((start + i) % n_trades)
                    
            resampled = trades.iloc[resampled_indices[:n_trades]].reset_index(drop=True)
            
        elif method == ResamplingMethod.STATIONARY_BOOTSTRAP:
            # Stationary bootstrap with random block lengths
            avg_block_size = block_size or max(1, int(n_trades ** (1/3)))
            p = 1.0 / avg_block_size  # Probability of ending block
            
            resampled_indices: List[int] = []
            while len(resampled_indices) < n_trades:
                # Start new block
                current_idx = self._rng.randint(0, n_trades)
                
                # Continue block until random stop
                while len(resampled_indices) < n_trades:
                    resampled_indices.append(current_idx)
                    if self._rng.random() < p:  # End block
                        break
                    current_idx = (current_idx + 1) % n_trades
                    
            resampled = trades.iloc[resampled_indices[:n_trades]].reset_index(drop=True)
            
        else:
            raise ValueError(f"Unknown resampling method: {method}")
            
        return resampled
    
    def calculate_metric_distributions(self,
                                     simulation_results: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
        """
        Calculate distributions of metrics across simulations
        
        Args:
            simulation_results: List of metric dictionaries
            
        Returns:
            Dictionary mapping metrics to value arrays
        """
        if not simulation_results:
            return {}
            
        # Get all metric names
        metric_names = list(simulation_results[0].keys())
        if 'equity_curve' in metric_names:
            metric_names.remove('equity_curve')
            
        distributions = {}
        for metric in metric_names:
            values = [r[metric] for r in simulation_results if metric in r]
            distributions[metric] = np.array(values)
            
        return distributions
    
    def calculate_confidence_interval(self,
                                    values: Union[List[float], np.ndarray],
                                    confidence_level: float) -> Dict[str, float]:
        """
        Calculate confidence interval using percentile method
        
        Args:
            values: Array of values
            confidence_level: Confidence level (e.g., 0.95)
            
        Returns:
            Dictionary with lower, upper, mean, median
        """
        values = np.array(values)
        alpha = 1 - confidence_level
        
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            'lower': np.percentile(values, lower_percentile),
            'upper': np.percentile(values, upper_percentile),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values)
        }
    
    def bootstrap_returns(self,
                        returns: pd.Series,
                        n_days: Optional[int] = None) -> pd.Series:
        """
        Bootstrap resample returns series
        
        Args:
            returns: Original returns series
            n_days: Number of days to simulate
            
        Returns:
            Bootstrapped returns series
        """
        if n_days is None:
            n_days = len(returns)
            
        # Resample returns with replacement
        bootstrapped_returns = self._rng.choice(returns.values, size=n_days, replace=True)
        
        # Create new series with same frequency
        new_index = pd.date_range(
            start=returns.index[0],
            periods=n_days,
            freq=returns.index.freq or 'D'
        )
        
        return pd.Series(bootstrapped_returns, index=new_index)
    
    def calculate_risk_of_ruin(self,
                             equity_curves: List[pd.Series],
                             ruin_threshold: float = 0.5) -> float:
        """
        Calculate risk of ruin from equity curves
        
        Args:
            equity_curves: List of simulated equity curves
            ruin_threshold: Fraction of capital loss considered ruin
            
        Returns:
            Probability of ruin
        """
        if not equity_curves:
            return 0.0
            
        ruin_count = 0
        for curve in equity_curves:
            initial_capital = curve.iloc[0]
            min_equity = curve.min()
            
            if min_equity / initial_capital <= ruin_threshold:
                ruin_count += 1
                
        return ruin_count / len(equity_curves)
    
    def run_single_simulation(self,
                            backtest_result: BacktestResult,
                            simulation_id: int) -> Dict[str, Any]:
        """
        Run a single Monte Carlo simulation
        
        Args:
            backtest_result: Original backtest result
            simulation_id: Simulation identifier
            
        Returns:
            Dictionary of simulated metrics
        """
        # Resample trades
        original_trades = backtest_result.trades
        if len(original_trades) == 0:
            # No trades to resample
            return backtest_result.metrics.copy()
            
        resampled_trades = self.resample_trades(
            original_trades,
            self.resampling_method
        )
        
        # Calculate cumulative PnL from resampled trades
        initial_capital = backtest_result.portfolio.init_cash
        cumulative_pnl = resampled_trades['pnl'].cumsum()
        
        # Create equity values including initial capital
        equity_values = [initial_capital]
        equity_values.extend((initial_capital + cumulative_pnl).values)
        
        # Create simulated equity curve with proper dates
        dates = pd.date_range(
            start=backtest_result.equity_curve.index[0],
            end=backtest_result.equity_curve.index[-1],
            periods=len(equity_values)
        )
        equity_series = pd.Series(equity_values, index=dates)
        
        # Calculate metrics from simulated results
        total_return = (equity_series.iloc[-1] / initial_capital) - 1
        returns = equity_series.pct_change().dropna()
        
        # Calculate Sharpe ratio
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
            
        # Calculate max drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Calculate win rate
        winning_trades = resampled_trades[resampled_trades['pnl'] > 0]
        win_rate = len(winning_trades) / len(resampled_trades) if len(resampled_trades) > 0 else 0
        
        # Calculate profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(resampled_trades[resampled_trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        return {
            'simulation_id': simulation_id,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades_count': len(resampled_trades),
            'equity_curve': equity_series
        }
    
    def run_validation(self,
                      backtest_result: BacktestResult,
                      n_simulations: Optional[int] = None,
                      n_jobs: int = -1) -> MonteCarloResult:
        """
        Run full Monte Carlo validation
        
        Args:
            backtest_result: Original backtest result
            n_simulations: Override number of simulations
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            
        Returns:
            MonteCarloResult object
        """
        n_sims = n_simulations or self.n_simulations
        
        self.logger.info(f"Running {n_sims} Monte Carlo simulations")
        
        # Store original metrics
        original_metrics = backtest_result.metrics.copy()
        
        # Run simulations
        simulation_results = []
        equity_curves = []
        
        if n_jobs == 1:
            # Sequential execution
            for i in range(n_sims):
                result = self.run_single_simulation(backtest_result, i)
                simulation_results.append(result)
                if 'equity_curve' in result:
                    equity_curves.append(result['equity_curve'])
                    
        else:
            # Parallel execution
            # Handle n_jobs=-1 to use all CPUs
            if n_jobs == -1:
                max_workers = None  # Let ProcessPoolExecutor decide
            else:
                max_workers = n_jobs
                
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.run_single_simulation, backtest_result, i): i
                    for i in range(n_sims)
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        simulation_results.append(result)
                        if 'equity_curve' in result:
                            equity_curves.append(result['equity_curve'])
                    except Exception as e:
                        self.logger.error(f"Simulation failed: {e}")
        
        # Calculate distributions
        distributions = self.calculate_metric_distributions(simulation_results)
        
        # Calculate confidence intervals for each metric
        confidence_intervals: Dict[str, Dict[float, Dict[str, float]]] = {}
        for metric, values in distributions.items():
            confidence_intervals[metric] = {}
            for cl in self.confidence_levels:
                confidence_intervals[metric][cl] = self.calculate_confidence_interval(values, cl)
        
        # Create result object
        mc_result = MonteCarloResult(
            original_metrics=original_metrics,
            simulation_results=simulation_results,
            confidence_intervals=confidence_intervals,
            equity_curves=equity_curves if equity_curves else None
        )
        
        # Calculate risk metrics
        mc_result.get_risk_metrics()
        
        self.logger.info("Monte Carlo validation completed")
        
        return mc_result
    
    def test_statistical_significance(self,
                                    results_strategy: List[Dict[str, float]],
                                    results_baseline: List[Dict[str, float]],
                                    metric: str = 'sharpe_ratio',
                                    test_type: str = 'two-sided') -> float:
        """
        Test statistical significance between two sets of results
        
        Args:
            results_strategy: Results from strategy
            results_baseline: Results from baseline
            metric: Metric to compare
            test_type: 'two-sided', 'greater', or 'less'
            
        Returns:
            p-value from statistical test
        """
        # Extract metric values
        values_strategy = [r[metric] for r in results_strategy if metric in r]
        values_baseline = [r[metric] for r in results_baseline if metric in r]
        
        if not values_strategy or not values_baseline:
            self.logger.warning(f"No values found for metric {metric}")
            return 1.0
        
        # Use Welch's t-test (doesn't assume equal variances)
        if test_type == 'two-sided':
            alternative = 'two-sided'
        elif test_type == 'greater':
            alternative = 'greater'
        elif test_type == 'less':
            alternative = 'less'
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Suppress the FutureWarning about alternative parameter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            t_stat, p_value = stats.ttest_ind(
                values_strategy,
                values_baseline,
                equal_var=False,  # Welch's t-test
                alternative=alternative
            )
        
        return p_value
    
    def compare_strategies(self,
                         mc_results: List[Tuple[str, MonteCarloResult]],
                         metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        Compare multiple strategies using Monte Carlo results
        
        Args:
            mc_results: List of (strategy_name, MonteCarloResult) tuples
            metric: Metric to compare
            
        Returns:
            DataFrame with comparison statistics
        """
        comparison_data = []
        
        for name, result in mc_results:
            # Get confidence interval for the metric
            if metric in result.confidence_intervals:
                ci_95 = result.confidence_intervals[metric].get(0.95, {})
                ci_99 = result.confidence_intervals[metric].get(0.99, {})
                
                row = {
                    'strategy': name,
                    'original': result.original_metrics.get(metric, np.nan),
                    'mean': ci_95.get('mean', np.nan),
                    'median': ci_95.get('median', np.nan),
                    'std': ci_95.get('std', np.nan),
                    'ci_95_lower': ci_95.get('lower', np.nan),
                    'ci_95_upper': ci_95.get('upper', np.nan),
                    'ci_99_lower': ci_99.get('lower', np.nan),
                    'ci_99_upper': ci_99.get('upper', np.nan)
                }
                
                # Add risk metrics if available
                if result.risk_metrics:
                    row['sharpe_below_zero_prob'] = result.risk_metrics.get('sharpe_below_zero_probability', np.nan)
                    
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by mean value
        df = df.sort_values('mean', ascending=False)
        
        return df