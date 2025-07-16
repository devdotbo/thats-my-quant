"""
Walk-Forward Validation Framework

This module provides walk-forward analysis for robust out-of-sample
validation of trading strategies, helping detect overfitting and
ensuring strategy robustness.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats

from src.backtesting.engines.vectorbt_engine import VectorBTEngine
from src.utils.logging import get_logger


class WindowType(Enum):
    """Types of walk-forward windows"""
    ROLLING = "rolling"      # Fixed IS/OOS size, rolling forward
    ANCHORED = "anchored"    # Growing IS, fixed OOS size
    EXPANDING = "expanding"  # Growing both IS and OOS


class OptimizationMetric(Enum):
    """Metrics to optimize during parameter selection"""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"  # Minimize this one


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window"""
    start_date: datetime
    in_sample_end: datetime
    out_sample_end: datetime
    window_id: int
    
    def __post_init__(self):
        """Validate window dates"""
        if self.in_sample_end <= self.start_date:
            raise ValueError("In-sample end must be after start date")
        if self.out_sample_end <= self.in_sample_end:
            raise ValueError("Out-of-sample end must be after in-sample end")
    
    @property
    def in_sample_days(self) -> int:
        """Number of days in in-sample period"""
        return (self.in_sample_end - self.start_date).days
    
    @property
    def out_sample_days(self) -> int:
        """Number of days in out-of-sample period"""
        return (self.out_sample_end - self.in_sample_end).days


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation"""
    window_results: List[Dict[str, Any]]
    strategy_class: type
    param_grid: Dict[str, List[Any]]
    optimization_metric: OptimizationMetric
    summary_stats: Optional[Dict[str, float]] = None
    
    def get_summary(self) -> Dict[str, float]:
        """Calculate summary statistics"""
        if self.summary_stats is not None:
            return self.summary_stats
            
        n_windows = len(self.window_results)
        if n_windows == 0:
            return {}
            
        # Extract scores
        is_scores = [r['in_sample_score'] for r in self.window_results 
                    if 'in_sample_score' in r]
        oos_scores = [r['out_sample_score'] for r in self.window_results 
                     if 'out_sample_score' in r]
        
        # Calculate averages
        avg_is = np.mean(is_scores) if is_scores else 0
        avg_oos = np.mean(oos_scores) if oos_scores else 0
        
        # Overfitting ratio: (OOS - IS) / |IS|
        overfitting_ratio = (avg_oos - avg_is) / abs(avg_is) if avg_is != 0 else 0
        
        # Consistency score: correlation between IS and OOS
        consistency = 0
        if len(is_scores) == len(oos_scores) and len(is_scores) > 1:
            consistency = np.corrcoef(is_scores, oos_scores)[0, 1]
            
        self.summary_stats = {
            'n_windows': n_windows,
            'avg_in_sample_score': avg_is,
            'avg_out_sample_score': avg_oos,
            'overfitting_ratio': overfitting_ratio,
            'consistency_score': consistency,
            'is_score_std': np.std(is_scores) if is_scores else 0,
            'oos_score_std': np.std(oos_scores) if oos_scores else 0
        }
        
        return self.summary_stats
    
    def get_parameter_stability(self) -> Dict[str, Dict[str, Any]]:
        """Analyze parameter stability across windows"""
        param_history = {}
        
        # Collect parameter values across windows
        for result in self.window_results:
            if 'best_params' not in result:
                continue
                
            for param, value in result['best_params'].items():
                if param not in param_history:
                    param_history[param] = []
                param_history[param].append(value)
        
        # Calculate stability metrics
        stability_metrics = {}
        for param, values in param_history.items():
            if not values:
                continue
                
            # Find mode (most common value)
            unique, counts = np.unique(values, return_counts=True)
            mode_idx = np.argmax(counts)
            mode_value = unique[mode_idx]
            mode_frequency = counts[mode_idx] / len(values)
            
            stability_metrics[param] = {
                'mode': mode_value,
                'frequency': mode_frequency,
                'unique_values': len(unique),
                'changes': sum(1 for i in range(1, len(values)) if values[i] != values[i-1])
            }
            
        return stability_metrics
    
    def analyze_performance_decay(self) -> Dict[str, Any]:
        """Analyze if performance decays over time"""
        # Extract OOS scores in chronological order
        oos_scores = []
        for result in self.window_results:
            if 'out_sample_score' in result:
                oos_scores.append(result['out_sample_score'])
                
        if len(oos_scores) < 3:
            return {'insufficient_data': True}
            
        # Linear regression on scores vs time
        x = np.arange(len(oos_scores))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, oos_scores)
        
        return {
            'trend_coefficient': slope,
            'trend_pvalue': p_value,
            'r_squared': r_value ** 2,
            'performance_declining': slope < 0 and p_value < 0.05,
            'avg_early_performance': np.mean(oos_scores[:len(oos_scores)//2]),
            'avg_late_performance': np.mean(oos_scores[len(oos_scores)//2:])
        }
    
    def export_to_csv(self, filepath: Union[str, Path]) -> None:
        """Export results to CSV"""
        rows = []
        for result in self.window_results:
            row = {
                'window_id': result.get('window', {}).window_id if 'window' in result else None,
                'start_date': result.get('window', {}).start_date if 'window' in result else None,
                'in_sample_end': result.get('window', {}).in_sample_end if 'window' in result else None,
                'out_sample_end': result.get('window', {}).out_sample_end if 'window' in result else None,
                'in_sample_score': result.get('in_sample_score', None),
                'out_sample_score': result.get('out_sample_score', None)
            }
            
            # Add best parameters
            if 'best_params' in result:
                for param, value in result['best_params'].items():
                    row[f'param_{param}'] = value
                    
            # Add key metrics
            for metric_type in ['in_sample_metrics', 'out_sample_metrics']:
                if metric_type in result:
                    for metric, value in result[metric_type].items():
                        row[f'{metric_type}_{metric}'] = value
                        
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        
    def export_to_json(self, filepath: Union[str, Path]) -> None:
        """Export results to JSON"""
        export_data = {
            'strategy_class': self.strategy_class.__name__,
            'param_grid': self.param_grid,
            'optimization_metric': self.optimization_metric.value,
            'summary': self.get_summary(),
            'parameter_stability': self.get_parameter_stability(),
            'performance_decay': self.analyze_performance_decay(),
            'window_results': []
        }
        
        # Convert window results to serializable format
        for result in self.window_results:
            window_data = result.copy()
            if 'window' in window_data:
                window = window_data['window']
                window_data['window'] = {
                    'window_id': window.window_id,
                    'start_date': window.start_date.isoformat(),
                    'in_sample_end': window.in_sample_end.isoformat(),
                    'out_sample_end': window.out_sample_end.isoformat()
                }
            export_data['window_results'].append(window_data)
            
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


class WalkForwardValidator:
    """
    Walk-Forward Validation for Trading Strategies
    
    Implements walk-forward analysis to validate strategy performance
    out-of-sample and detect overfitting.
    """
    
    def __init__(self,
                 in_sample_months: int = 12,
                 out_sample_months: int = 3,
                 window_type: WindowType = WindowType.ROLLING,
                 optimization_metric: OptimizationMetric = OptimizationMetric.SHARPE_RATIO,
                 min_periods: int = 252):
        """
        Initialize walk-forward validator
        
        Args:
            in_sample_months: Months for in-sample optimization
            out_sample_months: Months for out-of-sample testing
            window_type: Type of window (rolling, anchored, expanding)
            optimization_metric: Metric to optimize
            min_periods: Minimum periods required for validation
        """
        self.in_sample_months = in_sample_months
        self.out_sample_months = out_sample_months
        self.window_type = window_type
        self.optimization_metric = optimization_metric
        self.min_periods = min_periods
        self.logger = get_logger(self.__class__.__name__)
        self.custom_scoring_func: Optional[Callable] = None
        
    def set_custom_scoring(self, scoring_func: Callable[[Dict[str, float]], float]) -> None:
        """Set a custom scoring function"""
        self.custom_scoring_func = scoring_func
        
    def create_windows(self,
                      data: pd.DataFrame,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> List[WalkForwardWindow]:
        """
        Create walk-forward windows from data
        
        Args:
            data: Historical data
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            List of WalkForwardWindow objects
        """
        if start_date is None:
            start_date = data.index[0]
        if end_date is None:
            end_date = data.index[-1]
            
        # Ensure we have datetime objects
        if hasattr(start_date, 'to_pydatetime'):
            start_date = start_date.to_pydatetime()
        if hasattr(end_date, 'to_pydatetime'):
            end_date = end_date.to_pydatetime()
            
        # Check data sufficiency
        total_months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
        required_months = self.in_sample_months + self.out_sample_months
        
        if total_months < required_months:
            raise ValueError(f"Insufficient data: {total_months} months available, "
                           f"{required_months} months required")
        
        windows = []
        window_id = 0
        
        if self.window_type == WindowType.ROLLING:
            # Rolling windows
            current_start = start_date
            
            while True:
                # Calculate window dates using relativedelta for accurate month arithmetic
                is_end = current_start + relativedelta(months=self.in_sample_months)
                oos_end = is_end + relativedelta(months=self.out_sample_months)
                
                # Adjust to month end
                is_end = self._get_month_end(is_end - timedelta(days=1))
                oos_end = self._get_month_end(oos_end - timedelta(days=1))
                
                if oos_end > end_date:
                    break
                    
                windows.append(WalkForwardWindow(
                    start_date=current_start,
                    in_sample_end=is_end,
                    out_sample_end=oos_end,
                    window_id=window_id
                ))
                
                window_id += 1
                # Move to next window
                current_start = current_start + relativedelta(months=self.out_sample_months)
                
        elif self.window_type == WindowType.ANCHORED:
            # Anchored windows (growing in-sample)
            anchor_start = start_date
            current_is_months = self.in_sample_months
            
            while True:
                is_end = anchor_start + relativedelta(months=current_is_months)
                oos_end = is_end + relativedelta(months=self.out_sample_months)
                
                # Adjust to month end
                is_end = self._get_month_end(is_end - timedelta(days=1))
                oos_end = self._get_month_end(oos_end - timedelta(days=1))
                
                if oos_end > end_date:
                    break
                    
                windows.append(WalkForwardWindow(
                    start_date=anchor_start,
                    in_sample_end=is_end,
                    out_sample_end=oos_end,
                    window_id=window_id
                ))
                
                window_id += 1
                current_is_months += self.out_sample_months
                
        return windows
    
    def _get_month_end(self, date: datetime) -> datetime:
        """Get the last day of the month"""
        # Handle cases where the date might already be past month end
        # First, normalize to a valid date in the target month
        year = date.year
        month = date.month
        
        # Get first day of next month
        if month == 12:
            next_month_first = datetime(year + 1, 1, 1)
        else:
            next_month_first = datetime(year, month + 1, 1)
            
        # Last day of current month is day before first of next month
        return next_month_first - timedelta(days=1)
    
    def optimize_parameters(self,
                          strategy_class: type,
                          data: pd.DataFrame,
                          param_grid: Dict[str, List[Any]],
                          initial_capital: float = 100000) -> Tuple[Dict[str, Any], float]:
        """
        Optimize strategy parameters on given data
        
        Args:
            strategy_class: Strategy class to optimize
            data: Historical data for optimization
            param_grid: Parameter grid to search
            initial_capital: Initial capital for backtesting
            
        Returns:
            Tuple of (best_parameters, best_score)
        """
        engine = VectorBTEngine()
        
        # Run optimization
        # For Calmar ratio, optimize on Sharpe and calculate Calmar ourselves
        metric_to_use = ('sharpe_ratio' if self.optimization_metric == OptimizationMetric.CALMAR_RATIO 
                        else self.optimization_metric.value)
        
        optimization_result = engine.optimize_parameters(
            strategy_class=strategy_class,
            data=data,
            param_grid=param_grid,
            metric=metric_to_use,
            initial_capital=initial_capital
        )
        
        # Get best parameters and metric value
        if self.optimization_metric == OptimizationMetric.CALMAR_RATIO:
            # Recalculate best based on Calmar ratio
            df = optimization_result.results_df.copy()
            df['calmar_ratio'] = df['annualized_return'] / df['max_drawdown'].replace(0, 0.0001)
            # Replace -inf and inf with very negative/positive values
            df['calmar_ratio'] = df['calmar_ratio'].replace([np.inf, -np.inf], [100, -100])
            
            best_idx = df['calmar_ratio'].idxmax()
            best_row = df.iloc[best_idx]
            
            # Reconstruct best_params from the row
            param_cols = [col for col in df.columns if col not in [
                'total_return', 'annualized_return', 'sharpe_ratio', 'sortino_ratio',
                'max_drawdown', 'win_rate', 'profit_factor', 'trades_count', 
                'avg_trade_return', 'calmar_ratio', 'winning_trades', 'losing_trades',
                'avg_win', 'avg_loss'  # Add any other metric columns
            ]]
            best_params = {}
            for col in param_cols:
                value = best_row[col]
                # Convert numpy types to Python types
                if hasattr(value, 'item'):
                    value = value.item()
                best_params[col] = value
            best_metric_value = best_row['calmar_ratio']
        else:
            best_params = optimization_result.best_params
            best_metric_value = optimization_result.best_metric
        
        if best_metric_value == -np.inf or (isinstance(best_metric_value, float) and np.isnan(best_metric_value)):
            raise ValueError("No valid parameter combinations found")
        
        # For custom scoring, we need the full metrics
        if self.custom_scoring_func:
            # Find the best row based on the metric used
            if self.optimization_metric == OptimizationMetric.CALMAR_RATIO:
                # We already calculated best params above for Calmar
                best_metrics = optimization_result.results_df[
                    optimization_result.results_df['calmar_ratio'] == best_metric_value
                ].iloc[0].to_dict()
            else:
                best_idx = optimization_result.results_df[metric_to_use].idxmax()
                best_row = optimization_result.results_df.iloc[best_idx]
                best_metrics = best_row.to_dict()
            
            best_score = self.custom_scoring_func(best_metrics)
        else:
            best_score = best_metric_value
            
        return best_params, best_score
    
    def _calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate optimization score from metrics"""
        metric_name = self.optimization_metric.value
        
        # Handle special case for Calmar ratio
        if self.optimization_metric == OptimizationMetric.CALMAR_RATIO:
            # Calculate Calmar ratio: annualized return / max drawdown
            annual_return = metrics.get('annualized_return', 0)
            max_dd = metrics.get('max_drawdown', 1)  # Avoid division by zero
            if max_dd == 0:
                return 0 if annual_return <= 0 else 10  # Arbitrary high value for no drawdown
            return annual_return / max_dd
        
        if metric_name not in metrics:
            self.logger.warning(f"Metric {metric_name} not found, using Sharpe ratio")
            metric_name = 'sharpe_ratio'
            
        score = metrics.get(metric_name, 0)
        
        # Invert if we're minimizing (e.g., max drawdown)
        if self.optimization_metric == OptimizationMetric.MAX_DRAWDOWN:
            score = -abs(score)
            
        return score
    
    def run_window(self,
                  window: WalkForwardWindow,
                  data: pd.DataFrame,
                  strategy_class: type,
                  param_grid: Dict[str, List[Any]],
                  initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Run validation on a single window
        
        Args:
            window: Walk-forward window
            data: Full historical data
            strategy_class: Strategy class to test
            param_grid: Parameter grid
            initial_capital: Initial capital
            
        Returns:
            Dictionary with window results
        """
        # Extract in-sample data
        is_data = data[(data.index >= window.start_date) & 
                      (data.index <= window.in_sample_end)]
        
        # Extract out-of-sample data
        oos_data = data[(data.index > window.in_sample_end) & 
                       (data.index <= window.out_sample_end)]
        
        if len(is_data) < self.min_periods or len(oos_data) < 20:
            self.logger.warning(f"Insufficient data for window {window.window_id}")
            return {
                'window': window,
                'error': 'Insufficient data'
            }
        
        try:
            # Optimize on in-sample
            best_params, is_score = self.optimize_parameters(
                strategy_class=strategy_class,
                data=is_data,
                param_grid=param_grid,
                initial_capital=initial_capital
            )
            
            # Debug print for Calmar ratio issue
            if self.optimization_metric == OptimizationMetric.CALMAR_RATIO:
                self.logger.debug(f"Best params for Calmar: {best_params}")
                
            # Test on out-of-sample
            strategy = strategy_class(parameters=best_params)
            engine = VectorBTEngine()
            
            oos_result = engine.run_backtest(
                strategy=strategy,
                data=oos_data,
                initial_capital=initial_capital
            )
            
            oos_metrics = oos_result.metrics
            if self.custom_scoring_func:
                oos_score = self.custom_scoring_func(oos_metrics)
            else:
                oos_score = self._calculate_score(oos_metrics)
            
            # Get in-sample metrics for comparison
            is_result = engine.run_backtest(
                strategy=strategy,
                data=is_data,
                initial_capital=initial_capital
            )
            is_metrics = is_result.metrics
            
            return {
                'window': window,
                'best_params': best_params,
                'in_sample_metrics': is_metrics,
                'out_sample_metrics': oos_metrics,
                'in_sample_score': is_score,
                'out_sample_score': oos_score
            }
            
        except Exception as e:
            self.logger.error(f"Error in window {window.window_id}: {str(e)}")
            return {
                'window': window,
                'error': str(e)
            }
    
    def run_validation(self,
                      data: pd.DataFrame,
                      strategy_class: type,
                      param_grid: Dict[str, List[Any]],
                      initial_capital: float = 100000,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      n_jobs: int = -1) -> WalkForwardResult:
        """
        Run full walk-forward validation
        
        Args:
            data: Historical data
            strategy_class: Strategy class to validate
            param_grid: Parameter grid
            initial_capital: Initial capital
            start_date: Start date for validation
            end_date: End date for validation
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            
        Returns:
            WalkForwardResult object
        """
        # Create windows
        windows = self.create_windows(data, start_date, end_date)
        
        if not windows:
            raise ValueError("No valid windows created")
            
        self.logger.info(f"Running walk-forward validation with {len(windows)} windows")
        
        # Run windows in parallel
        if n_jobs == -1:
            n_jobs = None  # Use all available CPUs
        elif n_jobs == 1:
            # Sequential processing
            results = []
            for window in windows:
                result = self.run_window(
                    window=window,
                    data=data,
                    strategy_class=strategy_class,
                    param_grid=param_grid,
                    initial_capital=initial_capital
                )
                results.append(result)
        else:
            # Parallel processing
            results = []
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_to_window = {
                    executor.submit(
                        self.run_window,
                        window=window,
                        data=data,
                        strategy_class=strategy_class,
                        param_grid=param_grid,
                        initial_capital=initial_capital
                    ): window
                    for window in windows
                }
                
                for future in as_completed(future_to_window):
                    result = future.result()
                    results.append(result)
                    
        # Sort results by window ID
        def get_window_id(result):
            if 'window' in result and hasattr(result['window'], 'window_id'):
                return result['window'].window_id
            return -1
            
        results.sort(key=get_window_id)
        
        # Create result object
        return WalkForwardResult(
            window_results=results,
            strategy_class=strategy_class,
            param_grid=param_grid,
            optimization_metric=self.optimization_metric
        )