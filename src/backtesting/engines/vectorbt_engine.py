"""
VectorBT Backtesting Engine Wrapper
High-performance vectorized backtesting using VectorBT
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any, Optional, Union, List, Type
from dataclasses import dataclass
import itertools

from src.strategies.base import BaseStrategy
from src.utils.logging import get_logger


@dataclass
class BacktestResult:
    """Container for backtest results"""
    portfolio: Any  # vbt.Portfolio object
    trades: pd.DataFrame
    metrics: Dict[str, float]
    equity_curve: pd.Series
    signals: pd.Series
    positions: pd.Series


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    best_params: Dict[str, Any]
    best_metric: float
    results_df: pd.DataFrame
    all_results: List[BacktestResult]


class VectorBTEngine:
    """
    Wrapper for VectorBT backtesting engine
    
    Features:
    - Fast vectorized backtesting
    - Portfolio performance metrics
    - Parameter optimization
    - Multi-asset support
    - Transaction cost modeling
    """
    
    def __init__(self, freq: str = '1min'):
        """
        Initialize the backtesting engine
        
        Args:
            freq: Data frequency for the backtest
        """
        self.logger = get_logger("vectorbt_engine")
        self.freq = freq
        
        # Configure VectorBT settings
        vbt.settings.portfolio['init_cash'] = 10000
        vbt.settings.portfolio['fees'] = 0.001  # 0.1% default commission
        vbt.settings.portfolio['slippage'] = 0.0001  # 1 basis point slippage
        vbt.settings.portfolio['accumulate'] = False
        vbt.settings.portfolio['cash_sharing'] = True
    
    def run_backtest(self,
                    strategy: BaseStrategy,
                    data: pd.DataFrame,
                    initial_capital: float = 10000,
                    commission: float = 0.001,
                    slippage: float = 0.0001,
                    position_size: str = 'fixed',
                    position_size_params: Optional[Dict[str, Any]] = None) -> BacktestResult:
        """
        Run a backtest for a single strategy
        
        Args:
            strategy: Strategy instance to backtest
            data: Market data (OHLCV)
            initial_capital: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate
            position_size: Position sizing method
            position_size_params: Parameters for position sizing
            
        Returns:
            BacktestResult containing portfolio, trades, and metrics
        """
        self.logger.info(f"Running backtest for {strategy}")
        
        # Validate data
        strategy.validate_data(data)
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Calculate positions based on position sizing method
        if position_size == 'fixed':
            positions = self._calculate_fixed_positions(
                signals, data['close'], initial_capital, 
                position_size_params or {'size_pct': 0.1}
            )
        elif position_size == 'volatility':
            positions = self._calculate_volatility_positions(
                signals, data, initial_capital,
                position_size_params or {'target_vol': 0.02}
            )
        else:
            # Use strategy's own position calculation
            positions = strategy.calculate_positions(
                signals, initial_capital, risk_params=position_size_params
            )
        
        # Create portfolio using VectorBT
        portfolio = vbt.Portfolio.from_signals(
            close=data['close'],
            entries=signals > 0,
            exits=signals < 0,
            size=np.abs(positions),
            init_cash=initial_capital,
            fees=commission,
            slippage=slippage,
            freq=self.freq
        )
        
        # Extract trades
        trades_df = self._extract_trades(portfolio)
        
        # Calculate metrics
        metrics = self.calculate_metrics(portfolio, data)
        
        # Create result
        result = BacktestResult(
            portfolio=portfolio,
            trades=trades_df,
            metrics=metrics,
            equity_curve=portfolio.value(),
            signals=signals,
            positions=positions
        )
        
        # Add drawdown series to portfolio
        result.portfolio.drawdown_series = portfolio.drawdown()
        
        return result
    
    def run_multi_asset_backtest(self,
                               data: Dict[str, pd.DataFrame],
                               signals: pd.DataFrame,
                               initial_capital: float = 100000,
                               weights: Optional[Dict[str, float]] = None,
                               commission: float = 0.001) -> BacktestResult:
        """
        Run backtest for multiple assets
        
        Args:
            data: Dict of asset data {symbol: OHLCV DataFrame}
            signals: DataFrame with signals for each asset
            initial_capital: Starting capital
            weights: Portfolio weights for each asset
            commission: Commission rate
            
        Returns:
            BacktestResult for the portfolio
        """
        self.logger.info(f"Running multi-asset backtest for {len(data)} assets")
        
        # Use equal weights if not specified
        if weights is None:
            weights = {symbol: 1.0 / len(data) for symbol in data}
        
        # Prepare data for VectorBT
        close_prices = pd.DataFrame({
            symbol: df['close'] for symbol, df in data.items()
        })
        
        # Allocate capital based on weights
        asset_capital = {
            symbol: initial_capital * weight 
            for symbol, weight in weights.items()
        }
        
        # Calculate positions for each asset
        positions = pd.DataFrame()
        for symbol in data:
            if symbol in signals.columns:
                positions[symbol] = self._calculate_fixed_positions(
                    signals[symbol],
                    close_prices[symbol],
                    asset_capital[symbol],
                    {'size_pct': 0.1}
                )
        
        # Create multi-asset portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=close_prices,
            entries=signals > 0,
            exits=signals < 0,
            size=positions.abs(),
            init_cash=initial_capital,
            fees=commission,
            freq=self.freq,
            group_by=True,  # Group all assets into one portfolio
            cash_sharing=True  # Share cash between assets
        )
        
        # Calculate aggregate metrics
        metrics = self.calculate_metrics(portfolio, close_prices)
        
        # Get asset-level returns
        asset_returns = {}
        for symbol in data:
            asset_portfolio = vbt.Portfolio.from_signals(
                close=close_prices[symbol],
                entries=signals[symbol] > 0,
                exits=signals[symbol] < 0,
                size=positions[symbol].abs(),
                init_cash=asset_capital[symbol],
                fees=commission,
                freq=self.freq
            )
            asset_returns[symbol] = asset_portfolio.total_return()
        
        result = BacktestResult(
            portfolio=portfolio,
            trades=self._extract_trades(portfolio),
            metrics=metrics,
            equity_curve=portfolio.value(),
            signals=signals,
            positions=positions
        )
        
        # Add asset returns to result
        result.asset_returns = asset_returns
        
        return result
    
    def optimize_parameters(self,
                          strategy_class: Type[BaseStrategy],
                          data: pd.DataFrame,
                          param_grid: Dict[str, List[Any]],
                          metric: str = 'sharpe_ratio',
                          initial_capital: float = 10000,
                          commission: float = 0.001) -> OptimizationResult:
        """
        Optimize strategy parameters
        
        Args:
            strategy_class: Strategy class to optimize
            data: Market data
            param_grid: Parameter grid to search
            metric: Metric to optimize
            initial_capital: Starting capital
            commission: Commission rate
            
        Returns:
            OptimizationResult with best parameters and all results
        """
        self.logger.info(f"Optimizing parameters for {strategy_class.__name__}")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        results = []
        metrics_list = []
        
        # Test each combination
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            
            try:
                # Create strategy instance with parameters
                strategy = strategy_class(parameters=param_dict)
                
                # Run backtest
                result = self.run_backtest(
                    strategy=strategy,
                    data=data,
                    initial_capital=initial_capital,
                    commission=commission
                )
                
                # Store results
                result_dict = param_dict.copy()
                result_dict.update(result.metrics)
                results.append(result_dict)
                metrics_list.append(result.metrics.get(metric, -np.inf))
                
            except Exception as e:
                self.logger.warning(f"Failed to test parameters {param_dict}: {e}")
                metrics_list.append(-np.inf)
        
        # Find best parameters
        best_idx = np.argmax(metrics_list)
        best_params = dict(zip(param_names, param_combinations[best_idx]))
        best_metric_value = metrics_list[best_idx]
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        self.logger.info(f"Best {metric}: {best_metric_value:.4f} with params: {best_params}")
        
        return OptimizationResult(
            best_params=best_params,
            best_metric=best_metric_value,
            results_df=results_df,
            all_results=[]  # Could store all BacktestResults if needed
        )
    
    def calculate_metrics(self, 
                        portfolio: vbt.Portfolio,
                        data: Union[pd.DataFrame, pd.Series]) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            portfolio: VectorBT portfolio object
            data: Market data used in backtest
            
        Returns:
            Dictionary of performance metrics
        """
        # Basic returns
        total_return = portfolio.total_return()
        
        # Annualized metrics
        returns = portfolio.returns()
        periods_per_year = self._get_periods_per_year()
        
        # Handle empty returns
        if len(returns) == 0 or returns.std() == 0:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            annualized_return = 0.0
        else:
            annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(periods_per_year)
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                sortino_ratio = returns.mean() / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0
            else:
                sortino_ratio = sharpe_ratio * 2  # No downside
        
        # Drawdown (make it positive for consistency)
        max_drawdown = abs(portfolio.max_drawdown())
        
        # Trade statistics
        trades = portfolio.trades.records_readable
        trades_count = len(trades)
        
        if trades_count > 0:
            winning_trades = trades[trades['PnL'] > 0]
            losing_trades = trades[trades['PnL'] < 0]
            
            win_rate = len(winning_trades) / trades_count
            
            avg_win = winning_trades['PnL'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['PnL'].mean()) if len(losing_trades) > 0 else 0
            
            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if avg_loss > 0 and len(losing_trades) > 0 else np.inf
            
            avg_trade_return = trades['Return'].mean()
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_return = 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades_count': trades_count,
            'avg_trade_return': avg_trade_return
        }
        
        return metrics
    
    def generate_report(self, 
                       result: BacktestResult,
                       format: str = 'text') -> Union[str, Dict[str, Any]]:
        """
        Generate a performance report
        
        Args:
            result: Backtest result to report on
            format: 'text' or 'dict'
            
        Returns:
            Formatted report
        """
        metrics = result.metrics
        
        if format == 'dict':
            return {
                'metrics': metrics,
                'summary': {
                    'initial_capital': result.portfolio.init_cash,
                    'final_value': result.portfolio.final_value(),
                    'total_return_pct': metrics['total_return'] * 100,
                    'max_drawdown_pct': metrics['max_drawdown'] * 100,
                    'trades': metrics['trades_count']
                }
            }
        
        # Text report
        report = []
        report.append("=" * 50)
        report.append("BACKTEST PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append("")
        
        report.append("Returns:")
        report.append(f"  Total Return: {metrics['total_return']:.2%}")
        report.append(f"  Annualized Return: {metrics['annualized_return']:.2%}")
        report.append("")
        
        report.append("Risk Metrics:")
        report.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        report.append(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        report.append(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        report.append("")
        
        report.append("Trading Statistics:")
        report.append(f"  Number of Trades: {metrics['trades_count']}")
        report.append(f"  Win Rate: {metrics['win_rate']:.2%}")
        report.append(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        report.append(f"  Avg Trade Return: {metrics['avg_trade_return']:.2%}")
        report.append("")
        
        report.append("Portfolio Summary:")
        report.append(f"  Initial Capital: ${result.portfolio.init_cash:,.2f}")
        report.append(f"  Final Value: ${result.portfolio.final_value():,.2f}")
        report.append("")
        
        return "\n".join(report)
    
    # Helper methods
    def _calculate_fixed_positions(self, 
                                 signals: pd.Series,
                                 prices: pd.Series,
                                 capital: float,
                                 params: Dict[str, Any]) -> pd.Series:
        """Calculate fixed position sizes"""
        size_pct = params.get('size_pct', 0.1)
        position_value = capital * size_pct
        shares = position_value / prices
        
        return signals * shares
    
    def _calculate_volatility_positions(self,
                                      signals: pd.Series,
                                      data: pd.DataFrame,
                                      capital: float,
                                      params: Dict[str, Any]) -> pd.Series:
        """Calculate volatility-adjusted position sizes"""
        target_vol = params.get('target_vol', 0.02)
        lookback = params.get('lookback', 20)
        
        # Calculate rolling volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(lookback).std() * np.sqrt(252 * 390)  # Annualized
        
        # Avoid division by zero
        volatility = volatility.bfill()
        volatility[volatility == 0] = 0.01
        
        # Position size inversely proportional to volatility
        position_value = (capital * target_vol) / volatility
        shares = position_value / data['close']
        
        return signals * shares
    
    def _extract_trades(self, portfolio: vbt.Portfolio) -> pd.DataFrame:
        """Extract trade information from portfolio"""
        if len(portfolio.trades.records) == 0:
            return pd.DataFrame()
        
        trades = portfolio.trades.records_readable
        
        # Get actual column names
        column_mapping = {
            'Entry Timestamp': 'entry_time',
            'Exit Timestamp': 'exit_time',
            'Size': 'size',
            'Entry Price': 'entry_price',
            'Exit Price': 'exit_price',
            'PnL': 'pnl',
            'Return': 'return_pct'
        }
        
        # Create trades dataframe with available columns
        trades_df = pd.DataFrame()
        for old_col, new_col in column_mapping.items():
            if old_col in trades.columns:
                trades_df[new_col] = trades[old_col]
        
        # Add fixed columns
        trades_df['symbol'] = 'Asset'
        
        # Handle direction/side
        if 'Direction' in trades.columns:
            trades_df['side'] = trades['Direction'].str.lower()
        elif 'Side' in trades.columns:
            trades_df['side'] = trades['Side'].str.lower()
        else:
            trades_df['side'] = 'long'  # Default
        
        # Calculate commission if not present
        if 'Fees' in trades.columns:
            trades_df['commission'] = trades['Fees']
        else:
            trades_df['commission'] = 0.0
        
        return trades_df
    
    def _get_periods_per_year(self) -> int:
        """Get number of periods per year based on frequency"""
        freq_map = {
            '1min': 252 * 390,  # Trading minutes per year
            '5min': 252 * 78,
            '15min': 252 * 26,
            '30min': 252 * 13,
            '1H': 252 * 6.5,
            'D': 252,
            'W': 52,
            'M': 12
        }
        
        return freq_map.get(self.freq, 252)