"""
Portfolio Backtesting Module
Handles multi-strategy portfolio backtesting with various allocation methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings

from src.strategies.base import BaseStrategy
from src.backtesting.engines.vectorbt_engine import VectorBTEngine, BacktestResult
from src.utils.logging import get_logger


class AllocationMethod(Enum):
    """Portfolio allocation methods"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    CUSTOM_WEIGHTS = "custom_weights"
    INVERSE_VOLATILITY = "inverse_volatility"
    MEAN_VARIANCE = "mean_variance"


@dataclass
class PortfolioResult:
    """Container for portfolio backtest results"""
    strategy_results: Dict[str, BacktestResult]
    portfolio_metrics: Dict[str, float]
    weights: pd.DataFrame
    equity_curve: pd.Series
    returns: pd.Series
    correlation_matrix: pd.DataFrame
    strategy_contributions: pd.DataFrame
    
    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary statistics"""
        return {
            'portfolio_metrics': self.portfolio_metrics,
            'strategy_count': len(self.strategy_results),
            'total_return': self.portfolio_metrics['total_return'],
            'sharpe_ratio': self.portfolio_metrics['sharpe_ratio'],
            'max_drawdown': self.portfolio_metrics['max_drawdown'],
            'correlation_mean': self.correlation_matrix.mean().mean()
        }


class PortfolioBacktester:
    """
    Multi-strategy portfolio backtester
    
    Supports various allocation methods and rebalancing frequencies
    """
    
    def __init__(self, 
                 engine: Optional[VectorBTEngine] = None,
                 rebalance_frequency: str = 'monthly'):
        """
        Initialize portfolio backtester
        
        Args:
            engine: Backtesting engine to use (creates new if None)
            rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'never')
        """
        self.engine = engine or VectorBTEngine()
        self.rebalance_frequency = rebalance_frequency
        self.logger = get_logger('portfolio_backtester')
    
    def run_portfolio_backtest(self,
                             strategies: Dict[str, BaseStrategy],
                             data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                             initial_capital: float = 100000,
                             allocation_method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT,
                             custom_weights: Optional[Dict[str, float]] = None,
                             **kwargs) -> PortfolioResult:
        """
        Run portfolio backtest with multiple strategies
        
        Args:
            strategies: Dictionary of strategy_name -> strategy instance
            data: Market data (single DataFrame for all strategies or dict of DataFrames per strategy)
            initial_capital: Starting capital
            allocation_method: How to allocate capital between strategies
            custom_weights: Custom weights if using CUSTOM_WEIGHTS method
            **kwargs: Additional arguments passed to backtesting engine
            
        Returns:
            PortfolioResult with combined performance
        """
        self.logger.info(f"Running portfolio backtest with {len(strategies)} strategies")
        
        # Validate input
        if not strategies:
            raise ValueError("At least one strategy is required for portfolio backtesting")
        
        # Run individual strategy backtests
        strategy_results = self._run_individual_backtests(
            strategies, data, initial_capital, **kwargs
        )
        
        # Calculate portfolio weights
        weights = self._calculate_weights(
            strategy_results, allocation_method, custom_weights
        )
        
        # Apply rebalancing
        weights = self._apply_rebalancing(weights)
        
        # Calculate portfolio returns and metrics
        portfolio_returns = self._calculate_portfolio_returns(strategy_results, weights)
        portfolio_equity = self._calculate_equity_curve(portfolio_returns, initial_capital)
        portfolio_metrics = self._calculate_portfolio_metrics(portfolio_equity, portfolio_returns)
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlations(strategy_results)
        
        # Calculate strategy contributions
        contributions = self._calculate_contributions(strategy_results, weights, portfolio_returns)
        
        return PortfolioResult(
            strategy_results=strategy_results,
            portfolio_metrics=portfolio_metrics,
            weights=weights,
            equity_curve=portfolio_equity,
            returns=portfolio_returns,
            correlation_matrix=correlation_matrix,
            strategy_contributions=contributions
        )
    
    def _run_individual_backtests(self,
                                strategies: Dict[str, BaseStrategy],
                                data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                                initial_capital: float,
                                **kwargs) -> Dict[str, BacktestResult]:
        """Run backtest for each strategy"""
        results = {}
        
        # Determine data for each strategy
        if isinstance(data, pd.DataFrame):
            # Same data for all strategies
            strategy_data = {name: data for name in strategies.keys()}
        else:
            strategy_data = data
        
        # Run each strategy
        for name, strategy in strategies.items():
            self.logger.info(f"Backtesting strategy: {name}")
            
            # Get data for this strategy
            strat_data = strategy_data.get(name)
            if strat_data is None:
                self.logger.warning(f"No data for strategy {name}, skipping")
                continue
            
            # Run backtest
            result = self.engine.run_backtest(
                strategy=strategy,
                data=strat_data,
                initial_capital=initial_capital,
                **kwargs
            )
            
            results[name] = result
        
        return results
    
    def _calculate_weights(self,
                         strategy_results: Dict[str, BacktestResult],
                         method: AllocationMethod,
                         custom_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Calculate portfolio weights based on allocation method"""
        # Get union of all indices (not intersection) to ensure we have all timestamps
        all_indices = [result.equity_curve.index for result in strategy_results.values()]
        
        # Start with the first index
        combined_index = all_indices[0]
        
        # Union with all other indices
        for idx in all_indices[1:]:
            combined_index = combined_index.union(idx)
        
        # Sort the index
        common_index = combined_index.sort_values()
        
        if method == AllocationMethod.EQUAL_WEIGHT:
            return self._equal_weight_allocation(list(strategy_results.keys()), common_index)
        
        elif method == AllocationMethod.RISK_PARITY:
            return self._risk_parity_allocation(strategy_results, common_index)
        
        elif method == AllocationMethod.CUSTOM_WEIGHTS:
            if custom_weights is None:
                raise ValueError("custom_weights required for CUSTOM_WEIGHTS method")
            return self._custom_weight_allocation(custom_weights, common_index)
        
        elif method == AllocationMethod.INVERSE_VOLATILITY:
            return self._inverse_volatility_allocation(strategy_results, common_index)
        
        elif method == AllocationMethod.MEAN_VARIANCE:
            return self._mean_variance_allocation(strategy_results, common_index)
        
        else:
            raise ValueError(f"Unknown allocation method: {method}")
    
    def _equal_weight_allocation(self, 
                               strategy_names: List[str],
                               index: pd.DatetimeIndex) -> pd.DataFrame:
        """Equal weight allocation"""
        n_strategies = len(strategy_names)
        weight = 1.0 / n_strategies
        
        # Create weights array
        weights_array = np.full((len(index), n_strategies), weight)
        
        # Create DataFrame from array
        weights = pd.DataFrame(
            weights_array,
            index=index,
            columns=list(strategy_names)
        )
        
        return weights
    
    def _risk_parity_allocation(self,
                              strategy_results: Dict[str, BacktestResult],
                              index: pd.DatetimeIndex) -> pd.DataFrame:
        """Risk parity allocation - equal risk contribution"""
        # Calculate returns for each strategy
        returns_dict = {}
        for name, result in strategy_results.items():
            returns = result.equity_curve.pct_change().reindex(index).fillna(0)
            returns_dict[name] = returns
        
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate rolling volatilities
        window = 60  # 60-period rolling window
        volatilities = returns_df.rolling(window=window).std()
        
        # Inverse volatility weights (normalized)
        inv_vol = 1 / volatilities.replace(0, np.inf)
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        
        # Fill initial NaN values with equal weights
        weights = weights.fillna(1.0 / len(strategy_results))
        
        return weights
    
    def _custom_weight_allocation(self,
                                custom_weights: Dict[str, float],
                                index: pd.DatetimeIndex) -> pd.DataFrame:
        """Custom weight allocation"""
        # Normalize weights
        total_weight = sum(custom_weights.values())
        normalized_weights = {k: v/total_weight for k, v in custom_weights.items()}
        
        # Create DataFrame with proper structure
        weights = pd.DataFrame(
            index=index,
            columns=list(custom_weights.keys())
        )
        
        # Fill with normalized weights
        for strategy, weight in normalized_weights.items():
            weights[strategy] = weight
        
        return weights
    
    def _inverse_volatility_allocation(self,
                                     strategy_results: Dict[str, BacktestResult],
                                     index: pd.DatetimeIndex) -> pd.DataFrame:
        """Inverse volatility allocation"""
        # Similar to risk parity but simpler
        return self._risk_parity_allocation(strategy_results, index)
    
    def _mean_variance_allocation(self,
                                strategy_results: Dict[str, BacktestResult],
                                index: pd.DatetimeIndex) -> pd.DataFrame:
        """Mean-variance optimization (simplified)"""
        # Calculate returns
        returns_dict = {}
        for name, result in strategy_results.items():
            returns = result.equity_curve.pct_change().reindex(index).fillna(0)
            returns_dict[name] = returns
        
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate expected returns and covariance
        window = 252  # 1 year lookback
        weights_list = []
        
        for i in range(len(index)):
            if i < window:
                # Equal weights for initial period
                w = pd.Series(1.0 / len(strategy_results), index=returns_df.columns)
            else:
                # Get historical window
                hist_returns = returns_df.iloc[i-window:i]
                
                # Calculate mean returns and covariance
                mean_returns = hist_returns.mean()
                cov_matrix = hist_returns.cov()
                
                # Simple mean-variance optimization (maximize Sharpe)
                # This is a simplified version - real implementation would use optimization
                inv_cov = np.linalg.pinv(cov_matrix.values)
                raw_weights = inv_cov @ mean_returns.values
                
                # Normalize to sum to 1 and ensure non-negative
                raw_weights = np.maximum(raw_weights, 0)
                weight_sum = raw_weights.sum()
                if weight_sum > 0:
                    w = pd.Series(raw_weights / weight_sum, index=returns_df.columns)
                else:
                    # Fall back to equal weights if optimization fails
                    w = pd.Series(1.0 / len(returns_df.columns), index=returns_df.columns)
            
            weights_list.append(w)
        
        weights = pd.DataFrame(weights_list, index=index)
        return weights
    
    def _apply_rebalancing(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Apply rebalancing frequency to weights"""
        if self.rebalance_frequency == 'never':
            # Use initial weights throughout
            initial_weights = weights.iloc[0]
            return pd.DataFrame(
                np.tile(initial_weights.values, (len(weights), 1)),
                index=weights.index,
                columns=weights.columns
            )
        
        # Map frequency to pandas offset
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'ME',
            'quarterly': 'QE',
            'yearly': 'YE'
        }
        
        if self.rebalance_frequency not in freq_map:
            self.logger.warning(f"Unknown rebalance frequency: {self.rebalance_frequency}, using monthly")
            freq = 'ME'
        else:
            freq = freq_map[self.rebalance_frequency]
        
        # Group by rebalance period and forward fill
        # First resample to get end-of-period weights
        resampled = weights.resample(freq).last()
        
        # If resampling produced NaN (e.g., incomplete periods), use original weights
        if resampled.isna().all().all():
            return weights
        
        # Reindex back to original frequency and forward fill
        rebalanced = resampled.reindex(weights.index).ffill()
        
        # Fill any remaining NaN with original weights
        rebalanced = rebalanced.fillna(weights)
        
        return rebalanced
    
    def _calculate_portfolio_returns(self,
                                   strategy_results: Dict[str, BacktestResult],
                                   weights: pd.DataFrame) -> pd.Series:
        """Calculate weighted portfolio returns"""
        # Get returns for each strategy
        returns_dict = {}
        for name, result in strategy_results.items():
            returns = result.equity_curve.pct_change()
            returns_dict[name] = returns
        
        # Align all returns to common index
        returns_df = pd.DataFrame(returns_dict).reindex(weights.index).fillna(0)
        
        # Calculate weighted returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        return portfolio_returns
    
    def _calculate_equity_curve(self,
                              returns: pd.Series,
                              initial_capital: float) -> pd.Series:
        """Calculate equity curve from returns"""
        return initial_capital * (1 + returns).cumprod()
    
    def _calculate_portfolio_metrics(self,
                                   equity_curve: pd.Series,
                                   returns: pd.Series) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Sharpe ratio
        if returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std()
        else:
            sortino_ratio = 0
        
        # Max drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': abs(max_drawdown),
            'calmar_ratio': calmar_ratio,
            'volatility': returns.std() * np.sqrt(252)
        }
    
    def _calculate_correlations(self,
                              strategy_results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """Calculate correlation matrix between strategy returns"""
        returns_dict = {}
        for name, result in strategy_results.items():
            returns = result.equity_curve.pct_change().dropna()
            returns_dict[name] = returns
        
        returns_df = pd.DataFrame(returns_dict)
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def _calculate_contributions(self,
                               strategy_results: Dict[str, BacktestResult],
                               weights: pd.DataFrame,
                               portfolio_returns: pd.Series) -> pd.DataFrame:
        """Calculate strategy contributions to portfolio returns"""
        contributions = {}
        
        for name in strategy_results.keys():
            # Get strategy returns
            strat_returns = strategy_results[name].equity_curve.pct_change()
            strat_returns = strat_returns.reindex(weights.index).fillna(0)
            
            # Calculate contribution (weight * return)
            contribution = weights[name] * strat_returns
            contributions[name] = contribution
        
        contributions_df = pd.DataFrame(contributions)
        
        # Add summary statistics
        contributions_df['total'] = contributions_df.sum(axis=1)
        
        return contributions_df
    
    def analyze_portfolio(self, result: PortfolioResult) -> Dict[str, Any]:
        """
        Perform detailed portfolio analysis
        
        Args:
            result: Portfolio backtest result
            
        Returns:
            Dictionary with analysis metrics
        """
        analysis = {
            'performance': result.portfolio_metrics,
            'correlations': {
                'mean': result.correlation_matrix.mean().mean(),
                'max': result.correlation_matrix.values[
                    ~np.eye(result.correlation_matrix.shape[0], dtype=bool)
                ].max(),
                'min': result.correlation_matrix.values[
                    ~np.eye(result.correlation_matrix.shape[0], dtype=bool)
                ].min()
            },
            'diversification_ratio': self._calculate_diversification_ratio(result),
            'strategy_sharpe_ratios': {
                name: res.metrics['sharpe_ratio'] 
                for name, res in result.strategy_results.items()
            },
            'weight_statistics': {
                'mean': result.weights.mean().to_dict(),
                'std': result.weights.std().to_dict(),
                'min': result.weights.min().to_dict(),
                'max': result.weights.max().to_dict()
            }
        }
        
        return analysis
    
    def _calculate_diversification_ratio(self, result: PortfolioResult) -> float:
        """Calculate diversification ratio (weighted avg volatility / portfolio volatility)"""
        # Get individual volatilities
        individual_vols = {}
        for name, res in result.strategy_results.items():
            returns = res.equity_curve.pct_change().dropna()
            individual_vols[name] = returns.std() * np.sqrt(252)
        
        # Calculate weighted average volatility
        avg_weights = result.weights.mean()
        weighted_avg_vol = sum(
            avg_weights[name] * vol 
            for name, vol in individual_vols.items()
        )
        
        # Portfolio volatility
        portfolio_vol = result.returns.std() * np.sqrt(252)
        
        # Diversification ratio
        if portfolio_vol > 0:
            div_ratio = weighted_avg_vol / portfolio_vol
        else:
            div_ratio = 1.0
        
        return div_ratio