"""
Visualization Module
Comprehensive charting for strategy performance comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

from src.backtesting.engines.vectorbt_engine import BacktestResult
from src.utils.logging import get_logger


class StrategyVisualizer:
    """
    Comprehensive visualization tools for strategy comparison
    
    Features:
    - Multi-strategy equity curves
    - Performance metric heatmaps
    - Rolling performance comparison
    - Drawdown visualization
    - Return distributions
    - Correlation matrices
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.logger = get_logger('visualizer')
        
        # Set style with fallback
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default if style not found
            self.logger.warning(f"Style '{style}' not found, using default")
            if 'seaborn' in style:
                # Try modern seaborn style
                try:
                    plt.style.use('seaborn-v0_8')
                    self.style = 'seaborn-v0_8'
                except OSError:
                    pass
        
        try:
            sns.set_palette("husl")
        except:
            pass
    
    def plot_equity_curves(self,
                          results: Dict[str, BacktestResult],
                          benchmark: Optional[pd.Series] = None,
                          log_scale: bool = False,
                          show_drawdowns: bool = True,
                          interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot equity curves for multiple strategies
        
        Args:
            results: Dictionary of strategy_name -> BacktestResult
            benchmark: Optional benchmark equity curve
            log_scale: Use log scale for y-axis
            show_drawdowns: Show drawdown periods as shaded areas
            interactive: Use Plotly for interactive plot
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if interactive:
            return self._plot_equity_curves_plotly(results, benchmark, log_scale, show_drawdowns)
        else:
            return self._plot_equity_curves_matplotlib(results, benchmark, log_scale, show_drawdowns)
    
    def _plot_equity_curves_matplotlib(self,
                                     results: Dict[str, BacktestResult],
                                     benchmark: Optional[pd.Series],
                                     log_scale: bool,
                                     show_drawdowns: bool) -> plt.Figure:
        """Matplotlib version of equity curve plot"""
        fig = plt.figure(figsize=self.figsize)
        
        if show_drawdowns:
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            ax_equity = plt.subplot(gs[0])
            ax_drawdown = plt.subplot(gs[1], sharex=ax_equity)
        else:
            ax_equity = plt.gca()
        
        # Plot each strategy
        for name, result in results.items():
            equity = result.equity_curve
            ax_equity.plot(equity.index, equity.values, label=name, linewidth=2)
            
            if show_drawdowns:
                # Calculate drawdowns
                rolling_max = equity.expanding().max()
                drawdowns = (equity - rolling_max) / rolling_max * 100
                ax_drawdown.fill_between(drawdowns.index, 0, drawdowns.values, 
                                       alpha=0.3, label=f"{name} DD")
        
        # Plot benchmark if provided
        if benchmark is not None:
            ax_equity.plot(benchmark.index, benchmark.values, 
                         label='Benchmark', linestyle='--', color='black', linewidth=2)
        
        # Format equity axis
        ax_equity.set_title('Strategy Equity Curves', fontsize=14, fontweight='bold')
        ax_equity.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax_equity.legend(loc='best')
        ax_equity.grid(True, alpha=0.3)
        
        if log_scale:
            ax_equity.set_yscale('log')
        
        # Format drawdown axis
        if show_drawdowns:
            ax_drawdown.set_xlabel('Date', fontsize=12)
            ax_drawdown.set_ylabel('Drawdown (%)', fontsize=12)
            ax_drawdown.grid(True, alpha=0.3)
            ax_drawdown.set_title('Drawdowns', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def _plot_equity_curves_plotly(self,
                                 results: Dict[str, BacktestResult],
                                 benchmark: Optional[pd.Series],
                                 log_scale: bool,
                                 show_drawdowns: bool) -> go.Figure:
        """Plotly version of equity curve plot"""
        if show_drawdowns:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.1,
                               subplot_titles=('Equity Curves', 'Drawdowns'),
                               row_heights=[0.7, 0.3])
        else:
            fig = go.Figure()
        
        # Plot each strategy
        for name, result in results.items():
            equity = result.equity_curve
            
            # Equity curve
            fig.add_trace(
                go.Scatter(x=equity.index, y=equity.values, 
                          mode='lines', name=name, line=dict(width=2)),
                row=1 if show_drawdowns else None, 
                col=1 if show_drawdowns else None
            )
            
            if show_drawdowns:
                # Calculate drawdowns
                rolling_max = equity.expanding().max()
                drawdowns = (equity - rolling_max) / rolling_max * 100
                
                fig.add_trace(
                    go.Scatter(x=drawdowns.index, y=drawdowns.values,
                             fill='tozeroy', name=f"{name} DD",
                             line=dict(width=0.5)),
                    row=2, col=1
                )
        
        # Plot benchmark if provided
        if benchmark is not None:
            fig.add_trace(
                go.Scatter(x=benchmark.index, y=benchmark.values,
                          mode='lines', name='Benchmark',
                          line=dict(color='black', dash='dash', width=2)),
                row=1 if show_drawdowns else None,
                col=1 if show_drawdowns else None
            )
        
        # Update layout
        fig.update_layout(
            title='Strategy Performance Comparison',
            hovermode='x unified',
            height=600 if show_drawdowns else 400
        )
        
        # Update axes
        if show_drawdowns:
            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
        else:
            fig.update_yaxes(title_text="Portfolio Value ($)")
            fig.update_xaxes(title_text="Date")
        
        if log_scale:
            fig.update_yaxes(type="log", row=1 if show_drawdowns else None, col=1)
        
        return fig
    
    def plot_performance_heatmap(self,
                               metrics_df: pd.DataFrame,
                               metrics_to_show: Optional[List[str]] = None,
                               normalize: bool = True,
                               interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot heatmap of performance metrics across strategies
        
        Args:
            metrics_df: DataFrame with strategies as index and metrics as columns
            metrics_to_show: List of metrics to display (defaults to key metrics)
            normalize: Normalize metrics to 0-1 scale
            interactive: Use Plotly for interactive plot
            
        Returns:
            Figure object
        """
        if metrics_to_show is None:
            metrics_to_show = [
                'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                'total_return', 'max_drawdown', 'win_rate',
                'profit_factor', 'volatility'
            ]
        
        # Filter available metrics
        available_metrics = [m for m in metrics_to_show if m in metrics_df.columns]
        data = metrics_df[available_metrics].copy()
        
        # Normalize if requested
        if normalize:
            for col in data.columns:
                # Reverse normalize for metrics where lower is better
                if col in ['max_drawdown', 'volatility']:
                    data[col] = 1 - (data[col] - data[col].min()) / (data[col].max() - data[col].min())
                else:
                    data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        
        if interactive:
            return self._plot_heatmap_plotly(data, normalize)
        else:
            return self._plot_heatmap_matplotlib(data, normalize)
    
    def _plot_heatmap_matplotlib(self, data: pd.DataFrame, normalize: bool) -> plt.Figure:
        """Matplotlib version of performance heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(data.T, annot=True, fmt='.2f' if not normalize else '.3f',
                   cmap='RdYlGn', center=0.5 if normalize else None,
                   cbar_kws={'label': 'Normalized Score' if normalize else 'Value'},
                   ax=ax)
        
        ax.set_title('Strategy Performance Metrics Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('Metric', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def _plot_heatmap_plotly(self, data: pd.DataFrame, normalize: bool) -> go.Figure:
        """Plotly version of performance heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=data.T.values,
            x=data.index,
            y=data.columns,
            colorscale='RdYlGn',
            text=data.T.round(3 if normalize else 2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title='Normalized Score' if normalize else 'Value')
        ))
        
        fig.update_layout(
            title='Strategy Performance Metrics Heatmap',
            xaxis_title='Strategy',
            yaxis_title='Metric',
            height=600
        )
        
        return fig
    
    def plot_rolling_performance(self,
                               results: Dict[str, BacktestResult],
                               window: int = 252,
                               metrics: List[str] = None,
                               interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot rolling performance metrics
        
        Args:
            results: Strategy results
            window: Rolling window size (default 252 days = 1 year)
            metrics: Metrics to plot (defaults to ['sharpe', 'returns'])
            interactive: Use Plotly for interactive plot
            
        Returns:
            Figure object
        """
        if metrics is None:
            metrics = ['sharpe', 'returns']
        
        if interactive:
            return self._plot_rolling_plotly(results, window, metrics)
        else:
            return self._plot_rolling_matplotlib(results, window, metrics)
    
    def _plot_rolling_matplotlib(self,
                               results: Dict[str, BacktestResult],
                               window: int,
                               metrics: List[str]) -> plt.Figure:
        """Matplotlib version of rolling performance plot"""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), sharex=True)
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            for name, result in results.items():
                returns = result.equity_curve.pct_change().dropna()
                
                if metric == 'sharpe':
                    rolling_metric = returns.rolling(window).apply(
                        lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
                    )
                    ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
                    ax.set_title(f'{window}-Day Rolling Sharpe Ratio', fontsize=12)
                    
                elif metric == 'returns':
                    rolling_metric = returns.rolling(window).mean() * 252
                    ax.set_ylabel('Annualized Return', fontsize=12)
                    ax.set_title(f'{window}-Day Rolling Annualized Returns', fontsize=12)
                    
                elif metric == 'volatility':
                    rolling_metric = returns.rolling(window).std() * np.sqrt(252)
                    ax.set_ylabel('Annualized Volatility', fontsize=12)
                    ax.set_title(f'{window}-Day Rolling Volatility', fontsize=12)
                
                ax.plot(rolling_metric.index, rolling_metric.values, label=name, linewidth=2)
            
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def _plot_rolling_plotly(self,
                           results: Dict[str, BacktestResult],
                           window: int,
                           metrics: List[str]) -> go.Figure:
        """Plotly version of rolling performance plot"""
        n_metrics = len(metrics)
        
        subplot_titles = []
        for metric in metrics:
            if metric == 'sharpe':
                subplot_titles.append(f'{window}-Day Rolling Sharpe Ratio')
            elif metric == 'returns':
                subplot_titles.append(f'{window}-Day Rolling Returns')
            elif metric == 'volatility':
                subplot_titles.append(f'{window}-Day Rolling Volatility')
        
        fig = make_subplots(rows=n_metrics, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1, subplot_titles=subplot_titles)
        
        for idx, metric in enumerate(metrics):
            for name, result in results.items():
                returns = result.equity_curve.pct_change().dropna()
                
                if metric == 'sharpe':
                    rolling_metric = returns.rolling(window).apply(
                        lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
                    )
                elif metric == 'returns':
                    rolling_metric = returns.rolling(window).mean() * 252
                elif metric == 'volatility':
                    rolling_metric = returns.rolling(window).std() * np.sqrt(252)
                
                fig.add_trace(
                    go.Scatter(x=rolling_metric.index, y=rolling_metric.values,
                             mode='lines', name=f"{name} ({metric})",
                             showlegend=(idx == 0)),
                    row=idx + 1, col=1
                )
        
        fig.update_layout(height=300 * n_metrics, hovermode='x unified')
        fig.update_xaxes(title_text="Date", row=n_metrics, col=1)
        
        return fig
    
    def plot_return_distributions(self,
                                results: Dict[str, BacktestResult],
                                bins: int = 50,
                                show_stats: bool = True,
                                interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot return distributions for strategies
        
        Args:
            results: Strategy results
            bins: Number of histogram bins
            show_stats: Show distribution statistics
            interactive: Use Plotly for interactive plot
            
        Returns:
            Figure object
        """
        if interactive:
            return self._plot_distributions_plotly(results, bins, show_stats)
        else:
            return self._plot_distributions_matplotlib(results, bins, show_stats)
    
    def _plot_distributions_matplotlib(self,
                                     results: Dict[str, BacktestResult],
                                     bins: int,
                                     show_stats: bool) -> plt.Figure:
        """Matplotlib version of return distributions"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        all_returns = []
        
        for name, result in results.items():
            returns = result.equity_curve.pct_change().dropna() * 100  # Convert to percentage
            all_returns.extend(returns.values)
            
            # Plot histogram
            ax.hist(returns, bins=bins, alpha=0.5, label=name, density=True)
            
            if show_stats:
                # Add vertical lines for mean
                mean_return = returns.mean()
                ax.axvline(mean_return, linestyle='--', linewidth=2, 
                         label=f'{name} Mean: {mean_return:.2f}%')
        
        # Overlay normal distribution
        all_returns = np.array(all_returns)
        mu, sigma = np.mean(all_returns), np.std(all_returns)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'k-', linewidth=2, 
               label='Normal Distribution')
        
        ax.set_xlabel('Daily Returns (%)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Return Distributions', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        if show_stats:
            stats_text = []
            for name, result in results.items():
                returns = result.equity_curve.pct_change().dropna()
                stats_text.append(f"{name}:")
                stats_text.append(f"  Skew: {stats.skew(returns):.2f}")
                stats_text.append(f"  Kurt: {stats.kurtosis(returns):.2f}")
            
            textstr = '\n'.join(stats_text)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
    
    def _plot_distributions_plotly(self,
                                 results: Dict[str, BacktestResult],
                                 bins: int,
                                 show_stats: bool) -> go.Figure:
        """Plotly version of return distributions"""
        fig = go.Figure()
        
        for name, result in results.items():
            returns = result.equity_curve.pct_change().dropna() * 100  # Convert to percentage
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=returns,
                name=name,
                opacity=0.5,
                nbinsx=bins,
                histnorm='probability density'
            ))
            
            if show_stats:
                # Add vertical line for mean
                mean_return = returns.mean()
                fig.add_vline(x=mean_return, line_dash="dash",
                            annotation_text=f"{name} Mean: {mean_return:.2f}%")
        
        fig.update_layout(
            title='Return Distributions',
            xaxis_title='Daily Returns (%)',
            yaxis_title='Density',
            barmode='overlay',
            height=500
        )
        
        return fig
    
    def plot_correlation_matrix(self,
                              correlation_df: pd.DataFrame,
                              interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot correlation matrix between strategies
        
        Args:
            correlation_df: Correlation matrix DataFrame
            interactive: Use Plotly for interactive plot
            
        Returns:
            Figure object
        """
        if interactive:
            return self._plot_correlation_plotly(correlation_df)
        else:
            return self._plot_correlation_matplotlib(correlation_df)
    
    def _plot_correlation_matplotlib(self, correlation_df: pd.DataFrame) -> plt.Figure:
        """Matplotlib version of correlation matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_df, dtype=bool))
        
        # Create heatmap
        sns.heatmap(correlation_df, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Strategy Return Correlations', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_correlation_plotly(self, correlation_df: pd.DataFrame) -> go.Figure:
        """Plotly version of correlation matrix"""
        # Create mask for lower triangle
        mask = np.tril(np.ones_like(correlation_df, dtype=bool), k=-1)
        correlation_masked = correlation_df.where(~mask)
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_masked.values,
            x=correlation_df.columns,
            y=correlation_df.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_masked.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title='Strategy Return Correlations',
            height=600,
            width=700
        )
        
        return fig
    
    def create_performance_dashboard(self,
                                   results: Dict[str, BacktestResult],
                                   metrics_df: pd.DataFrame,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive performance dashboard
        
        Args:
            results: Strategy results
            metrics_df: Performance metrics DataFrame
            save_path: Optional path to save figure
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[2, 1.5, 1.5])
        
        # 1. Equity curves (top, spanning 2 columns)
        ax_equity = plt.subplot(gs[0, :2])
        for name, result in results.items():
            ax_equity.plot(result.equity_curve.index, result.equity_curve.values, 
                         label=name, linewidth=2)
        ax_equity.set_title('Equity Curves', fontsize=12, fontweight='bold')
        ax_equity.set_ylabel('Portfolio Value ($)')
        ax_equity.legend(loc='best')
        ax_equity.grid(True, alpha=0.3)
        
        # 2. Drawdowns (top right)
        ax_dd = plt.subplot(gs[0, 2])
        for name, result in results.items():
            equity = result.equity_curve
            rolling_max = equity.expanding().max()
            drawdowns = (equity - rolling_max) / rolling_max * 100
            ax_dd.plot(drawdowns.index, drawdowns.values, label=name, linewidth=1.5)
        ax_dd.set_title('Drawdowns', fontsize=12, fontweight='bold')
        ax_dd.set_ylabel('Drawdown (%)')
        ax_dd.grid(True, alpha=0.3)
        
        # 3. Performance metrics table (middle left)
        ax_table = plt.subplot(gs[1, 0])
        ax_table.axis('tight')
        ax_table.axis('off')
        
        # Select key metrics for table
        table_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'volatility']
        table_data = metrics_df[table_metrics].round(3)
        
        table = ax_table.table(cellText=table_data.values,
                             rowLabels=table_data.index,
                             colLabels=table_data.columns,
                             cellLoc='center',
                             loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax_table.set_title('Key Metrics', fontsize=12, fontweight='bold', pad=20)
        
        # 4. Return distributions (middle center)
        ax_dist = plt.subplot(gs[1, 1])
        for name, result in results.items():
            returns = result.equity_curve.pct_change().dropna() * 100
            ax_dist.hist(returns, bins=30, alpha=0.5, label=name, density=True)
        ax_dist.set_xlabel('Daily Returns (%)')
        ax_dist.set_ylabel('Density')
        ax_dist.set_title('Return Distributions', fontsize=12, fontweight='bold')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe (middle right)
        ax_sharpe = plt.subplot(gs[1, 2])
        window = 252
        for name, result in results.items():
            returns = result.equity_curve.pct_change().dropna()
            rolling_sharpe = returns.rolling(window).apply(
                lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
            )
            ax_sharpe.plot(rolling_sharpe.index, rolling_sharpe.values, 
                         label=name, linewidth=1.5)
        ax_sharpe.set_title(f'{window}-Day Rolling Sharpe', fontsize=12, fontweight='bold')
        ax_sharpe.set_ylabel('Sharpe Ratio')
        ax_sharpe.grid(True, alpha=0.3)
        
        # 6. Correlation heatmap (bottom, spanning all columns)
        ax_corr = plt.subplot(gs[2, :])
        returns_df = pd.DataFrame({
            name: result.equity_curve.pct_change().dropna()
            for name, result in results.items()
        })
        correlation = returns_df.corr()
        
        im = ax_corr.imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax_corr.set_xticks(range(len(correlation.columns)))
        ax_corr.set_yticks(range(len(correlation.index)))
        ax_corr.set_xticklabels(correlation.columns, rotation=45)
        ax_corr.set_yticklabels(correlation.index)
        
        # Add correlation values
        for i in range(len(correlation.index)):
            for j in range(len(correlation.columns)):
                text = ax_corr.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        ax_corr.set_title('Strategy Correlations', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax_corr, label='Correlation')
        
        plt.suptitle('Strategy Performance Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Dashboard saved to {save_path}")
        
        return fig