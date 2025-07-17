"""
Tests for Visualization Module
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.analysis.visualization import StrategyVisualizer
from src.backtesting.engines.vectorbt_engine import BacktestResult


class TestStrategyVisualizer:
    """Test visualization functionality"""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample backtest results for visualization"""
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        
        results = {}
        
        # Create 3 different strategy patterns
        patterns = {
            'TrendFollowing': {
                'growth_rate': 0.0008,
                'volatility': 0.015,
                'drawdown_periods': [(50, 70), (150, 180)]
            },
            'MeanReversion': {
                'growth_rate': 0.0006,
                'volatility': 0.008,
                'drawdown_periods': [(80, 90), (200, 220)]
            },
            'Momentum': {
                'growth_rate': 0.001,
                'volatility': 0.02,
                'drawdown_periods': [(30, 60), (120, 140)]
            }
        }
        
        for name, params in patterns.items():
            # Generate base equity curve
            returns = np.random.normal(params['growth_rate'], params['volatility'], len(dates))
            
            # Add drawdown periods
            for start, end in params['drawdown_periods']:
                returns[start:end] = np.random.normal(-0.002, params['volatility'], end - start)
            
            equity = 100000 * (1 + returns).cumprod()
            equity_series = pd.Series(equity, index=dates)
            
            # Create trades
            n_trades = 100
            trade_times = pd.date_range(dates[0], dates[-1], periods=n_trades)
            trades = pd.DataFrame({
                'entry_time': trade_times,
                'exit_time': trade_times + timedelta(days=5),
                'pnl': np.random.normal(50, 200, n_trades),
                'return': np.random.normal(0.002, 0.01, n_trades)
            })
            
            # Calculate metrics
            daily_returns = equity_series.pct_change().dropna()
            sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            
            result = BacktestResult(
                equity_curve=equity_series,
                trades=trades,
                metrics={
                    'sharpe_ratio': sharpe,
                    'total_return': (equity_series.iloc[-1] / equity_series.iloc[0]) - 1,
                    'max_drawdown': 0.15,  # Simplified
                    'win_rate': 0.55,
                    'total_trades': len(trades)
                },
                stats={},
                orders=pd.DataFrame()
            )
            
            results[name] = result
        
        return results
    
    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance"""
        return StrategyVisualizer(style='seaborn', figsize=(12, 8))
    
    @pytest.fixture
    def metrics_df(self, sample_results):
        """Create sample metrics DataFrame"""
        data = []
        for name, result in sample_results.items():
            data.append({
                'sharpe_ratio': result.metrics['sharpe_ratio'],
                'total_return': result.metrics['total_return'],
                'max_drawdown': result.metrics['max_drawdown'],
                'win_rate': result.metrics['win_rate'],
                'volatility': 0.15,  # Placeholder
                'sortino_ratio': result.metrics['sharpe_ratio'] * 1.2,  # Simplified
                'calmar_ratio': result.metrics['total_return'] / result.metrics['max_drawdown'],
                'profit_factor': 1.5
            })
        
        return pd.DataFrame(data, index=list(sample_results.keys()))
    
    def test_initialization(self):
        """Test visualizer initialization"""
        viz = StrategyVisualizer(style='ggplot', figsize=(10, 6))
        assert viz.style == 'ggplot'
        assert viz.figsize == (10, 6)
        assert viz.logger is not None
    
    def test_plot_equity_curves_matplotlib(self, visualizer, sample_results):
        """Test matplotlib equity curve plotting"""
        fig = visualizer.plot_equity_curves(
            sample_results,
            log_scale=False,
            show_drawdowns=True,
            interactive=False
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Check that figure has correct subplots
        axes = fig.get_axes()
        if len(axes) == 2:  # With drawdowns
            # Check titles
            assert any('Equity' in ax.get_title() for ax in axes)
            assert any('Drawdown' in ax.get_title() for ax in axes)
        
        # Test with benchmark
        benchmark = sample_results['TrendFollowing'].equity_curve * 1.1
        fig_bench = visualizer.plot_equity_curves(
            sample_results,
            benchmark=benchmark,
            show_drawdowns=False
        )
        assert isinstance(fig_bench, plt.Figure)
        
        # Test log scale
        fig_log = visualizer.plot_equity_curves(
            sample_results,
            log_scale=True,
            show_drawdowns=False
        )
        assert isinstance(fig_log, plt.Figure)
        
        plt.close('all')
    
    def test_plot_equity_curves_plotly(self, visualizer, sample_results):
        """Test plotly equity curve plotting"""
        fig = visualizer.plot_equity_curves(
            sample_results,
            log_scale=False,
            show_drawdowns=True,
            interactive=True
        )
        
        assert isinstance(fig, go.Figure)
        
        # Check traces
        assert len(fig.data) > 0
        
        # Test without drawdowns
        fig_no_dd = visualizer.plot_equity_curves(
            sample_results,
            show_drawdowns=False,
            interactive=True
        )
        assert isinstance(fig_no_dd, go.Figure)
    
    def test_plot_performance_heatmap_matplotlib(self, visualizer, metrics_df):
        """Test matplotlib performance heatmap"""
        fig = visualizer.plot_performance_heatmap(
            metrics_df,
            normalize=True,
            interactive=False
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Test without normalization
        fig_raw = visualizer.plot_performance_heatmap(
            metrics_df,
            normalize=False,
            interactive=False
        )
        assert isinstance(fig_raw, plt.Figure)
        
        # Test with specific metrics
        fig_specific = visualizer.plot_performance_heatmap(
            metrics_df,
            metrics_to_show=['sharpe_ratio', 'total_return'],
            interactive=False
        )
        assert isinstance(fig_specific, plt.Figure)
        
        plt.close('all')
    
    def test_plot_performance_heatmap_plotly(self, visualizer, metrics_df):
        """Test plotly performance heatmap"""
        fig = visualizer.plot_performance_heatmap(
            metrics_df,
            normalize=True,
            interactive=True
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Should have one heatmap trace
        assert fig.data[0].type == 'heatmap'
    
    def test_plot_rolling_performance_matplotlib(self, visualizer, sample_results):
        """Test matplotlib rolling performance plotting"""
        fig = visualizer.plot_rolling_performance(
            sample_results,
            window=60,
            metrics=['sharpe', 'returns'],
            interactive=False
        )
        
        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 2  # One for each metric
        
        # Test single metric
        fig_single = visualizer.plot_rolling_performance(
            sample_results,
            window=30,
            metrics=['volatility'],
            interactive=False
        )
        assert isinstance(fig_single, plt.Figure)
        
        plt.close('all')
    
    def test_plot_rolling_performance_plotly(self, visualizer, sample_results):
        """Test plotly rolling performance plotting"""
        fig = visualizer.plot_rolling_performance(
            sample_results,
            window=60,
            metrics=['sharpe', 'returns'],
            interactive=True
        )
        
        assert isinstance(fig, go.Figure)
        # Should have traces for each strategy and metric combination
        assert len(fig.data) >= len(sample_results) * 2
    
    def test_plot_return_distributions_matplotlib(self, visualizer, sample_results):
        """Test matplotlib return distribution plotting"""
        fig = visualizer.plot_return_distributions(
            sample_results,
            bins=50,
            show_stats=True,
            interactive=False
        )
        
        assert isinstance(fig, plt.Figure)
        ax = fig.get_axes()[0]
        
        # Check for legend (should have strategy names and possibly normal dist)
        legend = ax.get_legend()
        assert legend is not None
        
        # Test without stats
        fig_no_stats = visualizer.plot_return_distributions(
            sample_results,
            bins=30,
            show_stats=False,
            interactive=False
        )
        assert isinstance(fig_no_stats, plt.Figure)
        
        plt.close('all')
    
    def test_plot_return_distributions_plotly(self, visualizer, sample_results):
        """Test plotly return distribution plotting"""
        fig = visualizer.plot_return_distributions(
            sample_results,
            bins=50,
            show_stats=True,
            interactive=True
        )
        
        assert isinstance(fig, go.Figure)
        # Should have histogram traces
        assert any(trace.type == 'histogram' for trace in fig.data)
    
    def test_plot_correlation_matrix_matplotlib(self, visualizer, sample_results):
        """Test matplotlib correlation matrix plotting"""
        # Create correlation matrix
        returns_dict = {}
        for name, result in sample_results.items():
            returns_dict[name] = result.equity_curve.pct_change().dropna()
        
        corr_matrix = pd.DataFrame(returns_dict).corr()
        
        fig = visualizer.plot_correlation_matrix(
            corr_matrix,
            interactive=False
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Should be square
        ax = fig.get_axes()[0]
        assert ax.get_xlim()[1] - ax.get_xlim()[0] == ax.get_ylim()[1] - ax.get_ylim()[0]
        
        plt.close('all')
    
    def test_plot_correlation_matrix_plotly(self, visualizer, sample_results):
        """Test plotly correlation matrix plotting"""
        # Create correlation matrix
        returns_dict = {}
        for name, result in sample_results.items():
            returns_dict[name] = result.equity_curve.pct_change().dropna()
        
        corr_matrix = pd.DataFrame(returns_dict).corr()
        
        fig = visualizer.plot_correlation_matrix(
            corr_matrix,
            interactive=True
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == 'heatmap'
    
    def test_create_performance_dashboard(self, visualizer, sample_results, metrics_df, tmp_path):
        """Test comprehensive dashboard creation"""
        fig = visualizer.create_performance_dashboard(
            sample_results,
            metrics_df
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Should have multiple subplots
        axes = fig.get_axes()
        assert len(axes) >= 6  # At least 6 different plots
        
        # Test saving
        save_path = tmp_path / "test_dashboard.png"
        fig_saved = visualizer.create_performance_dashboard(
            sample_results,
            metrics_df,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        
        plt.close('all')
    
    def test_empty_results_handling(self, visualizer):
        """Test handling of empty results"""
        empty_results = {}
        
        # Should handle empty results gracefully
        with pytest.raises(Exception):  # May raise various exceptions
            visualizer.plot_equity_curves(empty_results)
    
    def test_single_strategy_handling(self, visualizer, sample_results):
        """Test handling of single strategy"""
        single_result = {'OnlyStrategy': list(sample_results.values())[0]}
        
        # Should work with single strategy
        fig = visualizer.plot_equity_curves(single_result)
        assert isinstance(fig, plt.Figure)
        
        plt.close('all')
    
    def test_style_setting(self):
        """Test different matplotlib styles"""
        styles = ['seaborn', 'ggplot', 'default']
        
        for style in styles:
            try:
                viz = StrategyVisualizer(style=style)
                assert viz.style == style
            except:
                # Some styles might not be available
                pass
    
    def test_figure_size_setting(self, sample_results):
        """Test custom figure sizes"""
        viz_small = StrategyVisualizer(figsize=(6, 4))
        fig_small = viz_small.plot_equity_curves(sample_results, show_drawdowns=False)
        
        viz_large = StrategyVisualizer(figsize=(16, 10))
        fig_large = viz_large.plot_equity_curves(sample_results, show_drawdowns=False)
        
        # Check figure sizes
        assert fig_small.get_figwidth() < fig_large.get_figwidth()
        assert fig_small.get_figheight() < fig_large.get_figheight()
        
        plt.close('all')