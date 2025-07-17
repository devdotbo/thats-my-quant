"""
Tests for Reporting Module  
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re

from src.analysis.reporting import PerformanceReporter
from src.analysis.performance_analyzer import PerformanceAnalyzer, ComparisonResult
from src.backtesting.engines.vectorbt_engine import BacktestResult


class TestPerformanceReporter:
    """Test reporting functionality"""
    
    @pytest.fixture
    def sample_comparison_result(self):
        """Create sample comparison result for testing"""
        # Create sample data
        strategies = ['StrategyA', 'StrategyB', 'StrategyC']
        
        # Strategy metrics
        metrics_data = {
            'sharpe_ratio': [1.5, 1.2, 0.8],
            'total_return': [0.25, 0.20, 0.15],
            'max_drawdown': [0.10, 0.15, 0.20],
            'volatility': [0.12, 0.15, 0.18],
            'sortino_ratio': [2.0, 1.5, 1.0],
            'calmar_ratio': [2.5, 1.33, 0.75],
            'win_rate': [0.60, 0.55, 0.50],
            'profit_factor': [1.8, 1.5, 1.2]
        }
        strategy_metrics = pd.DataFrame(metrics_data, index=strategies)
        
        # Rankings
        rankings_data = []
        for i, strategy in enumerate(strategies):
            rankings_data.append({
                'strategy': strategy,
                'sharpe_ratio': metrics_data['sharpe_ratio'][i],
                'sharpe_ratio_rank': i + 1,
                'total_return': metrics_data['total_return'][i],
                'total_return_rank': i + 1,
                'composite_rank': i + 1
            })
        rankings = pd.DataFrame(rankings_data)
        
        # Statistical tests
        statistical_tests = {
            'sharpe_difference': pd.DataFrame([
                {
                    'strategy_1': 'StrategyA',
                    'strategy_2': 'StrategyB',
                    'sharpe_1': 1.5,
                    'sharpe_2': 1.2,
                    'difference': 0.3,
                    't_statistic': 2.5,
                    'p_value': 0.02,
                    'significant_5%': True
                },
                {
                    'strategy_1': 'StrategyA',
                    'strategy_2': 'StrategyC',
                    'sharpe_1': 1.5,
                    'sharpe_2': 0.8,
                    'difference': 0.7,
                    't_statistic': 4.2,
                    'p_value': 0.001,
                    'significant_5%': True
                },
                {
                    'strategy_1': 'StrategyB',
                    'strategy_2': 'StrategyC',
                    'sharpe_1': 1.2,
                    'sharpe_2': 0.8,
                    'difference': 0.4,
                    't_statistic': 1.8,
                    'p_value': 0.08,
                    'significant_5%': False
                }
            ])
        }
        
        # Relative performance
        relative_performance = pd.DataFrame([
            {
                'strategy': 'StrategyB',
                'benchmark': 'StrategyA',
                'alpha': -0.02,
                'beta': 1.1,
                'information_ratio': -0.5,
                'tracking_error': 0.04,
                'total_outperformance': -0.05,
                'correlation': 0.85
            },
            {
                'strategy': 'StrategyC',
                'benchmark': 'StrategyA',
                'alpha': -0.08,
                'beta': 1.3,
                'information_ratio': -1.2,
                'tracking_error': 0.06,
                'total_outperformance': -0.10,
                'correlation': 0.75
            }
        ])
        
        # Correlation matrix
        correlation_matrix = pd.DataFrame(
            [[1.0, 0.85, 0.75],
             [0.85, 1.0, 0.80],
             [0.75, 0.80, 1.0]],
            index=strategies,
            columns=strategies
        )
        
        # Summary stats
        summary_stats = {
            'n_strategies': 3,
            'best_sharpe': 1.5,
            'best_sharpe_strategy': 'StrategyA',
            'best_return': 0.25,
            'best_return_strategy': 'StrategyA',
            'avg_sharpe': 1.17,
            'avg_return': 0.20,
            'avg_max_drawdown': 0.15,
            'sharpe_difference_significant_pairs': 2,
            'sharpe_difference_total_comparisons': 3
        }
        
        return ComparisonResult(
            strategy_metrics=strategy_metrics,
            rankings=rankings,
            statistical_tests=statistical_tests,
            relative_performance=relative_performance,
            correlation_matrix=correlation_matrix,
            summary_stats=summary_stats
        )
    
    @pytest.fixture
    def sample_backtest_results(self):
        """Create sample backtest results"""
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        
        results = {}
        for i, name in enumerate(['StrategyA', 'StrategyB', 'StrategyC']):
            # Create equity curve
            growth_rate = 0.001 - i * 0.0002
            returns = np.random.normal(growth_rate, 0.01 + i * 0.003, 252)
            equity = 100000 * (1 + returns).cumprod()
            equity_series = pd.Series(equity, index=dates)
            
            # Create trades
            n_trades = 100 - i * 10
            trades = pd.DataFrame({
                'entry_time': pd.date_range(dates[0], dates[-1], periods=n_trades),
                'exit_time': pd.date_range(dates[0], dates[-1], periods=n_trades),
                'pnl': np.random.normal(50 - i * 10, 100, n_trades),
                'return': np.random.normal(0.002 - i * 0.0005, 0.01, n_trades)
            })
            
            metrics = {
                'sharpe_ratio': 1.5 - i * 0.35,
                'total_return': 0.25 - i * 0.05,
                'max_drawdown': 0.10 + i * 0.05,
                'win_rate': 0.60 - i * 0.05,
                'total_trades': n_trades,
                'max_consecutive_wins': 8 - i,
                'max_consecutive_losses': 4 + i
            }
            
            results[name] = BacktestResult(
                equity_curve=equity_series,
                trades=trades,
                metrics=metrics,
                stats={},
                orders=pd.DataFrame()
            )
        
        return results
    
    @pytest.fixture
    def reporter(self):
        """Create reporter instance"""
        return PerformanceReporter(
            company_name="Test Trading Co",
            report_style="professional"
        )
    
    def test_initialization(self):
        """Test reporter initialization"""
        reporter = PerformanceReporter(
            company_name="My Company",
            report_style="minimal"
        )
        assert reporter.company_name == "My Company"
        assert reporter.report_style == "minimal"
        assert reporter.logger is not None
        assert reporter.visualizer is not None
    
    def test_generate_report_basic(self, reporter, sample_comparison_result, 
                                 sample_backtest_results):
        """Test basic report generation"""
        html = reporter.generate_report(
            sample_comparison_result,
            sample_backtest_results,
            report_title="Test Strategy Analysis"
        )
        
        assert isinstance(html, str)
        assert len(html) > 1000  # Should be substantial
        
        # Check basic structure
        assert '<!DOCTYPE html>' in html
        assert '<html>' in html
        assert '</html>' in html
        assert 'Test Strategy Analysis' in html
        assert 'Test Trading Co' in html
    
    def test_generate_report_sections(self, reporter, sample_comparison_result,
                                    sample_backtest_results):
        """Test individual report sections"""
        # Test with specific sections
        sections = ['executive_summary', 'performance_metrics', 'statistical_tests']
        
        html = reporter.generate_report(
            sample_comparison_result,
            sample_backtest_results,
            sections=sections
        )
        
        # Check sections are included
        assert 'Executive Summary' in html
        assert 'Performance Metrics' in html
        assert 'Statistical Analysis' in html
        
        # Check sections not included
        assert 'Risk Analysis' not in html
        assert 'Trade Analysis' not in html
    
    def test_executive_summary_section(self, reporter, sample_comparison_result):
        """Test executive summary generation"""
        html = reporter._generate_executive_summary(sample_comparison_result)
        
        assert 'Executive Summary' in html
        assert 'StrategyA' in html  # Best strategy
        assert '1.5' in html  # Best Sharpe
        assert '25' in html  # Best return (as percentage)
        
        # Check metric boxes
        assert 'metric-box' in html
        assert 'Strategies Analyzed' in html
        assert '3' in html  # Number of strategies
    
    def test_metrics_section(self, reporter, sample_comparison_result):
        """Test metrics section generation"""
        html = reporter._generate_metrics_section(sample_comparison_result)
        
        assert 'Performance Metrics' in html
        assert 'Strategy Rankings' in html
        assert 'Detailed Performance Metrics' in html
        
        # Check table structure
        assert '<table>' in html
        assert 'Sharpe Ratio' in html
        assert 'Total Return' in html
        
        # Check all strategies included
        for strategy in ['StrategyA', 'StrategyB', 'StrategyC']:
            assert strategy in html
    
    def test_statistical_tests_section(self, reporter, sample_comparison_result):
        """Test statistical tests section generation"""
        html = reporter._generate_statistical_tests_section(sample_comparison_result)
        
        assert 'Statistical Analysis' in html
        assert 'Sharpe Ratio Difference Tests' in html
        
        # Check test results
        assert 'P-Value' in html
        assert 'Significant' in html
        assert '0.02' in html  # P-value from test data
        assert 'Yes' in html  # Significant result
        
        # Check summary boxes
        assert 'Statistical Test Summary' in html
        assert '2/3' in html  # Significant pairs
    
    def test_risk_analysis_section(self, reporter, sample_comparison_result,
                                  sample_backtest_results):
        """Test risk analysis section generation"""
        html = reporter._generate_risk_analysis_section(
            sample_comparison_result,
            sample_backtest_results
        )
        
        assert 'Risk Analysis' in html
        assert 'Maximum Drawdowns' in html
        assert 'Risk Metrics Comparison' in html
        
        # Check metrics table
        assert 'Volatility' in html
        assert 'Sortino Ratio' in html
        assert 'Calmar Ratio' in html
        assert 'Skewness' in html
        assert 'Kurtosis' in html
        
        # Should have chart embedded
        assert 'img src="data:image/png;base64,' in html
    
    def test_correlation_section(self, reporter, sample_comparison_result):
        """Test correlation section generation"""
        html = reporter._generate_correlation_section(sample_comparison_result)
        
        assert 'Correlation Analysis' in html
        assert 'correlation matrix' in html
        assert 'diversification benefits' in html
        
        # Should have correlation heatmap
        assert 'img src="data:image/png;base64,' in html
    
    def test_trade_analysis_section(self, reporter, sample_backtest_results):
        """Test trade analysis section generation"""
        html = reporter._generate_trade_analysis_section(sample_backtest_results)
        
        assert 'Trade Analysis' in html
        assert 'Total Trades' in html
        assert 'Win Rate' in html
        assert 'Profit Factor' in html
        assert 'Max Consec. Wins' in html
        
        # Check all strategies included
        for strategy in sample_backtest_results.keys():
            assert strategy in html
    
    def test_recommendations_section(self, reporter, sample_comparison_result):
        """Test recommendations section generation"""
        html = reporter._generate_recommendations_section(sample_comparison_result)
        
        assert 'Recommendations' in html
        assert 'Best Overall Strategy' in html
        assert 'StrategyA' in html  # Best strategy
        assert 'Risk-Averse Investors' in html
        
        # Check risk warnings
        assert 'Important Considerations' in html
        assert 'Past performance' in html
        assert 'overfitting' in html
    
    def test_css_styles(self, reporter):
        """Test CSS style generation"""
        # Test professional style
        css = reporter._get_css_styles()
        assert 'font-family' in css
        assert 'background-color' in css
        assert 'metric-box' in css
        
        # Test minimal style
        reporter.report_style = 'minimal'
        css_minimal = reporter._get_css_styles()
        assert len(css_minimal) < len(css)
    
    def test_save_report(self, reporter, sample_comparison_result,
                       sample_backtest_results, tmp_path):
        """Test saving report to file"""
        output_path = tmp_path / "test_report.html"
        
        html = reporter.generate_report(
            sample_comparison_result,
            sample_backtest_results,
            output_path=output_path
        )
        
        assert output_path.exists()
        
        # Check file contents
        with open(output_path, 'r') as f:
            saved_html = f.read()
        
        assert saved_html == html
        assert len(saved_html) > 1000
    
    def test_generate_summary_json(self, reporter, sample_comparison_result, tmp_path):
        """Test JSON summary generation"""
        # Test without saving
        summary = reporter.generate_summary_json(sample_comparison_result)
        
        assert isinstance(summary, dict)
        assert 'generated_at' in summary
        assert 'summary_stats' in summary
        assert 'rankings' in summary
        assert 'strategy_metrics' in summary
        assert 'statistical_tests' in summary
        assert 'correlations' in summary
        
        # Test with saving
        output_path = tmp_path / "summary.json"
        summary_saved = reporter.generate_summary_json(
            sample_comparison_result,
            output_path=output_path
        )
        
        assert output_path.exists()
        
        # Load and verify
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['summary_stats'] == summary['summary_stats']
        assert len(loaded['rankings']) == 3
    
    def test_fig_to_base64(self, reporter):
        """Test figure to base64 conversion"""
        # Create simple figure
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        base64_str = reporter._fig_to_base64(fig)
        
        assert isinstance(base64_str, str)
        assert len(base64_str) > 100  # Should be substantial
        
        # Should be valid base64
        import base64
        try:
            base64.b64decode(base64_str)
            valid = True
        except:
            valid = False
        
        assert valid
        plt.close(fig)
    
    def test_pdf_export_warning(self, reporter, sample_comparison_result,
                               sample_backtest_results, tmp_path):
        """Test PDF export (just warning if pdfkit not installed)"""
        html = reporter.generate_report(
            sample_comparison_result,
            sample_backtest_results
        )
        
        output_path = tmp_path / "test_report.pdf"
        
        # This will likely log a warning if pdfkit not installed
        # Just test that it doesn't crash
        reporter.export_to_pdf(html, output_path)
        
        # Method should complete without raising
        assert True
    
    def test_empty_sections_handling(self, reporter):
        """Test handling of empty comparison result"""
        # Create minimal comparison result
        empty_result = ComparisonResult(
            strategy_metrics=pd.DataFrame(),
            rankings=pd.DataFrame(),
            statistical_tests={},
            relative_performance=pd.DataFrame(),
            correlation_matrix=pd.DataFrame(),
            summary_stats={}
        )
        
        # Should handle gracefully
        html = reporter.generate_report(
            empty_result,
            {},
            sections=['executive_summary']
        )
        
        assert isinstance(html, str)
        assert 'Executive Summary' in html
    
    def test_custom_report_title(self, reporter, sample_comparison_result,
                                sample_backtest_results):
        """Test custom report title"""
        custom_title = "My Custom Strategy Analysis Report 2024"
        
        html = reporter.generate_report(
            sample_comparison_result,
            sample_backtest_results,
            report_title=custom_title
        )
        
        assert custom_title in html
        
        # Should appear in both title tag and header
        assert f'<title>{custom_title}</title>' in html
        assert f'<h2>{custom_title}</h2>' in html