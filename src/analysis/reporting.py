"""
Reporting Module
Generate comprehensive HTML/PDF reports for strategy analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import warnings

from src.backtesting.engines.vectorbt_engine import BacktestResult
from src.analysis.performance_analyzer import ComparisonResult
from src.analysis.visualization import StrategyVisualizer
from src.utils.logging import get_logger


class PerformanceReporter:
    """
    Generate comprehensive performance reports
    
    Features:
    - HTML report generation with embedded charts
    - PDF export capability (requires wkhtmltopdf)
    - Customizable report sections
    - Statistical test summaries
    - Performance rankings
    """
    
    def __init__(self, 
                 company_name: str = "Quant Trading Co",
                 report_style: str = "professional"):
        """
        Initialize reporter
        
        Args:
            company_name: Company name for report header
            report_style: Report style ('professional', 'minimal', 'detailed')
        """
        self.company_name = company_name
        self.report_style = report_style
        self.logger = get_logger('reporter')
        self.visualizer = StrategyVisualizer()
    
    def generate_report(self,
                       comparison_result: ComparisonResult,
                       backtest_results: Dict[str, BacktestResult],
                       report_title: str = "Strategy Performance Analysis",
                       sections: Optional[List[str]] = None,
                       output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive HTML report
        
        Args:
            comparison_result: Results from performance comparison
            backtest_results: Original backtest results
            report_title: Title for the report
            sections: Sections to include (defaults to all)
            output_path: Optional path to save report
            
        Returns:
            HTML string
        """
        if sections is None:
            sections = [
                'executive_summary',
                'performance_metrics',
                'equity_curves',
                'statistical_tests',
                'risk_analysis',
                'correlation_analysis',
                'trade_analysis',
                'recommendations'
            ]
        
        # Generate HTML
        html_parts = [self._generate_header(report_title)]
        
        if 'executive_summary' in sections:
            html_parts.append(self._generate_executive_summary(comparison_result))
        
        if 'performance_metrics' in sections:
            html_parts.append(self._generate_metrics_section(comparison_result))
        
        if 'equity_curves' in sections:
            html_parts.append(self._generate_equity_curves_section(backtest_results))
        
        if 'statistical_tests' in sections:
            html_parts.append(self._generate_statistical_tests_section(comparison_result))
        
        if 'risk_analysis' in sections:
            html_parts.append(self._generate_risk_analysis_section(
                comparison_result, backtest_results))
        
        if 'correlation_analysis' in sections:
            html_parts.append(self._generate_correlation_section(comparison_result))
        
        if 'trade_analysis' in sections:
            html_parts.append(self._generate_trade_analysis_section(backtest_results))
        
        if 'recommendations' in sections:
            html_parts.append(self._generate_recommendations_section(comparison_result))
        
        html_parts.append(self._generate_footer())
        
        html = '\n'.join(html_parts)
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(html)
            
            self.logger.info(f"Report saved to {output_path}")
        
        return html
    
    def _generate_header(self, title: str) -> str:
        """Generate HTML header"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="utf-8">
            <style>
                {self._get_css_styles()}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>{self.company_name}</h1>
                    <h2>{title}</h2>
                    <p class="date">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </header>
        """
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the report"""
        if self.report_style == 'professional':
            return """
                body {
                    font-family: 'Arial', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                header {
                    text-align: center;
                    border-bottom: 3px solid #2c3e50;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }
                h1 {
                    color: #2c3e50;
                    margin: 10px 0;
                }
                h2 {
                    color: #34495e;
                    margin: 20px 0;
                    border-bottom: 2px solid #ecf0f1;
                    padding-bottom: 10px;
                }
                h3 {
                    color: #7f8c8d;
                    margin: 15px 0;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #2c3e50;
                    color: white;
                    font-weight: bold;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .metric-box {
                    display: inline-block;
                    margin: 10px;
                    padding: 20px;
                    background-color: #ecf0f1;
                    border-radius: 5px;
                    text-align: center;
                    min-width: 150px;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }
                .metric-label {
                    font-size: 14px;
                    color: #7f8c8d;
                }
                .chart-container {
                    margin: 20px 0;
                    text-align: center;
                }
                .alert {
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 5px;
                }
                .alert-success {
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    color: #155724;
                }
                .alert-warning {
                    background-color: #fff3cd;
                    border: 1px solid #ffeeba;
                    color: #856404;
                }
                .alert-danger {
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                    color: #721c24;
                }
                .summary-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }
                footer {
                    text-align: center;
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #7f8c8d;
                    font-size: 12px;
                }
            """
        else:
            # Minimal style
            return """
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                h1, h2, h3 { color: #333; }
                .chart-container { margin: 20px 0; }
            """
    
    def _generate_executive_summary(self, result: ComparisonResult) -> str:
        """Generate executive summary section"""
        summary = result.summary_stats
        best_strategy = result.rankings.iloc[0]['strategy']
        
        html = """
        <section class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
        """
        
        # Key metrics boxes
        metrics = [
            ('Strategies Analyzed', summary['n_strategies']),
            ('Best Strategy', best_strategy),
            ('Best Sharpe Ratio', f"{summary['best_sharpe']:.2f}"),
            ('Best Total Return', f"{summary['best_return']:.1%}"),
            ('Avg Max Drawdown', f"{summary['avg_max_drawdown']:.1%}")
        ]
        
        for label, value in metrics:
            html += f"""
                <div class="metric-box">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
            """
        
        html += """
            </div>
            <div class="summary-text">
                <p>This report analyzes {n_strategies} trading strategies over the backtesting period. 
                The best performing strategy is <strong>{best_strategy}</strong> with a Sharpe ratio of 
                {best_sharpe:.2f} and total return of {best_return:.1%}.</p>
            </div>
        </section>
        """.format(**summary, best_strategy=best_strategy)
        
        return html
    
    def _generate_metrics_section(self, result: ComparisonResult) -> str:
        """Generate performance metrics section"""
        html = """
        <section class="performance-metrics">
            <h2>Performance Metrics</h2>
        """
        
        # Rankings table
        html += """
            <h3>Strategy Rankings</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Strategy</th>
                        <th>Sharpe Ratio</th>
                        <th>Total Return</th>
                        <th>Max Drawdown</th>
                        <th>Volatility</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        rankings = result.rankings
        for idx, row in rankings.iterrows():
            html += f"""
                <tr>
                    <td>{row.get('composite_rank', idx+1):.0f}</td>
                    <td><strong>{row['strategy']}</strong></td>
                    <td>{row.get('sharpe_ratio', 0):.2f}</td>
                    <td>{row.get('total_return', 0):.1%}</td>
                    <td>{row.get('max_drawdown', 0):.1%}</td>
                    <td>{row.get('volatility', 0):.1%}</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        """
        
        # Detailed metrics table
        html += """
            <h3>Detailed Performance Metrics</h3>
            <div style="overflow-x: auto;">
        """
        
        # Convert metrics dataframe to HTML
        metrics_html = result.strategy_metrics.round(3).to_html(
            classes='metrics-table', 
            table_id='detailed-metrics'
        )
        html += metrics_html
        
        html += """
            </div>
        </section>
        """
        
        return html
    
    def _generate_equity_curves_section(self, results: Dict[str, BacktestResult]) -> str:
        """Generate equity curves section"""
        html = """
        <section class="equity-curves">
            <h2>Equity Curves</h2>
            <div class="chart-container">
        """
        
        # Generate equity curve plot
        fig = self.visualizer.plot_equity_curves(results, show_drawdowns=True)
        
        # Convert to base64 for embedding
        img_str = self._fig_to_base64(fig)
        html += f'<img src="data:image/png;base64,{img_str}" width="100%">'
        
        plt.close(fig)
        
        html += """
            </div>
        </section>
        """
        
        return html
    
    def _generate_statistical_tests_section(self, result: ComparisonResult) -> str:
        """Generate statistical tests section"""
        html = """
        <section class="statistical-tests">
            <h2>Statistical Analysis</h2>
        """
        
        # Sharpe difference tests
        if 'sharpe_difference' in result.statistical_tests:
            sharpe_tests = result.statistical_tests['sharpe_difference']
            
            html += """
                <h3>Sharpe Ratio Difference Tests</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Strategy 1</th>
                            <th>Strategy 2</th>
                            <th>Sharpe 1</th>
                            <th>Sharpe 2</th>
                            <th>Difference</th>
                            <th>P-Value</th>
                            <th>Significant</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for _, row in sharpe_tests.iterrows():
                sig_class = 'alert-success' if row['significant_5%'] else ''
                html += f"""
                    <tr class="{sig_class}">
                        <td>{row['strategy_1']}</td>
                        <td>{row['strategy_2']}</td>
                        <td>{row['sharpe_1']:.2f}</td>
                        <td>{row['sharpe_2']:.2f}</td>
                        <td>{row['difference']:.2f}</td>
                        <td>{row['p_value']:.4f}</td>
                        <td>{'Yes' if row['significant_5%'] else 'No'}</td>
                    </tr>
                """
            
            html += """
                    </tbody>
                </table>
            """
        
        # Summary of statistical tests
        test_summary = result.get_statistical_summary()
        html += """
            <h3>Statistical Test Summary</h3>
            <div class="summary-grid">
        """
        
        for test_name, summary in test_summary.items():
            html += f"""
                <div class="metric-box">
                    <div class="metric-value">{summary['significant_at_5%']}/{summary['total_comparisons']}</div>
                    <div class="metric-label">Significant {test_name.replace('_', ' ').title()}</div>
                </div>
            """
        
        html += """
            </div>
        </section>
        """
        
        return html
    
    def _generate_risk_analysis_section(self, 
                                      comparison_result: ComparisonResult,
                                      backtest_results: Dict[str, BacktestResult]) -> str:
        """Generate risk analysis section"""
        html = """
        <section class="risk-analysis">
            <h2>Risk Analysis</h2>
        """
        
        # Drawdown chart
        html += """
            <h3>Maximum Drawdowns</h3>
            <div class="chart-container">
        """
        
        # Create drawdown comparison chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        strategies = []
        max_drawdowns = []
        
        for name in comparison_result.strategy_metrics.index:
            strategies.append(name)
            max_drawdowns.append(comparison_result.strategy_metrics.loc[name, 'max_drawdown'])
        
        bars = ax.bar(strategies, max_drawdowns)
        
        # Color bars based on severity
        for i, (bar, dd) in enumerate(zip(bars, max_drawdowns)):
            if dd > 0.3:  # > 30% drawdown
                bar.set_color('red')
            elif dd > 0.2:  # > 20% drawdown
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        ax.set_ylabel('Maximum Drawdown (%)')
        ax.set_title('Maximum Drawdowns by Strategy')
        ax.set_ylim(0, max(max_drawdowns) * 1.1)
        
        # Add value labels on bars
        for bar, dd in zip(bars, max_drawdowns):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{dd:.1%}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img_str = self._fig_to_base64(fig)
        html += f'<img src="data:image/png;base64,{img_str}" width="100%">'
        plt.close(fig)
        
        html += """
            </div>
        """
        
        # Risk metrics table
        html += """
            <h3>Risk Metrics Comparison</h3>
            <table>
                <thead>
                    <tr>
                        <th>Strategy</th>
                        <th>Volatility</th>
                        <th>Max Drawdown</th>
                        <th>Sortino Ratio</th>
                        <th>Calmar Ratio</th>
                        <th>Skewness</th>
                        <th>Kurtosis</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for strategy in comparison_result.strategy_metrics.index:
            metrics = comparison_result.strategy_metrics.loc[strategy]
            html += f"""
                <tr>
                    <td><strong>{strategy}</strong></td>
                    <td>{metrics.get('volatility', 0):.1%}</td>
                    <td>{metrics.get('max_drawdown', 0):.1%}</td>
                    <td>{metrics.get('sortino_ratio', 0):.2f}</td>
                    <td>{metrics.get('calmar_ratio', 0):.2f}</td>
                    <td>{metrics.get('skewness', 0):.2f}</td>
                    <td>{metrics.get('kurtosis', 0):.2f}</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </section>
        """
        
        return html
    
    def _generate_correlation_section(self, result: ComparisonResult) -> str:
        """Generate correlation analysis section"""
        html = """
        <section class="correlation-analysis">
            <h2>Correlation Analysis</h2>
            <div class="chart-container">
        """
        
        # Generate correlation heatmap
        fig = self.visualizer.plot_correlation_matrix(result.correlation_matrix)
        
        img_str = self._fig_to_base64(fig)
        html += f'<img src="data:image/png;base64,{img_str}" width="80%">'
        plt.close(fig)
        
        html += """
            </div>
            <p>The correlation matrix shows the relationship between strategy returns. 
            Lower correlations indicate better diversification benefits when combining strategies.</p>
        </section>
        """
        
        return html
    
    def _generate_trade_analysis_section(self, results: Dict[str, BacktestResult]) -> str:
        """Generate trade analysis section"""
        html = """
        <section class="trade-analysis">
            <h2>Trade Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Strategy</th>
                        <th>Total Trades</th>
                        <th>Win Rate</th>
                        <th>Avg Win</th>
                        <th>Avg Loss</th>
                        <th>Profit Factor</th>
                        <th>Max Consec. Wins</th>
                        <th>Max Consec. Losses</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for name, result in results.items():
            metrics = result.metrics
            
            # Extract trade statistics
            total_trades = metrics.get('total_trades', 0)
            win_rate = metrics.get('win_rate', 0)
            
            # Calculate average win/loss if we have trade data
            if hasattr(result, 'trades') and result.trades is not None and len(result.trades) > 0:
                wins = result.trades[result.trades['pnl'] > 0]['pnl']
                losses = result.trades[result.trades['pnl'] < 0]['pnl']
                avg_win = wins.mean() if len(wins) > 0 else 0
                avg_loss = losses.mean() if len(losses) > 0 else 0
            else:
                avg_win = avg_loss = 0
            
            html += f"""
                <tr>
                    <td><strong>{name}</strong></td>
                    <td>{total_trades}</td>
                    <td>{win_rate:.1%}</td>
                    <td>${avg_win:,.2f}</td>
                    <td>${avg_loss:,.2f}</td>
                    <td>{metrics.get('profit_factor', 0):.2f}</td>
                    <td>{metrics.get('max_consecutive_wins', 0)}</td>
                    <td>{metrics.get('max_consecutive_losses', 0)}</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </section>
        """
        
        return html
    
    def _generate_recommendations_section(self, result: ComparisonResult) -> str:
        """Generate recommendations section"""
        html = """
        <section class="recommendations">
            <h2>Recommendations</h2>
        """
        
        # Get top strategies
        top_strategies = result.rankings.head(3)
        
        # Analyze characteristics
        recommendations = []
        
        # Best overall
        best_overall = top_strategies.iloc[0]['strategy']
        recommendations.append(
            f"<strong>Best Overall Strategy:</strong> {best_overall} demonstrates the best "
            f"risk-adjusted returns with a Sharpe ratio of {result.strategy_metrics.loc[best_overall, 'sharpe_ratio']:.2f}."
        )
        
        # Risk considerations
        low_risk_strategies = result.strategy_metrics.nsmallest(3, 'volatility').index.tolist()
        recommendations.append(
            f"<strong>For Risk-Averse Investors:</strong> Consider {', '.join(low_risk_strategies[:2])} "
            f"which show the lowest volatility."
        )
        
        # Diversification
        if len(result.correlation_matrix) > 2:
            # Find least correlated pair
            corr_matrix = result.correlation_matrix
            min_corr = 1.0
            best_pair = None
            
            for i in range(len(corr_matrix.index)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr = corr_matrix.iloc[i, j]
                    if corr < min_corr:
                        min_corr = corr
                        best_pair = (corr_matrix.index[i], corr_matrix.columns[j])
            
            if best_pair:
                recommendations.append(
                    f"<strong>Best Diversification:</strong> Combining {best_pair[0]} and {best_pair[1]} "
                    f"provides good diversification with correlation of {min_corr:.2f}."
                )
        
        # Create recommendation list
        html += "<ul>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul>"
        
        # Risk warnings
        html += """
            <div class="alert alert-warning">
                <h4>Important Considerations</h4>
                <ul>
                    <li>Past performance does not guarantee future results</li>
                    <li>Backtesting results may be subject to overfitting</li>
                    <li>Transaction costs and slippage estimates may differ in live trading</li>
                    <li>Market conditions change - strategies should be regularly re-evaluated</li>
                </ul>
            </div>
        </section>
        """
        
        return html
    
    def _generate_footer(self) -> str:
        """Generate HTML footer"""
        return """
            <footer>
                <p>This report is for informational purposes only and does not constitute investment advice.</p>
                <p>&copy; {year} {company}. All rights reserved.</p>
            </footer>
            </div>
        </body>
        </html>
        """.format(year=datetime.now().year, company=self.company_name)
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        buffer.close()
        return img_str
    
    def export_to_pdf(self, html: str, output_path: Path):
        """
        Export HTML report to PDF
        
        Note: Requires wkhtmltopdf to be installed
        
        Args:
            html: HTML string
            output_path: Path for PDF output
        """
        try:
            import pdfkit
            
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
                'no-outline': None
            }
            
            pdfkit.from_string(html, str(output_path), options=options)
            self.logger.info(f"PDF report saved to {output_path}")
            
        except ImportError:
            self.logger.warning("pdfkit not installed. Install with: pip install pdfkit")
            self.logger.warning("Also requires wkhtmltopdf: https://wkhtmltopdf.org/")
        except Exception as e:
            self.logger.error(f"Failed to generate PDF: {e}")
    
    def generate_summary_json(self,
                            comparison_result: ComparisonResult,
                            output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate JSON summary of results
        
        Args:
            comparison_result: Comparison results
            output_path: Optional path to save JSON
            
        Returns:
            Dictionary with summary data
        """
        summary = {
            'generated_at': datetime.now().isoformat(),
            'summary_stats': comparison_result.summary_stats,
            'rankings': comparison_result.rankings.to_dict('records'),
            'strategy_metrics': comparison_result.strategy_metrics.to_dict('index'),
            'statistical_tests': {
                test_name: results.to_dict('records')
                for test_name, results in comparison_result.statistical_tests.items()
            },
            'correlations': comparison_result.correlation_matrix.to_dict()
        }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"JSON summary saved to {output_path}")
        
        return summary