"""
Performance Comparison Demo
Example of using the analysis module to compare multiple strategies
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.analysis import (
    PerformanceAnalyzer,
    StatisticalTests,
    StrategyVisualizer,
    PerformanceReporter
)
from src.strategies.examples.moving_average import MovingAverageCrossover
from src.strategies.examples.orb import OpeningRangeBreakout
from src.backtesting.engines.vectorbt_engine import VectorBTEngine
from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngine


def main():
    """Demonstrate performance comparison framework"""
    
    print("Performance Comparison Framework Demo")
    print("=" * 50)
    
    # 1. Load and prepare data
    print("\n1. Loading data...")
    
    # Check if data exists
    data_path = Path("data/raw/minute_aggs/by_symbol/AAPL/AAPL_2024_01.csv.gz")
    if not data_path.exists():
        print("Error: Sample data not found. Please run the data downloader first.")
        return
    
    # Load and process data
    preprocessor = DataPreprocessor(
        raw_data_dir=Path("data/raw/minute_aggs/by_symbol"),
        processed_data_dir=Path("data/processed"),
        cache_dir=Path("data/cache")
    )
    
    # Process January data
    processed_data = preprocessor.process('AAPL', ['2024_01'])
    
    # Add features
    feature_engine = FeatureEngine()
    data_with_features = feature_engine.add_all_features(processed_data)
    
    print(f"   Loaded {len(data_with_features)} bars of AAPL data")
    
    # 2. Define strategies to compare
    print("\n2. Creating strategies...")
    
    strategies = {
        'MA_Fast': MovingAverageCrossover({
            'fast_period': 10,
            'slow_period': 30,
            'ma_type': 'EMA'
        }),
        'MA_Slow': MovingAverageCrossover({
            'fast_period': 20,
            'slow_period': 50,
            'ma_type': 'SMA'
        }),
        'ORB_5min': OpeningRangeBreakout({
            'range_minutes': 5,
            'profit_target_r': 3.0,
            'stop_type': 'range'
        }),
        'ORB_15min': OpeningRangeBreakout({
            'range_minutes': 15,
            'profit_target_r': 5.0,
            'stop_type': 'atr'
        })
    }
    
    print(f"   Created {len(strategies)} strategies for comparison")
    
    # 3. Run backtests
    print("\n3. Running backtests...")
    
    engine = VectorBTEngine()
    backtest_results = {}
    
    for name, strategy in strategies.items():
        print(f"   Backtesting {name}...")
        result = engine.run_backtest(
            strategy=strategy,
            data=data_with_features,
            initial_capital=100000,
            commission=0.001
        )
        backtest_results[name] = result
        print(f"     Sharpe: {result.metrics['sharpe_ratio']:.2f}, "
              f"Return: {result.metrics['total_return']:.1%}")
    
    # 4. Compare strategies
    print("\n4. Comparing strategies...")
    
    analyzer = PerformanceAnalyzer()
    comparison = analyzer.compare_strategies(backtest_results)
    
    print("\n   Strategy Rankings:")
    print(comparison.rankings[['strategy', 'sharpe_ratio', 'total_return', 'composite_rank']].to_string())
    
    # 5. Statistical testing
    print("\n5. Statistical significance tests...")
    
    stat_tests = StatisticalTests()
    
    # Test if best strategy is significantly better
    best_strategy = comparison.rankings.iloc[0]['strategy']
    second_best = comparison.rankings.iloc[1]['strategy']
    
    test_result = stat_tests.test_sharpe_difference(
        backtest_results[best_strategy].equity_curve.pct_change().dropna(),
        backtest_results[second_best].equity_curve.pct_change().dropna()
    )
    
    print(f"\n   {test_result.summary()}")
    
    # 6. Generate visualizations
    print("\n6. Creating visualizations...")
    
    visualizer = StrategyVisualizer()
    
    # Equity curves
    fig_equity = visualizer.plot_equity_curves(
        backtest_results,
        show_drawdowns=True,
        interactive=False
    )
    fig_equity.savefig('examples/output/equity_curves.png', dpi=150, bbox_inches='tight')
    print("   Saved equity curves to examples/output/equity_curves.png")
    
    # Performance heatmap
    fig_heatmap = visualizer.plot_performance_heatmap(
        comparison.strategy_metrics,
        normalize=True
    )
    fig_heatmap.savefig('examples/output/performance_heatmap.png', dpi=150, bbox_inches='tight')
    print("   Saved performance heatmap to examples/output/performance_heatmap.png")
    
    # 7. Generate report
    print("\n7. Generating HTML report...")
    
    reporter = PerformanceReporter(
        company_name="Quant Trading Demo",
        report_style="professional"
    )
    
    # Create output directory
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "strategy_comparison_report.html"
    
    reporter.generate_report(
        comparison,
        backtest_results,
        report_title="Strategy Comparison Analysis - AAPL January 2024",
        output_path=report_path
    )
    
    print(f"   Saved report to {report_path}")
    
    # Also save JSON summary
    json_path = output_dir / "strategy_comparison_summary.json"
    reporter.generate_summary_json(comparison, output_path=json_path)
    print(f"   Saved JSON summary to {json_path}")
    
    print("\n" + "=" * 50)
    print("Demo complete! Check the examples/output/ directory for results.")
    
    # Display final summary
    print("\nSummary Statistics:")
    print(f"  Best Strategy: {comparison.summary_stats['best_sharpe_strategy']}")
    print(f"  Best Sharpe Ratio: {comparison.summary_stats['best_sharpe']:.2f}")
    print(f"  Best Total Return: {comparison.summary_stats['best_return']:.1%}")
    print(f"  Average Max Drawdown: {comparison.summary_stats['avg_max_drawdown']:.1%}")


if __name__ == "__main__":
    main()