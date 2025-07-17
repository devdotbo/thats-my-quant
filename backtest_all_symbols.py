#!/usr/bin/env python3
"""
Systematic Backtesting Script
Runs all strategies on all available symbols and generates comprehensive reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import json
from datetime import datetime
from typing import Dict, List, Tuple
import concurrent.futures
from tqdm import tqdm

# Import our modules
from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngine
from src.strategies.examples.moving_average import MovingAverageCrossover
from src.strategies.examples.orb import OpeningRangeBreakout
from src.backtesting.engines.vectorbt_engine import VectorBTEngine, BacktestResult
from src.backtesting.costs import TransactionCostEngine
from src.analysis import PerformanceAnalyzer, PerformanceReporter, StrategyVisualizer
from src.validation.walk_forward import WalkForwardValidator
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_available_symbols() -> List[str]:
    """Get list of symbols with available data"""
    symbol_dirs = glob.glob('data/raw/minute_aggs/by_symbol/*')
    symbols = [Path(d).name for d in symbol_dirs if Path(d).is_dir()]
    return sorted(symbols)


def load_symbol_data(symbol: str, months: List[str] = None) -> pd.DataFrame:
    """Load and preprocess data for a symbol"""
    logger.info(f"Loading data for {symbol}")
    
    # Initialize preprocessor and feature engine
    preprocessor = DataPreprocessor(
        raw_data_dir=Path('data/raw/minute_aggs/by_symbol'),
        processed_data_dir=Path('data/processed'),
        cache_dir=Path('data/cache')
    )
    feature_engine = FeatureEngine()
    
    # If no months specified, get all available months
    if not months:
        # Get all available files for the symbol to extract months
        files = sorted(glob.glob(f'data/raw/minute_aggs/by_symbol/{symbol}/*.csv.gz'))
        
        if not files:
            logger.warning(f"No data files found for {symbol}")
            return pd.DataFrame()
            
        # Extract months from filenames (format: SYMBOL_YYYY_MM.csv.gz)
        months = []
        for f in files:
            filename = Path(f).stem.replace('.csv', '')  # Remove .csv.gz
            parts = filename.split('_')
            if len(parts) >= 3:
                month = f"{parts[-2]}_{parts[-1]}"  # YYYY_MM
                months.append(month)
        
        logger.info(f"Found {len(months)} months of data for {symbol}")
    
    # Process data
    try:
        processed = preprocessor.process(symbol=symbol, months=months)
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return pd.DataFrame()
    
    # Add features
    data_with_features = feature_engine.add_all_features(processed)
    
    logger.info(f"Loaded {len(data_with_features)} bars for {symbol}")
    return data_with_features


def create_strategies() -> Dict[str, List[Tuple[str, dict]]]:
    """Create strategy instances with different parameter sets"""
    strategies = {
        'MA_Fast': [
            ('MA_Fast_5_20', {'fast_period': 5, 'slow_period': 20, 'ma_type': 'ema'}),
            ('MA_Fast_10_30', {'fast_period': 10, 'slow_period': 30, 'ma_type': 'ema'}),
            ('MA_Fast_15_50', {'fast_period': 15, 'slow_period': 50, 'ma_type': 'ema'}),
        ],
        'MA_Slow': [
            ('MA_Slow_20_50', {'fast_period': 20, 'slow_period': 50, 'ma_type': 'sma'}),
            ('MA_Slow_30_100', {'fast_period': 30, 'slow_period': 100, 'ma_type': 'sma'}),
            ('MA_Slow_50_200', {'fast_period': 50, 'slow_period': 200, 'ma_type': 'sma'}),
        ],
        'ORB': [
            ('ORB_5min_3R', {'range_minutes': 5, 'profit_target_r': 3.0, 'stop_type': 'range'}),
            ('ORB_15min_5R', {'range_minutes': 15, 'profit_target_r': 5.0, 'stop_type': 'atr'}),
            ('ORB_30min_10R', {'range_minutes': 30, 'profit_target_r': 10.0, 'stop_type': 'range'}),
        ]
    }
    
    # Create strategy instances
    strategy_instances = {}
    
    for strategy_type, param_sets in strategies.items():
        for name, params in param_sets:
            if 'MA' in strategy_type:
                strategy_instances[name] = MovingAverageCrossover(params)
            else:  # ORB
                strategy_instances[name] = OpeningRangeBreakout(params)
    
    return strategy_instances


def backtest_symbol_strategy(
    symbol: str, 
    strategy_name: str,
    strategy,
    data: pd.DataFrame,
    engine: VectorBTEngine,
    initial_capital: float = 100000,
    commission: float = 0.0005,
    slippage: float = 0.0001
) -> Tuple[str, str, BacktestResult]:
    """Run backtest for a single symbol-strategy combination"""
    
    logger.info(f"Backtesting {strategy_name} on {symbol}")
    
    try:
        result = engine.run_backtest(
            strategy=strategy,
            data=data,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )
        
        logger.info(f"  {symbol}-{strategy_name}: Sharpe={result.metrics['sharpe_ratio']:.2f}, "
                   f"Return={result.metrics['total_return']:.1%}")
        
        return symbol, strategy_name, result
        
    except Exception as e:
        logger.error(f"Error backtesting {strategy_name} on {symbol}: {e}")
        return symbol, strategy_name, None


def run_all_backtests(
    symbols: List[str],
    strategies: Dict[str, object],
    initial_capital: float = 100000,
    max_workers: int = 4
) -> Dict[str, Dict[str, BacktestResult]]:
    """Run all backtests in parallel"""
    
    # Initialize backtesting engine
    engine = VectorBTEngine(freq='1min')
    
    # Results storage
    results = {}
    
    # Create tasks
    tasks = []
    for symbol in symbols:
        # Load data once per symbol
        data = load_symbol_data(symbol)
        if data.empty:
            continue
            
        results[symbol] = {}
        
        for strategy_name, strategy in strategies.items():
            # Include commission and slippage in task parameters
            tasks.append((symbol, strategy_name, strategy, data, engine, initial_capital, 0.0005, 0.0001))
    
    # Run backtests in parallel
    logger.info(f"Running {len(tasks)} backtests with {max_workers} workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in tasks:
            future = executor.submit(backtest_symbol_strategy, *task)
            futures.append(future)
        
        # Process results with progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            symbol, strategy_name, result = future.result()
            if result:
                results[symbol][strategy_name] = result
    
    return results


def analyze_results(results: Dict[str, Dict[str, BacktestResult]]) -> pd.DataFrame:
    """Analyze and summarize backtest results"""
    
    summary_data = []
    
    for symbol, strategy_results in results.items():
        for strategy_name, result in strategy_results.items():
            if result and result.metrics:
                summary_data.append({
                    'symbol': symbol,
                    'strategy': strategy_name,
                    'sharpe_ratio': result.metrics['sharpe_ratio'],
                    'total_return': result.metrics['total_return'],
                    'annual_return': result.metrics.get('annual_return', 0),
                    'max_drawdown': result.metrics['max_drawdown'],
                    'win_rate': result.metrics.get('win_rate', 0),
                    'total_trades': result.metrics.get('total_trades', 0),
                    'profit_factor': result.metrics.get('profit_factor', 0),
                    'volatility': result.metrics.get('volatility', 0)
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add rankings
    summary_df['sharpe_rank'] = summary_df['sharpe_ratio'].rank(ascending=False)
    summary_df['return_rank'] = summary_df['total_return'].rank(ascending=False)
    summary_df['composite_rank'] = (summary_df['sharpe_rank'] + summary_df['return_rank']) / 2
    
    # Sort by composite rank
    summary_df = summary_df.sort_values('composite_rank')
    
    return summary_df


def generate_report(
    results: Dict[str, Dict[str, BacktestResult]], 
    summary_df: pd.DataFrame,
    output_dir: Path
):
    """Generate comprehensive performance report"""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary to CSV
    summary_df.to_csv(output_dir / 'backtest_summary.csv', index=False)
    logger.info(f"Saved summary to {output_dir / 'backtest_summary.csv'}")
    
    # Get top 10 strategies
    top_strategies = summary_df.head(10)
    
    print("\n" + "="*80)
    print("TOP 10 STRATEGIES BY COMPOSITE RANK")
    print("="*80)
    print(f"{'Rank':<5} {'Symbol':<8} {'Strategy':<20} {'Sharpe':<8} {'Return':<10} {'Max DD':<10} {'Trades':<8}")
    print("-"*80)
    
    for idx, row in top_strategies.iterrows():
        print(f"{row['composite_rank']:<5.0f} {row['symbol']:<8} {row['strategy']:<20} "
              f"{row['sharpe_ratio']:<8.2f} {row['total_return']:<10.1%} "
              f"{row['max_drawdown']:<10.1%} {row['total_trades']:<8.0f}")
    
    # Best by category
    print("\n" + "="*80)
    print("BEST STRATEGIES BY CATEGORY")
    print("="*80)
    
    best_sharpe = summary_df.loc[summary_df['sharpe_ratio'].idxmax()]
    best_return = summary_df.loc[summary_df['total_return'].idxmax()]
    best_win_rate = summary_df.loc[summary_df['win_rate'].idxmax()]
    
    print(f"\nBest Sharpe Ratio: {best_sharpe['symbol']} - {best_sharpe['strategy']}")
    print(f"  Sharpe: {best_sharpe['sharpe_ratio']:.2f}, Return: {best_sharpe['total_return']:.1%}")
    
    print(f"\nBest Total Return: {best_return['symbol']} - {best_return['strategy']}")
    print(f"  Return: {best_return['total_return']:.1%}, Sharpe: {best_return['sharpe_ratio']:.2f}")
    
    print(f"\nBest Win Rate: {best_win_rate['symbol']} - {best_win_rate['strategy']}")
    print(f"  Win Rate: {best_win_rate['win_rate']:.1%}, Sharpe: {best_win_rate['sharpe_ratio']:.2f}")
    
    # Strategy type performance
    print("\n" + "="*80)
    print("STRATEGY TYPE PERFORMANCE")
    print("="*80)
    
    summary_df['strategy_type'] = summary_df['strategy'].apply(
        lambda x: 'MA' if 'MA' in x else 'ORB'
    )
    
    strategy_perf = summary_df.groupby('strategy_type').agg({
        'sharpe_ratio': ['mean', 'std'],
        'total_return': ['mean', 'std'],
        'win_rate': 'mean'
    }).round(3)
    
    print(strategy_perf)
    
    # Symbol performance
    print("\n" + "="*80)
    print("SYMBOL PERFORMANCE")
    print("="*80)
    
    symbol_perf = summary_df.groupby('symbol').agg({
        'sharpe_ratio': ['mean', 'max'],
        'total_return': ['mean', 'max']
    }).round(3)
    
    print(symbol_perf)
    
    # Generate detailed HTML report for top strategies
    if len(top_strategies) > 0:
        logger.info("Generating detailed HTML report for top strategies")
        
        # Get top 5 for detailed comparison
        top_5 = top_strategies.head(5)
        comparison_results = {}
        
        for _, row in top_5.iterrows():
            key = f"{row['symbol']}_{row['strategy']}"
            comparison_results[key] = results[row['symbol']][row['strategy']]
        
        # Use performance analyzer and reporter
        analyzer = PerformanceAnalyzer()
        reporter = PerformanceReporter(company_name="Systematic Backtest Results")
        
        comparison = analyzer.compare_strategies(comparison_results)
        
        html_report = reporter.generate_report(
            comparison,
            comparison_results,
            report_title="Top 5 Strategies Comparison",
            output_path=output_dir / "top_strategies_report.html"
        )
        
        logger.info(f"Saved detailed report to {output_dir / 'top_strategies_report.html'}")
    
    # Save full results as JSON
    results_json = {}
    for symbol, strategy_results in results.items():
        results_json[symbol] = {}
        for strategy_name, result in strategy_results.items():
            if result:
                results_json[symbol][strategy_name] = {
                    'metrics': result.metrics,
                    'total_trades': len(result.trades) if hasattr(result, 'trades') else 0
                }
    
    with open(output_dir / 'full_results.json', 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    
    logger.info(f"Saved full results to {output_dir / 'full_results.json'}")


def main():
    """Main execution function"""
    
    print("="*80)
    print("SYSTEMATIC BACKTESTING - ALL SYMBOLS")
    print("="*80)
    
    # Configuration
    initial_capital = 100000
    output_dir = Path('backtest_results') / datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get available symbols
    symbols = get_available_symbols()
    print(f"\nFound {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Create strategies
    strategies = create_strategies()
    print(f"\nCreated {len(strategies)} strategy variations")
    
    # Run backtests
    print(f"\nRunning {len(symbols) * len(strategies)} backtests...")
    results = run_all_backtests(symbols, strategies, initial_capital, max_workers=2)
    
    # Analyze results
    print("\nAnalyzing results...")
    summary_df = analyze_results(results)
    
    # Generate report
    print("\nGenerating reports...")
    generate_report(results, summary_df, output_dir)
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()