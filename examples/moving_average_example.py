"""
Example: Moving Average Crossover Strategy
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngine
from src.strategies.examples.moving_average import MovingAverageCrossover
from src.backtesting.engines.vectorbt_engine import VectorBTEngine
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Example of using Moving Average Crossover strategy"""
    
    # 1. Load and prepare data
    logger.info("Loading market data for SPY...")
    preprocessor = DataPreprocessor(
        raw_data_dir=Path("data/raw/minute_aggs/by_symbol"),
        processed_data_dir=Path("data/processed"),
        cache_dir=Path("data/cache")
    )
    
    # Load SPY data for January 2024
    data = preprocessor.load_processed("SPY", "2024_01")
    if data is None:
        logger.info("Processing SPY data...")
        data = preprocessor.process("SPY", months=["2024_01"])
    
    # 2. Add technical indicators needed by the strategy
    logger.info("Adding technical indicators...")
    feature_engine = FeatureEngine()
    
    # Add volume MA for volume filter
    data['volume_ma'] = data['volume'].rolling(20).mean()
    
    # Add ATR for volatility-based position sizing
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr'] = true_range.rolling(14).mean()
    
    # 3. Test different MA configurations
    configurations = [
        {
            'name': 'Fast SMA Crossover',
            'params': {
                'fast_period': 10,
                'slow_period': 30,
                'ma_type': 'sma',
                'position_sizing': 'fixed',
                'risk_per_trade': 0.02
            }
        },
        {
            'name': 'Slow EMA Crossover',
            'params': {
                'fast_period': 20,
                'slow_period': 50,
                'ma_type': 'ema',
                'position_sizing': 'fixed',
                'risk_per_trade': 0.02
            }
        },
        {
            'name': 'EMA with Volume Filter',
            'params': {
                'fast_period': 10,
                'slow_period': 30,
                'ma_type': 'ema',
                'use_volume_filter': True,
                'volume_threshold': 1.2,
                'position_sizing': 'fixed',
                'risk_per_trade': 0.02
            }
        },
        {
            'name': 'Volatility-Adjusted Sizing',
            'params': {
                'fast_period': 10,
                'slow_period': 30,
                'ma_type': 'ema',
                'position_sizing': 'volatility',
                'risk_per_trade': 0.02
            }
        },
        {
            'name': 'With Risk Management',
            'params': {
                'fast_period': 10,
                'slow_period': 30,
                'ma_type': 'ema',
                'use_stops': True,
                'stop_loss': 0.02,
                'take_profit': 0.05,
                'position_sizing': 'fixed',
                'risk_per_trade': 0.02
            }
        }
    ]
    
    # Initialize backtesting engine
    engine = VectorBTEngine(freq='1min')
    
    print("\n" + "="*80)
    print("MOVING AVERAGE CROSSOVER STRATEGY COMPARISON")
    print("="*80)
    print(f"Data: SPY - January 2024 ({len(data)} minute bars)")
    print(f"Initial Capital: $100,000")
    print("="*80)
    
    results = []
    
    for config in configurations:
        logger.info(f"Testing {config['name']}...")
        
        # Create strategy
        strategy = MovingAverageCrossover(parameters=config['params'])
        
        # Run backtest
        result = engine.run_backtest(
            strategy=strategy,
            data=data,
            initial_capital=100000,
            commission=0.001,  # 0.1% commission
            slippage=0.0001   # 1 basis point slippage
        )
        
        # Store results
        results.append({
            'name': config['name'],
            'result': result,
            'config': config
        })
        
        # Print summary
        print(f"\n{config['name']}:")
        print(f"  Parameters: Fast={config['params']['fast_period']}, "
              f"Slow={config['params']['slow_period']}, "
              f"Type={config['params']['ma_type']}")
        print(f"  Total Return: {result.metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {result.metrics['max_drawdown']:.2%}")
        print(f"  Number of Trades: {result.metrics['trades_count']}")
        print(f"  Win Rate: {result.metrics['win_rate']:.2%}")
    
    # Find best configuration
    print("\n" + "="*80)
    print("BEST CONFIGURATION BY METRIC")
    print("="*80)
    
    # Best by Sharpe Ratio
    best_sharpe = max(results, key=lambda x: x['result'].metrics['sharpe_ratio'])
    print(f"\nBest Sharpe Ratio: {best_sharpe['name']}")
    print(f"  Sharpe: {best_sharpe['result'].metrics['sharpe_ratio']:.2f}")
    print(f"  Return: {best_sharpe['result'].metrics['total_return']:.2%}")
    
    # Best by Total Return
    best_return = max(results, key=lambda x: x['result'].metrics['total_return'])
    print(f"\nBest Total Return: {best_return['name']}")
    print(f"  Return: {best_return['result'].metrics['total_return']:.2%}")
    print(f"  Sharpe: {best_return['result'].metrics['sharpe_ratio']:.2f}")
    
    # Lowest Drawdown
    best_dd = min(results, key=lambda x: x['result'].metrics['max_drawdown'])
    print(f"\nLowest Drawdown: {best_dd['name']}")
    print(f"  Drawdown: {best_dd['result'].metrics['max_drawdown']:.2%}")
    print(f"  Return: {best_dd['result'].metrics['total_return']:.2%}")
    
    # 4. Parameter optimization for best configuration
    print("\n" + "="*80)
    print("PARAMETER OPTIMIZATION")
    print("="*80)
    
    # Define parameter grid
    param_grid = {
        'fast_period': [5, 10, 15, 20],
        'slow_period': [20, 30, 40, 50],
        'ma_type': ['ema'],  # Keep EMA fixed
        'position_sizing': ['fixed'],
        'risk_per_trade': [0.02]
    }
    
    logger.info("Running parameter optimization...")
    optimization_result = engine.optimize_parameters(
        strategy_class=MovingAverageCrossover,
        data=data,
        param_grid=param_grid,
        metric='sharpe_ratio',
        initial_capital=100000
    )
    
    print(f"\nOptimal Parameters Found:")
    print(f"  Fast Period: {optimization_result.best_params['fast_period']}")
    print(f"  Slow Period: {optimization_result.best_params['slow_period']}")
    print(f"  Best Sharpe: {optimization_result.best_metric:.2f}")
    
    # Show top 5 parameter combinations
    print("\nTop 5 Parameter Combinations:")
    print("-" * 60)
    print("Fast | Slow | Sharpe | Return | Max DD | Trades")
    print("-" * 60)
    
    top_5 = optimization_result.results_df.nlargest(5, 'sharpe_ratio')
    for _, row in top_5.iterrows():
        print(f"{int(row['fast_period']):4d} | {int(row['slow_period']):4d} | "
              f"{row['sharpe_ratio']:6.2f} | {row['total_return']:6.2%} | "
              f"{row['max_drawdown']:6.2%} | {int(row['trades_count']):6d}")
    
    # 5. Run final backtest with optimized parameters
    print("\n" + "="*80)
    print("OPTIMIZED STRATEGY PERFORMANCE")
    print("="*80)
    
    optimized_strategy = MovingAverageCrossover(parameters=optimization_result.best_params)
    final_result = engine.run_backtest(
        strategy=optimized_strategy,
        data=data,
        initial_capital=100000,
        commission=0.001
    )
    
    # Generate detailed report
    report = engine.generate_report(final_result, format='text')
    print(report)
    
    # Save results
    output_dir = Path("results/strategies")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save equity curve
    equity_curve_path = output_dir / "ma_crossover_equity_curve.csv"
    final_result.equity_curve.to_csv(equity_curve_path)
    print(f"\nEquity curve saved to: {equity_curve_path}")
    
    # Save trades
    if len(final_result.trades) > 0:
        trades_path = output_dir / "ma_crossover_trades.csv"
        final_result.trades.to_csv(trades_path, index=False)
        print(f"Trades saved to: {trades_path}")
        
        # Show sample trades
        print("\nSample Trades:")
        print(final_result.trades.head())
    
    logger.info("Moving average strategy example completed")


if __name__ == "__main__":
    main()