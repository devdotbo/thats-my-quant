#!/usr/bin/env python3
"""
Find Profitable Trading Parameters
Run comprehensive Bayesian optimization to find actually profitable strategies
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json

from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngine
from src.backtesting.engines.vectorbt_engine import VectorBTEngine
from src.strategies.examples.moving_average import MovingAverageCrossover
from src.strategies.examples.orb import OpeningRangeBreakout
from src.utils.logging import get_logger

logger = get_logger("find_profitable")


def load_symbol_data(symbol: str) -> pd.DataFrame:
    """Load and preprocess data for a symbol"""
    preprocessor = DataPreprocessor(
        raw_data_dir=Path('data/raw/minute_aggs/by_symbol'),
        processed_data_dir=Path('data/processed'),
        cache_dir=Path('data/cache')
    )
    
    processed = preprocessor.process(symbol=symbol)
    feature_engine = FeatureEngine()
    data_with_features = feature_engine.add_all_features(processed)
    
    return data_with_features


def find_profitable_ma_params(symbol: str, n_trials: int = 300):
    """Find profitable Moving Average parameters"""
    logger.info(f"Searching for profitable MA parameters for {symbol}")
    
    # Load data
    data = load_symbol_data(symbol)
    logger.info(f"Loaded {len(data)} bars of data")
    
    # Create engine
    engine = VectorBTEngine()
    
    # Comprehensive parameter space
    param_space = {
        # Try faster MAs for trending markets
        'fast_period': {'type': 'int', 'low': 3, 'high': 30},
        'slow_period': {'type': 'int', 'low': 10, 'high': 100},
        'ma_type': {'type': 'categorical', 'choices': ['sma', 'ema']},
        # Wider stop loss range
        'stop_loss': {'type': 'float', 'low': 0.005, 'high': 0.10, 'step': 0.005},
        # Aggressive take profits
        'take_profit': {'type': 'float', 'low': 0.01, 'high': 0.20, 'step': 0.01},
        # Test with and without volume filter
        'volume_filter': {'type': 'categorical', 'choices': [True, False]}
    }
    
    # Run optimization
    result = engine.optimize_parameters_bayesian(
        strategy_class=MovingAverageCrossover,
        data=data,
        param_space=param_space,
        metric='sharpe_ratio',
        n_trials=n_trials,
        n_jobs=1
    )
    
    return result


def find_profitable_orb_params(symbol: str, n_trials: int = 300):
    """Find profitable ORB parameters"""
    logger.info(f"Searching for profitable ORB parameters for {symbol}")
    
    # Load data
    data = load_symbol_data(symbol)
    
    # Create engine
    engine = VectorBTEngine()
    
    # Comprehensive parameter space
    param_space = {
        # Very short opening ranges for volatile markets
        'opening_minutes': {'type': 'int', 'low': 1, 'high': 30, 'step': 1},
        'stop_type': {'type': 'categorical', 'choices': ['range', 'atr', 'fixed']},
        # Lower R multiples for higher win rate
        'profit_target_r': {'type': 'float', 'low': 0.5, 'high': 5.0, 'step': 0.25},
        'buffer_pct': {'type': 'float', 'low': 0.0, 'high': 0.02, 'step': 0.001},
        'atr_period': {'type': 'int', 'low': 5, 'high': 20},
        'atr_multiplier': {'type': 'float', 'low': 0.25, 'high': 2.0, 'step': 0.25},
        'exit_at_close': {'type': 'categorical', 'choices': [True, False]}
    }
    
    # Run optimization
    result = engine.optimize_parameters_bayesian(
        strategy_class=OpeningRangeBreakout,
        data=data,
        param_space=param_space,
        metric='sharpe_ratio',
        n_trials=n_trials,
        n_jobs=1
    )
    
    return result


def main():
    """Find profitable parameters for multiple symbols"""
    
    # Focus on symbols with strong trends or volatility
    symbols = ['NVDA', 'TSLA', 'AAPL', 'SPY', 'META']
    n_trials = 300  # Enough to explore the space well
    
    # Create output directory
    output_dir = Path('profitable_params')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    all_results = []
    profitable_strategies = []
    
    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimizing {symbol}")
        logger.info(f"{'='*60}")
        
        # Try MA strategy
        ma_result = find_profitable_ma_params(symbol, n_trials)
        result_entry = {
            'symbol': symbol,
            'strategy': 'MovingAverageCrossover',
            'best_sharpe': ma_result.best_metric,
            'best_params': ma_result.best_params,
            'n_trials': len(ma_result.results_df)
        }
        all_results.append(result_entry)
        
        logger.info(f"\n{symbol} MA Results:")
        logger.info(f"Best Sharpe: {ma_result.best_metric:.3f}")
        logger.info(f"Best params: {ma_result.best_params}")
        
        if ma_result.best_metric > 0:
            logger.info("🎉 PROFITABLE STRATEGY FOUND!")
            profitable_strategies.append(result_entry)
        
        # Try ORB strategy
        orb_result = find_profitable_orb_params(symbol, n_trials)
        result_entry = {
            'symbol': symbol,
            'strategy': 'OpeningRangeBreakout',
            'best_sharpe': orb_result.best_metric,
            'best_params': orb_result.best_params,
            'n_trials': len(orb_result.results_df)
        }
        all_results.append(result_entry)
        
        logger.info(f"\n{symbol} ORB Results:")
        logger.info(f"Best Sharpe: {orb_result.best_metric:.3f}")
        logger.info(f"Best params: {orb_result.best_params}")
        
        if orb_result.best_metric > 0:
            logger.info("🎉 PROFITABLE STRATEGY FOUND!")
            profitable_strategies.append(result_entry)
        
        # Save intermediate results
        with open(output_dir / f'results_{timestamp}.json', 'w') as f:
            json.dump({
                'all_results': all_results,
                'profitable_strategies': profitable_strategies
            }, f, indent=2)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*60)
    
    if profitable_strategies:
        logger.info(f"\n🎉 Found {len(profitable_strategies)} PROFITABLE strategies!")
        for strategy in profitable_strategies:
            logger.info(f"\n{strategy['symbol']} - {strategy['strategy']}")
            logger.info(f"Sharpe Ratio: {strategy['best_sharpe']:.3f}")
            logger.info(f"Parameters: {strategy['best_params']}")
    else:
        logger.info("\nNo profitable strategies found with current parameter ranges.")
        logger.info("Suggestions:")
        logger.info("1. Try different symbols (growth stocks, crypto)")
        logger.info("2. Expand parameter ranges")
        logger.info("3. Implement mean reversion strategies")
        logger.info("4. Add market regime filters")
    
    # Save final results
    with open(output_dir / f'final_results_{timestamp}.json', 'w') as f:
        json.dump({
            'all_results': all_results,
            'profitable_strategies': profitable_strategies,
            'summary': {
                'total_optimizations': len(all_results),
                'profitable_count': len(profitable_strategies),
                'best_overall': max(all_results, key=lambda x: x['best_sharpe']) if all_results else None
            }
        }, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}")
    
    # Print the single best strategy
    if all_results:
        best = max(all_results, key=lambda x: x['best_sharpe'])
        logger.info(f"\nBEST OVERALL STRATEGY:")
        logger.info(f"{best['symbol']} - {best['strategy']}")
        logger.info(f"Sharpe Ratio: {best['best_sharpe']:.3f}")
        logger.info(f"Parameters: {best['best_params']}")


if __name__ == "__main__":
    main()