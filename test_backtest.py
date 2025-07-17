#!/usr/bin/env python3
"""
Test backtesting script with fewer symbols
"""
import sys
sys.path.append('.')

from backtest_all_symbols import *

def test_main():
    """Test with just 2 symbols"""
    
    print("="*80)
    print("TEST BACKTESTING - 2 SYMBOLS")
    print("="*80)
    
    # Configuration
    initial_capital = 100000
    output_dir = Path('backtest_results') / 'test'
    
    # Use just 2 symbols for testing
    symbols = ['AAPL', 'SPY']
    print(f"\nTesting with {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Create just a few strategies
    strategies = {
        'MA_Fast_10_30': MovingAverageCrossover({'fast_period': 10, 'slow_period': 30, 'ma_type': 'ema'}),
        'ORB_5min_3R': OpeningRangeBreakout({'range_minutes': 5, 'profit_target_r': 3.0, 'stop_type': 'range'})
    }
    print(f"\nCreated {len(strategies)} strategy variations")
    
    # Run backtests
    print(f"\nRunning {len(symbols) * len(strategies)} backtests...")
    results = run_all_backtests(symbols, strategies, initial_capital, max_workers=1)
    
    # Analyze results
    print("\nAnalyzing results...")
    summary_df = analyze_results(results)
    
    # Generate report
    print("\nGenerating reports...")
    generate_report(results, summary_df, output_dir)
    
    print(f"\n{'='*80}")
    print(f"TEST COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_main()