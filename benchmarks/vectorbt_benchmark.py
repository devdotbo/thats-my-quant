"""
VectorBT Performance Benchmark
Tests VectorBT backtesting performance with realistic market data
"""

import numpy as np
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# VectorBT will be imported after installation check
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    print("VectorBT not installed. Please install with: pip install vectorbt")


class VectorBTBenchmark:
    """Benchmark VectorBT performance for backtesting operations"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'vectorbt_available': VECTORBT_AVAILABLE,
            'vectorbt_version': vbt.__version__ if VECTORBT_AVAILABLE else None,
            'benchmarks': {}
        }
    
    def generate_market_data(self, n_symbols: int = 10, n_days: int = 252, 
                           freq: str = 'D') -> pd.DataFrame:
        """Generate realistic OHLCV market data"""
        if freq == 'D':
            periods = n_days
            dates = pd.date_range('2022-01-01', periods=periods, freq='B')  # Business days
        elif freq == '1min':
            # 390 minutes per trading day (9:30 AM - 4:00 PM)
            periods = n_days * 390
            dates = pd.date_range('2022-01-01 09:30:00', periods=periods, freq='1min')
            # Filter out non-trading hours
            dates = dates[(dates.hour >= 9) & (dates.hour < 16)]
            dates = dates[~((dates.hour == 9) & (dates.minute < 30))]
            dates = dates[:periods]  # Ensure we have the right number of periods
        else:
            raise ValueError(f"Unsupported frequency: {freq}")
        
        data = {}
        
        for i in range(n_symbols):
            symbol = f'STOCK_{i}'
            
            # Generate realistic price movement
            returns = np.random.randn(len(dates)) * 0.02  # 2% daily volatility
            price = 100 * np.exp(np.cumsum(returns))
            
            # Generate OHLCV data
            noise = np.random.randn(len(dates))
            data[('Open', symbol)] = price * (1 + noise * 0.001)
            data[('High', symbol)] = price * (1 + np.abs(noise) * 0.005)
            data[('Low', symbol)] = price * (1 - np.abs(noise) * 0.005)
            data[('Close', symbol)] = price
            data[('Volume', symbol)] = np.random.randint(1000000, 10000000, len(dates))
        
        df = pd.DataFrame(data, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df
    
    def benchmark_simple_ma_crossover(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """Benchmark a simple moving average crossover strategy"""
        if not VECTORBT_AVAILABLE:
            return 0.0, {}
        
        close = data['Close']
        
        start = time.perf_counter()
        
        # Calculate moving averages
        fast_ma = vbt.MA.run(close, 10, short_name='fast')
        slow_ma = vbt.MA.run(close, 30, short_name='slow')
        
        # Generate signals
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        # Run backtest
        pf = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.001,
            freq='D'
        )
        
        # Get performance metrics
        stats = pf.stats()
        
        elapsed = time.perf_counter() - start
        
        metrics = {
            'total_return': float(stats.get('Total Return [%]', 0)),
            'sharpe_ratio': float(stats.get('Sharpe Ratio', 0)),
            'max_drawdown': float(stats.get('Max Drawdown [%]', 0)),
            'win_rate': float(stats.get('Win Rate [%]', 0)),
            'trades': int(stats.get('Total Trades', 0))
        }
        
        return elapsed, metrics
    
    def benchmark_parameter_optimization(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """Benchmark parameter optimization performance"""
        if not VECTORBT_AVAILABLE:
            return 0.0, {}
        
        close = data['Close'].iloc[:, :5]  # Use only 5 symbols for optimization
        
        start = time.perf_counter()
        
        # Define parameter ranges
        fast_periods = np.arange(10, 50, 5)
        slow_periods = np.arange(50, 200, 10)
        
        # Run optimization
        fast_ma = vbt.MA.run(close, fast_periods, short_name='fast')
        slow_ma = vbt.MA.run(close, slow_periods, short_name='slow')
        
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        pf = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.001,
            freq='D'
        )
        
        # Find best parameters
        returns = pf.total_return()
        best_idx = returns.idxmax()
        best_return = float(returns.max())
        
        elapsed = time.perf_counter() - start
        
        metrics = {
            'total_combinations': len(fast_periods) * len(slow_periods),
            'best_params': str(best_idx),
            'best_return': best_return,
            'optimization_time': elapsed
        }
        
        return elapsed, metrics
    
    def benchmark_vectorized_indicators(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """Benchmark calculation of multiple technical indicators"""
        if not VECTORBT_AVAILABLE:
            return 0.0, {}
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        start = time.perf_counter()
        
        # Calculate various indicators
        indicators = {}
        
        # Moving averages
        indicators['MA_20'] = vbt.MA.run(close, 20).ma
        indicators['MA_50'] = vbt.MA.run(close, 50).ma
        indicators['MA_200'] = vbt.MA.run(close, 200).ma
        
        # RSI
        indicators['RSI'] = vbt.RSI.run(close, 14).rsi
        
        # Bollinger Bands
        bb = vbt.BBANDS.run(close, 20, 2)
        indicators['BB_upper'] = bb.upper
        indicators['BB_middle'] = bb.middle
        indicators['BB_lower'] = bb.lower
        
        # MACD
        macd = vbt.MACD.run(close, 12, 26, 9)
        indicators['MACD'] = macd.macd
        indicators['MACD_signal'] = macd.signal
        
        # ATR
        indicators['ATR'] = vbt.ATR.run(high, low, close, 14).atr
        
        elapsed = time.perf_counter() - start
        
        metrics = {
            'indicators_calculated': len(indicators),
            'total_values': sum(ind.size for ind in indicators.values()),
            'calc_time': elapsed
        }
        
        return elapsed, metrics
    
    def benchmark_portfolio_simulation(self, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """Benchmark a more complex portfolio simulation"""
        if not VECTORBT_AVAILABLE:
            return 0.0, {}
        
        close = data['Close']
        
        start = time.perf_counter()
        
        # Generate more complex signals using RSI
        rsi = vbt.RSI.run(close, 14).rsi
        entries = rsi < 30  # Oversold
        exits = rsi > 70    # Overbought
        
        # Run portfolio simulation with more realistic parameters
        pf = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.001,
            size=0.1,  # 10% position size
            size_type='targetpercent',
            group_by=True,  # Treat as single portfolio
            cash_sharing=True,
            call_seq='auto',  # Automatic call sequencing
            freq='D'
        )
        
        # Calculate various metrics
        metrics_dict = {
            'total_return': float(pf.total_return()),
            'sharpe_ratio': float(pf.sharpe_ratio()),
            'sortino_ratio': float(pf.sortino_ratio()),
            'calmar_ratio': float(pf.calmar_ratio()),
            'max_drawdown': float(pf.max_drawdown()),
            'trades': int(pf.count()),
            'win_rate': float(pf.win_rate()) if pf.count() > 0 else 0.0,
            'expectancy': float(pf.expectancy()) if pf.count() > 0 else 0.0
        }
        
        elapsed = time.perf_counter() - start
        
        return elapsed, metrics_dict
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run complete VectorBT benchmark suite"""
        print("="*60)
        print("VectorBT Performance Benchmark")
        print("="*60)
        
        if not VECTORBT_AVAILABLE:
            print("ERROR: VectorBT not installed!")
            print("Please install with: pip install vectorbt")
            return self.results
        
        print(f"VectorBT Version: {vbt.__version__}")
        print("="*60)
        
        # Test different data sizes
        test_configs = [
            ('10_symbols_1y_daily', 10, 252, 'D'),
            ('10_symbols_5y_daily', 10, 1260, 'D'),
            ('100_symbols_1y_daily', 100, 252, 'D'),
            ('10_symbols_1y_minute', 10, 252, '1min'),
        ]
        
        for config_name, n_symbols, n_days, freq in test_configs:
            print(f"\nTesting configuration: {config_name}")
            print(f"  Symbols: {n_symbols}, Days: {n_days}, Frequency: {freq}")
            
            # Generate data
            print("  Generating market data...")
            data_start = time.perf_counter()
            data = self.generate_market_data(n_symbols, n_days, freq)
            data_time = time.perf_counter() - data_start
            print(f"  Data generation: {data_time:.3f}s")
            
            config_results = {
                'data_shape': str(data.shape),
                'data_generation_time': data_time,
                'tests': {}
            }
            
            # Skip minute data for some tests due to size
            if freq == '1min' and n_symbols > 10:
                print("  Skipping minute data for large dataset")
                continue
            
            # Run benchmarks
            print("  Running simple MA crossover...")
            ma_time, ma_metrics = self.benchmark_simple_ma_crossover(data)
            config_results['tests']['ma_crossover'] = {
                'time': ma_time,
                'metrics': ma_metrics
            }
            print(f"    Time: {ma_time:.3f}s")
            
            if n_symbols <= 10 and n_days <= 252:  # Only optimize on smaller datasets
                print("  Running parameter optimization...")
                opt_time, opt_metrics = self.benchmark_parameter_optimization(data)
                config_results['tests']['optimization'] = {
                    'time': opt_time,
                    'metrics': opt_metrics
                }
                print(f"    Time: {opt_time:.3f}s")
                print(f"    Combinations: {opt_metrics.get('total_combinations', 0)}")
            
            print("  Calculating technical indicators...")
            ind_time, ind_metrics = self.benchmark_vectorized_indicators(data)
            config_results['tests']['indicators'] = {
                'time': ind_time,
                'metrics': ind_metrics
            }
            print(f"    Time: {ind_time:.3f}s")
            
            print("  Running portfolio simulation...")
            port_time, port_metrics = self.benchmark_portfolio_simulation(data)
            config_results['tests']['portfolio'] = {
                'time': port_time,
                'metrics': port_metrics
            }
            print(f"    Time: {port_time:.3f}s")
            
            self.results['benchmarks'][config_name] = config_results
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save benchmark results to file"""
        results_dir = Path('benchmarks/results')
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = results_dir / f'vectorbt_benchmark_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filename}")
    
    def _print_summary(self):
        """Print performance summary and recommendations"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        # Check critical benchmark: 1 year minute data
        minute_test = self.results['benchmarks'].get('10_symbols_1y_minute', {})
        if minute_test:
            ma_time = minute_test.get('tests', {}).get('ma_crossover', {}).get('time', float('inf'))
            print(f"\nCritical Test - 1 Year Minute Data (10 symbols):")
            print(f"  MA Crossover Backtest: {ma_time:.3f}s")
            
            if ma_time < 5.0:
                print(f"  ✅ PASSED: Meets <5s target ({ma_time:.3f}s)")
            else:
                print(f"  ❌ FAILED: Exceeds 5s target ({ma_time:.3f}s)")
        
        # Daily data performance
        daily_test = self.results['benchmarks'].get('10_symbols_1y_daily', {})
        if daily_test:
            tests = daily_test.get('tests', {})
            print(f"\nDaily Data Performance (1 year, 10 symbols):")
            for test_name, test_data in tests.items():
                print(f"  {test_name}: {test_data.get('time', 0):.3f}s")
        
        # Optimization performance
        opt_test = self.results['benchmarks'].get('10_symbols_1y_daily', {}).get('tests', {}).get('optimization', {})
        if opt_test:
            opt_metrics = opt_test.get('metrics', {})
            print(f"\nOptimization Performance:")
            print(f"  Combinations tested: {opt_metrics.get('total_combinations', 0)}")
            print(f"  Time: {opt_test.get('time', 0):.3f}s")
            print(f"  Speed: {opt_metrics.get('total_combinations', 0) / opt_test.get('time', 1):.0f} combinations/sec")
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        # Performance recommendations
        if minute_test and ma_time < 5.0:
            print("✅ VectorBT performance meets requirements for minute data")
            print("✅ Recommended as primary backtesting engine")
        else:
            print("⚠️  VectorBT performance may not meet minute data requirements")
            print("   Consider optimizing data handling or using daily data")
        
        print("\n✅ VectorBT successfully installed and tested")
        print("✅ Ready for strategy development and backtesting")


def main():
    """Run VectorBT benchmarks"""
    if not VECTORBT_AVAILABLE:
        print("ERROR: VectorBT not installed!")
        print("Please run: pip install vectorbt")
        return
    
    benchmark = VectorBTBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Final status
    minute_test = results['benchmarks'].get('10_symbols_1y_minute', {})
    if minute_test:
        ma_time = minute_test.get('tests', {}).get('ma_crossover', {}).get('time', float('inf'))
        if ma_time < 5.0:
            print("\n✅ All critical performance targets met!")
        else:
            print("\n⚠️  Performance optimization may be needed")


if __name__ == "__main__":
    main()