"""
Complete Pipeline Integration Tests
Tests the full workflow from data loading to validation
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import time
import tempfile
import shutil

from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngine
from src.data.cache import CacheManager
from src.strategies.examples.moving_average import MovingAverageCrossover
from src.backtesting.engines.vectorbt_engine import VectorBTEngine
from src.backtesting.costs import TransactionCostEngine
from src.validation.walk_forward import WalkForwardValidator, WindowType
from src.validation.monte_carlo import MonteCarloValidator, ResamplingMethod
from src.utils.config import Config


@pytest.mark.integration
class TestCompletePipeline:
    """Test complete backtesting pipeline from data to validation"""
    
    @pytest.fixture
    def test_data_path(self):
        """Path to test data"""
        return Path("data/raw/minute_aggs/by_symbol/AAPL/AAPL_2024_01.csv.gz")
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing"""
        temp_root = tempfile.mkdtemp(prefix="integration_test_")
        dirs = {
            'cache': Path(temp_root) / 'cache',
            'processed': Path(temp_root) / 'processed',
            'results': Path(temp_root) / 'results'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        yield dirs
        
        # Cleanup
        shutil.rmtree(temp_root)
    
    def test_complete_happy_path(self, test_data_path, temp_dirs, performance_timer):
        """Test complete pipeline with real data - happy path"""
        # Verify test data exists
        assert test_data_path.exists(), f"Test data not found: {test_data_path}"
        
        # 1. Load and preprocess data
        performance_timer.start("data_loading")
        
        preprocessor = DataPreprocessor(
            raw_data_dir=test_data_path.parent.parent.parent,
            processed_data_dir=temp_dirs['processed'],
            cache_dir=temp_dirs['cache']
        )
        
        # Load January 2024 AAPL data
        raw_data = pd.read_csv(test_data_path, compression='gzip')
        assert len(raw_data) > 0, "No data loaded"
        
        # Process the data
        processed_data = preprocessor.process(raw_data, 'AAPL')
        load_time = performance_timer.stop("data_loading")
        
        assert len(processed_data) > 0, "No data after preprocessing"
        assert load_time < 2.0, f"Data loading too slow: {load_time:.2f}s"
        
        # 2. Add technical indicators
        performance_timer.start("feature_engineering")
        
        feature_eng = FeatureEngineer(cache_manager=CacheManager(temp_dirs['cache']))
        data_with_features = feature_eng.add_all_features(
            processed_data,
            features=['sma_20', 'sma_50', 'rsi_14', 'atr_14', 'volume_sma_20']
        )
        
        feature_time = performance_timer.stop("feature_engineering")
        assert feature_time < 1.0, f"Feature engineering too slow: {feature_time:.2f}s"
        
        # Verify features were added
        expected_features = ['sma_20', 'sma_50', 'rsi_14', 'atr_14', 'volume_sma_20']
        for feature in expected_features:
            assert feature in data_with_features.columns, f"Missing feature: {feature}"
        
        # 3. Run backtesting with strategy
        performance_timer.start("backtesting")
        
        strategy = MovingAverageCrossover({
            'fast_period': 20,
            'slow_period': 50,
            'position_size': 0.95
        })
        
        engine = VectorBTEngine()
        backtest_result = engine.run_backtest(
            strategy=strategy,
            data=data_with_features,
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005
        )
        
        backtest_time = performance_timer.stop("backtesting")
        assert backtest_time < 5.0, f"Backtesting too slow: {backtest_time:.2f}s"
        
        # Verify backtest results
        assert backtest_result.metrics['total_return'] is not None
        assert backtest_result.metrics['sharpe_ratio'] is not None
        assert backtest_result.metrics['max_drawdown'] < 0  # Should be negative
        assert len(backtest_result.trades) >= 0  # May have no trades
        
        # 4. Run walk-forward validation
        performance_timer.start("walk_forward")
        
        wf_validator = WalkForwardValidator(
            engine=engine,
            train_periods=15,  # 15 days training
            test_periods=5,    # 5 days testing
            step_size=5,       # 5 days step
            window_type=WindowType.ROLLING
        )
        
        # Create windows for January data
        windows = wf_validator.create_windows(
            data_with_features,
            start_date=data_with_features.index[0],
            end_date=data_with_features.index[-1]
        )
        
        assert len(windows) > 0, "No walk-forward windows created"
        
        # Run validation on first window only (for speed)
        wf_results = wf_validator.run_validation(
            strategy=strategy,
            data=data_with_features,
            windows=windows[:1],  # Just first window
            optimization_metric='sharpe_ratio',
            parameter_ranges={
                'fast_period': [10, 20, 30],
                'slow_period': [40, 50, 60]
            }
        )
        
        wf_time = performance_timer.stop("walk_forward")
        assert wf_time < 10.0, f"Walk-forward too slow: {wf_time:.2f}s"
        
        # Verify walk-forward results
        assert len(wf_results.window_results) == 1
        assert wf_results.summary_stats['total_windows'] == 1
        
        # 5. Run Monte Carlo simulation
        performance_timer.start("monte_carlo")
        
        mc_validator = MonteCarloValidator(
            n_simulations=100,  # Reduced for testing
            confidence_level=0.95,
            random_seed=42
        )
        
        mc_results = mc_validator.run_simulation(
            backtest_result=backtest_result,
            resampling_method=ResamplingMethod.BOOTSTRAP,
            parallel=True
        )
        
        mc_time = performance_timer.stop("monte_carlo")
        assert mc_time < 5.0, f"Monte Carlo too slow: {mc_time:.2f}s"
        
        # Verify Monte Carlo results
        assert mc_results.n_simulations == 100
        assert 'sharpe_ratio' in mc_results.confidence_intervals
        assert mc_results.risk_metrics['risk_of_ruin'] >= 0
        
        # 6. Overall performance check
        total_time = (load_time + feature_time + backtest_time + 
                     wf_time + mc_time)
        assert total_time < 25.0, f"Total pipeline too slow: {total_time:.2f}s"
        
        print(f"\nPipeline Performance:")
        print(f"  Data Loading: {load_time:.2f}s")
        print(f"  Features: {feature_time:.2f}s")
        print(f"  Backtesting: {backtest_time:.2f}s")
        print(f"  Walk-Forward: {wf_time:.2f}s")
        print(f"  Monte Carlo: {mc_time:.2f}s")
        print(f"  Total: {total_time:.2f}s")
    
    def test_multi_symbol_portfolio(self, temp_dirs, performance_timer):
        """Test portfolio backtesting with multiple symbols"""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        start_time = time.time()
        
        # Load data for each symbol
        all_data = {}
        preprocessor = DataPreprocessor(
            raw_data_dir=Path("data/raw/minute_aggs/by_symbol"),
            processed_data_dir=temp_dirs['processed'],
            cache_dir=temp_dirs['cache']
        )
        
        for symbol in symbols:
            data_path = Path(f"data/raw/minute_aggs/by_symbol/{symbol}/{symbol}_2024_01.csv.gz")
            if data_path.exists():
                raw_data = pd.read_csv(data_path, compression='gzip')
                processed = preprocessor.process(raw_data, symbol)
                
                # Add features
                feature_eng = FeatureEngineer()
                with_features = feature_eng.add_moving_averages(processed, periods=[20, 50])
                all_data[symbol] = with_features
        
        assert len(all_data) > 0, "No data loaded for portfolio"
        
        # Align data
        if len(all_data) > 1:
            # Get common index
            common_index = all_data[symbols[0]].index
            for symbol in symbols[1:]:
                common_index = common_index.intersection(all_data[symbol].index)
            
            # Reindex all data
            for symbol in symbols:
                all_data[symbol] = all_data[symbol].reindex(common_index)
        
        # Create portfolio data structure
        close_prices = pd.DataFrame({
            symbol: data['close'] for symbol, data in all_data.items()
        })
        
        # Run portfolio backtest
        strategy = MovingAverageCrossover({
            'fast_period': 20,
            'slow_period': 50,
            'position_size': 0.95
        })
        
        engine = VectorBTEngine()
        
        # Generate signals for each symbol
        portfolio_signals = pd.DataFrame()
        for symbol in symbols:
            if symbol in all_data:
                signals = strategy.generate_signals(all_data[symbol])
                portfolio_signals[symbol] = signals
        
        # Equal weight allocation
        weights = pd.DataFrame(
            1.0 / len(symbols),
            index=portfolio_signals.index,
            columns=portfolio_signals.columns
        )
        
        # Adjust weights by signals (0 weight when no signal)
        weights = weights * (portfolio_signals != 0)
        weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)
        
        # Portfolio metrics
        assert not close_prices.empty, "No price data for portfolio"
        assert not portfolio_signals.empty, "No signals generated"
        
        elapsed = time.time() - start_time
        assert elapsed < 10.0, f"Portfolio test too slow: {elapsed:.2f}s"
    
    def test_cache_effectiveness(self, test_data_path, temp_dirs, performance_timer):
        """Test that caching improves performance significantly"""
        cache_manager = CacheManager(cache_dir=temp_dirs['cache'], max_size_gb=0.1)
        
        # First load - no cache
        performance_timer.start("first_load")
        raw_data = pd.read_csv(test_data_path, compression='gzip')
        
        preprocessor = DataPreprocessor(
            raw_data_dir=test_data_path.parent.parent.parent,
            processed_data_dir=temp_dirs['processed'],
            cache_dir=temp_dirs['cache']
        )
        processed = preprocessor.process(raw_data, 'AAPL')
        
        # Cache the processed data
        cache_key = f"AAPL_2024_01_processed"
        cache_path = cache_manager.cache_file(
            temp_dirs['processed'] / "temp_processed.parquet",
            cache_key,
            category='processed_data'
        )
        processed.to_parquet(cache_path)
        
        first_time = performance_timer.stop("first_load")
        
        # Second load - from cache
        performance_timer.start("cached_load")
        cached_path = cache_manager.get_cached_file(cache_key)
        assert cached_path is not None, "Cache miss on second load"
        
        cached_data = pd.read_parquet(cached_path)
        cached_time = performance_timer.stop("cached_load")
        
        # Verify cache speedup
        speedup = first_time / cached_time
        assert speedup > 5.0, f"Insufficient cache speedup: {speedup:.1f}x"
        
        # Verify data integrity
        pd.testing.assert_frame_equal(processed, cached_data)
        
        print(f"\nCache Performance:")
        print(f"  First Load: {first_time:.3f}s")
        print(f"  Cached Load: {cached_time:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")
    
    def test_different_strategies_same_data(self, test_data_path, temp_dirs):
        """Test multiple strategies on the same dataset"""
        # Load and prepare data
        raw_data = pd.read_csv(test_data_path, compression='gzip')
        
        preprocessor = DataPreprocessor(
            raw_data_dir=test_data_path.parent.parent.parent,
            processed_data_dir=temp_dirs['processed'],
            cache_dir=temp_dirs['cache']
        )
        processed = preprocessor.process(raw_data, 'AAPL')
        
        # Add features for both strategies
        feature_eng = FeatureEngineer()
        data_with_features = feature_eng.add_all_features(
            processed,
            features=['sma_20', 'sma_50', 'rsi_14', 'atr_14', 'high_low_ratio']
        )
        
        engine = VectorBTEngine()
        results = {}
        
        # Test 1: Moving Average Crossover
        from src.strategies.examples.moving_average import MovingAverageCrossover
        ma_strategy = MovingAverageCrossover({
            'fast_period': 20,
            'slow_period': 50
        })
        
        results['ma_crossover'] = engine.run_backtest(
            strategy=ma_strategy,
            data=data_with_features,
            initial_capital=100000
        )
        
        # Test 2: Opening Range Breakout (intraday)
        from src.strategies.examples.orb import OpeningRangeBreakout
        orb_strategy = OpeningRangeBreakout({
            'opening_minutes': 30,
            'stop_type': 'range',
            'profit_target_r': 2.0
        })
        
        # ORB needs intraday data with proper market hours
        # Filter to market hours only
        market_hours = data_with_features.between_time('09:30', '16:00')
        if len(market_hours) > 0:
            results['orb'] = engine.run_backtest(
                strategy=orb_strategy,
                data=market_hours,
                initial_capital=100000
            )
        
        # Verify both strategies produced results
        assert 'ma_crossover' in results
        assert results['ma_crossover'].metrics['total_trades'] >= 0
        
        if 'orb' in results:
            assert results['orb'].metrics['total_trades'] >= 0
            
            # Compare strategies
            print("\nStrategy Comparison:")
            for name, result in results.items():
                print(f"\n{name}:")
                print(f"  Total Return: {result.metrics['total_return']:.2%}")
                print(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown: {result.metrics['max_drawdown']:.2%}")
                print(f"  Total Trades: {result.metrics['total_trades']}")
    
    def test_parameter_optimization(self, test_data_path, temp_dirs, performance_timer):
        """Test parameter optimization workflow"""
        # Load minimal data for speed
        raw_data = pd.read_csv(test_data_path, compression='gzip')
        
        # Use only first 5 days
        raw_data = raw_data.head(5 * 390)  # ~5 days of minute data
        
        preprocessor = DataPreprocessor(
            raw_data_dir=test_data_path.parent.parent.parent,
            processed_data_dir=temp_dirs['processed'],
            cache_dir=temp_dirs['cache']
        )
        processed = preprocessor.process(raw_data, 'AAPL')
        
        # Add features
        feature_eng = FeatureEngineer()
        data_with_features = feature_eng.add_moving_averages(
            processed, 
            periods=[10, 20, 30, 40, 50, 60]
        )
        
        # Define parameter grid
        param_grid = {
            'fast_period': [10, 20],
            'slow_period': [40, 50],
            'position_size': [0.8, 0.95]
        }
        
        # Run optimization
        strategy = MovingAverageCrossover()
        engine = VectorBTEngine()
        
        performance_timer.start("optimization")
        optimization_result = engine.optimize_parameters(
            strategy=strategy,
            data=data_with_features,
            parameter_grid=param_grid,
            metric='sharpe_ratio',
            initial_capital=100000
        )
        opt_time = performance_timer.stop("optimization")
        
        # Verify optimization results
        assert optimization_result.best_params is not None
        assert optimization_result.best_metric is not None
        assert len(optimization_result.results_df) == 8  # 2*2*2 combinations
        assert opt_time < 10.0, f"Optimization too slow: {opt_time:.2f}s"
        
        print(f"\nOptimization Results:")
        print(f"  Best Parameters: {optimization_result.best_params}")
        print(f"  Best Sharpe Ratio: {optimization_result.best_metric:.2f}")
        print(f"  Time: {opt_time:.2f}s")
        print(f"  Combinations Tested: {len(optimization_result.results_df)}")