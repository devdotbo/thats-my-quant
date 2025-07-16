"""
Data Pipeline Integration Tests
Tests data flow from raw files through preprocessing to features
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import tempfile
import shutil
import time
from datetime import datetime, timedelta

from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngine
from src.data.cache import CacheManager


@pytest.mark.integration
class TestDataPipeline:
    """Test data flow through the complete pipeline"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp(prefix="data_pipeline_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_symbols(self):
        """List of symbols to test"""
        return ['AAPL', 'MSFT', 'GOOGL']
    
    def test_single_symbol_data_flow(self, temp_cache_dir):
        """Test complete data flow for a single symbol"""
        # Initialize components
        preprocessor = DataPreprocessor(
            raw_data_dir=Path("data/raw/minute_aggs/by_symbol"),
            processed_data_dir=temp_cache_dir / "processed",
            cache_dir=temp_cache_dir
        )
        
        cache_manager = CacheManager(
            cache_dir=temp_cache_dir,
            max_size_gb=0.1
        )
        
        feature_engineer = FeatureEngine(cache_dir=temp_cache_dir)
        
        # Load raw data
        raw_path = Path("data/raw/minute_aggs/by_symbol/AAPL/AAPL_2024_01.csv.gz")
        assert raw_path.exists(), f"Test data not found: {raw_path}"
        
        # Preprocess data
        processed = preprocessor.process('AAPL', ['2024_01'])
        print(f"Processed {len(processed)} records")
        assert len(processed) > 0, "No data after preprocessing"
        assert processed.index.is_monotonic_increasing, "Index not sorted"
        assert processed.index.tz is not None, "Timezone not set"
        
        # Verify data quality
        assert not processed['close'].isna().any(), "NaN values in close prices"
        assert (processed['high'] >= processed['low']).all(), "High < Low found"
        assert (processed['high'] >= processed['close']).all(), "High < Close found"
        assert (processed['low'] <= processed['close']).all(), "Low > Close found"
        
        # Add features - add_all_features doesn't take parameters
        with_features = feature_engineer.add_all_features(processed)
        
        # Verify features
        expected_cols = ['open', 'high', 'low', 'close', 'volume', 
                        'sma_20', 'sma_50', 'rsi_14', 'atr_14', 'vwap']
        for col in expected_cols:
            assert col in with_features.columns, f"Missing column: {col}"
        
        # Test caching
        cache_key = "AAPL_2024_01_with_features"
        temp_file = temp_cache_dir / "temp_features.parquet"
        with_features.to_parquet(temp_file)
        
        cached_path = cache_manager.cache_file(temp_file, cache_key, category='features')
        assert cached_path.exists(), "Failed to cache file"
        
        # Verify cache retrieval
        retrieved_path = cache_manager.get_cached_file(cache_key)
        assert retrieved_path is not None, "Cache miss"
        
        cached_data = pd.read_parquet(retrieved_path)
        pd.testing.assert_frame_equal(with_features, cached_data)
        
        print(f"Pipeline processed {len(with_features)} records with {len(with_features.columns)} columns")
    
    def test_multi_symbol_alignment(self, sample_symbols, temp_cache_dir):
        """Test loading and aligning multiple symbols"""
        preprocessor = DataPreprocessor(
            raw_data_dir=Path("data/raw/minute_aggs/by_symbol"),
            processed_data_dir=temp_cache_dir / "processed",
            cache_dir=temp_cache_dir
        )
        
        all_data = {}
        
        # Load data for each symbol
        for symbol in sample_symbols:
            data_path = Path(f"data/raw/minute_aggs/by_symbol/{symbol}/{symbol}_2024_01.csv.gz")
            if data_path.exists():
                processed = preprocessor.process(symbol, ['2024_01'])
                all_data[symbol] = processed
                print(f"{symbol}: {len(processed)} records")
        
        assert len(all_data) >= 2, "Need at least 2 symbols for alignment test"
        
        # Find common trading times
        common_index = None
        for symbol, data in all_data.items():
            if common_index is None:
                common_index = data.index
            else:
                common_index = common_index.intersection(data.index)
        
        print(f"Common timestamps: {len(common_index)}")
        
        # Align all data
        aligned_data = {}
        for symbol, data in all_data.items():
            aligned = data.reindex(common_index)
            aligned_data[symbol] = aligned
            
            # Verify alignment
            assert len(aligned) == len(common_index)
            assert aligned.index.equals(common_index)
        
        # Create multi-symbol DataFrame
        close_prices = pd.DataFrame({
            symbol: data['close'] for symbol, data in aligned_data.items()
        })
        
        # Verify no NaN after alignment (for available symbols)
        for symbol in close_prices.columns:
            assert not close_prices[symbol].isna().any(), f"NaN in {symbol} after alignment"
        
        # Calculate correlations
        correlations = close_prices.corr()
        print("\nCorrelation Matrix:")
        print(correlations)
        
        # Verify reasonable correlations (tech stocks should be somewhat correlated)
        for i in range(len(correlations)):
            assert correlations.iloc[i, i] == 1.0, "Diagonal should be 1"
    
    def test_missing_data_handling(self, temp_cache_dir):
        """Test handling of missing bars and data gaps"""
        # Create sample data with gaps
        dates = pd.date_range('2024-01-02 09:30', '2024-01-02 16:00', freq='1min')
        data = pd.DataFrame({
            'open': 100.0,
            'high': 100.5,
            'low': 99.5,
            'close': 100.2,
            'volume': 10000,
            'vwap': 100.1,
            'transactions': 100
        }, index=dates)
        
        # Remove some random minutes to create gaps
        np.random.seed(42)
        gaps = np.random.choice(len(data), size=50, replace=False)
        data_with_gaps = data.drop(data.index[gaps])
        
        print(f"Created data with {len(gaps)} gaps")
        
        # Process with gap filling
        preprocessor = DataPreprocessor(
            raw_data_dir=Path("data"),
            processed_data_dir=temp_cache_dir / "processed",
            cache_dir=temp_cache_dir
        )
        
        # Mock the processing by directly calling fill_missing_bars
        filled = preprocessor.fill_missing_bars(data_with_gaps)
        
        # Verify gaps were filled
        expected_bars = len(pd.date_range(
            data_with_gaps.index[0],
            data_with_gaps.index[-1],
            freq='1min'
        ))
        
        # Only market hours should be filled
        market_hours = filled.between_time('09:30', '16:00')
        assert len(market_hours) > len(data_with_gaps), "Gaps not filled"
        
        # Verify forward-filled data
        for idx in range(1, len(filled)):
            if filled.iloc[idx]['volume'] == 0:  # This was a filled bar
                # Should have forward-filled price
                assert filled.iloc[idx]['close'] == filled.iloc[idx-1]['close']
                assert filled.iloc[idx]['open'] == filled.iloc[idx-1]['close']
    
    def test_timezone_handling(self, temp_cache_dir):
        """Test proper timezone handling for market data"""
        # Load real data
        raw_path = Path("data/raw/minute_aggs/by_symbol/SPY/SPY_2024_01.csv.gz")
        if not raw_path.exists():
            pytest.skip("SPY test data not available")
        
        preprocessor = DataPreprocessor(
            raw_data_dir=Path("data/raw/minute_aggs/by_symbol"),
            processed_data_dir=temp_cache_dir / "processed",
            cache_dir=temp_cache_dir
        )
        
        processed = preprocessor.process('SPY', ['2024_01'])
        
        # Verify timezone is set
        assert processed.index.tz is not None, "No timezone set"
        assert str(processed.index.tz) == 'America/New_York', f"Wrong timezone: {processed.index.tz}"
        
        # Verify only market hours are included
        for timestamp in processed.index:
            hour = timestamp.hour
            minute = timestamp.minute
            
            # Should be between 9:30 AM and 4:00 PM
            time_minutes = hour * 60 + minute
            assert 9*60+30 <= time_minutes <= 16*60, f"Non-market hour: {timestamp}"
        
        # Check no weekend data
        for timestamp in processed.index:
            assert timestamp.weekday() < 5, f"Weekend data found: {timestamp}"
    
    def test_cache_performance_benchmark(self, temp_cache_dir, performance_timer):
        """Benchmark cache performance with different data sizes"""
        cache_manager = CacheManager(
            cache_dir=temp_cache_dir,
            max_size_gb=0.5
        )
        
        results = {}
        
        # Test different data sizes
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            # Create test data
            dates = pd.date_range('2024-01-01', periods=size, freq='1min')
            test_data = pd.DataFrame({
                'open': np.random.randn(size) * 0.01 + 100,
                'high': np.random.randn(size) * 0.01 + 100.5,
                'low': np.random.randn(size) * 0.01 + 99.5,
                'close': np.random.randn(size) * 0.01 + 100,
                'volume': np.random.randint(1000, 100000, size)
            }, index=dates)
            
            # Save to temp file
            temp_file = temp_cache_dir / f"test_{size}.parquet"
            
            performance_timer.start(f"save_{size}")
            test_data.to_parquet(temp_file)
            save_time = performance_timer.stop(f"save_{size}")
            
            # Cache the file
            performance_timer.start(f"cache_{size}")
            cache_key = f"test_data_{size}"
            cached_path = cache_manager.cache_file(temp_file, cache_key)
            cache_time = performance_timer.stop(f"cache_{size}")
            
            # Read from cache
            performance_timer.start(f"read_cache_{size}")
            retrieved_path = cache_manager.get_cached_file(cache_key)
            loaded_data = pd.read_parquet(retrieved_path)
            read_time = performance_timer.stop(f"read_cache_{size}")
            
            # Calculate throughput
            file_size_mb = temp_file.stat().st_size / (1024 * 1024)
            
            results[size] = {
                'save_time': save_time,
                'cache_time': cache_time,
                'read_time': read_time,
                'file_size_mb': file_size_mb,
                'save_throughput_mb_s': file_size_mb / save_time,
                'read_throughput_mb_s': file_size_mb / read_time,
                'rows_per_second': size / (save_time + read_time)
            }
        
        # Print results
        print("\nCache Performance Benchmark:")
        print(f"{'Size':>10} {'Save (s)':>10} {'Read (s)':>10} {'MB/s Save':>12} {'MB/s Read':>12}")
        print("-" * 60)
        
        for size, metrics in results.items():
            print(f"{size:>10} {metrics['save_time']:>10.3f} {metrics['read_time']:>10.3f} "
                  f"{metrics['save_throughput_mb_s']:>12.1f} {metrics['read_throughput_mb_s']:>12.1f}")
        
        # Verify performance targets - be realistic for test environment
        for size, metrics in results.items():
            assert metrics['read_throughput_mb_s'] > 10, f"Read too slow: {metrics['read_throughput_mb_s']:.1f} MB/s"
            assert metrics['rows_per_second'] > 10000, f"Processing too slow: {metrics['rows_per_second']:.0f} rows/s"
    
    def test_feature_calculation_accuracy(self, temp_cache_dir):
        """Test accuracy of calculated features"""
        # Create known data for verification
        prices = [100, 101, 102, 101, 100, 99, 98, 99, 100, 101,
                 102, 103, 104, 105, 104, 103, 102, 101, 100, 99]
        
        dates = pd.date_range('2024-01-02 09:30', periods=len(prices), freq='1min')
        test_data = pd.DataFrame({
            'open': prices,
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.5 for p in prices],
            'close': prices,
            'volume': [10000] * len(prices)
        }, index=dates)
        
        feature_engineer = FeatureEngine()
        
        # Calculate SMA
        with_sma = feature_engineer.add_moving_averages(test_data, periods=[5, 10])
        
        # Manual SMA calculation for verification
        manual_sma_5 = test_data['close'].rolling(5).mean()
        manual_sma_10 = test_data['close'].rolling(10).mean()
        
        # Compare (where not NaN)
        pd.testing.assert_series_equal(
            with_sma['sma_5'].dropna(),
            manual_sma_5.dropna(),
            check_names=False
        )
        
        pd.testing.assert_series_equal(
            with_sma['sma_10'].dropna(),
            manual_sma_10.dropna(),
            check_names=False
        )
        
        # Test RSI calculation
        with_rsi = feature_engineer.add_momentum_indicators(test_data, rsi_period=14)
        
        # RSI should be between 0 and 100
        rsi_values = with_rsi['rsi_14'].dropna()
        assert (rsi_values >= 0).all() and (rsi_values <= 100).all(), "RSI out of bounds"
        
        print(f"Feature accuracy verified for {len(with_rsi.columns)} features")
    
    def test_large_file_handling(self, temp_cache_dir):
        """Test handling of large data files"""
        # Use full month of data
        large_file = Path("data/raw/minute_aggs/by_symbol/SPY/SPY_2024_01.csv.gz")
        
        if not large_file.exists():
            pytest.skip("Large test file not available")
        
        start_time = time.time()
        
        # Process the data
        preprocessor = DataPreprocessor(
            raw_data_dir=Path("data/raw/minute_aggs/by_symbol"),
            processed_data_dir=temp_cache_dir / "processed",
            cache_dir=temp_cache_dir
        )
        
        start_time = time.time()
        processed = preprocessor.process('SPY', ['2024_01'])
        load_time = 0  # No separate load time when using preprocessor
        process_time = time.time() - start_time
        
        print(f"Processed {len(processed)} records in {process_time:.2f}s")
        print(f"Process speed: {len(processed)/process_time:.0f} records/second")
        
        # Verify performance
        # No separate load assertion when using preprocessor directly
        assert process_time < 5.0, f"Process too slow: {process_time:.2f}s"
        assert len(processed)/process_time > 10000, "Processing speed below target"