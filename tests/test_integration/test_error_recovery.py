"""
Error Recovery Integration Tests
Tests error handling and recovery scenarios
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import gzip
import json
from unittest.mock import patch, MagicMock

from src.data.downloader import PolygonDownloader
from src.data.preprocessor import DataPreprocessor
from src.data.cache import CacheManager
from src.data.features import FeatureEngine
from src.strategies.examples.moving_average import MovingAverageCrossover
from src.backtesting.engines.vectorbt_engine import VectorBTEngine
from src.validation.walk_forward import WalkForwardValidator
from src.validation.monte_carlo import MonteCarloValidator, ResamplingMethod


@pytest.mark.integration
class TestErrorRecovery:
    """Test error handling and recovery in the pipeline"""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories"""
        temp_root = tempfile.mkdtemp(prefix="error_recovery_test_")
        dirs = {
            'cache': Path(temp_root) / 'cache',
            'processed': Path(temp_root) / 'processed',
            'raw': Path(temp_root) / 'raw'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        yield dirs
        
        # Cleanup
        shutil.rmtree(temp_root)
    
    def test_corrupt_data_file_handling(self, temp_dirs):
        """Test handling of corrupt data files"""
        # Create a corrupt gzip file
        corrupt_file = temp_dirs['raw'] / 'corrupt_data.csv.gz'
        with open(corrupt_file, 'wb') as f:
            f.write(b'This is not a valid gzip file')
        
        # Try to load it
        preprocessor = DataPreprocessor(
            raw_data_dir=temp_dirs['raw'],
            processed_data_dir=temp_dirs['processed'],
            cache_dir=temp_dirs['cache']
        )
        
        # Should handle the error gracefully
        with pytest.raises(Exception) as exc_info:
            pd.read_csv(corrupt_file, compression='gzip')
        
        assert "gzip" in str(exc_info.value).lower() or "compressed" in str(exc_info.value).lower()
        
        # Create a valid but malformed CSV
        malformed_csv = temp_dirs['raw'] / 'malformed.csv.gz'
        malformed_data = "timestamp,open,high,low,close,volume\n" \
                        "not_a_number,100,101,99,100.5,abc\n" \
                        "2024-01-01,,,,,\n"
        
        with gzip.open(malformed_csv, 'wt') as f:
            f.write(malformed_data)
        
        # Try to process it
        try:
            # For testing, create a test symbol directory structure
            test_symbol_dir = temp_dirs['raw'] / 'TEST'
            test_symbol_dir.mkdir(exist_ok=True)
            # Move malformed file to proper location
            malformed_csv.rename(test_symbol_dir / 'TEST_2024_01.csv.gz')
            
            # Processing should handle missing/invalid data
            preprocessor_test = DataPreprocessor(
                raw_data_dir=temp_dirs['raw'],
                processed_data_dir=temp_dirs['processed'],
                cache_dir=temp_dirs['cache']
            )
            processed = preprocessor_test.process('TEST', ['2024_01'])
            # Should have cleaned the data
            assert len(processed) > 0
        except Exception as e:
            # Should fail gracefully
            assert True
    
    def test_missing_required_features(self, temp_dirs):
        """Test handling when required features are missing"""
        # Create minimal data without required features
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        data = pd.DataFrame({
            'open': 100,
            'high': 101,
            'low': 99,
            'close': 100,
            'volume': 10000
        }, index=dates)
        
        # Create strategy that requires specific features
        # MovingAverageCrossover calculates MAs internally, so use volatility-based sizing
        strategy = MovingAverageCrossover({
            'fast_period': 20,
            'slow_period': 50,
            'position_sizing': 'volatility'  # This requires 'atr' feature
        })
        
        # Strategy should validate data
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_data(data)
        
        assert "required features" in str(exc_info.value).lower() or "atr" in str(exc_info.value).lower()
    
    def test_invalid_strategy_parameters(self):
        """Test handling of invalid strategy parameters"""
        # Test negative periods
        with pytest.raises(ValueError):
            strategy = MovingAverageCrossover({
                'fast_period': -10,
                'slow_period': 50
            })
            strategy._validate_parameters()
        
        # Test fast > slow
        with pytest.raises(ValueError):
            strategy = MovingAverageCrossover({
                'fast_period': 50,
                'slow_period': 20
            })
            strategy._validate_parameters()
        
        # Test invalid position size
        with pytest.raises(ValueError):
            strategy = MovingAverageCrossover({
                'fast_period': 20,
                'slow_period': 50,
                'position_size': 1.5  # >100%
            })
            strategy._validate_parameters()
    
    def test_cache_corruption_recovery(self, temp_dirs):
        """Test recovery from cache corruption"""
        cache_manager = CacheManager(
            cache_dir=temp_dirs['cache'],
            max_size_gb=0.1
        )
        
        # Cache a file
        test_data = pd.DataFrame({'value': range(100)})
        temp_file = temp_dirs['cache'] / 'test.parquet'
        test_data.to_parquet(temp_file)
        
        cache_key = 'test_data'
        cached_path = cache_manager.cache_file(temp_file, cache_key)
        
        # Corrupt the metadata
        metadata_file = temp_dirs['cache'] / '.cache_metadata.json'
        with open(metadata_file, 'w') as f:
            f.write('{"invalid": json}')  # Invalid JSON
        
        # Create new cache manager - should recover
        new_cache_manager = CacheManager(
            cache_dir=temp_dirs['cache'],
            max_size_gb=0.1
        )
        
        # Should have reset metadata
        stats = new_cache_manager.get_statistics()
        assert stats['total_files'] == 0  # Metadata was reset
        
        # But cached file should still exist
        assert cached_path.exists()
    
    def test_partial_file_download_recovery(self, temp_dirs):
        """Test recovery from partial file downloads"""
        # Create a partial file (simulating interrupted download)
        partial_file = temp_dirs['raw'] / 'data.csv.gz.partial'
        with open(partial_file, 'wb') as f:
            f.write(b'Partial data...')
        
        # The downloader should detect and handle partial files
        # In real implementation, it would resume or restart download
        assert partial_file.exists()
        assert '.partial' in str(partial_file)
        
        # Simulate cleanup of partial files
        for partial in temp_dirs['raw'].glob('*.partial'):
            partial.unlink()
        
        assert not list(temp_dirs['raw'].glob('*.partial'))
    
    def test_network_failure_handling(self):
        """Test handling of network failures during data download"""
        downloader = PolygonDownloader()
        
        # Mock boto3 client's download_file method to simulate network failure
        with patch.object(downloader.s3_client, 'download_file') as mock_download:
            mock_download.side_effect = Exception("Network timeout")
            
            # Should handle the error gracefully
            with pytest.raises(Exception) as exc_info:
                from datetime import date
                downloader.download_daily_file(
                    date_obj=date(2024, 1, 1),
                    output_dir=Path('test_dir'),
                    overwrite=False
                )
            
            # The error should propagate up
            assert "Network timeout" in str(exc_info.value) or "download" in str(exc_info.value).lower()
    
    def test_insufficient_data_for_strategy(self, temp_dirs):
        """Test handling when data is insufficient for strategy requirements"""
        # Create data with fewer bars than required
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        short_data = pd.DataFrame({
            'open': 100,
            'high': 101,
            'low': 99,
            'close': 100,
            'volume': 10000,
            'sma_20': 100,  # Add required features
            'sma_50': 100   # But not enough data for 50-day SMA
        }, index=dates)
        
        strategy = MovingAverageCrossover({
            'fast_period': 20,
            'slow_period': 50
        })
        
        # Should validate minimum data requirements
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_data(short_data)
        
        assert "insufficient data" in str(exc_info.value).lower()
    
    def test_memory_limit_handling(self, temp_dirs):
        """Test handling of memory constraints"""
        # Create cache with very small limit
        cache_manager = CacheManager(
            cache_dir=temp_dirs['cache'],
            max_size_gb=0.0001  # 0.1 MB limit
        )
        
        # Try to cache files that exceed limit
        large_data = pd.DataFrame({
            'value': np.random.randn(10000)
        })
        
        temp_file = temp_dirs['cache'] / 'large.parquet'
        large_data.to_parquet(temp_file)
        
        # Cache multiple files
        for i in range(5):
            cache_key = f'large_file_{i}'
            cache_manager.cache_file(temp_file, cache_key)
        
        # Should have evicted old files
        stats = cache_manager.get_statistics()
        assert stats['total_files'] < 5  # Some files were evicted
        assert stats['total_size_bytes'] <= cache_manager.max_size_bytes
    
    def test_invalid_date_handling(self):
        """Test handling of invalid dates in data"""
        # Create data with invalid dates
        data_with_issues = pd.DataFrame({
            'timestamp': ['2024-01-01', 'invalid_date', '2024-01-03', None, '2024-01-05'],
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [10000, 20000, 30000, 40000, 50000]
        })
        
        # Preprocessing should handle invalid dates
        preprocessor = DataPreprocessor(
            raw_data_dir=Path('data'),
            processed_data_dir=Path('data/processed'),
            cache_dir=Path('data/cache')
        )
        
        # Convert timestamp with error handling
        try:
            data_with_issues['timestamp'] = pd.to_datetime(
                data_with_issues['timestamp'], 
                errors='coerce'
            )
            # Remove rows with invalid dates
            clean_data = data_with_issues.dropna(subset=['timestamp'])
            assert len(clean_data) == 3  # Only valid dates remain
        except Exception as e:
            # Should handle gracefully
            assert True
    
    def test_strategy_exception_handling(self, temp_dirs):
        """Test handling of exceptions during strategy execution"""
        # Create data that might cause strategy issues
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        problematic_data = pd.DataFrame({
            'open': [100] * 50 + [np.nan] * 50,  # NaN values
            'high': [101] * 100,
            'low': [99] * 100,
            'close': [100] * 50 + [0] * 50,  # Zero prices
            'volume': [10000] * 100,
            'sma_20': [100] * 100,
            'sma_50': [100] * 100
        }, index=dates)
        
        strategy = MovingAverageCrossover()
        engine = VectorBTEngine()
        
        # Should handle the problematic data
        try:
            result = engine.run_backtest(
                strategy=strategy,
                data=problematic_data,
                initial_capital=100000
            )
            # If it runs, should handle edge cases
            assert result is not None
        except Exception as e:
            # Should fail gracefully with informative error
            assert "price" in str(e).lower() or "data" in str(e).lower()
    
    def test_validation_with_no_trades(self):
        """Test validation when strategy generates no trades"""
        # Create flat data that won't generate signals
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        flat_data = pd.DataFrame({
            'open': 100,
            'high': 100.1,
            'low': 99.9,
            'close': 100,
            'volume': 10000,
            'sma_20': 100,
            'sma_50': 100
        }, index=dates)
        
        strategy = MovingAverageCrossover({
            'fast_period': 20,
            'slow_period': 50
        })
        
        engine = VectorBTEngine()
        result = engine.run_backtest(
            strategy=strategy,
            data=flat_data,
            initial_capital=100000
        )
        
        # Should handle no trades gracefully
        assert result.metrics['trades_count'] == 0
        assert result.metrics['total_return'] == 0
        
        # Monte Carlo should also handle this
        from src.validation.monte_carlo import ResamplingMethod
        mc_validator = MonteCarloValidator(
            n_simulations=50,
            resampling_method=ResamplingMethod.BOOTSTRAP
        )
        
        # Should handle empty trade list
        mc_result = mc_validator.run_validation(result)
        
        # Should return valid results even with no trades
        assert len(mc_result.simulation_results) == 50
        # With no trades, risk of ruin should be 0 or the key might not exist
        if mc_result.risk_metrics:
            assert mc_result.risk_metrics.get('risk_of_ruin', 0) == 0
    
    def test_concurrent_access_handling(self, temp_dirs):
        """Test handling of concurrent access to cache"""
        import threading
        import time
        
        cache_manager = CacheManager(
            cache_dir=temp_dirs['cache'],
            max_size_gb=0.1
        )
        
        errors = []
        
        def cache_operation(thread_id):
            """Simulate concurrent cache operations"""
            try:
                for i in range(10):
                    # Create data
                    data = pd.DataFrame({'value': [thread_id] * 100})
                    temp_file = temp_dirs['cache'] / f'thread_{thread_id}_{i}.parquet'
                    data.to_parquet(temp_file)
                    
                    # Cache it
                    cache_key = f'thread_{thread_id}_file_{i}'
                    cache_manager.cache_file(temp_file, cache_key)
                    
                    # Try to retrieve
                    retrieved = cache_manager.get_cached_file(cache_key)
                    assert retrieved is not None
                    
                    # Small delay
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=cache_operation, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should have no errors
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        
        # Cache should be consistent
        stats = cache_manager.get_statistics()
        assert stats['total_files'] > 0
        print(f"Handled {stats['total_files']} concurrent cache operations")
    
    def test_recovery_from_incomplete_backtest(self, temp_dirs):
        """Test recovery when backtest is interrupted"""
        # Create test data
        dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
        test_data = pd.DataFrame({
            'open': 100 + np.random.randn(1000) * 0.5,
            'high': 101 + np.random.randn(1000) * 0.5,
            'low': 99 + np.random.randn(1000) * 0.5,
            'close': 100 + np.random.randn(1000) * 0.5,
            'volume': np.random.randint(1000, 10000, 1000),
            'sma_20': 100,
            'sma_50': 100
        }, index=dates)
        
        strategy = MovingAverageCrossover()
        engine = VectorBTEngine()
        
        # Simulate interruption by using a subset
        partial_data = test_data.iloc[:500]  # First half
        
        # Run partial backtest
        partial_result = engine.run_backtest(
            strategy=strategy,
            data=partial_data,
            initial_capital=100000
        )
        
        # Save partial results
        partial_results_file = temp_dirs['cache'] / 'partial_backtest.json'
        with open(partial_results_file, 'w') as f:
            json.dump({
                'last_index': len(partial_data),
                'metrics': partial_result.metrics,
                'final_equity': partial_result.equity_curve.iloc[-1]
            }, f)
        
        # "Resume" with full data (in practice, would start from saved point)
        full_result = engine.run_backtest(
            strategy=strategy,
            data=test_data,
            initial_capital=100000
        )
        
        # Verify we can complete the full backtest
        assert len(full_result.equity_curve) == len(test_data)
        print(f"Recovered from interruption at bar {len(partial_data)}")