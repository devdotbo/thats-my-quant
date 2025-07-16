"""
Tests for data preprocessor module
Uses real extracted data for testing
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time
import pytz
import gzip
import tempfile
import shutil

from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test data preprocessing functionality"""
    
    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create sample data directory with test files"""
        symbol_dir = tmp_path / "SPY"
        symbol_dir.mkdir()
        
        # Create sample data that mimics real Polygon format
        # January 2, 2024 was a Tuesday (market open)
        sample_data = pd.DataFrame({
            'ticker': ['SPY'] * 10,
            'volume': [1000, 2000, 1500, 3000, 2500, 99999999, 1800, 2200, 1900, 2100],  # One outlier
            'open': [476.25, 476.30, 476.35, 476.40, 476.45, 476.50, 476.55, 476.60, 476.65, 476.70],
            'close': [476.30, 476.35, 476.40, 476.45, 476.50, 1000.00, 476.60, 476.65, 476.70, 476.75],  # One outlier
            'high': [476.35, 476.40, 476.45, 476.50, 476.55, 1000.00, 476.65, 476.70, 476.75, 476.80],
            'low': [476.20, 476.25, 476.30, 476.35, 476.40, 476.45, 476.50, 476.55, 476.60, 476.65],
            # Timestamps for 9:30, 9:31, 9:32, skip 9:33, 9:34...
            'window_start': [
                1704205800000000000,  # 2024-01-02 09:30:00 EST
                1704205860000000000,  # 2024-01-02 09:31:00 EST
                1704205920000000000,  # 2024-01-02 09:32:00 EST
                # Skip 9:33 to test missing data
                1704206040000000000,  # 2024-01-02 09:34:00 EST
                1704206100000000000,  # 2024-01-02 09:35:00 EST
                1704206160000000000,  # 2024-01-02 09:36:00 EST (with outlier)
                1704206220000000000,  # 2024-01-02 09:37:00 EST
                1704206280000000000,  # 2024-01-02 09:38:00 EST
                1704206340000000000,  # 2024-01-02 09:39:00 EST
                1704206400000000000,  # 2024-01-02 09:40:00 EST
            ],
            'transactions': [50, 100, 75, 150, 125, 200, 90, 110, 95, 105]
        })
        
        # Save as gzipped CSV
        output_file = symbol_dir / "SPY_2024_01.csv.gz"
        with gzip.open(output_file, 'wt') as f:
            sample_data.to_csv(f, index=False)
        
        return tmp_path
    
    @pytest.fixture
    def preprocessor(self, sample_data_dir, tmp_path):
        """Create preprocessor instance with test directories"""
        return DataPreprocessor(
            raw_data_dir=sample_data_dir,
            processed_data_dir=tmp_path / "processed",
            cache_dir=tmp_path / "cache"
        )
    
    @pytest.fixture
    def real_data_preprocessor(self):
        """Create preprocessor for real extracted data (integration test)"""
        raw_dir = Path("data/raw/minute_aggs/by_symbol")
        if not raw_dir.exists():
            pytest.skip("Real data not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            return DataPreprocessor(
                raw_data_dir=raw_dir,
                processed_data_dir=Path(tmpdir) / "processed",
                cache_dir=Path(tmpdir) / "cache"
            )
    
    def test_load_symbol_data(self, preprocessor):
        """Test loading and combining symbol data files"""
        df = preprocessor.load_symbol_data("SPY", months=["2024_01"])
        
        assert len(df) == 10
        assert list(df.columns) == ['ticker', 'volume', 'open', 'close', 'high', 'low', 'window_start', 'transactions']
        assert df['ticker'].unique()[0] == 'SPY'
    
    def test_convert_timestamps(self, preprocessor):
        """Test nanosecond timestamp conversion"""
        df = preprocessor.load_symbol_data("SPY", months=["2024_01"])
        df_converted = preprocessor.convert_timestamps(df)
        
        # Check that index is datetime
        assert isinstance(df_converted.index, pd.DatetimeIndex)
        
        # Check first timestamp is correct (9:30 AM EST)
        expected_time = pd.Timestamp('2024-01-02 09:30:00', tz='America/New_York')
        assert df_converted.index[0] == expected_time
        
        # Check all times are in market hours
        for ts in df_converted.index:
            assert 9 <= ts.hour <= 16
            if ts.hour == 16:
                assert ts.minute == 0  # 4:00 PM
    
    def test_fill_missing_bars(self, preprocessor):
        """Test filling missing minute bars during market hours"""
        df = preprocessor.load_symbol_data("SPY", months=["2024_01"])
        df = preprocessor.convert_timestamps(df)
        df_filled = preprocessor.fill_missing_bars(df)
        
        # Check that missing 9:33 bar is filled
        expected_time = pd.Timestamp('2024-01-02 09:33:00', tz='America/New_York')
        assert expected_time in df_filled.index
        
        # Check forward fill worked correctly
        filled_bar = df_filled.loc[expected_time]
        prev_bar = df_filled.loc[pd.Timestamp('2024-01-02 09:32:00', tz='America/New_York')]
        
        # Volume should be 0 for filled bars
        assert filled_bar['volume'] == 0
        assert filled_bar['transactions'] == 0
        
        # Prices should be forward filled
        assert filled_bar['close'] == prev_bar['close']
        assert filled_bar['open'] == prev_bar['close']
        assert filled_bar['high'] == prev_bar['close']
        assert filled_bar['low'] == prev_bar['close']
    
    def test_clean_outliers(self, preprocessor):
        """Test IQR-based outlier cleaning"""
        df = preprocessor.load_symbol_data("SPY", months=["2024_01"])
        df = preprocessor.convert_timestamps(df)
        df_cleaned = preprocessor.clean_outliers(df)
        
        # Check that the outlier at index 5 is handled
        # The close price of 1000.00 should be replaced
        assert df_cleaned['close'].max() < 500  # Original had 1000
        assert df_cleaned['volume'].max() < 10000000  # Original had 99999999
        
        # Check that non-outlier values are preserved
        assert df_cleaned['close'].min() > 470
        assert df_cleaned['close'].max() < 480
    
    def test_validate_data(self, preprocessor):
        """Test data validation"""
        df = preprocessor.load_symbol_data("SPY", months=["2024_01"])
        df = preprocessor.convert_timestamps(df)
        df = preprocessor.fill_missing_bars(df)
        df = preprocessor.clean_outliers(df)
        
        # Should pass validation
        assert preprocessor.validate_data(df) is True
        
        # Test with invalid data - negative prices
        df_invalid = df.copy()
        df_invalid.loc[df_invalid.index[0], 'close'] = -100
        
        with pytest.raises(ValueError, match="Negative prices found"):
            preprocessor.validate_data(df_invalid)
        
        # Test with invalid data - NaN values
        df_invalid = df.copy()
        df_invalid.loc[df_invalid.index[0], 'close'] = np.nan
        
        with pytest.raises(ValueError, match="NaN values found"):
            preprocessor.validate_data(df_invalid)
    
    def test_process_full_pipeline(self, preprocessor):
        """Test full preprocessing pipeline"""
        processed_df = preprocessor.process("SPY", months=["2024_01"])
        
        # Check output format
        assert isinstance(processed_df.index, pd.DatetimeIndex)
        assert processed_df.index.tz is not None  # Should have timezone
        
        # Check required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            assert col in processed_df.columns
        
        # Check no missing values in critical columns
        for col in ['open', 'high', 'low', 'close']:
            assert processed_df[col].isna().sum() == 0
        
        # Check data is clean
        assert processed_df['close'].max() < 500  # Outlier removed
        assert processed_df['close'].min() > 0    # No negative prices
    
    def test_save_and_load_processed(self, preprocessor):
        """Test saving and loading processed data"""
        # Process data
        processed_df = preprocessor.process("SPY", months=["2024_01"])
        
        # Save it
        output_path = preprocessor.save_processed(processed_df, "SPY", "2024_01")
        assert output_path.exists()
        assert output_path.suffix == '.parquet'
        
        # Load it back
        loaded_df = pd.read_parquet(output_path)
        
        # Compare (ignore frequency info which might not be preserved)
        pd.testing.assert_frame_equal(processed_df, loaded_df, check_freq=False)
    
    @pytest.mark.integration
    def test_real_data_processing(self, real_data_preprocessor):
        """Integration test with real extracted SPY data"""
        # Process January 2024 SPY data
        processed_df = real_data_preprocessor.process("SPY", months=["2024_01"])
        
        # Basic sanity checks
        assert len(processed_df) > 0
        assert isinstance(processed_df.index, pd.DatetimeIndex)
        
        # Check we have expected number of trading days (roughly)
        # January 2024 had ~21 trading days
        expected_bars = 21 * 390  # 390 minutes per trading day
        assert len(processed_df) > expected_bars * 0.9  # Allow 10% tolerance
        
        # Check price ranges are reasonable for SPY
        assert 400 < processed_df['close'].mean() < 600
        assert processed_df['volume'].mean() > 0
        
        # Check no outliers remain
        price_std = processed_df['close'].std()
        price_mean = processed_df['close'].mean()
        assert processed_df['close'].max() < price_mean + 5 * price_std
        assert processed_df['close'].min() > price_mean - 5 * price_std
    
    def test_process_multiple_months(self, preprocessor):
        """Test processing multiple months of data"""
        # This will only have January data in test fixture
        processed_df = preprocessor.process("SPY", months=["2024_01", "2024_02"])
        
        # Should still work with only January data available
        assert len(processed_df) > 0
        assert processed_df.index.month.unique()[0] == 1
    
    def test_timezone_handling(self, preprocessor):
        """Test proper timezone handling for market hours"""
        df = preprocessor.load_symbol_data("SPY", months=["2024_01"])
        df = preprocessor.convert_timestamps(df)
        
        # All timestamps should be in EST/EDT
        assert str(df.index.tz) == 'America/New_York'
        
        # No data outside market hours
        for ts in df.index:
            market_time = ts.time()
            assert market_time >= time(9, 30) or market_time <= time(16, 0)