"""
Tests for Polygon.io data downloader
Real integration tests - no mocking!
"""

import pytest
from datetime import date
import pandas as pd

from src.data.downloader import PolygonDownloader


class TestPolygonDownloader:
    """Test Polygon data downloader with real S3 connections"""
    
    @pytest.fixture
    def downloader(self):
        """Create downloader instance"""
        return PolygonDownloader()
    
    @pytest.fixture
    def test_output_dir(self, tmp_path):
        """Create temporary output directory"""
        output_dir = tmp_path / "test_downloads"
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    @pytest.mark.requires_polygon
    def test_s3_connection(self, downloader):
        """Test real S3 connection to Polygon"""
        # This should connect successfully with correct credentials
        assert downloader.s3_client is not None
        
        # Test listing objects (real S3 call)
        response = downloader.s3_client.list_objects_v2(
            Bucket=downloader.bucket,
            Prefix='us_stocks_sip/',
            MaxKeys=1
        )
        
        # Should have at least one result or empty Contents
        assert 'ResponseMetadata' in response
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
    
    @pytest.mark.requires_polygon
    def test_list_available_paths(self, downloader):
        """Test listing available data paths"""
        # Real S3 listing
        paths = downloader.list_available_paths('us_stocks_sip/')
        
        # Should find some data directories
        assert isinstance(paths, list)
        assert len(paths) > 0
        
        # Should include known data types
        path_strings = [str(p) for p in paths]
        assert any('minute' in p for p in path_strings)
    
    @pytest.mark.requires_polygon
    def test_find_data_structure(self, downloader):
        """Test discovering actual data structure"""
        # Find minute aggregate paths
        minute_paths = downloader.find_data_type_paths('minute')
        
        assert len(minute_paths) > 0
        print(f"Found minute data paths: {minute_paths}")
        
        # Verify path format
        for path in minute_paths:
            assert 'minute' in path.lower()
    
    @pytest.mark.requires_polygon
    def test_build_daily_s3_key(self, downloader):
        """Test building correct S3 keys for daily files"""
        # Test with date-based structure
        test_date = date(2024, 1, 2)
        key = downloader.build_daily_s3_key(
            date_obj=test_date,
            data_type='minute_aggs'
        )
        
        assert isinstance(key, str)
        assert '2024-01-02' in key
        assert 'minute_aggs_v1' in key
        assert key.endswith('.csv.gz')
    
    @pytest.mark.requires_polygon
    @pytest.mark.slow
    def test_download_daily_file(self, downloader, test_output_dir):
        """Test downloading a real daily file from S3"""
        # Try to download a recent trading day
        test_date = date(2024, 1, 2)  # Known trading day
        
        output_file = downloader.download_daily_file(
            date_obj=test_date,
            output_dir=test_output_dir,
            data_type='minute_aggs'
        )
        
        if output_file:
            # File was downloaded successfully
            assert output_file.exists()
            assert output_file.stat().st_size > 0
            
            # Verify it's a valid gzip file
            assert output_file.suffix == '.gz'
            
            # Should be at least several MB (contains all symbols)
            assert output_file.stat().st_size > 1_000_000  # > 1MB
            
            # Try to read it as CSV
            df = pd.read_csv(output_file, compression='gzip', nrows=100)
            assert len(df) > 0
            assert 'ticker' in df.columns  # Should have ticker column
            
            # Should contain at least one symbol
            unique_symbols = df['ticker'].unique()
            assert len(unique_symbols) >= 1  # Some test days might have limited data
        else:
            # File might not exist for this date - that's OK
            pytest.skip(f"No data available for {test_date}")
    
    @pytest.mark.requires_polygon
    def test_check_daily_file_exists(self, downloader):
        """Test checking if a daily file exists in S3"""
        # Check for a known trading day
        test_date = date(2024, 1, 2)
        exists = downloader.check_daily_file_exists(
            date_obj=test_date,
            data_type='minute_aggs'
        )
        
        # Should return boolean
        assert isinstance(exists, bool)
        
        # Weekend should not exist
        weekend_date = date(2024, 1, 6)  # Saturday
        weekend_exists = downloader.check_daily_file_exists(
            date_obj=weekend_date,
            data_type='minute_aggs'
        )
        assert not weekend_exists
    
    @pytest.mark.requires_polygon
    @pytest.mark.slow
    def test_extract_symbols_from_daily_file(self, downloader, test_output_dir):
        """Test extracting specific symbols from a daily file"""
        # First download a daily file
        test_date = date(2024, 1, 2)
        daily_file = downloader.download_daily_file(
            date_obj=test_date,
            output_dir=test_output_dir / 'daily',
            data_type='minute_aggs'
        )
        
        if not daily_file:
            pytest.skip(f"No data available for {test_date}")
        
        # Extract specific symbols
        symbols = ['SPY', 'AAPL', 'MSFT']
        extracted = downloader.extract_symbols_from_daily_file(
            daily_file=daily_file,
            symbols=symbols,
            output_dir=test_output_dir / 'symbols'
        )
        
        # Should extract requested symbols
        assert len(extracted) == len(symbols)
        
        for symbol in symbols:
            assert symbol in extracted
            assert extracted[symbol].exists()
            assert extracted[symbol].suffix == '.gz'
            
            # Verify extracted data
            df = pd.read_csv(extracted[symbol], compression='gzip', nrows=10)
            assert len(df) > 0
            assert df['ticker'].iloc[0] == symbol
    
    @pytest.mark.requires_polygon
    @pytest.mark.slow
    def test_download_and_extract_symbols(self, downloader, test_output_dir):
        """Test complete workflow: download and extract"""
        # Test with a small date range
        symbols = ['SPY', 'QQQ']
        start_date = date(2024, 1, 2)
        end_date = date(2024, 1, 3)
        
        symbol_files = downloader.download_and_extract_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            output_dir=test_output_dir / 'extracted',
            daily_cache_dir=test_output_dir / 'daily_cache'
        )
        
        # Should return dict of symbol to list of files
        assert isinstance(symbol_files, dict)
        assert all(symbol in symbol_files for symbol in symbols)
        
        # Each symbol should have files for the date range
        for symbol, files in symbol_files.items():
            assert len(files) > 0
            for file in files:
                assert file.exists()
                assert file.suffix == '.gz'
    
    @pytest.mark.requires_polygon
    def test_get_available_dates(self, downloader):
        """Test listing available dates for a given month"""
        # List dates for a specific month
        dates = downloader.get_available_dates(
            year=2024,
            month=1,
            data_type='minute_aggs'
        )
        
        if dates:
            assert isinstance(dates, list)
            assert len(dates) > 0
            # Should have trading days (not weekends)
            for d in dates:
                assert d.weekday() < 5  # Monday=0, Friday=4
    
    @pytest.mark.requires_polygon
    @pytest.mark.slow
    def test_download_date_range(self, downloader, test_output_dir):
        """Test downloading daily files for a date range"""
        # Download just a few days for testing
        files = downloader.download_date_range(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
            output_dir=test_output_dir,
            data_type='minute_aggs'
        )
        
        # Should return list of downloaded files
        assert isinstance(files, list)
        
        # Should skip weekends
        if files:
            # Verify each file
            for file in files:
                assert file.exists()
                assert file.stat().st_size > 1_000_000  # Daily files are large
    
    @pytest.mark.requires_polygon
    @pytest.mark.slow
    def test_concurrent_downloads(self, downloader, test_output_dir):
        """Test downloading multiple symbols concurrently"""
        symbols = ['SPY', 'QQQ', 'AAPL']
        
        # Download concurrently
        results = downloader.download_multiple_symbols(
            symbols=symbols,
            year=2024,
            month=1,
            output_dir=test_output_dir,
            max_concurrent=3,
            data_type='minute_aggs'
        )
        
        # Should get results for each symbol
        assert len(results) == len(symbols)
        
        # Check each result
        successful = sum(1 for r in results.values() if r['success'])
        print(f"Successfully downloaded {successful}/{len(symbols)} symbols")
        
        # At least some should succeed
        assert successful > 0
    
    @pytest.mark.requires_polygon
    def test_retry_on_failure(self, downloader, test_output_dir):
        """Test retry logic on download failure"""
        # Try to download non-existent data to test retry
        output_file = downloader.download_single_file(
            symbol='INVALID_SYMBOL_XYZ',
            year=2024,
            month=1,
            output_dir=test_output_dir,
            data_type='minute_aggs',
            max_retries=2
        )
        
        # Should handle gracefully and return None
        assert output_file is None
    
    @pytest.mark.requires_polygon
    def test_validate_downloaded_data(self, downloader, test_output_dir):
        """Test validation of downloaded data"""
        # Download a file first
        output_file = downloader.download_single_file(
            symbol='SPY',
            year=2024,
            month=1,
            output_dir=test_output_dir,
            data_type='minute_aggs'
        )
        
        if output_file and output_file.exists():
            # Validate the downloaded file
            is_valid = downloader.validate_data_file(output_file)
            assert isinstance(is_valid, bool)
            
            if is_valid:
                # Read and check structure
                df = pd.read_csv(output_file, compression='gzip', nrows=100)
                
                # Should have expected columns
                expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                actual_cols = df.columns.tolist()
                
                # Check if most expected columns are present
                matching_cols = sum(1 for col in expected_cols if any(col in c.lower() for c in actual_cols))
                assert matching_cols >= 4  # At least 4 of the expected columns


@pytest.mark.requires_polygon
class TestPolygonDownloaderCLI:
    """Test command line interface"""
    
    def test_cli_parsing(self):
        """Test CLI argument parsing"""
        from src.data.cli import parse_args
        
        # Test basic arguments
        args = parse_args([
            '--symbols', 'SPY,AAPL',
            '--start-date', '2024-01-01',
            '--end-date', '2024-01-31',
            '--output-dir', '/tmp/test'
        ])
        
        assert args.symbols == 'SPY,AAPL'
        assert args.start_date == '2024-01-01'
        assert args.end_date == '2024-01-31'
        assert args.output_dir == '/tmp/test'