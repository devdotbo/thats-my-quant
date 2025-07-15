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
    def test_build_s3_key(self, downloader):
        """Test building correct S3 keys for data"""
        # Test with discovered path structure
        key = downloader.build_s3_key(
            symbol='SPY',
            year=2024,
            month=1,
            data_type='minute_aggs'
        )
        
        assert isinstance(key, str)
        assert 'SPY' in key
        assert '2024' in key
        assert '01' in key  # Month should be zero-padded
    
    @pytest.mark.requires_polygon
    @pytest.mark.slow
    def test_download_single_file(self, downloader, test_output_dir):
        """Test downloading a real file from S3"""
        # Try to download a small recent file
        # Using a recent date that should exist
        symbol = 'SPY'
        year = 2024
        month = 1
        
        output_file = downloader.download_single_file(
            symbol=symbol,
            year=year,
            month=month,
            output_dir=test_output_dir,
            data_type='minute_aggs'
        )
        
        if output_file:
            # File was downloaded successfully
            assert output_file.exists()
            assert output_file.stat().st_size > 0
            
            # Verify it's a valid gzip file
            assert output_file.suffix == '.gz'
            
            # Try to read it as CSV
            df = pd.read_csv(output_file, compression='gzip', nrows=5)
            assert len(df) > 0
            assert 'timestamp' in df.columns or 'time' in df.columns
        else:
            # File might not exist for this date - that's OK
            pytest.skip(f"No data available for {symbol} {year}-{month}")
    
    @pytest.mark.requires_polygon
    def test_check_file_exists(self, downloader):
        """Test checking if a file exists in S3"""
        # Check for a known symbol in recent data
        exists = downloader.check_file_exists(
            symbol='SPY',
            year=2024,
            month=1,
            data_type='minute_aggs'
        )
        
        # Should return boolean
        assert isinstance(exists, bool)
    
    @pytest.mark.requires_polygon
    @pytest.mark.slow
    def test_download_with_progress(self, downloader, test_output_dir, capsys):
        """Test download with progress tracking"""
        # Download with progress callback
        def progress_callback(bytes_downloaded, total_bytes):
            if total_bytes > 0:
                pct = (bytes_downloaded / total_bytes) * 100
                print(f"\rProgress: {pct:.1f}%", end='')
        
        output_file = downloader.download_single_file(
            symbol='AAPL',
            year=2024,
            month=1,
            output_dir=test_output_dir,
            data_type='minute_aggs',
            progress_callback=progress_callback
        )
        
        # Check that progress was reported
        captured = capsys.readouterr()
        if output_file:
            assert 'Progress:' in captured.out
    
    @pytest.mark.requires_polygon
    def test_get_available_symbols(self, downloader):
        """Test listing available symbols for a given date"""
        # List symbols for a specific month
        symbols = downloader.get_available_symbols(
            year=2024,
            month=1,
            data_type='minute_aggs'
        )
        
        if symbols:
            assert isinstance(symbols, list)
            assert len(symbols) > 0
            # Should include major symbols
            assert any(s in symbols for s in ['SPY', 'AAPL', 'MSFT'])
    
    @pytest.mark.requires_polygon
    def test_download_date_range(self, downloader, test_output_dir):
        """Test downloading data for a date range"""
        # Download just a few days for testing
        files = downloader.download_date_range(
            symbol='SPY',
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
            output_dir=test_output_dir,
            data_type='minute_aggs'
        )
        
        # Should return list of downloaded files
        assert isinstance(files, list)
        
        # Verify each file
        for file in files:
            assert file.exists()
            assert file.stat().st_size > 0
    
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