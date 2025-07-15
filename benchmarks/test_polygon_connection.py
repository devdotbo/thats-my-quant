"""
Test Polygon.io S3 Connection
Verifies credentials and downloads sample data
"""

import os
import boto3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PolygonConnectionTest:
    """Test Polygon.io S3 flat files connection"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'credentials_found': False,
            'connection_successful': False
        }
        
        # Load credentials
        self.api_key = os.getenv('polygon_io_api_key')
        self.s3_access_key = os.getenv('polygon_io_s3_access_key_id')
        self.s3_secret = os.getenv('polygon_io_s3_access_secret')
        self.s3_endpoint = os.getenv('polygon_io_s3_endpoint', 'https://files.polygon.io')
        self.s3_bucket = os.getenv('polygon_io_s3_bucket', 'flatfiles')
        
        # Check if credentials exist
        self.results['credentials_found'] = all([
            self.api_key,
            self.s3_access_key,
            self.s3_secret
        ])
    
    def test_s3_connection(self) -> bool:
        """Test basic S3 connection"""
        print("\nTesting S3 Connection...")
        
        if not self.results['credentials_found']:
            print("❌ ERROR: Missing Polygon.io credentials in .env file")
            print("Please ensure the following are set:")
            print("  - polygon_io_api_key")
            print("  - polygon_io_s3_access_key_id")
            print("  - polygon_io_s3_access_secret")
            self.results['tests']['s3_connection'] = {
                'status': 'failed',
                'error': 'Missing credentials'
            }
            return False
        
        try:
            # Create S3 client
            # IMPORTANT: Use API key as S3 secret (discovered via rclone testing)
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.s3_endpoint,
                aws_access_key_id=self.s3_access_key,
                aws_secret_access_key=self.api_key  # Use API key, not S3 secret!
            )
            
            # Test listing objects (limited to check connection)
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix='us_stocks_sip/',
                MaxKeys=1
            )
            
            self.results['connection_successful'] = True
            self.results['tests']['s3_connection'] = {
                'status': 'success',
                'bucket': self.s3_bucket,
                'endpoint': self.s3_endpoint
            }
            
            print("✅ S3 connection successful!")
            return True
            
        except Exception as e:
            print(f"❌ S3 connection failed: {str(e)}")
            self.results['tests']['s3_connection'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def list_available_data(self) -> Dict[str, Any]:
        """List available data types and structure"""
        print("\nExploring available data structure...")
        
        if not self.results['connection_successful']:
            print("Skipping - no S3 connection")
            return {}
        
        try:
            # Check different data types
            data_types = {
                'trades': 'us_stocks_sip/trades/',
                'quotes': 'us_stocks_sip/quotes/',
                'minute_aggs': 'us_stocks_sip/minute_aggs/',
                'day_aggs': 'us_stocks_sip/day_aggs/'
            }
            
            available = {}
            
            for data_type, prefix in data_types.items():
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=prefix,
                    Delimiter='/',
                    MaxKeys=1
                )
                
                if 'CommonPrefixes' in response or 'Contents' in response:
                    available[data_type] = True
                    print(f"  ✅ {data_type}: Available")
                else:
                    available[data_type] = False
                    print(f"  ❌ {data_type}: Not found")
            
            self.results['tests']['data_availability'] = available
            return available
            
        except Exception as e:
            print(f"❌ Error listing data: {str(e)}")
            self.results['tests']['data_availability'] = {'error': str(e)}
            return {}
    
    def download_sample_data(self, symbol: str = 'AAPL', 
                           date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Download sample minute aggregate data for testing"""
        print(f"\nDownloading sample data for {symbol}...")
        
        if not self.results['connection_successful']:
            print("Skipping - no S3 connection")
            return None
        
        if date is None:
            # Use a recent weekday
            date = datetime.now() - timedelta(days=5)
            while date.weekday() > 4:  # Skip weekends
                date -= timedelta(days=1)
        
        date_str = date.strftime('%Y-%m-%d')
        
        try:
            # Construct S3 key for minute aggregates
            year = date.strftime('%Y')
            month = date.strftime('%m')
            day = date.strftime('%d')
            
            key = f'us_stocks_sip/minute_aggs/{year}/{month}/{symbol}.csv.gz'
            
            print(f"  Attempting to download: {key}")
            
            # Download to temporary file
            temp_dir = Path('temp_data')
            temp_dir.mkdir(exist_ok=True)
            
            local_file = temp_dir / f'{symbol}_{date_str}_minute.csv.gz'
            
            self.s3_client.download_file(
                self.s3_bucket,
                key,
                str(local_file)
            )
            
            print(f"  ✅ Downloaded to: {local_file}")
            
            # Read the data
            df = pd.read_csv(local_file, compression='gzip')
            
            # Clean up
            local_file.unlink()
            temp_dir.rmdir()
            
            self.results['tests']['sample_download'] = {
                'status': 'success',
                'symbol': symbol,
                'date': date_str,
                'shape': str(df.shape),
                'columns': list(df.columns)
            }
            
            print(f"  Data shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            return df
            
        except self.s3_client.exceptions.NoSuchKey:
            print(f"  ⚠️  No data found for {symbol} on {date_str}")
            print("  This might be a weekend/holiday or future date")
            self.results['tests']['sample_download'] = {
                'status': 'no_data',
                'symbol': symbol,
                'date': date_str
            }
            return None
            
        except Exception as e:
            print(f"  ❌ Download failed: {str(e)}")
            self.results['tests']['sample_download'] = {
                'status': 'failed',
                'error': str(e)
            }
            return None
    
    def test_data_format(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Test and validate data format"""
        print("\nValidating data format...")
        
        if df is None or df.empty:
            print("  No data to validate")
            return {}
        
        format_info = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Expected columns for minute aggregates
        expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"  ⚠️  Missing expected columns: {missing_columns}")
        else:
            print("  ✅ All expected columns present")
        
        # Check data quality
        print(f"  Shape: {df.shape}")
        print(f"  Memory usage: {format_info['memory_usage_mb']:.2f} MB")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        self.results['tests']['data_format'] = format_info
        return format_info
    
    def estimate_storage_requirements(self) -> Dict[str, float]:
        """Estimate storage requirements for different scenarios"""
        print("\nEstimating storage requirements...")
        
        # Based on sample data, estimate storage needs
        estimates = {
            '1_symbol_1_year_minute': 0.1,  # ~100MB
            '10_symbols_1_year_minute': 1.0,  # ~1GB
            '10_symbols_5_years_minute': 5.0,  # ~5GB
            '100_symbols_1_year_minute': 10.0,  # ~10GB
            '10_symbols_1_year_trades': 5.0,  # ~5GB (trades are larger)
            '10_symbols_1_year_quotes': 10.0,  # ~10GB (quotes are largest)
        }
        
        print("\nEstimated storage requirements:")
        for scenario, size_gb in estimates.items():
            print(f"  {scenario}: ~{size_gb:.1f} GB")
        
        total_recommended = 50.0
        print(f"\nRecommended allocation: {total_recommended:.0f} GB")
        print("(Leaves 50GB for cache and results)")
        
        self.results['tests']['storage_estimates'] = estimates
        return estimates
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete connection test suite"""
        print("="*60)
        print("Polygon.io S3 Connection Test")
        print("="*60)
        
        # Test connection
        if not self.test_s3_connection():
            self._save_results()
            return self.results
        
        # List available data
        available_data = self.list_available_data()
        
        # Download sample data
        sample_df = self.download_sample_data('SPY')
        
        # Validate data format
        if sample_df is not None:
            self.test_data_format(sample_df)
        
        # Estimate storage
        self.estimate_storage_requirements()
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save test results to file"""
        results_dir = Path('benchmarks/results')
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = results_dir / f'polygon_connection_test_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filename}")
    
    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        if self.results['credentials_found']:
            print("✅ Credentials found in .env")
        else:
            print("❌ Missing credentials - check .env file")
        
        if self.results['connection_successful']:
            print("✅ S3 connection successful")
        else:
            print("❌ S3 connection failed")
        
        # Data availability
        data_avail = self.results['tests'].get('data_availability', {})
        if data_avail and not data_avail.get('error'):
            print("\nAvailable data types:")
            for dtype, available in data_avail.items():
                if available:
                    print(f"  ✅ {dtype}")
        
        # Sample download
        sample = self.results['tests'].get('sample_download', {})
        if sample.get('status') == 'success':
            print("\n✅ Successfully downloaded sample data")
            print(f"  Symbol: {sample.get('symbol')}")
            print(f"  Shape: {sample.get('shape')}")
        
        print("\n" + "="*60)
        
        if self.results['connection_successful']:
            print("✅ Polygon.io S3 connection verified and ready!")
            print("✅ You can proceed with data downloads")
        else:
            print("❌ Connection issues need to be resolved")
            print("Please check your credentials and try again")


def main():
    """Run Polygon.io connection tests"""
    tester = PolygonConnectionTest()
    results = tester.run_all_tests()
    
    # Return status
    if results['connection_successful']:
        print("\n✅ Day 1 Gate: Polygon.io connection verified!")
    else:
        print("\n❌ Day 1 Gate: Connection issues must be resolved")


if __name__ == "__main__":
    main()