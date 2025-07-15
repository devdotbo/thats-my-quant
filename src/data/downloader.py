"""
Polygon.io S3 Data Downloader
Downloads historical market data from Polygon.io flat files

IMPORTANT: Polygon data is organized by DATE, not symbol.
Each daily file contains ALL symbols for that day.
Path structure: us_stocks_sip/minute_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz
"""

import os
import boto3
from botocore.exceptions import ClientError
import time
from pathlib import Path
from datetime import date, timedelta
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from src.utils.config import get_config
from src.utils.logging import get_logger

# Load environment variables from .env file
load_dotenv()


class PolygonDownloader:
    """Download market data from Polygon.io S3 flat files"""
    
    def __init__(self):
        """Initialize S3 client with Polygon credentials"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Get credentials from environment
        self.api_key = os.getenv('polygon_io_api_key')
        self.s3_access_key = os.getenv('polygon_io_s3_access_key_id')
        
        if not self.api_key or not self.s3_access_key:
            raise ValueError("Missing Polygon.io credentials in environment")
        
        # Initialize S3 client
        # IMPORTANT: Use API key as S3 secret (discovered through testing)
        self.s3_client = boto3.client(
            's3',
            endpoint_url='https://files.polygon.io',
            aws_access_key_id=self.s3_access_key,
            aws_secret_access_key=self.api_key  # Use API key as secret!
        )
        
        self.bucket = 'flatfiles'
        self.logger.info("Initialized Polygon S3 downloader")
        
        # Data type paths (will be discovered dynamically)
        self.data_paths = {
            'minute_aggs': 'us_stocks_sip/minute_aggs_v1',
            'trades': 'us_stocks_sip/trades_v1',
            'quotes': 'us_stocks_sip/quotes_v1',
            'day_aggs': 'us_stocks_sip/day_aggs_v1'
        }
    
    def list_available_paths(self, prefix: str = '') -> List[str]:
        """List available paths in the bucket"""
        paths = []
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket,
                Prefix=prefix,
                Delimiter='/'
            )
            
            for page in page_iterator:
                if 'CommonPrefixes' in page:
                    for prefix_info in page['CommonPrefixes']:
                        paths.append(prefix_info['Prefix'])
            
        except ClientError as e:
            self.logger.error(f"Error listing paths: {e}")
            
        return paths
    
    def find_data_type_paths(self, data_type: str) -> List[str]:
        """Find all paths containing a specific data type"""
        matching_paths = []
        
        # Search in us_stocks_sip
        paths = self.list_available_paths('us_stocks_sip/')
        
        for path in paths:
            if data_type.lower() in path.lower():
                matching_paths.append(path)
                self.logger.info(f"Found {data_type} path: {path}")
        
        return matching_paths
    
    def build_daily_s3_key(self, date_obj: date, data_type: str = 'minute_aggs') -> str:
        """Build S3 key for a daily file containing all symbols"""
        # Get base path for data type
        base_path = self.data_paths.get(data_type, f'us_stocks_sip/{data_type}_v1')
        
        # Format: base_path/YYYY/MM/YYYY-MM-DD.csv.gz
        key = f"{base_path}/{date_obj.year:04d}/{date_obj.month:02d}/{date_obj.strftime('%Y-%m-%d')}.csv.gz"
        
        return key
    
    def check_daily_file_exists(self, date_obj: date, data_type: str = 'minute_aggs') -> bool:
        """Check if a daily file exists in S3"""
        key = self.build_daily_s3_key(date_obj, data_type)
        
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                self.logger.error(f"Error checking file {key}: {e}")
                return False
    
    def download_daily_file(self, date_obj: date, output_dir: Path, 
                          data_type: str = 'minute_aggs',
                          progress_callback: Optional[Callable] = None,
                          max_retries: int = 3) -> Optional[Path]:
        """Download a daily file containing all symbols from S3 with retry logic"""
        key = self.build_daily_s3_key(date_obj, data_type)
        output_file = output_dir / f"{date_obj.strftime('%Y-%m-%d')}.csv.gz"
        
        # Skip if already downloaded
        if output_file.exists():
            self.logger.info(f"Daily file already exists: {output_file}")
            return output_file
        
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Downloading {key} (attempt {attempt + 1}/{max_retries})")
                
                # Download with progress tracking if callback provided
                if progress_callback:
                    response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
                    total_size = int(response['ContentLength'])
                    
                    with open(output_file, 'wb') as f:
                        bytes_downloaded = 0
                        for chunk in response['Body'].iter_chunks(chunk_size=8192):
                            f.write(chunk)
                            bytes_downloaded += len(chunk)
                            progress_callback(bytes_downloaded, total_size)
                else:
                    # Simple download
                    self.s3_client.download_file(self.bucket, key, str(output_file))
                
                self.logger.info(f"Successfully downloaded {output_file}")
                return output_file
                
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    self.logger.warning(f"File not found: {key}")
                    return None
                else:
                    self.logger.error(f"Download error (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        return None
            
            except Exception as e:
                self.logger.error(f"Unexpected error downloading {key}: {e}")
                return None
    
    def extract_symbols_from_daily_file(self, daily_file: Path, symbols: List[str],
                                      output_dir: Path, data_type: str = 'minute_aggs') -> Dict[str, Path]:
        """Extract specific symbols from a daily file"""
        import gzip
        import csv
        from io import StringIO
        
        extracted_files = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract date from filename
        date_str = daily_file.stem  # e.g., "2024-01-02"
        
        try:
            # Read the compressed file
            with gzip.open(daily_file, 'rt') as f:
                # Read header
                header = f.readline().strip()
                
                # Create output files for each symbol
                symbol_files = {}
                symbol_writers = {}
                
                for symbol in symbols:
                    output_file = output_dir / f"{symbol}_{date_str}.csv"
                    symbol_files[symbol] = output_file
                    file_handle = open(output_file, 'w')
                    writer = csv.writer(file_handle)
                    writer.writerow(header.split(','))
                    symbol_writers[symbol] = (file_handle, writer)
                
                # Process data line by line
                line_count = 0
                for line in f:
                    line_count += 1
                    if line_count % 100000 == 0:
                        self.logger.info(f"Processed {line_count} lines from {daily_file.name}")
                    
                    # Parse CSV line
                    row = line.strip().split(',')
                    if len(row) > 0:
                        ticker = row[0]  # First column is ticker
                        if ticker in symbols:
                            _, writer = symbol_writers[ticker]
                            writer.writerow(row)
                
                # Close all files
                for symbol, (file_handle, _) in symbol_writers.items():
                    file_handle.close()
                    
                    # Compress the extracted file
                    csv_file = symbol_files[symbol]
                    gz_file = csv_file.with_suffix('.csv.gz')
                    
                    with open(csv_file, 'rb') as f_in:
                        with gzip.open(gz_file, 'wb') as f_out:
                            f_out.writelines(f_in)
                    
                    # Remove uncompressed file
                    csv_file.unlink()
                    
                    extracted_files[symbol] = gz_file
                    self.logger.info(f"Extracted {symbol} to {gz_file}")
                
        except Exception as e:
            self.logger.error(f"Error extracting symbols from {daily_file}: {e}")
            
        return extracted_files
    
    def get_available_dates(self, year: int, month: int, 
                          data_type: str = 'minute_aggs') -> List[date]:
        """List available dates for a specific month"""
        dates = []
        prefix = f"{self.data_paths.get(data_type)}/{year:04d}/{month:02d}/"
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=1000
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Extract date from filename
                    filename = obj['Key'].split('/')[-1]
                    if filename.endswith('.csv.gz'):
                        # Format: YYYY-MM-DD.csv.gz
                        date_str = filename.replace('.csv.gz', '')
                        try:
                            date_obj = date.fromisoformat(date_str)
                            dates.append(date_obj)
                        except ValueError:
                            self.logger.warning(f"Invalid date format: {date_str}")
            
        except ClientError as e:
            self.logger.error(f"Error listing dates: {e}")
        
        return sorted(dates)
    
    def get_symbols_from_daily_file(self, daily_file: Path) -> List[str]:
        """Get list of unique symbols from a daily file (for discovery)"""
        import gzip
        
        symbols = set()
        
        try:
            with gzip.open(daily_file, 'rt') as f:
                # Skip header
                f.readline()
                
                # Read limited lines for discovery
                for i, line in enumerate(f):
                    if i > 10000:  # Sample first 10k lines
                        break
                    
                    row = line.strip().split(',')
                    if len(row) > 0:
                        ticker = row[0]
                        symbols.add(ticker)
            
            self.logger.info(f"Found {len(symbols)} unique symbols in {daily_file.name}")
            
        except Exception as e:
            self.logger.error(f"Error reading symbols from {daily_file}: {e}")
        
        return sorted(list(symbols))
    
    def download_date_range(self, start_date: date, end_date: date,
                          output_dir: Path, data_type: str = 'minute_aggs') -> List[Path]:
        """Download daily files for a date range"""
        downloaded_files = []
        
        # Iterate through days in range
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends (Saturday=5, Sunday=6)
            if current_date.weekday() < 5:
                output_file = self.download_daily_file(
                    date_obj=current_date,
                    output_dir=output_dir,
                    data_type=data_type
                )
                
                if output_file:
                    downloaded_files.append(output_file)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        return downloaded_files
    
    def download_and_extract_symbols(self, symbols: List[str], start_date: date, 
                                   end_date: date, output_dir: Path,
                                   daily_cache_dir: Optional[Path] = None,
                                   data_type: str = 'minute_aggs') -> Dict[str, List[Path]]:
        """Complete workflow: download daily files and extract specific symbols"""
        if daily_cache_dir is None:
            daily_cache_dir = output_dir / 'daily_cache'
        
        # Download daily files
        self.logger.info(f"Downloading daily files from {start_date} to {end_date}")
        daily_files = self.download_date_range(start_date, end_date, daily_cache_dir, data_type)
        
        # Extract symbols from each daily file
        symbol_files = {symbol: [] for symbol in symbols}
        
        for daily_file in daily_files:
            self.logger.info(f"Extracting symbols from {daily_file}")
            extracted = self.extract_symbols_from_daily_file(
                daily_file, symbols, output_dir, data_type
            )
            
            for symbol, file_path in extracted.items():
                symbol_files[symbol].append(file_path)
        
        return symbol_files
    
    def download_multiple_days_concurrent(self, dates: List[date], output_dir: Path,
                                        max_concurrent: int = 5,
                                        data_type: str = 'minute_aggs') -> Dict[str, Dict[str, Any]]:
        """Download multiple daily files concurrently"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit download tasks
            future_to_date = {
                executor.submit(
                    self.download_daily_file,
                    date_obj, output_dir, data_type
                ): date_obj
                for date_obj in dates
            }
            
            # Process completed downloads
            with tqdm(total=len(dates), desc="Downloading daily files") as pbar:
                for future in as_completed(future_to_date):
                    date_obj = future_to_date[future]
                    
                    try:
                        output_file = future.result()
                        results[date_obj.strftime('%Y-%m-%d')] = {
                            'success': output_file is not None,
                            'file': output_file,
                            'error': None
                        }
                    except Exception as e:
                        results[date_obj.strftime('%Y-%m-%d')] = {
                            'success': False,
                            'file': None,
                            'error': str(e)
                        }
                    
                    pbar.update(1)
        
        # Log summary
        successful = sum(1 for r in results.values() if r['success'])
        self.logger.info(f"Downloaded {successful}/{len(dates)} daily files successfully")
        
        return results
    
    def validate_data_file(self, file_path: Path) -> bool:
        """Validate downloaded data file"""
        try:
            # Try to read first few rows
            df = pd.read_csv(file_path, compression='gzip', nrows=10)
            
            # Check if it has data
            if len(df) == 0:
                return False
            
            # Check for some expected columns (flexible due to different formats)
            if len(df.columns) < 4:  # Should have at least OHLC
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating {file_path}: {e}")
            return False


# Example usage for date-based structure:
if __name__ == "__main__":
    from datetime import date
    from pathlib import Path
    
    # Initialize downloader
    downloader = PolygonDownloader()
    
    # Example 1: Download a single day
    single_day = downloader.download_daily_file(
        date_obj=date(2024, 1, 2),
        output_dir=Path("data/raw/minute_aggs/daily_files")
    )
    print(f"Downloaded: {single_day}")
    
    # Example 2: Extract specific symbols from daily file
    if single_day:
        symbols = ["SPY", "AAPL", "MSFT"]
        extracted = downloader.extract_symbols_from_daily_file(
            daily_file=single_day,
            symbols=symbols,
            output_dir=Path("data/raw/minute_aggs/by_symbol")
        )
        print(f"Extracted: {extracted}")
    
    # Example 3: Download date range and extract symbols
    symbol_files = downloader.download_and_extract_symbols(
        symbols=["SPY", "QQQ", "AAPL"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 5),
        output_dir=Path("data/raw/minute_aggs/by_symbol"),
        daily_cache_dir=Path("data/raw/minute_aggs/daily_cache")
    )
    print(f"Symbol files: {symbol_files}")