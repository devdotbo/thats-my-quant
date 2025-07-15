"""
Polygon.io S3 Data Downloader
Downloads historical market data from Polygon.io flat files
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

from src.utils.config import get_config
from src.utils.logging import get_logger


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
    
    def build_s3_key(self, symbol: str, year: int, month: int, 
                     data_type: str = 'minute_aggs') -> str:
        """Build S3 key for a specific file"""
        # Get base path for data type
        base_path = self.data_paths.get(data_type, f'us_stocks_sip/{data_type}_v1')
        
        # Format: base_path/YYYY/MM/SYMBOL.csv.gz
        key = f"{base_path}/{year:04d}/{month:02d}/{symbol.upper()}.csv.gz"
        
        return key
    
    def check_file_exists(self, symbol: str, year: int, month: int,
                         data_type: str = 'minute_aggs') -> bool:
        """Check if a specific file exists in S3"""
        key = self.build_s3_key(symbol, year, month, data_type)
        
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                self.logger.error(f"Error checking file {key}: {e}")
                return False
    
    def download_single_file(self, symbol: str, year: int, month: int,
                           output_dir: Path, data_type: str = 'minute_aggs',
                           progress_callback: Optional[Callable] = None,
                           max_retries: int = 3) -> Optional[Path]:
        """Download a single file from S3 with retry logic"""
        key = self.build_s3_key(symbol, year, month, data_type)
        output_file = output_dir / f"{symbol}_{year:04d}_{month:02d}_{data_type}.csv.gz"
        
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
    
    def get_available_symbols(self, year: int, month: int, 
                            data_type: str = 'minute_aggs') -> List[str]:
        """List available symbols for a specific month"""
        symbols = []
        prefix = f"{self.data_paths.get(data_type)}/{year:04d}/{month:02d}/"
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=1000
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Extract symbol from filename
                    filename = obj['Key'].split('/')[-1]
                    if filename.endswith('.csv.gz'):
                        symbol = filename.replace('.csv.gz', '')
                        symbols.append(symbol)
            
        except ClientError as e:
            self.logger.error(f"Error listing symbols: {e}")
        
        return sorted(symbols)
    
    def download_date_range(self, symbol: str, start_date: date, end_date: date,
                          output_dir: Path, data_type: str = 'minute_aggs') -> List[Path]:
        """Download data for a date range"""
        downloaded_files = []
        
        # Iterate through months in range
        current_date = start_date.replace(day=1)
        end_month = end_date.replace(day=1)
        
        while current_date <= end_month:
            output_file = self.download_single_file(
                symbol=symbol,
                year=current_date.year,
                month=current_date.month,
                output_dir=output_dir,
                data_type=data_type
            )
            
            if output_file:
                downloaded_files.append(output_file)
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return downloaded_files
    
    def download_multiple_symbols(self, symbols: List[str], year: int, month: int,
                                output_dir: Path, max_concurrent: int = 5,
                                data_type: str = 'minute_aggs') -> Dict[str, Dict[str, Any]]:
        """Download multiple symbols concurrently"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit download tasks
            future_to_symbol = {
                executor.submit(
                    self.download_single_file,
                    symbol, year, month, output_dir, data_type
                ): symbol
                for symbol in symbols
            }
            
            # Process completed downloads
            with tqdm(total=len(symbols), desc="Downloading symbols") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    
                    try:
                        output_file = future.result()
                        results[symbol] = {
                            'success': output_file is not None,
                            'file': output_file,
                            'error': None
                        }
                    except Exception as e:
                        results[symbol] = {
                            'success': False,
                            'file': None,
                            'error': str(e)
                        }
                    
                    pbar.update(1)
        
        # Log summary
        successful = sum(1 for r in results.values() if r['success'])
        self.logger.info(f"Downloaded {successful}/{len(symbols)} symbols successfully")
        
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