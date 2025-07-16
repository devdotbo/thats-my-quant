"""
Data Preprocessor Module
Handles cleaning and preparation of raw market data for backtesting
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
import gzip
from datetime import time
import pytz

from src.utils.logging import get_logger
from src.data.cache import CacheManager


class DataPreprocessor:
    """
    Preprocesses raw market data for backtesting
    
    Features:
    - Converts nanosecond timestamps to datetime
    - Fills missing minute bars during market hours
    - Removes outliers using IQR method
    - Validates data integrity
    - Saves processed data in Parquet format
    """
    
    def __init__(self, 
                 raw_data_dir: Union[str, Path],
                 processed_data_dir: Union[str, Path],
                 cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the data preprocessor
        
        Args:
            raw_data_dir: Directory containing raw symbol data
            processed_data_dir: Directory to save processed data
            cache_dir: Optional cache directory for processed files
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("preprocessor")
        
        # Initialize cache if directory provided
        self.cache_manager = None
        if cache_dir:
            self.cache_manager = CacheManager(cache_dir=Path(cache_dir))
        
        # Market hours in EST/EDT
        self.market_tz = pytz.timezone('America/New_York')
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
    
    def load_symbol_data(self, symbol: str, months: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load and combine symbol data from monthly CSV files
        
        Args:
            symbol: Stock symbol to load
            months: List of months to load (format: YYYY_MM). If None, loads all available
            
        Returns:
            Combined DataFrame with all requested data
            
        Example:
            >>> df = preprocessor.load_symbol_data("SPY", months=["2024_01", "2024_02"])
            >>> print(f"Loaded {len(df)} rows for SPY")
        """
        symbol_dir = self.raw_data_dir / symbol
        
        if not symbol_dir.exists():
            raise ValueError(f"No data found for symbol {symbol} at {symbol_dir}")
        
        # Get list of files to load
        if months:
            files = [symbol_dir / f"{symbol}_{month}.csv.gz" for month in months]
            files = [f for f in files if f.exists()]
        else:
            files = sorted(symbol_dir.glob(f"{symbol}_*.csv.gz"))
        
        if not files:
            raise ValueError(f"No data files found for {symbol}")
        
        self.logger.info(f"Loading {len(files)} files for {symbol}")
        
        # Load and combine all files
        dfs = []
        for file in files:
            with gzip.open(file, 'rt') as f:
                df = pd.read_csv(f)
                dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Loaded {len(combined_df)} rows for {symbol}")
        
        return combined_df
    
    def convert_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert nanosecond timestamps to datetime index
        
        Args:
            df: DataFrame with 'window_start' column in nanoseconds
            
        Returns:
            DataFrame with datetime index in market timezone
        """
        # Convert nanoseconds to datetime
        df['datetime'] = pd.to_datetime(df['window_start'], unit='ns')
        
        # Localize to UTC first, then convert to market timezone
        df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(self.market_tz)
        
        # Set as index
        df = df.set_index('datetime').sort_index()
        
        # Remove the original timestamp column
        df = df.drop(columns=['window_start'])
        
        self.logger.debug(f"Converted timestamps: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def fill_missing_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing minute bars during market hours
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with missing bars filled
        """
        if len(df) == 0:
            return df
        
        # Get the date range from the data
        start_time = df.index[0].replace(hour=9, minute=30, second=0, microsecond=0)
        end_time = df.index[-1].replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Create complete market hours index for the same day(s)
        market_hours = pd.date_range(
            start=start_time,
            end=end_time,
            freq='1min',
            tz=df.index.tz
        )
        
        # Filter to market hours only
        market_hours = market_hours[
            (market_hours.time >= self.market_open) & 
            (market_hours.time <= self.market_close)
        ]
        
        # Filter to weekdays only (no weekends)
        market_hours = market_hours[market_hours.weekday < 5]
        
        # Reindex to include all market minutes
        df_filled = df.reindex(market_hours)
        
        # Forward fill price data
        price_cols = ['open', 'high', 'low', 'close']
        df_filled[price_cols] = df_filled[price_cols].ffill()
        
        # For filled bars, set open=high=low=close
        is_filled = df_filled['volume'].isna()
        for col in ['open', 'high', 'low']:
            df_filled.loc[is_filled, col] = df_filled.loc[is_filled, 'close']
        
        # Set volume and transactions to 0 for filled bars
        df_filled['volume'] = df_filled['volume'].fillna(0)
        df_filled['transactions'] = df_filled['transactions'].fillna(0)
        
        # Keep ticker column if present
        if 'ticker' in df_filled.columns:
            df_filled['ticker'] = df_filled['ticker'].ffill()
        
        bars_added = len(df_filled) - len(df)
        if bars_added > 0:
            self.logger.info(f"Added {bars_added} missing bars")
        
        return df_filled
    
    def clean_outliers(self, df: pd.DataFrame, iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using IQR method
        
        Args:
            df: DataFrame with price and volume data
            iqr_multiplier: Multiplier for IQR range (default 1.5)
            
        Returns:
            DataFrame with outliers replaced
        """
        df_cleaned = df.copy()
        
        # Clean price outliers
        for col in ['open', 'high', 'low', 'close']:
            if col in df_cleaned.columns:
                # Calculate IQR
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                # Find outliers
                outliers = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                n_outliers = outliers.sum()
                
                if n_outliers > 0:
                    # Replace outliers with rolling median
                    rolling_median = df_cleaned[col].rolling(window=5, center=True, min_periods=1).median()
                    df_cleaned.loc[outliers, col] = rolling_median[outliers]
                    self.logger.info(f"Cleaned {n_outliers} outliers in {col}")
        
        # Clean volume outliers
        if 'volume' in df_cleaned.columns:
            Q1 = df_cleaned['volume'].quantile(0.25)
            Q3 = df_cleaned['volume'].quantile(0.75)
            IQR = Q3 - Q1
            
            upper_bound = Q3 + iqr_multiplier * IQR
            outliers = df_cleaned['volume'] > upper_bound
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                # Cap volume at upper bound
                df_cleaned.loc[outliers, 'volume'] = upper_bound
                self.logger.info(f"Cleaned {n_outliers} volume outliers")
        
        return df_cleaned
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data integrity
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] < 0).any():
                raise ValueError(f"Negative prices found in {col}")
        
        # Check for NaN values in critical columns
        for col in ['open', 'high', 'low', 'close']:
            if df[col].isna().any():
                raise ValueError(f"NaN values found in {col}")
        
        # Check OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        if invalid_ohlc.any():
            raise ValueError("Invalid OHLC relationships found")
        
        # Check datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")
        
        if df.index.tz is None:
            raise ValueError("Index must have timezone info")
        
        self.logger.info("Data validation passed")
        return True
    
    def process(self, symbol: str, months: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run full preprocessing pipeline
        
        Args:
            symbol: Stock symbol to process
            months: List of months to process
            
        Returns:
            Cleaned and validated DataFrame ready for backtesting
            
        Example:
            >>> processed = preprocessor.process("SPY", months=["2024_01"])
            >>> print(f"Processed {len(processed)} bars for SPY")
        """
        self.logger.info(f"Processing {symbol}")
        
        # Load data
        df = self.load_symbol_data(symbol, months)
        
        # Convert timestamps
        df = self.convert_timestamps(df)
        
        # Fill missing bars
        df = self.fill_missing_bars(df)
        
        # Clean outliers
        df = self.clean_outliers(df)
        
        # Validate
        self.validate_data(df)
        
        # Keep only required columns
        final_cols = ['open', 'high', 'low', 'close', 'volume', 'transactions']
        df = df[[col for col in final_cols if col in df.columns]]
        
        self.logger.info(f"Processing complete: {len(df)} bars")
        
        return df
    
    def save_processed(self, df: pd.DataFrame, symbol: str, period: str) -> Path:
        """
        Save processed data to Parquet format
        
        Args:
            df: Processed DataFrame
            symbol: Stock symbol
            period: Period identifier (e.g., "2024_01" or "2024")
            
        Returns:
            Path to saved file
        """
        # Create symbol directory
        symbol_dir = self.processed_data_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        # Save as Parquet
        output_file = symbol_dir / f"{symbol}_{period}.parquet"
        df.to_parquet(
            output_file,
            compression='snappy',
            index=True  # Preserve datetime index
        )
        
        self.logger.info(f"Saved processed data to {output_file}")
        
        # Add to cache if available
        if self.cache_manager:
            self.cache_manager.cache_file(
                output_file,
                f"processed/{symbol}/{period}",
                category="processed_data"
            )
        
        return output_file
    
    def load_processed(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Load previously processed data
        
        Args:
            symbol: Stock symbol
            period: Period identifier
            
        Returns:
            DataFrame if found, None otherwise
        """
        # Check cache first
        if self.cache_manager:
            cached_path = self.cache_manager.get_cached_file(f"processed/{symbol}/{period}")
            if cached_path and cached_path.exists():
                return pd.read_parquet(cached_path)
        
        # Check processed directory
        file_path = self.processed_data_dir / symbol / f"{symbol}_{period}.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
        
        return None