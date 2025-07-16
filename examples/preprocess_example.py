"""
Example: Using the data preprocessor with real market data
"""

from pathlib import Path
import pandas as pd

from src.data.preprocessor import DataPreprocessor
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Example of preprocessing market data"""
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        raw_data_dir=Path("data/raw/minute_aggs/by_symbol"),
        processed_data_dir=Path("data/processed"),
        cache_dir=Path("data/cache")
    )
    
    # Process January 2024 SPY data
    logger.info("Processing SPY data for January 2024...")
    df = preprocessor.process("SPY", months=["2024_01"])
    
    # Display basic statistics
    print("\n=== Processed Data Summary ===")
    print(f"Symbol: SPY")
    print(f"Period: January 2024")
    print(f"Total bars: {len(df):,}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nPrice Statistics:")
    print(f"  Mean: ${df['close'].mean():.2f}")
    print(f"  Min:  ${df['close'].min():.2f}")
    print(f"  Max:  ${df['close'].max():.2f}")
    print(f"  Std:  ${df['close'].std():.2f}")
    print(f"\nVolume Statistics:")
    print(f"  Total: {df['volume'].sum():,.0f}")
    print(f"  Mean:  {df['volume'].mean():,.0f}")
    print(f"  Max:   {df['volume'].max():,.0f}")
    
    # Check for missing bars (filled with 0 volume)
    filled_bars = len(df[df['volume'] == 0])
    print(f"\nFilled bars: {filled_bars} ({filled_bars/len(df)*100:.1f}%)")
    
    # Save processed data
    output_path = preprocessor.save_processed(df, "SPY", "2024_01")
    print(f"\nSaved to: {output_path}")
    
    # Show sample of data
    print("\n=== Sample Data (first 5 market minutes) ===")
    print(df.head())
    
    # Demonstrate loading cached data
    print("\n=== Loading from cache ===")
    cached_df = preprocessor.load_processed("SPY", "2024_01")
    if cached_df is not None:
        print(f"Successfully loaded {len(cached_df)} bars from cache")
    

if __name__ == "__main__":
    main()