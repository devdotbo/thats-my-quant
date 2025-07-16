"""
Example: Adding technical indicators and features to market data
"""

from pathlib import Path
import pandas as pd

from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngine
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Example of feature engineering on preprocessed data"""
    
    # First, load preprocessed data
    preprocessor = DataPreprocessor(
        raw_data_dir=Path("data/raw/minute_aggs/by_symbol"),
        processed_data_dir=Path("data/processed"),
        cache_dir=Path("data/cache")
    )
    
    # Load January 2024 SPY data (already preprocessed)
    logger.info("Loading preprocessed SPY data...")
    df = preprocessor.load_processed("SPY", "2024_01")
    
    if df is None:
        # If not cached, process it first
        logger.info("Processing SPY data for January 2024...")
        df = preprocessor.process("SPY", months=["2024_01"])
    
    print(f"\n=== Original Data ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Initialize feature engine
    feature_engine = FeatureEngine(cache_dir=Path("data/cache"))
    
    # Add specific features
    logger.info("\nAdding technical indicators...")
    
    # 1. Moving averages
    df_with_ma = feature_engine.add_moving_averages(
        df, 
        periods=[5, 20, 50],
        ma_types=['sma', 'ema']
    )
    print(f"\nAdded moving averages: {[c for c in df_with_ma.columns if 'ma_' in c]}")
    
    # 2. RSI
    df_with_rsi = feature_engine.add_momentum_indicators(
        df_with_ma,
        indicators=['rsi'],
        rsi_period=14
    )
    print(f"Added RSI: {df_with_rsi['rsi_14'].iloc[-1]:.2f}")
    
    # 3. Bollinger Bands
    df_with_bb = feature_engine.add_volatility_indicators(
        df_with_rsi,
        indicators=['bollinger'],
        bb_period=20,
        bb_std=2
    )
    print(f"Added Bollinger Bands - Width: ${df_with_bb['bb_width'].iloc[-1]:.2f}")
    
    # 4. VWAP
    df_with_vwap = feature_engine.add_volume_analytics(
        df_with_bb,
        indicators=['vwap'],
        vwap_period=None  # Session VWAP
    )
    print(f"Added VWAP: ${df_with_vwap['vwap'].iloc[-1]:.2f}")
    
    # Or add all features at once
    logger.info("\nAdding all features at once...")
    df_all_features = feature_engine.add_all_features(df, cache_key="SPY_2024_01_features")
    
    print(f"\n=== Feature-Enhanced Data ===")
    print(f"Shape: {df_all_features.shape}")
    print(f"Total features added: {len(df_all_features.columns) - len(df.columns)}")
    
    # Show some statistics
    print("\n=== Feature Statistics (last bar) ===")
    last_bar = df_all_features.iloc[-1]
    print(f"Close Price: ${last_bar['close']:.2f}")
    print(f"SMA 20: ${last_bar['sma_20']:.2f}")
    print(f"RSI 14: {last_bar['rsi_14']:.2f}")
    print(f"ATR 14: ${last_bar['atr_14']:.2f}")
    print(f"BB %B: {last_bar['bb_percent']:.2f}")
    print(f"VWAP: ${last_bar['vwap']:.2f}")
    print(f"20-day Volatility: {last_bar['volatility_20']:.2%}")
    
    # Check for any extreme values
    print("\n=== Data Quality Check ===")
    numeric_cols = df_all_features.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df_all_features[col].isin([float('inf'), float('-inf')]).any():
            print(f"WARNING: {col} contains infinite values")
        nan_count = df_all_features[col].isna().sum()
        if nan_count > 0:
            print(f"{col}: {nan_count} NaN values ({nan_count/len(df_all_features)*100:.1f}%)")
    
    # Save enhanced data
    output_path = Path("data/processed/SPY/SPY_2024_01_with_features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all_features.to_parquet(output_path, compression='snappy')
    print(f"\nSaved feature-enhanced data to: {output_path}")


if __name__ == "__main__":
    main()