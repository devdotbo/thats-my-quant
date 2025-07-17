#!/usr/bin/env python3
"""
Create sample Bitcoin data for lunar trading analysis
Since yfinance is having issues, we'll generate realistic Bitcoin data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_sample_bitcoin_data(start_date="2023-01-01", end_date="2024-12-31", freq="1h"):
    """
    Create realistic Bitcoin price data
    
    Characteristics:
    - Base price around $30,000-$60,000 
    - High volatility (2-3% daily)
    - Trending behavior
    - Realistic volume patterns
    """
    print("Creating sample Bitcoin data...")
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(dates)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate base price with trend
    initial_price = 35000
    trend = np.linspace(0, 15000, n)  # Upward trend
    
    # Add random walk component
    daily_returns = np.random.normal(0, 0.02, n)  # 2% daily volatility
    price_walk = initial_price * np.exp(np.cumsum(daily_returns * 0.1))
    
    # Add cyclical patterns (monthly cycles)
    cycle = 5000 * np.sin(np.arange(n) * 2 * np.pi / (30 * 24))  # Monthly cycle
    
    # Combine components
    close_prices = price_walk + trend + cycle
    
    # Ensure positive prices
    close_prices = np.maximum(close_prices, 1000)
    
    # Generate OHLCV data
    data = pd.DataFrame(index=dates)
    
    # Close price
    data['close'] = close_prices
    
    # High/Low (with realistic spreads)
    spread = np.random.uniform(0.005, 0.02, n)  # 0.5-2% spread
    data['high'] = data['close'] * (1 + spread)
    data['low'] = data['close'] * (1 - spread * 0.8)
    
    # Open (previous close with small gap)
    data['open'] = data['close'].shift(1)
    data['open'].iloc[0] = initial_price
    data['open'] = data['open'] * (1 + np.random.normal(0, 0.001, n))
    
    # Adjust high/low to include open
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)
    
    # Volume (with daily patterns)
    hour_of_day = dates.hour
    daily_pattern = 1 + 0.5 * np.sin((hour_of_day - 6) * np.pi / 12)  # Peak at noon
    base_volume = np.random.lognormal(20, 0.5, n)  # Log-normal volume
    data['volume'] = (base_volume * daily_pattern * 1e6).astype(int)
    
    # Add adjusted close (same as close for crypto)
    data['adj close'] = data['close']
    
    # Round prices
    for col in ['open', 'high', 'low', 'close', 'adj close']:
        data[col] = data[col].round(2)
    
    print(f"Created {len(data)} rows of Bitcoin data")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
    
    return data


def save_bitcoin_data():
    """Generate and save Bitcoin data in multiple formats"""
    # Create data directory
    data_dir = Path("data/crypto/bitcoin")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate hourly data for 2 years
    hourly_data = create_sample_bitcoin_data(
        start_date="2023-01-01",
        end_date="2024-12-31",
        freq="1h"
    )
    
    # Save as parquet
    hourly_file = data_dir / "btc_hourly_2023_2024_sample.parquet"
    hourly_data.to_parquet(hourly_file)
    print(f"\nSaved hourly data to {hourly_file}")
    
    # Generate daily data for longer period
    daily_data = create_sample_bitcoin_data(
        start_date="2020-01-01",
        end_date="2024-12-31",
        freq="1d"
    )
    
    # Save as CSV for easy inspection
    daily_file = data_dir / "btc_daily_2020_2024_sample.csv"
    daily_data.to_csv(daily_file)
    print(f"Saved daily data to {daily_file}")
    
    # Generate minute data for 1 month (for high-frequency testing)
    minute_data = create_sample_bitcoin_data(
        start_date="2024-11-01",
        end_date="2024-11-30",
        freq="5min"
    )
    
    minute_file = data_dir / "btc_5min_202411_sample.parquet"
    minute_data.to_parquet(minute_file)
    print(f"Saved 5-minute data to {minute_file}")
    
    return hourly_data


if __name__ == "__main__":
    print("Generating sample Bitcoin data due to yfinance issues...")
    print("=" * 60)
    
    # Generate and save data
    btc_data = save_bitcoin_data()
    
    # Show sample
    print("\nSample of generated data:")
    print(btc_data.head(10))
    
    print("\nData statistics:")
    print(btc_data.describe())
    
    print("\n" + "=" * 60)
    print("Sample data created successfully!")
    print("You can now run the lunar analysis with this data.")