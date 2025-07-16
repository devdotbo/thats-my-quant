#!/usr/bin/env python3
"""Parallel extraction of symbols from daily files"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.downloader import PolygonDownloader
from pathlib import Path
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def extract_month(args):
    """Extract symbols for a single month"""
    year, month, symbols = args
    
    downloader = PolygonDownloader()
    daily_dir = Path(f"data/raw/minute_aggs/daily_files/{year:04d}/{month:02d}")
    output_base = Path("data/raw/minute_aggs/by_symbol")
    
    if not daily_dir.exists():
        return f"{year}-{month:02d}: No data"
    
    daily_files = sorted(daily_dir.glob("*.csv.gz"))
    if not daily_files:
        return f"{year}-{month:02d}: No files"
    
    # Process each daily file
    total_rows = {symbol: 0 for symbol in symbols}
    
    for daily_file in daily_files:
        # Extract symbols from this daily file
        extracted = downloader.extract_symbols_from_daily_file(
            daily_file=daily_file,
            symbols=symbols,
            output_dir=output_base / "temp"
        )
        
        # Count rows for each symbol
        for symbol, temp_file in extracted.items():
            if temp_file.exists():
                import pandas as pd
                df = pd.read_csv(temp_file, compression='gzip')
                total_rows[symbol] += len(df)
    
    # Combine all daily files for each symbol into monthly file
    for symbol in symbols:
        temp_files = sorted((output_base / "temp").glob(f"{symbol}_*.csv.gz"))
        if temp_files:
            # Combine into monthly file
            output_file = output_base / symbol / f"{symbol}_{year}_{month:02d}.csv.gz"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Read all temp files and combine
            dfs = []
            for temp_file in temp_files:
                df = pd.read_csv(temp_file, compression='gzip')
                dfs.append(df)
                temp_file.unlink()  # Remove temp file
            
            # Save combined file
            combined = pd.concat(dfs, ignore_index=True)
            combined.to_csv(output_file, compression='gzip', index=False)
    
    # Clean up temp directory
    if (output_base / "temp").exists():
        (output_base / "temp").rmdir()
    
    result = f"{year}-{month:02d}: "
    result += ", ".join(f"{s}={total_rows[s]}" for s in symbols)
    return result


def main():
    YEAR = 2024
    SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM"]
    
    print("=" * 60)
    print("Parallel Symbol Extraction")
    print(f"Year: {YEAR}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print("=" * 60)
    
    # Create tasks for each month
    tasks = [(YEAR, month, SYMBOLS) for month in range(1, 13)]
    
    start_time = time.time()
    
    # Process in parallel (limit workers to avoid memory issues)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(extract_month, task): task for task in tasks}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                task = futures[future]
                print(f"Error processing {task[0]}-{task[1]:02d}: {e}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal extraction time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()