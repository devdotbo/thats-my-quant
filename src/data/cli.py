"""
Command line interface for Polygon data downloader
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.data.downloader import PolygonDownloader
from src.utils.logging import setup_logging, get_logger


def parse_args(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Download market data from Polygon.io'
    )
    
    parser.add_argument(
        '--symbols',
        required=True,
        help='Comma-separated list of symbols (e.g., SPY,AAPL,MSFT)'
    )
    
    parser.add_argument(
        '--start-date',
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--data-type',
        default='minute_aggs',
        choices=['minute_aggs', 'trades', 'quotes', 'day_aggs'],
        help='Type of data to download'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data/raw',
        help='Output directory for downloaded files'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=5,
        help='Maximum concurrent downloads'
    )
    
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args(args)


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = get_logger(__name__)
    
    logger.info("Starting Polygon data download")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Data type: {args.data_type}")
    
    # Parse inputs
    symbols = [s.strip() for s in args.symbols.split(',')]
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    output_dir = Path(args.output_dir)
    
    # Create downloader
    try:
        downloader = PolygonDownloader()
    except ValueError as e:
        logger.error(f"Failed to initialize downloader: {e}")
        logger.error("Please ensure Polygon.io credentials are set in .env file")
        sys.exit(1)
    
    # Download data
    all_results = {}
    total_files = 0
    
    # Process each month in the date range
    current_date = start_date.replace(day=1)
    end_month = end_date.replace(day=1)
    
    while current_date <= end_month:
        logger.info(f"\nDownloading data for {current_date.strftime('%Y-%m')}")
        
        # Download all symbols for this month
        results = downloader.download_multiple_symbols(
            symbols=symbols,
            year=current_date.year,
            month=current_date.month,
            output_dir=output_dir / args.data_type,
            max_concurrent=args.max_concurrent,
            data_type=args.data_type
        )
        
        # Track results
        for symbol, result in results.items():
            if symbol not in all_results:
                all_results[symbol] = []
            all_results[symbol].append(result)
            if result['success']:
                total_files += 1
        
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    
    for symbol in symbols:
        if symbol in all_results:
            successful = sum(1 for r in all_results[symbol] if r['success'])
            total = len(all_results[symbol])
            logger.info(f"{symbol}: {successful}/{total} months downloaded")
    
    logger.info(f"\nTotal files downloaded: {total_files}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # List downloaded files
    downloaded_files = list(output_dir.glob(f"**/*.csv.gz"))
    if downloaded_files:
        logger.info(f"\nDownloaded files:")
        for f in sorted(downloaded_files)[:10]:  # Show first 10
            logger.info(f"  {f.name}")
        if len(downloaded_files) > 10:
            logger.info(f"  ... and {len(downloaded_files) - 10} more")
    
    return 0 if total_files > 0 else 1


if __name__ == "__main__":
    sys.exit(main())