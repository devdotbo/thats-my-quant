
import yfinance as yf
import pandas as pd
from pathlib import Path
from src.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO", format="simple")
logger = get_logger(__name__)

# --- Configuration ---
SYMBOL = "BTC-USD"
START_DATE = "2014-01-01"  # yfinance has data starting from 2014-09-17 for BTC-USD
END_DATE = pd.to_datetime("today").strftime("%Y-%m-%d")

# Define output path
output_dir = Path(f"data/raw/crypto")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"{SYMBOL}.csv"


def download_btc_data():
    """
    Downloads the full historical daily data for BTC-USD from Yahoo Finance.
    """
    logger.info(f"Downloading historical data for {SYMBOL}...")
    logger.info(f"Date Range: {START_DATE} to {END_DATE}")

    try:
        # Download data using yfinance
        btc_data = yf.download(SYMBOL, start=START_DATE, end=END_DATE, progress=True)

        if btc_data.empty:
            logger.error("No data downloaded. The symbol may be incorrect or there might be no data for the specified date range.")
            return

        # yfinance column names are capitalized, convert to lowercase to match project convention
        btc_data.columns = [col.lower() for col in btc_data.columns]
        btc_data.rename(columns={"adj close": "adj_close"}, inplace=True)

        # Save to CSV
        btc_data.to_csv(output_file)
        logger.info(f"Successfully downloaded {len(btc_data)} rows of data.")
        logger.info(f"Data saved to: {output_file}")

    except Exception as e:
        logger.error(f"An error occurred during download: {e}", exc_info=True)

if __name__ == "__main__":
    download_btc_data()
