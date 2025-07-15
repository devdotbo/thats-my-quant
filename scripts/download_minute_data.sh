#!/bin/bash
# Download minute aggregates data from Polygon.io
# Data is organized by date, not by symbol

YEAR=2024
SYMBOLS="SPY QQQ AAPL MSFT GOOGL AMZN TSLA NVDA META JPM"
OUTPUT_DIR="data/raw/minute_aggs"

echo "============================================================"
echo "Downloading Polygon.io Minute Data for $YEAR"
echo "Symbols: $SYMBOLS"
echo "============================================================"

# Create output directories
mkdir -p $OUTPUT_DIR/daily_files
mkdir -p $OUTPUT_DIR/by_symbol

# Function to download a month
download_month() {
    local month=$1
    echo ""
    echo "Downloading data for $YEAR-$month..."
    
    # Download all daily files for the month
    rclone copy s3polygon:flatfiles/us_stocks_sip/minute_aggs_v1/$YEAR/$month/ \
        $OUTPUT_DIR/daily_files/$YEAR/$month/ \
        --progress \
        --transfers 5
    
    echo "Completed downloading $YEAR-$month"
}

# Download all months
for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
    download_month $month
done

echo ""
echo "============================================================"
echo "Download Complete!"
echo "============================================================"
echo ""
echo "Daily files location: $OUTPUT_DIR/daily_files/"
echo ""
echo "To extract specific symbols, run:"
echo "./scripts/extract_symbols.sh"
echo ""

# Show disk usage
echo "Disk usage:"
du -sh $OUTPUT_DIR/daily_files/