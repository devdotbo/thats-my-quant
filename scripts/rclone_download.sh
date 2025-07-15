#!/bin/bash
# Rclone download script for Polygon.io flat files

# Configuration
SYMBOLS="SPY QQQ AAPL MSFT GOOGL AMZN TSLA NVDA META JPM"
YEAR="2024"
OUTPUT_DIR="data/raw/minute_aggs"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting Polygon.io data download..."
echo "Symbols: $SYMBOLS"
echo "Year: $YEAR"
echo "Output: $OUTPUT_DIR"
echo "=================================="

# Download minute aggregates for each symbol
for symbol in $SYMBOLS; do
    echo "Downloading $symbol for year $YEAR..."
    
    # Download all months for the symbol
    rclone copy \
        polygon:flatfiles/us_stocks_sip/minute_aggs/$YEAR/ \
        $OUTPUT_DIR/$YEAR/ \
        --include "*/${symbol}.csv.gz" \
        --progress \
        --transfers 5
    
    echo "Completed $symbol"
    echo "----------------------------------"
done

echo "Download complete!"
echo "Data location: $OUTPUT_DIR"

# Show downloaded files
echo ""
echo "Downloaded files:"
find $OUTPUT_DIR -name "*.csv.gz" -type f | sort