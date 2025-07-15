#!/bin/bash
# Download just a few days of data for testing

echo "============================================================"
echo "Downloading Sample Days from January 2024"
echo "============================================================"

# Create output directory
mkdir -p data/raw/minute_aggs/daily_files/2024/01

# Download specific days
echo "Downloading 2024-01-02..."
rclone copy s3polygon:flatfiles/us_stocks_sip/minute_aggs_v1/2024/01/2024-01-02.csv.gz \
    data/raw/minute_aggs/daily_files/2024/01/ -v

echo "Downloading 2024-01-03..."
rclone copy s3polygon:flatfiles/us_stocks_sip/minute_aggs_v1/2024/01/2024-01-03.csv.gz \
    data/raw/minute_aggs/daily_files/2024/01/ -v

echo ""
echo "Download complete!"
echo ""

# Show what was downloaded
echo "Downloaded files:"
ls -lh data/raw/minute_aggs/daily_files/2024/01/
echo ""

# Extract our symbols from these files
echo "Extracting symbols..."
mkdir -p data/raw/minute_aggs/by_symbol

SYMBOLS="SPY QQQ AAPL MSFT GOOGL AMZN TSLA NVDA META JPM"

for symbol in $SYMBOLS; do
    echo "Extracting $symbol..."
    mkdir -p data/raw/minute_aggs/by_symbol/$symbol
    
    # Extract from each day file
    for file in data/raw/minute_aggs/daily_files/2024/01/*.csv.gz; do
        if [ -f "$file" ]; then
            date=$(basename "$file" .csv.gz)
            gunzip -c "$file" | grep -E "^ticker,|^$symbol," | gzip > data/raw/minute_aggs/by_symbol/$symbol/${symbol}_${date}.csv.gz
        fi
    done
done

echo ""
echo "Extraction complete!"
echo ""
echo "Sample data for SPY:"
gunzip -c data/raw/minute_aggs/by_symbol/SPY/SPY_2024-01-02.csv.gz | head -10