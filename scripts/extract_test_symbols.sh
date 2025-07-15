#!/bin/bash
# Extract specific symbols from the test file

echo "Extracting symbols from 2024-01-02 data..."

SYMBOLS="SPY QQQ AAPL MSFT GOOGL AMZN TSLA NVDA META JPM"
INPUT_FILE="data/raw/minute_aggs/daily_files/2024/01/2024-01-02.csv.gz"
OUTPUT_DIR="data/raw/minute_aggs/by_symbol"

# Create output directories
for symbol in $SYMBOLS; do
    mkdir -p $OUTPUT_DIR/$symbol
done

# Extract each symbol
for symbol in $SYMBOLS; do
    echo "Extracting $symbol..."
    gunzip -c $INPUT_FILE | grep -E "^ticker,|^$symbol," | gzip > $OUTPUT_DIR/$symbol/${symbol}_2024_01.csv.gz
    
    # Count rows
    rows=$(gunzip -c $OUTPUT_DIR/$symbol/${symbol}_2024_01.csv.gz | wc -l)
    echo "  Found $((rows-1)) rows for $symbol"
done

echo ""
echo "Sample SPY data:"
gunzip -c $OUTPUT_DIR/SPY/SPY_2024_01.csv.gz | head -10

echo ""
echo "File sizes:"
ls -lh $OUTPUT_DIR/*/*.csv.gz