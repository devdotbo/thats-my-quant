#!/bin/bash
# Download Bitcoin minute data from Polygon.io crypto flat files
# Based on the global_crypto data structure discovered in flatfiles.txt

# Configuration
YEARS="2021 2022 2023 2024"  # Start with recent years
OUTPUT_DIR="data/crypto/bitcoin"
CRYPTO_PATH="global_crypto/minute_aggs_v1"

echo "============================================================"
echo "Polygon.io Bitcoin Data Download Script"
echo "============================================================"
echo "Data path: $CRYPTO_PATH"
echo "Years: $YEARS"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directories
mkdir -p $OUTPUT_DIR/daily_files
mkdir -p $OUTPUT_DIR/processed

# Function to download a month of crypto data
download_crypto_month() {
    local year=$1
    local month=$2
    echo ""
    echo "Downloading crypto data for $year-$month..."
    
    # Create year/month directory
    mkdir -p $OUTPUT_DIR/daily_files/$year/$month
    
    # Download all daily files for the month
    rclone copy s3polygon:flatfiles/$CRYPTO_PATH/$year/$month/ \
        $OUTPUT_DIR/daily_files/$year/$month/ \
        --progress \
        --transfers 5 \
        --max-backlog 10000
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed downloading $year-$month"
    else
        echo "✗ Failed to download $year-$month"
        echo "  This might be due to access restrictions on crypto data"
        echo "  Check your Polygon.io subscription level"
    fi
}

# Function to extract Bitcoin pairs from daily files
extract_bitcoin_pairs() {
    local year=$1
    local month=$2
    local output_file="$OUTPUT_DIR/processed/btc_${year}_${month}.csv"
    
    echo "Extracting Bitcoin pairs from $year-$month..."
    
    # Common Bitcoin trading pairs to look for
    local btc_pairs="BTC-USD|BTCUSD|X:BTCUSD|BTC-USDT|BTCUSDT|BTC-EUR|BTCEUR"
    
    # Process each daily file in the month
    for file in $OUTPUT_DIR/daily_files/$year/$month/*.csv.gz; do
        if [ -f "$file" ]; then
            # Extract Bitcoin rows (case insensitive)
            zgrep -E -i "^($btc_pairs)," "$file" >> "$output_file.tmp" 2>/dev/null
        fi
    done
    
    # Sort and deduplicate
    if [ -f "$output_file.tmp" ]; then
        sort -u "$output_file.tmp" > "$output_file"
        rm "$output_file.tmp"
        echo "✓ Extracted Bitcoin data to $output_file"
        
        # Show sample of tickers found
        echo "  Sample tickers found:"
        cut -d',' -f1 "$output_file" | sort -u | head -5 | sed 's/^/    - /'
    fi
}

# Main download loop
echo ""
echo "Starting downloads..."
echo "===================="

for year in $YEARS; do
    echo ""
    echo "Processing year $year"
    echo "--------------------"
    
    # Download each month
    for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
        download_crypto_month $year $month
        
        # If download succeeded, extract Bitcoin data
        if [ -d "$OUTPUT_DIR/daily_files/$year/$month" ] && [ "$(ls -A $OUTPUT_DIR/daily_files/$year/$month)" ]; then
            extract_bitcoin_pairs $year $month
        fi
    done
done

echo ""
echo "============================================================"
echo "Download Process Complete!"
echo "============================================================"
echo ""
echo "Daily files location: $OUTPUT_DIR/daily_files/"
echo "Processed Bitcoin data: $OUTPUT_DIR/processed/"
echo ""

# Show disk usage
echo "Storage summary:"
echo "----------------"
if [ -d "$OUTPUT_DIR/daily_files" ]; then
    echo "Daily files:"
    du -sh $OUTPUT_DIR/daily_files/ 2>/dev/null || echo "  No data downloaded"
fi
if [ -d "$OUTPUT_DIR/processed" ]; then
    echo "Processed Bitcoin data:"
    du -sh $OUTPUT_DIR/processed/ 2>/dev/null || echo "  No data processed"
fi

echo ""
echo "Next steps:"
echo "-----------"
echo "1. Check if crypto data downloaded successfully"
echo "2. If access denied, verify your Polygon.io subscription includes crypto data"
echo "3. Run moon phase analysis on the extracted Bitcoin data"
echo ""
echo "To test with a single day first:"
echo "  rclone copy s3polygon:flatfiles/$CRYPTO_PATH/2024/01/2024-01-01.csv.gz tmp/"
echo "  zcat tmp/2024-01-01.csv.gz | grep -i btc | head"