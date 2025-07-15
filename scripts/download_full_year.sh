#!/bin/bash
# Download full year of minute data efficiently

YEAR=2024
OUTPUT_DIR="data/raw/minute_aggs"
SYMBOLS="SPY QQQ AAPL MSFT GOOGL AMZN TSLA NVDA META JPM"

echo "============================================================"
echo "Polygon.io Full Year Download Script"
echo "Year: $YEAR"
echo "Symbols: $SYMBOLS"
echo "============================================================"
echo ""
echo "This will download all daily files for $YEAR."
echo "Estimated size: ~300-400MB per month (3.6-4.8GB total)"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Create directories
mkdir -p $OUTPUT_DIR/daily_files/$YEAR
mkdir -p logs

# Download function
download_month() {
    local month=$1
    echo ""
    echo "============================================================"
    echo "Downloading $YEAR-$month..."
    echo "============================================================"
    
    mkdir -p $OUTPUT_DIR/daily_files/$YEAR/$month
    
    # Download with progress
    rclone copy s3polygon:flatfiles/us_stocks_sip/minute_aggs_v1/$YEAR/$month/ \
        $OUTPUT_DIR/daily_files/$YEAR/$month/ \
        --progress \
        --transfers 5 \
        --stats 10s
    
    # Count files
    file_count=$(ls $OUTPUT_DIR/daily_files/$YEAR/$month/*.csv.gz 2>/dev/null | wc -l)
    total_size=$(du -sh $OUTPUT_DIR/daily_files/$YEAR/$month/ | cut -f1)
    
    echo ""
    echo "Completed $YEAR-$month: $file_count files, $total_size"
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

# Summary
total_files=$(find $OUTPUT_DIR/daily_files/$YEAR -name "*.csv.gz" | wc -l)
total_size=$(du -sh $OUTPUT_DIR/daily_files/$YEAR | cut -f1)

echo "Total files downloaded: $total_files"
echo "Total size: $total_size"
echo ""
echo "Daily files location: $OUTPUT_DIR/daily_files/$YEAR/"
echo ""
echo "Next step: Run ./scripts/extract_symbols_year.sh to extract individual symbols"