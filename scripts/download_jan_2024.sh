#!/bin/bash
# Quick download of just January 2024 data for testing

echo "============================================================"
echo "Downloading January 2024 Minute Data"
echo "============================================================"

# Create output directory
mkdir -p data/raw/minute_aggs/daily_files/2024/01

# Download January 2024
echo "Starting download..."
rclone copy s3polygon:flatfiles/us_stocks_sip/minute_aggs_v1/2024/01/ \
    data/raw/minute_aggs/daily_files/2024/01/ \
    --progress \
    --transfers 5

echo ""
echo "Download complete!"
echo ""

# Show what was downloaded
echo "Downloaded files:"
ls -lh data/raw/minute_aggs/daily_files/2024/01/ | head -10
echo ""

# Count files
echo "Total files: $(ls data/raw/minute_aggs/daily_files/2024/01/*.csv.gz | wc -l)"
echo "Total size: $(du -sh data/raw/minute_aggs/daily_files/2024/01/)"