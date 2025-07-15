#!/bin/bash
# Quick test of the Polygon downloader

echo "Testing Polygon data downloader..."
echo "=================================="

# Download just January 2024 data for a few symbols
python -m src.data \
    --symbols SPY,AAPL,MSFT \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --data-type minute_aggs \
    --output-dir data/test \
    --log-level INFO

echo ""
echo "Checking downloaded files:"
echo "--------------------------"
find data/test -name "*.csv.gz" -type f | sort

echo ""
echo "Disk usage:"
echo "-----------"
du -sh data/test/