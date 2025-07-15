#!/bin/bash
# Script to monitor download progress of full year data

echo "=== Polygon Data Download Progress ==="
echo

# Check if download is still running
if ps aux | grep -q "[d]ownload_full_year.sh"; then
    echo "✅ Download script is still running"
    PID=$(ps aux | grep "[d]ownload_full_year.sh" | awk '{print $2}')
    echo "   Process ID: $PID"
else
    echo "❌ Download script is not running"
fi

echo
echo "=== Downloaded Months ==="

# Count downloaded months
YEAR=2024
DATA_DIR="data/raw/minute_aggs/daily_files/$YEAR"

if [ -d "$DATA_DIR" ]; then
    # List months with file count
    for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
        if [ -d "$DATA_DIR/$month" ]; then
            file_count=$(ls "$DATA_DIR/$month"/*.csv.gz 2>/dev/null | wc -l)
            size=$(du -sh "$DATA_DIR/$month" 2>/dev/null | cut -f1)
            echo "✅ $YEAR-$month: $file_count files, $size"
        else
            echo "⏳ $YEAR-$month: Not downloaded yet"
        fi
    done
    
    echo
    echo "=== Total Progress ==="
    total_months=$(ls -d "$DATA_DIR"/* 2>/dev/null | wc -l)
    total_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
    echo "Downloaded: $total_months/12 months"
    echo "Total size: $total_size"
    
    # Estimate time remaining
    if [ $total_months -gt 0 ] && [ $total_months -lt 12 ]; then
        # Rough estimate based on current progress
        remaining=$((12 - total_months))
        echo "Remaining: $remaining months"
        echo "Estimated time: ~$(($remaining * 7)) minutes"
    fi
else
    echo "Data directory not found: $DATA_DIR"
fi

echo
echo "=== Next Steps ==="
echo "When download completes, run:"
echo "  ./scripts/extract_symbols_year.sh"