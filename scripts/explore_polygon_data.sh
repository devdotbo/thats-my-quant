#!/bin/bash
# Explore Polygon.io data structure

echo "============================================================"
echo "Polygon.io Bucket Structure Explorer"
echo "============================================================"

# Create data directories if not exists
mkdir -p data/references

# List main directories
echo -e "\n1. Main directories in flatfiles bucket:"
echo "----------------------------------------"
rclone lsd s3polygon:flatfiles

# List stock data structure
echo -e "\n2. US Stocks SIP structure:"
echo "----------------------------"
rclone lsd s3polygon:flatfiles/us_stocks_sip/

# Find minute aggregates versions
echo -e "\n3. Looking for minute aggregates:"
echo "----------------------------------"
rclone ls s3polygon:flatfiles/us_stocks_sip/ --include "*minute*" --max-depth 1 | head -20

# Check for versioned paths
echo -e "\n4. Checking data versions (v1, v2, etc):"
echo "-----------------------------------------"
rclone lsd s3polygon:flatfiles/us_stocks_sip/ | grep -E "v[0-9]"

# Sample path for SPY
echo -e "\n5. Finding SPY data paths:"
echo "--------------------------"
echo "Searching for SPY files (this may take a moment)..."
rclone ls s3polygon:flatfiles --include "**/SPY.csv.gz" --max-depth 4 | head -10

# Save directory structure for reference
echo -e "\n6. Saving complete directory structure..."
echo "-----------------------------------------"
echo "Saving to: data/references/polygon_dirs.txt"
rclone lsd s3polygon:flatfiles -R > data/references/polygon_dirs.txt

# Save a smaller sample of actual files
echo -e "\nSaving sample file listing to: data/references/polygon_sample_files.txt"
rclone ls s3polygon:flatfiles/us_stocks_sip/ --max-depth 3 | head -1000 > data/references/polygon_sample_files.txt

echo -e "\n============================================================"
echo "Exploration complete!"
echo "Check these files for reference:"
echo "  - data/references/polygon_dirs.txt (directory structure)"
echo "  - data/references/polygon_sample_files.txt (sample files)"
echo "============================================================"

# Show summary of what we found
echo -e "\nQuick summary of available data:"
if [ -f data/references/polygon_dirs.txt ]; then
    echo "Total directories found: $(wc -l < data/references/polygon_dirs.txt)"
    echo ""
    echo "Data types found:"
    grep -E "(trades|quotes|minute|day)" data/references/polygon_dirs.txt | sort | uniq | head -10
fi