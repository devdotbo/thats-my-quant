#!/bin/bash
# Test script to check Polygon crypto data access

echo "Testing Polygon.io Crypto Data Access"
echo "====================================="

# Test 1: List crypto directories
echo -e "\n1. Checking available crypto data types:"
rclone lsd s3polygon:flatfiles/global_crypto/ 2>/dev/null | grep -v "NOTICE" || echo "Failed to list crypto directories"

# Test 2: Check minute aggregates structure
echo -e "\n2. Checking crypto minute aggregates years:"
rclone lsd s3polygon:flatfiles/global_crypto/minute_aggs_v1/ 2>/dev/null | grep -v "NOTICE" | head -10 || echo "Failed to list years"

# Test 3: Try to download a small recent file
echo -e "\n3. Testing download of a single day (2024-01-01):"
mkdir -p tmp/crypto_test
rclone copy s3polygon:flatfiles/global_crypto/minute_aggs_v1/2024/01/2024-01-01.csv.gz tmp/crypto_test/ 2>&1 | grep -v "NOTICE"

# Check if download succeeded
if [ -f "tmp/crypto_test/2024-01-01.csv.gz" ]; then
    echo "✓ Download successful!"
    echo -e "\n4. Examining file contents:"
    echo "File size: $(ls -lh tmp/crypto_test/2024-01-01.csv.gz | awk '{print $5}')"
    echo -e "\nFirst 10 lines:"
    zcat tmp/crypto_test/2024-01-01.csv.gz | head -10
    echo -e "\nBitcoin entries (first 5):"
    zcat tmp/crypto_test/2024-01-01.csv.gz | grep -i btc | head -5
else
    echo "✗ Download failed - crypto data may require additional permissions"
    echo ""
    echo "Alternative data sources to consider:"
    echo "1. CryptoDataDownload.com - Free historical crypto data"
    echo "2. Binance API - Free with rate limits"
    echo "3. CoinGecko API - Free tier available"
    echo "4. Yahoo Finance - BTC-USD historical data"
fi

echo -e "\n====================================="