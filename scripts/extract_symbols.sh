#!/bin/bash
# Extract specific symbols from daily minute aggregate files

YEAR=2024
SYMBOLS="SPY QQQ AAPL MSFT GOOGL AMZN TSLA NVDA META JPM"
INPUT_DIR="data/raw/minute_aggs/daily_files"
OUTPUT_DIR="data/raw/minute_aggs/by_symbol"

echo "============================================================"
echo "Extracting Symbols from Daily Files"
echo "Symbols: $SYMBOLS"
echo "============================================================"

# Create output directories for each symbol
for symbol in $SYMBOLS; do
    mkdir -p $OUTPUT_DIR/$symbol
done

# Process each month
for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
    echo ""
    echo "Processing $YEAR-$month..."
    
    # Check if month directory exists
    if [ ! -d "$INPUT_DIR/$YEAR/$month" ]; then
        echo "  Skipping $YEAR-$month (no data)"
        continue
    fi
    
    # Process each daily file in the month
    for file in $INPUT_DIR/$YEAR/$month/*.csv.gz; do
        if [ ! -f "$file" ]; then
            continue
        fi
        
        filename=$(basename "$file")
        date="${filename%.csv.gz}"
        
        echo "  Processing $date..."
        
        # Extract data for each symbol
        for symbol in $SYMBOLS; do
            # Extract symbol data and save to monthly file
            gunzip -c "$file" | grep -E "^ticker,|^$symbol," | gzip > $OUTPUT_DIR/$symbol/${symbol}_${YEAR}_${month}.csv.gz.tmp
            
            # If this is the first file of the month, move it
            if [ ! -f "$OUTPUT_DIR/$symbol/${symbol}_${YEAR}_${month}.csv.gz" ]; then
                mv $OUTPUT_DIR/$symbol/${symbol}_${YEAR}_${month}.csv.gz.tmp $OUTPUT_DIR/$symbol/${symbol}_${YEAR}_${month}.csv.gz
            else
                # Otherwise, append to existing file (skip header)
                gunzip -c $OUTPUT_DIR/$symbol/${symbol}_${YEAR}_${month}.csv.gz.tmp | tail -n +2 | gzip >> $OUTPUT_DIR/$symbol/${symbol}_${YEAR}_${month}.csv.gz
                rm $OUTPUT_DIR/$symbol/${symbol}_${YEAR}_${month}.csv.gz.tmp
            fi
        done
    done
    
    echo "  Completed $YEAR-$month"
done

echo ""
echo "============================================================"
echo "Extraction Complete!"
echo "============================================================"
echo ""
echo "Symbol files location: $OUTPUT_DIR/"
echo ""

# Show results
for symbol in $SYMBOLS; do
    echo "$symbol files:"
    ls -lh $OUTPUT_DIR/$symbol/*.csv.gz 2>/dev/null | head -5
    echo ""
done