#!/bin/bash
# Extract symbols from full year of daily files

YEAR=2024
SYMBOLS="SPY QQQ AAPL MSFT GOOGL AMZN TSLA NVDA META JPM"
INPUT_DIR="data/raw/minute_aggs/daily_files"
OUTPUT_DIR="data/raw/minute_aggs/by_symbol"

echo "============================================================"
echo "Extracting Symbols from $YEAR Daily Files"
echo "Symbols: $SYMBOLS"
echo "============================================================"

# Create output directories
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
    
    # Count files in month
    file_count=$(ls $INPUT_DIR/$YEAR/$month/*.csv.gz 2>/dev/null | wc -l)
    if [ $file_count -eq 0 ]; then
        echo "  No files found in $YEAR-$month"
        continue
    fi
    
    echo "  Found $file_count daily files"
    
    # Process each symbol
    for symbol in $SYMBOLS; do
        echo -n "  Extracting $symbol..."
        
        # Create temporary file for this month
        temp_file="$OUTPUT_DIR/$symbol/${symbol}_${YEAR}_${month}_temp.csv"
        output_file="$OUTPUT_DIR/$symbol/${symbol}_${YEAR}_${month}.csv.gz"
        
        # Extract header once
        first_file=$(ls $INPUT_DIR/$YEAR/$month/*.csv.gz | head -1)
        gunzip -c "$first_file" | head -1 > "$temp_file"
        
        # Extract symbol data from all files in the month
        for file in $INPUT_DIR/$YEAR/$month/*.csv.gz; do
            gunzip -c "$file" | grep "^$symbol," >> "$temp_file"
        done
        
        # Compress the result
        gzip -c "$temp_file" > "$output_file"
        rm "$temp_file"
        
        # Count rows
        rows=$(gunzip -c "$output_file" | wc -l)
        echo " $((rows-1)) rows"
    done
    
    echo "  Completed $YEAR-$month"
done

echo ""
echo "============================================================"
echo "Extraction Complete!"
echo "============================================================"
echo ""

# Summary
for symbol in $SYMBOLS; do
    echo "$symbol:"
    total_rows=0
    file_count=0
    
    for file in $OUTPUT_DIR/$symbol/*.csv.gz; do
        if [ -f "$file" ]; then
            rows=$(gunzip -c "$file" | wc -l)
            total_rows=$((total_rows + rows - 1))  # Subtract header
            file_count=$((file_count + 1))
        fi
    done
    
    total_size=$(du -sh $OUTPUT_DIR/$symbol/ 2>/dev/null | cut -f1)
    echo "  Files: $file_count"
    echo "  Total rows: $total_rows"
    echo "  Total size: $total_size"
    echo ""
done

echo "All extracted data location: $OUTPUT_DIR/"