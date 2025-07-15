# Polygon.io Data Structure Insights

## Critical Discovery: Data Organized by DATE, Not Symbol

### File Structure
```
flatfiles/
└── us_stocks_sip/
    └── minute_aggs_v1/
        └── YYYY/           # Year (e.g., 2024)
            └── MM/         # Month (e.g., 01)
                ├── YYYY-MM-DD.csv.gz  # Daily file with ALL symbols
                ├── 2024-01-02.csv.gz
                ├── 2024-01-03.csv.gz
                └── ...
```

### Key Insights

1. **Data Organization**
   - Each file contains minute data for ALL symbols traded that day
   - Files are named by date: `YYYY-MM-DD.csv.gz`
   - No per-symbol files available directly from Polygon
   - Must download full daily files and extract needed symbols

2. **File Format**
```csv
ticker,volume,open,close,high,low,window_start,transactions
A,356,139.03,139.03,139.03,139.03,1704200400000000000,1
AAPL,12345,180.25,180.50,180.75,180.00,1704200400000000000,150
SPY,20460,476.25,476.31,476.36,476,1704186000000000000,84
...
```

3. **Data Sizes**
   - **Daily file**: ~15-20MB compressed (contains ~5000+ symbols)
   - **Monthly data**: ~300-400MB compressed (20-22 trading days)
   - **Yearly data**: ~3.6-4.8GB compressed (~252 trading days)
   - **Single symbol/month**: ~100-500KB after extraction
   - **Single symbol/year**: ~1-6MB after extraction

4. **Extraction Results** (from 2024-01-02)
   - SPY: 821 minute bars
   - AAPL: 847 minute bars
   - MSFT: 613 minute bars
   - QQQ: 829 minute bars
   - (390 expected for full day, extra likely pre/post market)

### Download Strategy

1. **Download by Date Range**
   ```bash
   # Download all daily files for a month
   rclone copy s3polygon:flatfiles/us_stocks_sip/minute_aggs_v1/2024/01/ \
       data/raw/minute_aggs/daily_files/2024/01/
   ```

2. **Extract Symbols**
   ```bash
   # Extract specific symbols from daily files
   for file in data/raw/minute_aggs/daily_files/2024/01/*.csv.gz; do
       gunzip -c "$file" | grep -E "^ticker,|^SPY," >> SPY_2024_01.csv
   done
   ```

3. **Storage Efficiency**
   - Download once: 3.6-4.8GB for full year
   - Extract 10 symbols: ~10-60MB total
   - 100x reduction in storage for our use case

### Credential Configuration

**Important**: Polygon uses the API key as the S3 secret access key
```python
s3_client = boto3.client(
    's3',
    endpoint_url='https://files.polygon.io',
    aws_access_key_id='your-s3-access-key-id',
    aws_secret_access_key='your-polygon-api-key'  # Use API key here!
)
```

### Rclone Configuration
```
[s3polygon]
type = s3
env_auth = false
access_key_id = your-s3-access-key-id
secret_access_key = your-polygon-api-key  # Use API key as secret
endpoint = https://files.polygon.io
```

### Implications for Implementation

1. **Downloader Updates Needed**
   - Change from symbol-based to date-based downloads
   - Implement extraction process after download
   - Consider caching daily files for multiple symbol extraction

2. **Cache Strategy**
   - Cache daily files temporarily during extraction
   - Store extracted symbol files permanently
   - Implement cleanup of old daily files

3. **Performance Considerations**
   - Parallel downloads of daily files
   - Stream processing for extraction (don't load full file)
   - Consider keeping frequently used daily files cached