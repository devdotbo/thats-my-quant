# Data Architecture and Pipeline Design

## Overview

This document outlines the data architecture for efficiently handling large-scale financial market data from Polygon.io's flat files, including storage, processing, and retrieval strategies optimized for backtesting performance.

## Data Sources and Types

### Polygon.io Flat Files Structure

#### Available Data Types
1. **Trades** - Every individual trade
   - Timestamp (nanosecond precision)
   - Price
   - Size
   - Exchange
   - Conditions

2. **Quotes** - Best bid/ask updates
   - Timestamp
   - Bid price/size
   - Ask price/size
   - Exchange

3. **Aggregates** - OHLCV bars
   - Minute aggregates
   - Daily aggregates
   - Volume
   - Trade count
   - VWAP

#### File Organization
```
flatfiles/
├── us_stocks_sip/
│   ├── trades/
│   │   └── 2024/01/01/AAPL.csv.gz
│   ├── quotes/
│   │   └── 2024/01/01/AAPL.csv.gz
│   └── aggregates/
│       ├── minute/2024/01/01/AAPL.csv.gz
│       └── day/2024/01/AAPL.csv.gz
```

## Data Pipeline Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Polygon.io S3  │────▶│ Download     │────▶│  Raw Storage    │
│   Flat Files    │     │  Manager     │     │  (Compressed)   │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Backtesting   │◀────│ Cache Layer  │◀────│  Processing     │
│     Engine      │     │  (Parquet)   │     │    Engine       │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## Storage Strategy

### Three-Tier Storage System

#### Tier 1: Raw Compressed Storage
- **Format**: Original CSV.gz files from Polygon
- **Location**: `data/raw/{data_type}/{year}/{month}/{day}/{symbol}.csv.gz`
- **Retention**: 30 days (configurable)
- **Purpose**: Backup and reprocessing

#### Tier 2: Processed Cache
- **Format**: Apache Parquet with compression
- **Location**: `data/cache/{data_type}/{symbol}/{year}_{month}.parquet`
- **Retention**: Unlimited (manual cleanup)
- **Purpose**: Fast loading for backtesting

#### Tier 3: Memory Cache
- **Format**: Pandas DataFrames in RAM
- **Size**: Configurable (default 8GB)
- **Purpose**: Ultra-fast repeated access

### Storage Format Comparison

| Format | Read Speed | Write Speed | Compression | Query Support |
|--------|------------|-------------|-------------|---------------|
| CSV.gz | Slow | Slow | Good | No |
| Parquet | Fast | Medium | Excellent | Yes |
| HDF5 | Fast | Fast | Good | Limited |
| Feather | Very Fast | Very Fast | Poor | No |

**Decision**: Use Parquet for optimal balance of speed, compression, and features.

## Data Models

### Trade Data Schema
```python
@dataclass
class Trade:
    timestamp: pd.Timestamp  # Nanosecond precision
    symbol: str
    price: float
    size: int
    exchange: str
    conditions: List[int]
    
    class Meta:
        indexes = ['timestamp', 'symbol']
```

### Quote Data Schema
```python
@dataclass
class Quote:
    timestamp: pd.Timestamp
    symbol: str
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    bid_exchange: str
    ask_exchange: str
    
    class Meta:
        indexes = ['timestamp', 'symbol']
```

### Aggregate Data Schema
```python
@dataclass
class Aggregate:
    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trades: int
    
    class Meta:
        indexes = ['timestamp', 'symbol']
```

## Download Manager Implementation

```python
class PolygonDataDownloader:
    """Manages data downloads from Polygon.io S3"""
    
    def __init__(self, config: Config):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=config.polygon_s3_endpoint,
            aws_access_key_id=config.polygon_s3_access_key,
            aws_secret_access_key=config.polygon_s3_secret
        )
        self.bucket = config.polygon_s3_bucket
        
    async def download_batch(self, 
                           symbols: List[str], 
                           dates: List[datetime],
                           data_type: str) -> None:
        """Download multiple files concurrently"""
        tasks = []
        for symbol in symbols:
            for date in dates:
                task = self.download_file(symbol, date, data_type)
                tasks.append(task)
        
        await asyncio.gather(*tasks)
        
    def download_file(self, symbol: str, date: datetime, 
                     data_type: str) -> Path:
        """Download single file from S3"""
        key = self._build_s3_key(symbol, date, data_type)
        local_path = self._get_local_path(symbol, date, data_type)
        
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(
                self.bucket, key, str(local_path)
            )
        
        return local_path
```

## Data Processing Pipeline

### Processing Steps

1. **Decompression and Loading**
```python
def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load compressed CSV with proper dtypes"""
    dtypes = {
        'symbol': 'category',
        'exchange': 'category',
        'conditions': 'string'
    }
    
    df = pd.read_csv(
        file_path,
        compression='gzip',
        dtype=dtypes,
        parse_dates=['timestamp'],
        date_parser=lambda x: pd.to_datetime(x, unit='ns')
    )
    
    return df
```

2. **Data Cleaning**
```python
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate data"""
    # Remove duplicates
    df = df.drop_duplicates(['timestamp', 'symbol'])
    
    # Filter out erroneous prices
    df = df[df['price'] > 0]
    df = df[df['price'] < df['price'].quantile(0.9999)]
    
    # Handle missing values
    df = df.dropna(subset=['price', 'size'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    return df
```

3. **Feature Engineering**
```python
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for analysis"""
    # Spread calculation
    if 'bid_price' in df.columns:
        df['spread'] = df['ask_price'] - df['bid_price']
        df['spread_pct'] = df['spread'] / df['mid_price']
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
    
    # Volume features
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    return df
```

4. **Conversion to Parquet**
```python
def save_to_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Save processed data to Parquet format"""
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=True,
        partition_cols=['year', 'month']
    )
```

## Cache Management

### Smart Caching Strategy

```python
class DataCache:
    """Intelligent data caching with LRU eviction"""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory = max_memory_gb * 1024 * 1024 * 1024
        self.cache = OrderedDict()
        self.memory_usage = 0
        
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache, update LRU order"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
        
    def put(self, key: str, data: pd.DataFrame) -> None:
        """Add data to cache with memory management"""
        data_size = data.memory_usage(deep=True).sum()
        
        # Evict old data if necessary
        while (self.memory_usage + data_size > self.max_memory 
               and len(self.cache) > 0):
            self._evict_oldest()
            
        self.cache[key] = data
        self.memory_usage += data_size
        
    def _evict_oldest(self) -> None:
        """Remove least recently used item"""
        key, data = self.cache.popitem(last=False)
        self.memory_usage -= data.memory_usage(deep=True).sum()
```

### Cache Key Strategy
```python
def build_cache_key(symbol: str, data_type: str, 
                   start_date: datetime, end_date: datetime) -> str:
    """Build consistent cache keys"""
    return f"{symbol}_{data_type}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
```

## Data Quality Assurance

### Validation Checks

```python
class DataValidator:
    """Validate data quality and integrity"""
    
    def validate_trades(self, df: pd.DataFrame) -> ValidationResult:
        """Validate trade data"""
        checks = []
        
        # Check for gaps
        time_diff = df['timestamp'].diff()
        max_gap = time_diff.max()
        if max_gap > pd.Timedelta(minutes=5):
            checks.append(ValidationError(
                "Large time gaps detected", 
                severity="WARNING"
            ))
            
        # Check for outliers
        price_std = df['price'].std()
        price_mean = df['price'].mean()
        outliers = df[
            np.abs(df['price'] - price_mean) > 5 * price_std
        ]
        if len(outliers) > 0:
            checks.append(ValidationError(
                f"{len(outliers)} price outliers detected",
                severity="WARNING"
            ))
            
        # Check monotonic timestamps
        if not df['timestamp'].is_monotonic_increasing:
            checks.append(ValidationError(
                "Timestamps not monotonic",
                severity="ERROR"
            ))
            
        return ValidationResult(checks)
```

## Performance Optimization

### Parallel Processing
```python
def process_symbols_parallel(symbols: List[str], 
                           processor: Callable) -> Dict[str, pd.DataFrame]:
    """Process multiple symbols in parallel"""
    with multiprocessing.Pool() as pool:
        results = pool.map(processor, symbols)
    
    return dict(zip(symbols, results))
```

### Chunked Loading
```python
def load_large_dataset(file_path: Path, chunk_size: int = 1_000_000):
    """Load large files in chunks to manage memory"""
    chunks = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        processed_chunk = process_chunk(chunk)
        chunks.append(processed_chunk)
        
    return pd.concat(chunks, ignore_index=True)
```

### Index Optimization
```python
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame for fast lookups"""
    # Set multi-index for common queries
    df = df.set_index(['timestamp', 'symbol']).sort_index()
    
    # Convert to categorical for memory savings
    for col in ['exchange', 'conditions']:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    # Downcast numerics where possible
    df = downcast_numeric_dtypes(df)
    
    return df
```

## Data Update Strategy

### Incremental Updates
```python
class IncrementalUpdater:
    """Handle incremental data updates efficiently"""
    
    def update_data(self, symbol: str) -> None:
        """Update data with only new records"""
        # Get last timestamp in cache
        last_timestamp = self.get_last_timestamp(symbol)
        
        # Download only new data
        new_data = self.download_since(symbol, last_timestamp)
        
        # Merge with existing data
        if not new_data.empty:
            self.merge_and_save(symbol, new_data)
```

## Monitoring and Metrics

### Data Pipeline Metrics
```python
@dataclass
class PipelineMetrics:
    download_time: float
    processing_time: float
    records_processed: int
    errors_count: int
    cache_hit_rate: float
    memory_usage: float
    
    def log_metrics(self):
        """Log metrics for monitoring"""
        logger.info(f"Pipeline metrics: {self.__dict__}")
```

## Best Practices

1. **Always validate data** before processing
2. **Use appropriate data types** to minimize memory usage
3. **Implement retry logic** for network operations
4. **Monitor cache performance** and adjust size as needed
5. **Regularly clean old data** to manage storage
6. **Use async I/O** for concurrent downloads
7. **Compress data** when storing long-term
8. **Index appropriately** for query patterns

## Configuration

### data_config.yaml
```yaml
storage:
  raw_data_dir: "./data/raw"
  cache_dir: "./data/cache"
  max_cache_memory_gb: 8
  retention_days: 30
  
processing:
  chunk_size: 1000000
  parallel_workers: 4
  
download:
  concurrent_downloads: 10
  retry_attempts: 3
  timeout_seconds: 300
  
validation:
  max_price_deviation_std: 5
  max_time_gap_minutes: 5
  min_daily_trades: 100
```