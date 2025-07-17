# Technical Challenges: Detailed Issues and Solutions

## Overview

This document catalogs all technical challenges encountered during development, their root causes, and solutions or workarounds applied.

## 1. Multi-Core Parallelization Failures

### Challenge: Optuna n_jobs Parameter Ignored

**Symptom:**
```python
study.optimize(objective, n_trials=100, n_jobs=16)
# Still runs single-threaded
```

**Root Cause:**
- Objective function captures too much state
- VectorBT engine not pickleable
- Large data objects can't be serialized efficiently

**Failed Solution:**
```python
def objective(trial):
    # This captures 'self' and 'data' from closure
    strategy = self.create_strategy(trial.params)
    result = self.engine.run_backtest(strategy, self.data)
    return result.sharpe_ratio
```

**Workaround:**
Accept single-threaded execution. Still 10-100x faster than grid search.

### Challenge: ProcessPoolExecutor Overhead

**Symptom:**
```python
with ProcessPoolExecutor(max_workers=16) as executor:
    results = executor.map(optimize_worker, tasks)
# Slower than single-threaded
```

**Root Cause:**
- Process spawn time: ~2 seconds
- Data serialization: 100MB per process
- Total overhead: 32+ seconds before first computation

**Measurement:**
```
Single-threaded: 50 trials in 100 seconds (0.5 trials/sec)
16 processes: 50 trials in 150 seconds (0.33 trials/sec)
```

**Lesson Learned:**
Process parallelization only helps if: `computation_time >> serialization_time + spawn_time`

### Challenge: SQLite Lock Contention

**Symptom:**
```
sqlite3.OperationalError: database is locked
```

**Root Cause:**
- SQLite allows multiple readers but only one writer
- Optuna trials write results immediately
- 16 processes = constant write contention

**Failed Mitigation:**
```python
# Tried increasing timeout
storage = "sqlite:///optimization.db?timeout=30"
# Still caused serialization of writes
```

**Conclusion:**
SQLite not suitable for high-concurrency optimization.

## 2. VectorBT State Management

### Challenge: Global Settings Conflict

**Code:**
```python
vbt.settings.portfolio['init_cash'] = 10000
# This is GLOBAL - affects all instances
```

**Issue:**
- Multiple strategies can't have different settings
- Parallel processes overwrite each other's settings
- Not thread-safe or process-safe

**Solution:**
```python
# Use instance parameters instead
portfolio = vbt.Portfolio.from_signals(
    init_cash=10000,  # Instance-specific
    ...
)
```

### Challenge: Memory Usage with Large Datasets

**Symptom:**
```
Process killed: Memory usage exceeded 32GB
```

**Root Cause:**
- VectorBT keeps full price history in memory
- Calculates metrics on entire arrays
- No streaming/chunking support

**Solution:**
```python
# Process in smaller chunks
def backtest_in_chunks(data, chunk_size=50000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        result = engine.run_backtest(strategy, chunk)
        results.append(result)
    return combine_results(results)
```

## 3. Data Pipeline Issues

### Challenge: Polygon.io Structure Discovery

**Initial Assumption:**
```
Data organized by symbol: /AAPL/2024/01/data.csv
```

**Reality:**
```
Data organized by date: /2024/01/2024-01-15.csv.gz
Each file contains ALL symbols for that day
```

**Impact:**
- Had to rewrite entire downloader
- Download 15GB to get 100MB of symbol data
- Required post-processing extraction step

**Solution:**
```python
def download_and_extract_symbols(date, symbols):
    # Download full day file
    daily_file = download_date(date)
    
    # Extract only needed symbols
    df = pd.read_csv(daily_file)
    symbol_data = df[df['sym'].isin(symbols)]
    
    # Save by symbol
    for symbol in symbols:
        symbol_df = symbol_data[symbol_data['sym'] == symbol]
        save_symbol_data(symbol, date, symbol_df)
```

### Challenge: Timestamp Timezone Handling

**Issue:**
```python
# Polygon provides epoch nanoseconds
timestamp = 1704204000000000000
# Must convert to tz-aware datetime
```

**Multiple Attempts:**
```python
# Attempt 1: Naive conversion (wrong!)
dt = pd.to_datetime(timestamp, unit='ns')

# Attempt 2: UTC assumption (wrong for US markets!)
dt = pd.to_datetime(timestamp, unit='ns', utc=True)

# Correct: US/Eastern timezone
dt = pd.to_datetime(timestamp, unit='ns').tz_localize('UTC').tz_convert('US/Eastern')
```

## 4. Strategy Implementation Pitfalls

### Challenge: Parameter Validation Mismatch

**Error:**
```
ValidationError: ma_type must be one of ['sma', 'ema', 'wma']
# But Optuna suggested 'SMA' (uppercase)
```

**Root Cause:**
- Strategy expects lowercase
- Optuna doesn't know this constraint
- Validation happens after trial starts

**Solution:**
```python
# In strategy
ma_type = ma_type.lower()  # Normalize input

# Or in Optuna space
'ma_type': {'type': 'categorical', 'choices': ['sma', 'ema']}  # Use lowercase
```

### Challenge: Position Sizing Type Error

**Error:**
```python
TypeError: '>' not supported between instances of 'str' and 'float'
```

**Root Cause:**
```python
# Strategy returns string
position_size = 'volatility'  # This is wrong!

# VectorBT expects float
if position_size > 0:  # Fails here
```

**Fix:**
Remove position_size from parameter space, handle internally.

## 5. Testing Philosophy Challenges

### Challenge: No Mocking Policy

**Policy:**
```python
# NEVER mock external services
# Use real connections always
```

**Challenges:**
1. Tests require internet connection
2. Tests can fail due to service outages
3. Tests use real API quotas
4. Tests are slower

**Benefits Realized:**
1. Caught Polygon.io authentication issue
2. Discovered data structure mismatch
3. Found timezone handling bugs
4. Revealed performance bottlenecks

**Conclusion:**
Policy justified despite inconveniences.

## 6. Performance Optimization Challenges

### Challenge: Feature Calculation Speed

**Initial:**
```python
# Calculate each indicator separately
df['sma_20'] = df['close'].rolling(20).mean()
df['sma_50'] = df['close'].rolling(50).mean()
# 30 seconds for 100k bars
```

**Optimized:**
```python
# Vectorized operations
close_array = df['close'].values
sma_20 = np.convolve(close_array, np.ones(20)/20, mode='valid')
# 0.1 seconds for 100k bars
```

### Challenge: Backtest Memory Usage

**Issue:**
Portfolio object stores all intermediate calculations

**Solution:**
```python
# Clear unnecessary data after metrics calculation
portfolio.close()  # Frees memory
del portfolio._trades  # Remove trade details
gc.collect()  # Force garbage collection
```

## 7. API Design Mismatches

### Challenge: CommissionModel Parameter Names

**Error:**
```
TypeError: unexpected keyword argument 'model_type'
```

**Expected:**
```python
CommissionModel(model_type='percentage')
```

**Actual:**
```python
CommissionModel(commission_type='percentage')
```

**Lesson:**
Always check actual API, not assumed API.

## 8. Development Environment Issues

### Challenge: Apple Silicon Compatibility

**Issues:**
1. NumPy/SciPy not using Accelerate framework
2. Some packages no ARM64 wheels
3. Conda vs pip conflicts

**Solution:**
```bash
# Use conda for numerical packages
conda install numpy scipy pandas "blas=*=*accelerate*"

# Use pip for everything else
pip install vectorbt optuna
```

## Key Learnings

1. **Don't Assume Parallelization Helps**
   - Measure first
   - Consider overhead
   - Single-threaded can be optimal

2. **Real Tests Catch Real Bugs**
   - Mocking would have hidden Polygon issues
   - Integration tests essential
   - Performance tests prevent regressions

3. **API Documentation Lies**
   - Always verify with actual code
   - Parameter names change
   - Examples may be outdated

4. **Premature Optimization is Evil**
   - VectorBT already optimized
   - Focus on algorithmic improvements
   - Profile before optimizing

5. **State Management is Hard**
   - Global state breaks parallelization
   - Immutable data structures help
   - Explicit is better than implicit

## Recommendations for Future Development

1. **Stick with Single-Threaded**
   - It works
   - It's fast enough
   - It's debuggable

2. **Use Real Services in Tests**
   - Despite the pain
   - Catches real issues
   - Builds confidence

3. **Profile Everything**
   - Memory usage
   - CPU time
   - I/O patterns

4. **Document Workarounds**
   - Future you will thank you
   - Others can learn
   - Prevents repeated mistakes

## 9. Cryptocurrency Data Challenges

### Challenge: Polygon.io Crypto Access Denied

**Symptom:**
```bash
rclone copy s3polygon:flatfiles/global_crypto/minute_aggs_v1/2024/01/2024-01-01.csv.gz tmp/
# ERROR: 403 Forbidden
```

**Root Cause:**
- Crypto data requires separate subscription tier
- Not included in standard market data plans
- No clear documentation about this requirement

**Discovery Process:**
```bash
# Could list directories
rclone ls s3polygon:flatfiles/global_crypto/minute_aggs_v1/2024/01/
# Could see file sizes
3015708 2024-01-01.csv.gz
# But couldn't download
```

**Workaround:**
Switch to alternative data sources (yfinance, CoinGecko, etc.)

### Challenge: YFinance Crypto Data Failure

**Symptom:**
```python
yf.download("BTC-USD", period="1mo")
# Returns empty DataFrame
# JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Root Cause:**
- Yahoo Finance API changes/issues
- Affects all tickers (stocks and crypto)
- Possibly rate limiting or API deprecation

**Attempted Solutions:**
```python
# Tried multiple ticker formats
tickers = ["BTC-USD", "ETH-USD", "BTCUSD", "BTC=F"]
# All failed with same error
```

**Final Solution:**
Created sample data generator for development/testing

### Challenge: 24/7 Market Adaptations

**Issue:**
Crypto markets trade continuously, unlike stocks

**Required Changes:**
1. Remove market hours checks
2. Adjust volatility calculations for continuous trading
3. Modify position sizing for different risk profile
4. Handle weekend/holiday trading

**Code Adaptations:**
```python
# Stock strategy assumes market hours
if is_market_open():
    trade()

# Crypto strategy always ready
trade()  # No time restrictions
```

### Challenge: Astronomical Calculations

**Issue:**
Need precise moon phase calculations for trading signals

**Solution:**
Used `ephem` library for astronomical computations

**Implementation Details:**
```python
# Calculate moon phase as fraction
lunation = (current_date - previous_new_moon) / (next_new_moon - previous_new_moon)
# 0.0 = new moon, 0.5 = full moon
```

**Lessons Learned:**
- Don't try to calculate astronomically yourself
- Established libraries handle edge cases
- Time zones matter for precise timing