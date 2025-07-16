# Performance Benchmarking & Monitoring

## System Specifications

- **Chip**: M3 Max Pro
- **Memory**: 128GB unified memory
- **Memory Bandwidth**: ~400 GB/s (theoretical)
- **CPU**: Performance + Efficiency cores
- **GPU**: 40-core GPU
- **Neural Engine**: 16-core

## Performance Targets

| Metric | Target | Critical Threshold | Status |
|--------|--------|-------------------|---------|
| 1-year minute backtest | <5 seconds | >10 seconds | ✅ 0.045s |
| Data loading (100MB) | <1 second | >3 seconds | ✅ 0.017s |
| Feature calculation | >1M rows/sec | <500K rows/sec | Pending |
| Parameter optimization (1000) | <30 minutes | >60 minutes | Pending |
| Memory usage (typical) | <32GB | >64GB | ✅ Verified |
| Memory usage (peak) | <64GB | >96GB | ✅ Verified |
| Cache operations | <10ms | >50ms | ✅ Verified |

## Benchmark Results (M3 Max Pro)

### Hardware Performance
- **Memory Read**: 22.5 GB/s (actual)
- **Memory Write**: 15.2 GB/s 
- **CPU GFLOPS**: 346.3 (NumPy with Accelerate)
- **I/O Performance**: 
  - Save: 7002 MB/s
  - Load: 5926 MB/s

### VectorBT Performance
- **1 Year Minute Data**: 0.045 seconds
- **5 Years Daily Data**: 0.012 seconds
- **Parameter Optimization**: 8.45 seconds (100 combos)

## Key Performance Indicators (KPIs)

### Speed Metrics
- Backtest execution time
- Data loading throughput
- Feature calculation rate
- Optimization time per parameter

### Resource Metrics
- Memory usage (process and system)
- CPU utilization
- Disk I/O rates
- Cache hit rates

### Quality Metrics
- Test coverage (>80%)
- Type coverage (100%)
- Code complexity (<10)
- Documentation coverage (100% public APIs)

## Monitoring Implementation

### Performance Monitor Usage
```python
from src.monitoring.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

# Monitor backtest performance
metrics = monitor.monitor_backtest(
    strategy=strategy,
    data=data,
    params={'fast_ma': 20, 'slow_ma': 50}
)

# Monitor resource usage
resources = monitor.monitor_resource_usage()
print(f"Memory: {resources['process_memory_rss_gb']:.1f}GB")
print(f"CPU: {resources['cpu_percent']:.1f}%")
```

### Continuous Monitoring
```python
# Start background monitoring
from src.monitoring.continuous_monitor import ContinuousMonitor

monitor = ContinuousMonitor(interval_seconds=60)
monitor.start()

# Run your backtests...

# Stop monitoring
monitor.stop()
```

## Performance Optimization Guidelines

### 1. Memory Optimization
- Use chunked data loading for large datasets
- Clear unused dataframes with `del df; gc.collect()`
- Use appropriate dtypes (float32 vs float64)
- Leverage memory mapping for read-only data

### 2. CPU Optimization
- Vectorize operations with NumPy/Pandas
- Use VectorBT's built-in vectorized functions
- Parallelize parameter optimization
- Profile with `cProfile` to find bottlenecks

### 3. I/O Optimization
- Use Parquet format with snappy compression
- Implement read caching for frequently accessed data
- Batch write operations
- Use SSD for data storage

### 4. VectorBT Specific
- Pre-calculate indicators when possible
- Use `vectorbt.portfolio.from_signals()` for simple strategies
- Leverage built-in performance metrics
- Use appropriate chunk_len for memory management

## Baseline Performance Tests

Run after any major changes:
```bash
# Full benchmark suite
python benchmarks/run_all_benchmarks.py

# Compare to baseline
python benchmarks/compare_baseline.py

# Generate performance report
python benchmarks/generate_report.py
```

## Alert Thresholds

- **Memory**: Alert when process uses >64GB
- **CPU**: Alert when sustained >95% for 60s
- **Disk**: Alert when <10GB free space
- **Backtest**: Alert when >2x baseline time