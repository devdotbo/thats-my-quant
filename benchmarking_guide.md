# Apple Silicon Benchmarking Guide

## Overview

This guide provides comprehensive benchmarking procedures for the M3 Max Pro (128GB) to determine optimal configurations for quantitative trading workloads.

## System Specifications

- **Chip**: M3 Max Pro
- **Memory**: 128GB unified memory
- **Memory Bandwidth**: ~400 GB/s
- **CPU Cores**: Performance + Efficiency cores
- **GPU Cores**: 40-core GPU
- **Neural Engine**: 16-core

## Benchmarking Suite

### 1. Hardware Capability Tests

#### Memory Bandwidth Test
```python
# benchmarks/memory_bandwidth.py
import numpy as np
import time
from typing import Dict

def test_memory_bandwidth(size_gb: float = 1.0) -> Dict[str, float]:
    """Test memory bandwidth for different operations"""
    size = int(size_gb * 1024 * 1024 * 1024 / 8)  # Convert to float64 count
    
    results = {}
    
    # Sequential read
    arr = np.ones(size, dtype=np.float64)
    start = time.perf_counter()
    _ = arr.sum()
    elapsed = time.perf_counter() - start
    results['sequential_read_gbps'] = (size * 8 / 1e9) / elapsed
    
    # Sequential write
    start = time.perf_counter()
    arr[:] = 2.0
    elapsed = time.perf_counter() - start
    results['sequential_write_gbps'] = (size * 8 / 1e9) / elapsed
    
    # Random access
    indices = np.random.randint(0, size, size // 100)
    start = time.perf_counter()
    _ = arr[indices].sum()
    elapsed = time.perf_counter() - start
    results['random_read_gbps'] = (len(indices) * 8 / 1e9) / elapsed
    
    return results
```

#### CPU Performance Test
```python
# benchmarks/cpu_performance.py
import numpy as np
import time
from numba import jit
import multiprocessing

def test_numpy_performance(size: int = 10000) -> Dict[str, float]:
    """Test NumPy operations with Accelerate framework"""
    results = {}
    
    # Matrix multiplication
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)
    
    start = time.perf_counter()
    C = np.dot(A, B)
    elapsed = time.perf_counter() - start
    
    flops = 2 * size**3  # Matrix multiplication FLOPs
    results['matmul_gflops'] = flops / elapsed / 1e9
    
    # FFT performance
    data = np.random.randn(size * 1000)
    start = time.perf_counter()
    fft_result = np.fft.fft(data)
    elapsed = time.perf_counter() - start
    results['fft_msamples_per_sec'] = len(data) / elapsed / 1e6
    
    return results

@jit(nopython=True)
def monte_carlo_pi(n: int) -> float:
    """Numba JIT compiled Monte Carlo"""
    count = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y <= 1:
            count += 1
    return 4.0 * count / n

def test_parallel_performance() -> Dict[str, float]:
    """Test parallel processing capabilities"""
    n_simulations = 100_000_000
    n_cores = multiprocessing.cpu_count()
    
    # Single core
    start = time.perf_counter()
    pi_single = monte_carlo_pi(n_simulations)
    single_time = time.perf_counter() - start
    
    # Multi-core (simplified for example)
    results = {
        'single_core_msims_per_sec': n_simulations / single_time / 1e6,
        'cpu_cores': n_cores,
        'theoretical_speedup': n_cores
    }
    
    return results
```

#### GPU Performance Test
```python
# benchmarks/gpu_performance.py
import torch
import tensorflow as tf
import time

def test_pytorch_mps() -> Dict[str, float]:
    """Test PyTorch Metal Performance Shaders"""
    if not torch.backends.mps.is_available():
        return {'mps_available': False}
    
    results = {'mps_available': True}
    device = torch.device("mps")
    
    # Matrix multiplication benchmark
    sizes = [1000, 2000, 4000, 8000]
    for size in sizes:
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)
        
        # Warmup
        for _ in range(5):
            C = torch.matmul(A, B)
        torch.mps.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            C = torch.matmul(A, B)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        
        flops = 10 * 2 * size**3
        results[f'matmul_{size}_tflops'] = flops / elapsed / 1e12
    
    return results

def test_tensorflow_metal() -> Dict[str, float]:
    """Test TensorFlow Metal acceleration"""
    results = {}
    
    # Check if Metal is available
    gpus = tf.config.list_physical_devices('GPU')
    results['metal_gpus'] = len(gpus)
    
    if len(gpus) > 0:
        # Simple conv2d benchmark
        batch_size = 32
        image_size = 224
        channels = 3
        
        # Create dummy data
        images = tf.random.normal([batch_size, image_size, image_size, channels])
        
        # Build simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10)
        ])
        
        # Compile and warmup
        model.compile(optimizer='adam', loss='mse')
        for _ in range(5):
            _ = model(images, training=False)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = model(images, training=False)
        elapsed = time.perf_counter() - start
        
        results['conv2d_images_per_sec'] = (100 * batch_size) / elapsed
    
    return results
```

### 2. Backtesting Framework Benchmarks

#### VectorBT Performance
```python
# benchmarks/vectorbt_performance.py
import vectorbt as vbt
import pandas as pd
import numpy as np
import time

def generate_test_data(n_symbols: int = 10, n_days: int = 252) -> pd.DataFrame:
    """Generate realistic OHLCV data"""
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    data = {}
    
    for i in range(n_symbols):
        symbol = f'STOCK_{i}'
        prices = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))
        
        data[('Open', symbol)] = prices * (1 + np.random.randn(n_days) * 0.001)
        data[('High', symbol)] = prices * (1 + np.abs(np.random.randn(n_days)) * 0.005)
        data[('Low', symbol)] = prices * (1 - np.abs(np.random.randn(n_days)) * 0.005)
        data[('Close', symbol)] = prices
        data[('Volume', symbol)] = np.random.randint(1000000, 10000000, n_days)
    
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df

def benchmark_vectorbt_strategies() -> Dict[str, float]:
    """Benchmark common VectorBT operations"""
    results = {}
    
    # Generate data
    data_sizes = [
        (10, 252),    # 10 symbols, 1 year daily
        (10, 1260),   # 10 symbols, 5 years daily
        (100, 252),   # 100 symbols, 1 year daily
        (10, 78840),  # 10 symbols, 1 year minute (6.5h * 252 days)
    ]
    
    for n_symbols, n_periods in data_sizes:
        data = generate_test_data(n_symbols, n_periods)
        close = data['Close']
        
        # Simple moving average crossover
        start = time.perf_counter()
        fast_ma = vbt.MA.run(close, 10)
        slow_ma = vbt.MA.run(close, 30)
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        pf = vbt.Portfolio.from_signals(
            close, 
            entries, 
            exits,
            init_cash=100000,
            fees=0.001
        )
        
        stats = pf.stats()
        elapsed = time.perf_counter() - start
        
        key = f'ma_cross_{n_symbols}sym_{n_periods}per'
        results[key] = elapsed
        
        # Parameter optimization
        if n_symbols <= 10 and n_periods <= 1260:
            start = time.perf_counter()
            
            fast_periods = np.arange(10, 50, 5)
            slow_periods = np.arange(50, 200, 10)
            
            pf_opt = vbt.Portfolio.from_signals(
                close,
                entries=vbt.MA.run(close, fast_periods).ma_crossed_above(
                    vbt.MA.run(close, slow_periods)
                ),
                exits=vbt.MA.run(close, fast_periods).ma_crossed_below(
                    vbt.MA.run(close, slow_periods)
                ),
                init_cash=100000,
                fees=0.001
            )
            
            best_params = pf_opt.sharpe_ratio().idxmax()
            elapsed = time.perf_counter() - start
            
            key = f'optimization_{n_symbols}sym_{n_periods}per'
            results[key] = elapsed
    
    return results
```

### 3. Data Pipeline Benchmarks

#### I/O Performance
```python
# benchmarks/io_performance.py
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import time
import os
from pathlib import Path

def benchmark_data_formats(size_mb: int = 100) -> Dict[str, Dict[str, float]]:
    """Compare different data storage formats"""
    # Generate test data
    n_rows = int(size_mb * 1024 * 1024 / 100)  # Approximate row count
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_rows, freq='1min'),
        'open': np.random.randn(n_rows) * 10 + 100,
        'high': np.random.randn(n_rows) * 10 + 101,
        'low': np.random.randn(n_rows) * 10 + 99,
        'close': np.random.randn(n_rows) * 10 + 100,
        'volume': np.random.randint(1000, 1000000, n_rows),
        'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL'], n_rows)
    })
    
    results = {}
    temp_dir = Path('temp_benchmark')
    temp_dir.mkdir(exist_ok=True)
    
    # CSV
    csv_path = temp_dir / 'test.csv'
    start = time.perf_counter()
    data.to_csv(csv_path, index=False)
    write_time = time.perf_counter() - start
    
    start = time.perf_counter()
    df_csv = pd.read_csv(csv_path)
    read_time = time.perf_counter() - start
    
    file_size = os.path.getsize(csv_path) / 1024 / 1024
    results['csv'] = {
        'write_time': write_time,
        'read_time': read_time,
        'file_size_mb': file_size,
        'write_mbps': size_mb / write_time,
        'read_mbps': size_mb / read_time
    }
    
    # Parquet
    parquet_path = temp_dir / 'test.parquet'
    start = time.perf_counter()
    data.to_parquet(parquet_path, compression='snappy')
    write_time = time.perf_counter() - start
    
    start = time.perf_counter()
    df_parquet = pd.read_parquet(parquet_path)
    read_time = time.perf_counter() - start
    
    file_size = os.path.getsize(parquet_path) / 1024 / 1024
    results['parquet'] = {
        'write_time': write_time,
        'read_time': read_time,
        'file_size_mb': file_size,
        'write_mbps': size_mb / write_time,
        'read_mbps': size_mb / read_time
    }
    
    # Cleanup
    csv_path.unlink()
    parquet_path.unlink()
    temp_dir.rmdir()
    
    return results
```

### 4. ML Strategy Benchmarks

#### Feature Engineering Performance
```python
# benchmarks/ml_performance.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import time

def benchmark_feature_engineering(n_rows: int = 1_000_000) -> Dict[str, float]:
    """Benchmark feature calculation speed"""
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': np.random.randn(n_rows) * 10 + 100,
        'high': np.random.randn(n_rows) * 10 + 101,
        'low': np.random.randn(n_rows) * 10 + 99,
        'close': np.random.randn(n_rows) * 10 + 100,
        'volume': np.random.randint(1000, 1000000, n_rows)
    })
    
    results = {}
    
    # Technical indicators
    start = time.perf_counter()
    
    # Price features
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['volatility'] = data['returns'].rolling(20).std()
    
    # Moving averages
    for period in [10, 20, 50, 200]:
        data[f'ma_{period}'] = data['close'].rolling(period).mean()
    
    # Volume features
    data['volume_ma'] = data['volume'].rolling(20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['rsi'] = 100 - (100 / (1 + gain / loss))
    
    elapsed = time.perf_counter() - start
    results['feature_engineering_time'] = elapsed
    results['features_per_second'] = (n_rows * 15) / elapsed  # 15 features
    
    return results

def benchmark_ml_models(n_samples: int = 100_000) -> Dict[str, float]:
    """Benchmark ML model training and inference"""
    # Generate features and labels
    X = np.random.randn(n_samples, 20)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    results = {}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    
    start = time.perf_counter()
    rf.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    
    start = time.perf_counter()
    predictions = rf.predict(X_test)
    inference_time = time.perf_counter() - start
    
    results['rf_train_time'] = train_time
    results['rf_inference_time'] = inference_time
    results['rf_predictions_per_sec'] = len(X_test) / inference_time
    
    # XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',  # Use histogram-based algorithm
        'device': 'cpu'  # Can change to 'gpu' if supported
    }
    
    start = time.perf_counter()
    model = xgb.train(params, dtrain, num_boost_round=100)
    train_time = time.perf_counter() - start
    
    start = time.perf_counter()
    predictions = model.predict(dtest)
    inference_time = time.perf_counter() - start
    
    results['xgb_train_time'] = train_time
    results['xgb_inference_time'] = inference_time
    results['xgb_predictions_per_sec'] = len(X_test) / inference_time
    
    return results
```

## Running the Benchmarks

### Complete Benchmark Suite
```python
# benchmarks/run_all_benchmarks.py
import json
from datetime import datetime
from pathlib import Path

def run_all_benchmarks():
    """Run complete benchmark suite and save results"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'system': 'M3 Max Pro 128GB',
        'benchmarks': {}
    }
    
    print("Running memory bandwidth tests...")
    results['benchmarks']['memory'] = test_memory_bandwidth(1.0)
    
    print("Running CPU performance tests...")
    results['benchmarks']['cpu'] = test_numpy_performance(5000)
    results['benchmarks']['parallel'] = test_parallel_performance()
    
    print("Running GPU performance tests...")
    results['benchmarks']['pytorch_mps'] = test_pytorch_mps()
    results['benchmarks']['tensorflow_metal'] = test_tensorflow_metal()
    
    print("Running VectorBT benchmarks...")
    results['benchmarks']['vectorbt'] = benchmark_vectorbt_strategies()
    
    print("Running I/O benchmarks...")
    results['benchmarks']['io'] = benchmark_data_formats(100)
    
    print("Running ML benchmarks...")
    results['benchmarks']['features'] = benchmark_feature_engineering(1_000_000)
    results['benchmarks']['ml_models'] = benchmark_ml_models(100_000)
    
    # Save results
    output_path = Path('benchmarks/results')
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(output_path / f'benchmark_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    generate_summary_report(results)
    
    return results

def generate_summary_report(results: Dict):
    """Generate human-readable summary"""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    # Memory bandwidth
    mem = results['benchmarks']['memory']
    print(f"\nMemory Bandwidth:")
    print(f"  Sequential Read:  {mem['sequential_read_gbps']:.1f} GB/s")
    print(f"  Sequential Write: {mem['sequential_write_gbps']:.1f} GB/s")
    
    # CPU performance
    cpu = results['benchmarks']['cpu']
    print(f"\nCPU Performance:")
    print(f"  Matrix Multiply: {cpu['matmul_gflops']:.1f} GFLOPS")
    print(f"  FFT: {cpu['fft_msamples_per_sec']:.1f} MSamples/sec")
    
    # GPU performance
    if results['benchmarks']['pytorch_mps'].get('mps_available'):
        mps = results['benchmarks']['pytorch_mps']
        print(f"\nGPU Performance (PyTorch MPS):")
        print(f"  MatMul 4K: {mps.get('matmul_4000_tflops', 0):.2f} TFLOPS")
    
    # VectorBT performance
    vbt = results['benchmarks']['vectorbt']
    print(f"\nVectorBT Performance:")
    for key, time in vbt.items():
        print(f"  {key}: {time:.3f} seconds")
    
    # Decision recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if mem['sequential_read_gbps'] > 300:
        print("✓ Memory bandwidth excellent for large dataset processing")
    
    if cpu['matmul_gflops'] > 1000:
        print("✓ CPU matrix operations highly optimized")
    
    if results['benchmarks']['pytorch_mps'].get('mps_available'):
        print("✓ GPU acceleration available for ML workloads")
    else:
        print("⚠ GPU acceleration not available")
    
    vbt_1year_minute = vbt.get('ma_cross_10sym_78840per', float('inf'))
    if vbt_1year_minute < 5:
        print(f"✓ VectorBT can backtest 1 year minute data in {vbt_1year_minute:.1f}s")
    else:
        print(f"⚠ VectorBT slower than target: {vbt_1year_minute:.1f}s > 5s")

if __name__ == "__main__":
    run_all_benchmarks()
```

## Performance Targets

### Minimum Requirements
- Memory bandwidth: >200 GB/s
- Matrix operations: >500 GFLOPS
- VectorBT 1-year minute: <5 seconds
- Data loading: >100 MB/s

### Optimal Performance
- Memory bandwidth: >350 GB/s
- Matrix operations: >1 TFLOPS
- VectorBT 1-year minute: <2 seconds
- Data loading: >500 MB/s

## Decision Matrix

Based on benchmark results:

### Use GPU (MPS/Metal) if:
- GPU speedup >2x for matrix operations
- ML model training time critical
- Complex neural networks required

### Use CPU-only if:
- GPU speedup <1.5x
- Simple strategies dominate
- Maximum compatibility needed

### Use VectorBT if:
- Backtest time <5s for target data
- Parameter optimization required
- Vectorized operations possible

### Use Backtrader if:
- Complex event handling needed
- Live trading planned
- VectorBT limitations encountered

## Continuous Monitoring

Create `benchmarks/monitor.py` to track performance over time:

```python
def monitor_system_health():
    """Regular performance monitoring"""
    metrics = {
        'memory_usage_gb': get_memory_usage(),
        'cpu_usage_pct': get_cpu_usage(),
        'disk_free_gb': get_disk_free_space(),
        'backtest_time': measure_standard_backtest()
    }
    
    # Alert if degradation detected
    if metrics['backtest_time'] > 5.0:
        alert("Performance degradation detected")
    
    return metrics
```

## Troubleshooting

### Common Issues

1. **MPS Not Available**
   - Update macOS to 12.3+
   - Install latest PyTorch
   - Check Metal compatibility

2. **Slow NumPy Operations**
   - Verify Accelerate framework
   - Check BLAS configuration
   - Use conda-forge channel

3. **Memory Pressure**
   - Reduce batch sizes
   - Implement streaming
   - Clear caches regularly

4. **I/O Bottlenecks**
   - Use Parquet format
   - Enable compression
   - Optimize chunk sizes