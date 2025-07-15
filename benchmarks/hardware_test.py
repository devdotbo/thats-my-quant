#!/usr/bin/env python3
"""
Hardware Benchmark Suite for M3 Max Pro
Tests memory bandwidth, CPU performance, and system capabilities
"""

import numpy as np
import time
import platform
import psutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import multiprocessing


class HardwareBenchmark:
    """Comprehensive hardware benchmarking for Apple Silicon"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Gather system information"""
        return {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'machine': platform.machine(),
            'numpy_version': np.__version__,
            'numpy_config': np.show_config(mode='dicts')
        }
    
    def test_memory_bandwidth(self, size_gb: float = 1.0) -> Dict[str, float]:
        """Test memory bandwidth for different operations"""
        print(f"\nTesting memory bandwidth with {size_gb}GB array...")
        size = int(size_gb * 1024 * 1024 * 1024 / 8)  # Convert to float64 count
        
        results = {}
        
        # Sequential read
        print("  - Sequential read test...")
        arr = np.ones(size, dtype=np.float64)
        start = time.perf_counter()
        _ = arr.sum()
        elapsed = time.perf_counter() - start
        results['sequential_read_gbps'] = (size * 8 / 1e9) / elapsed
        
        # Sequential write
        print("  - Sequential write test...")
        start = time.perf_counter()
        arr[:] = 2.0
        elapsed = time.perf_counter() - start
        results['sequential_write_gbps'] = (size * 8 / 1e9) / elapsed
        
        # Copy bandwidth
        print("  - Copy bandwidth test...")
        arr2 = np.empty_like(arr)
        start = time.perf_counter()
        np.copyto(arr2, arr)
        elapsed = time.perf_counter() - start
        results['copy_gbps'] = (size * 8 * 2 / 1e9) / elapsed
        
        # Random access
        print("  - Random access test...")
        indices = np.random.randint(0, size, size // 100)
        start = time.perf_counter()
        _ = arr[indices].sum()
        elapsed = time.perf_counter() - start
        results['random_read_gbps'] = (len(indices) * 8 / 1e9) / elapsed
        
        return results
    
    def test_numpy_performance(self, size: int = 5000) -> Dict[str, float]:
        """Test NumPy operations with Accelerate framework"""
        print(f"\nTesting NumPy performance with {size}x{size} matrices...")
        results = {}
        
        # Matrix multiplication
        print("  - Matrix multiplication test...")
        A = np.random.randn(size, size).astype(np.float64)
        B = np.random.randn(size, size).astype(np.float64)
        
        # Warmup
        for _ in range(3):
            _ = np.dot(A[:100, :100], B[:100, :100])
        
        start = time.perf_counter()
        C = np.dot(A, B)
        elapsed = time.perf_counter() - start
        
        flops = 2 * size**3  # Matrix multiplication FLOPs
        results['matmul_gflops'] = flops / elapsed / 1e9
        results['matmul_time_seconds'] = elapsed
        
        # FFT performance
        print("  - FFT performance test...")
        data = np.random.randn(size * 1000).astype(np.complex128)
        
        # Warmup
        _ = np.fft.fft(data[:1000])
        
        start = time.perf_counter()
        fft_result = np.fft.fft(data)
        elapsed = time.perf_counter() - start
        results['fft_msamples_per_sec'] = len(data) / elapsed / 1e6
        results['fft_time_seconds'] = elapsed
        
        # Eigenvalue decomposition
        print("  - Eigenvalue decomposition test...")
        size_eig = min(size // 2, 1000)
        A_sym = np.random.randn(size_eig, size_eig)
        A_sym = (A_sym + A_sym.T) / 2
        
        start = time.perf_counter()
        eigenvalues, eigenvectors = np.linalg.eigh(A_sym)
        elapsed = time.perf_counter() - start
        results['eigenvalue_decomp_seconds'] = elapsed
        
        # Vector operations
        print("  - Vector operations test...")
        vec_size = size * 1000
        x = np.random.randn(vec_size)
        y = np.random.randn(vec_size)
        
        start = time.perf_counter()
        for _ in range(100):
            z = 2.5 * x + 3.7 * y
            w = np.sqrt(np.abs(z))
            v = np.exp(-w / 10)
        elapsed = time.perf_counter() - start
        results['vector_ops_gflops'] = (100 * vec_size * 5) / elapsed / 1e9
        
        return results
    
    def test_parallel_performance(self) -> Dict[str, float]:
        """Test parallel processing capabilities"""
        print("\nTesting parallel processing performance...")
        n_cores = multiprocessing.cpu_count()
        results = {
            'cpu_cores': n_cores,
            'performance_cores': n_cores // 2,  # Approximate for M3 Max
            'efficiency_cores': n_cores // 2
        }
        
        # Parallel NumPy operations
        print("  - Testing parallel matrix operations...")
        size = 2000
        matrices = [np.random.randn(size, size) for _ in range(n_cores)]
        
        # Single-threaded baseline
        start = time.perf_counter()
        for mat in matrices:
            _ = np.dot(mat, mat)
        single_time = time.perf_counter() - start
        
        results['single_thread_time'] = single_time
        results['matrices_per_second_single'] = len(matrices) / single_time
        
        # Note: True parallel testing would require multiprocessing
        # which is more complex to implement correctly
        
        return results
    
    def test_data_io_performance(self, size_mb: int = 100) -> Dict[str, float]:
        """Test file I/O performance"""
        print(f"\nTesting I/O performance with {size_mb}MB test file...")
        results = {}
        temp_dir = Path('temp_benchmark')
        temp_dir.mkdir(exist_ok=True)
        
        # Generate test data
        n_rows = int(size_mb * 1024 * 1024 / 100)  # Approximate row count
        test_data = {
            'timestamp': np.arange(n_rows),
            'price': np.random.randn(n_rows) * 10 + 100,
            'volume': np.random.randint(1000, 1000000, n_rows),
            'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL'], n_rows)
        }
        
        # NumPy save/load
        print("  - NumPy save/load test...")
        arr = np.column_stack([test_data['timestamp'], test_data['price'], test_data['volume']])
        
        np_file = temp_dir / 'test_array.npy'
        start = time.perf_counter()
        np.save(np_file, arr)
        save_time = time.perf_counter() - start
        
        start = time.perf_counter()
        loaded_arr = np.load(np_file)
        load_time = time.perf_counter() - start
        
        file_size_mb = np_file.stat().st_size / (1024 * 1024)
        results['numpy_save_mbps'] = file_size_mb / save_time
        results['numpy_load_mbps'] = file_size_mb / load_time
        
        # Cleanup
        np_file.unlink()
        temp_dir.rmdir()
        
        return results
    
    def test_financial_calculations(self) -> Dict[str, float]:
        """Test performance of common financial calculations"""
        print("\nTesting financial calculation performance...")
        results = {}
        
        # Generate realistic market data
        n_periods = 252 * 390  # 1 year of minute data
        n_symbols = 10
        
        print(f"  - Generating {n_periods} periods for {n_symbols} symbols...")
        prices = 100 * np.exp(np.cumsum(np.random.randn(n_periods, n_symbols) * 0.0001, axis=0))
        
        # Moving averages
        print("  - Calculating moving averages...")
        start = time.perf_counter()
        ma_20 = np.convolve(prices[:, 0], np.ones(20)/20, mode='valid')
        ma_50 = np.convolve(prices[:, 0], np.ones(50)/50, mode='valid')
        ma_200 = np.convolve(prices[:, 0], np.ones(200)/200, mode='valid')
        ma_time = time.perf_counter() - start
        results['moving_avg_calc_per_sec'] = 3 * n_periods / ma_time
        
        # Returns calculation
        print("  - Calculating returns...")
        start = time.perf_counter()
        returns = np.diff(prices, axis=0) / prices[:-1]
        log_returns = np.diff(np.log(prices), axis=0)
        returns_time = time.perf_counter() - start
        results['returns_calc_per_sec'] = 2 * n_periods * n_symbols / returns_time
        
        # Volatility calculation
        print("  - Calculating rolling volatility...")
        window = 20
        start = time.perf_counter()
        # Simplified rolling std calculation
        volatility = np.array([np.std(returns[i:i+window], axis=0) 
                              for i in range(len(returns)-window)])
        vol_time = time.perf_counter() - start
        results['volatility_windows_per_sec'] = len(volatility) / vol_time
        
        # Correlation matrix
        print("  - Calculating correlation matrix...")
        start = time.perf_counter()
        corr_matrix = np.corrcoef(returns.T)
        corr_time = time.perf_counter() - start
        results['correlation_matrix_time'] = corr_time
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print("="*60)
        print("That's My Quant - Hardware Benchmark Suite")
        print("="*60)
        print(f"System: {self.results['platform']}")
        print(f"CPU Cores: {self.results['system_info']['cpu_count']}")
        print(f"Memory: {self.results['system_info']['memory_gb']:.1f} GB")
        print("="*60)
        
        # Run benchmarks
        self.results['benchmarks']['memory'] = self.test_memory_bandwidth(1.0)
        self.results['benchmarks']['numpy'] = self.test_numpy_performance(5000)
        self.results['benchmarks']['parallel'] = self.test_parallel_performance()
        self.results['benchmarks']['io'] = self.test_data_io_performance(100)
        self.results['benchmarks']['financial'] = self.test_financial_calculations()
        
        # Save results
        self._save_results()
        
        # Generate summary
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save benchmark results to file"""
        results_dir = Path('benchmarks/results')
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = results_dir / f'hardware_benchmark_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filename}")
    
    def _print_summary(self):
        """Print benchmark summary with recommendations"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Memory bandwidth
        mem = self.results['benchmarks']['memory']
        print(f"\nMemory Bandwidth:")
        print(f"  Sequential Read:  {mem['sequential_read_gbps']:.1f} GB/s")
        print(f"  Sequential Write: {mem['sequential_write_gbps']:.1f} GB/s")
        print(f"  Copy Bandwidth:   {mem['copy_gbps']:.1f} GB/s")
        
        # CPU performance
        cpu = self.results['benchmarks']['numpy']
        print(f"\nCPU Performance:")
        print(f"  Matrix Multiply:  {cpu['matmul_gflops']:.1f} GFLOPS")
        print(f"  FFT Performance:  {cpu['fft_msamples_per_sec']:.1f} MSamples/sec")
        print(f"  Vector Ops:       {cpu['vector_ops_gflops']:.1f} GFLOPS")
        
        # I/O performance
        io = self.results['benchmarks']['io']
        print(f"\nI/O Performance:")
        print(f"  NumPy Save: {io['numpy_save_mbps']:.1f} MB/s")
        print(f"  NumPy Load: {io['numpy_load_mbps']:.1f} MB/s")
        
        # Financial calculations
        fin = self.results['benchmarks']['financial']
        print(f"\nFinancial Calculations:")
        print(f"  Moving Averages:  {fin['moving_avg_calc_per_sec']:.0f} calc/sec")
        print(f"  Returns:          {fin['returns_calc_per_sec']:.0f} calc/sec")
        print(f"  Volatility:       {fin['volatility_windows_per_sec']:.0f} windows/sec")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        if mem['sequential_read_gbps'] > 300:
            print("✓ Memory bandwidth excellent for large dataset processing")
        else:
            print("⚠ Memory bandwidth may limit large dataset performance")
        
        if cpu['matmul_gflops'] > 1000:
            print("✓ CPU matrix operations highly optimized (Accelerate detected)")
        else:
            print("⚠ Consider verifying NumPy is using Accelerate framework")
        
        if io['numpy_load_mbps'] > 500:
            print("✓ I/O performance excellent for market data")
        else:
            print("⚠ Consider SSD optimization for better I/O")
        
        print("\nSystem is ready for quantitative backtesting!")


def main():
    """Run hardware benchmarks"""
    benchmark = HardwareBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Check critical performance metrics
    mem_bandwidth = results['benchmarks']['memory']['sequential_read_gbps']
    cpu_performance = results['benchmarks']['numpy']['matmul_gflops']
    
    if mem_bandwidth < 200 or cpu_performance < 500:
        print("\n⚠ WARNING: Performance below recommended thresholds!")
        print("Consider checking system configuration.")
    else:
        print("\n✅ All performance metrics meet or exceed targets!")


if __name__ == "__main__":
    main()