"""
I/O Performance Benchmark Suite
Tests data loading and saving speeds for various formats
"""

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq
import tempfile
import shutil
from typing import Dict, List, Tuple
import logging

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class IOBenchmark:
    """Benchmark I/O operations for different data formats"""
    
    def __init__(self):
        """Initialize benchmark suite"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        self.temp_dir = None
    
    def setup(self):
        """Create temporary directory for tests"""
        self.temp_dir = tempfile.mkdtemp(prefix="io_benchmark_")
        self.logger.info(f"Created temp directory: {self.temp_dir}")
    
    def teardown(self):
        """Clean up temporary directory"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info("Cleaned up temp directory")
    
    def generate_test_data(self, rows: int = 1_000_000) -> pd.DataFrame:
        """Generate test market data"""
        self.logger.info(f"Generating test data with {rows:,} rows")
        
        # Generate realistic OHLCV data
        dates = pd.date_range('2023-01-01', periods=rows, freq='1min')
        base_price = 100.0
        
        # Random walk for prices
        returns = np.random.normal(0, 0.001, rows)
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV from close prices
        data = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices * (1 + np.random.uniform(-0.001, 0.001, rows)),
            'high': close_prices * (1 + np.random.uniform(0, 0.002, rows)),
            'low': close_prices * (1 + np.random.uniform(-0.002, 0, rows)),
            'close': close_prices,
            'volume': np.random.randint(1000, 100000, rows),
            'vwap': close_prices * (1 + np.random.uniform(-0.0005, 0.0005, rows)),
            'transactions': np.random.randint(10, 1000, rows)
        })
        
        return data
    
    def benchmark_csv(self, data: pd.DataFrame) -> Dict[str, float]:
        """Benchmark CSV I/O operations"""
        results = {}
        file_path = Path(self.temp_dir) / "test_data.csv"
        
        # Write CSV
        start = time.time()
        data.to_csv(file_path, index=False)
        write_time = time.time() - start
        results['write_time'] = write_time
        
        # File size
        file_size = file_path.stat().st_size / (1024**2)  # MB
        results['file_size_mb'] = file_size
        
        # Read CSV
        start = time.time()
        _ = pd.read_csv(file_path)
        read_time = time.time() - start
        results['read_time'] = read_time
        
        # Calculate speeds
        results['write_speed_mb_s'] = file_size / write_time
        results['read_speed_mb_s'] = file_size / read_time
        
        return results
    
    def benchmark_parquet(self, data: pd.DataFrame) -> Dict[str, float]:
        """Benchmark Parquet I/O operations"""
        results = {}
        file_path = Path(self.temp_dir) / "test_data.parquet"
        
        # Write Parquet
        start = time.time()
        data.to_parquet(file_path, compression='snappy')
        write_time = time.time() - start
        results['write_time'] = write_time
        
        # File size
        file_size = file_path.stat().st_size / (1024**2)  # MB
        results['file_size_mb'] = file_size
        
        # Read Parquet
        start = time.time()
        _ = pd.read_parquet(file_path)
        read_time = time.time() - start
        results['read_time'] = read_time
        
        # Calculate speeds
        results['write_speed_mb_s'] = file_size / write_time
        results['read_speed_mb_s'] = file_size / read_time
        
        return results
    
    def benchmark_numpy(self, data: pd.DataFrame) -> Dict[str, float]:
        """Benchmark NumPy binary I/O operations"""
        results = {}
        file_path = Path(self.temp_dir) / "test_data.npy"
        
        # Convert to numpy array
        np_data = data.select_dtypes(include=[np.number]).values
        
        # Write NumPy
        start = time.time()
        np.save(file_path, np_data)
        write_time = time.time() - start
        results['write_time'] = write_time
        
        # File size
        file_size = file_path.stat().st_size / (1024**2)  # MB
        results['file_size_mb'] = file_size
        
        # Read NumPy
        start = time.time()
        _ = np.load(file_path)
        read_time = time.time() - start
        results['read_time'] = read_time
        
        # Calculate speeds
        results['write_speed_mb_s'] = file_size / write_time
        results['read_speed_mb_s'] = file_size / read_time
        
        return results
    
    def benchmark_chunked_reading(self, data: pd.DataFrame) -> Dict[str, float]:
        """Benchmark chunked reading for large files"""
        results = {}
        file_path = Path(self.temp_dir) / "test_chunked.csv"
        
        # Write large CSV
        data.to_csv(file_path, index=False)
        file_size = file_path.stat().st_size / (1024**2)  # MB
        
        # Read in chunks
        chunk_size = 10000
        start = time.time()
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunks.append(len(chunk))
        read_time = time.time() - start
        
        results['chunk_size'] = chunk_size
        results['num_chunks'] = len(chunks)
        results['read_time'] = read_time
        results['read_speed_mb_s'] = file_size / read_time
        
        return results
    
    def run_all_benchmarks(self, data_sizes: List[int] = None) -> Dict:
        """Run all I/O benchmarks"""
        if data_sizes is None:
            data_sizes = [10_000, 100_000, 1_000_000]
        
        all_results = {}
        
        for size in data_sizes:
            self.logger.info(f"\nRunning benchmarks for {size:,} rows")
            
            # Generate test data
            data = self.generate_test_data(size)
            
            size_results = {
                'rows': size,
                'memory_mb': data.memory_usage(deep=True).sum() / (1024**2),
                'formats': {}
            }
            
            # CSV benchmark
            self.logger.info("Benchmarking CSV...")
            size_results['formats']['csv'] = self.benchmark_csv(data)
            
            # Parquet benchmark
            self.logger.info("Benchmarking Parquet...")
            size_results['formats']['parquet'] = self.benchmark_parquet(data)
            
            # NumPy benchmark
            self.logger.info("Benchmarking NumPy...")
            size_results['formats']['numpy'] = self.benchmark_numpy(data)
            
            # Chunked reading (only for largest size)
            if size == max(data_sizes):
                self.logger.info("Benchmarking chunked reading...")
                size_results['chunked'] = self.benchmark_chunked_reading(data)
            
            all_results[f'size_{size}'] = size_results
        
        return all_results
    
    def print_results(self, results: Dict):
        """Print benchmark results in readable format"""
        print("\n" + "="*80)
        print("I/O Performance Benchmark Results")
        print("="*80)
        
        for size_key, size_data in results.items():
            if size_key.startswith('size_'):
                rows = size_data['rows']
                memory = size_data['memory_mb']
                
                print(f"\nData Size: {rows:,} rows ({memory:.1f} MB in memory)")
                print("-" * 60)
                
                # Format comparison
                formats = size_data['formats']
                
                # Print header
                print(f"{'Format':<10} {'File Size':<12} {'Write Speed':<15} {'Read Speed':<15}")
                print(f"{'------':<10} {'---------':<12} {'-----------':<15} {'----------':<15}")
                
                # Print each format
                for fmt, data in formats.items():
                    file_size = f"{data['file_size_mb']:.1f} MB"
                    write_speed = f"{data['write_speed_mb_s']:.1f} MB/s"
                    read_speed = f"{data['read_speed_mb_s']:.1f} MB/s"
                    print(f"{fmt:<10} {file_size:<12} {write_speed:<15} {read_speed:<15}")
                
                # Print chunked results if available
                if 'chunked' in size_data:
                    chunked = size_data['chunked']
                    print(f"\nChunked Reading: {chunked['num_chunks']} chunks, "
                          f"{chunked['read_speed_mb_s']:.1f} MB/s")
    
    def save_results(self, results: Dict, output_dir: Path = None):
        """Save benchmark results to JSON"""
        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"io_benchmark_{timestamp}.json"
        
        # Add metadata
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'python_version': pd.__version__,
                'pandas_version': pd.__version__,
                'numpy_version': np.__version__
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to: {output_file}")
        return output_file


def main():
    """Run I/O performance benchmarks"""
    benchmark = IOBenchmark()
    
    try:
        benchmark.setup()
        
        # Run benchmarks
        results = benchmark.run_all_benchmarks(
            data_sizes=[10_000, 100_000, 500_000]
        )
        
        # Display results
        benchmark.print_results(results)
        
        # Save results
        benchmark.save_results(results)
        
        # Performance validation
        print("\n" + "="*80)
        print("Performance Validation")
        print("="*80)
        
        # Check against targets (using 100k rows as reference)
        if 'size_100000' in results:
            parquet_read = results['size_100000']['formats']['parquet']['read_speed_mb_s']
            csv_read = results['size_100000']['formats']['csv']['read_speed_mb_s']
            
            print(f"Parquet read speed: {parquet_read:.1f} MB/s")
            print(f"CSV read speed: {csv_read:.1f} MB/s")
            
            target_speed = 100  # MB/s target
            if parquet_read >= target_speed:
                print(f"✅ Parquet exceeds target speed of {target_speed} MB/s")
            else:
                print(f"⚠️  Parquet below target speed of {target_speed} MB/s")
    
    finally:
        benchmark.teardown()


if __name__ == "__main__":
    main()