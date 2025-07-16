"""
Run All Benchmarks
Executes all performance benchmarks and generates a comprehensive report
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BenchmarkRunner:
    """Orchestrates running all benchmark tests"""
    
    def __init__(self):
        """Initialize benchmark runner"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.benchmarks_dir = Path(__file__).parent
        self.results_dir = self.benchmarks_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Define benchmark scripts to run
        self.benchmark_scripts = [
            {
                'name': 'Hardware Performance',
                'script': 'hardware_test.py',
                'description': 'Tests CPU, memory, and system capabilities'
            },
            {
                'name': 'VectorBT Performance',
                'script': 'vectorbt_benchmark.py',
                'description': 'Tests backtesting engine performance'
            },
            {
                'name': 'I/O Performance',
                'script': 'io_performance.py',
                'description': 'Tests data loading and saving speeds'
            },
            {
                'name': 'Polygon Connection',
                'script': 'test_polygon_connection.py',
                'description': 'Tests data provider connectivity'
            }
        ]
        
        self.all_results = {}
    
    def run_benchmark(self, benchmark_info: Dict) -> Dict:
        """Run a single benchmark script"""
        script_path = self.benchmarks_dir / benchmark_info['script']
        
        if not script_path.exists():
            self.logger.warning(f"Benchmark script not found: {script_path}")
            return {
                'status': 'missing',
                'error': f"Script {benchmark_info['script']} not found"
            }
        
        self.logger.info(f"Running {benchmark_info['name']}...")
        print(f"\n{'='*60}")
        print(f"Running: {benchmark_info['name']}")
        print(f"Description: {benchmark_info['description']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run the benchmark script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                # Find the latest result file for this benchmark
                latest_result = self.find_latest_result(benchmark_info['script'])
                
                return {
                    'status': 'success',
                    'execution_time': execution_time,
                    'result_file': str(latest_result) if latest_result else None,
                    'stdout': result.stdout[-1000:],  # Last 1000 chars
                    'stderr': result.stderr
                }
            else:
                return {
                    'status': 'failed',
                    'execution_time': execution_time,
                    'returncode': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'error': f"Benchmark timed out after 300 seconds"
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def find_latest_result(self, script_name: str) -> Optional[Path]:
        """Find the most recent result file for a benchmark"""
        # Map script names to result file patterns
        patterns = {
            'hardware_test.py': 'hardware_benchmark_*.json',
            'vectorbt_benchmark.py': 'vectorbt_benchmark_*.json',
            'io_performance.py': 'io_benchmark_*.json',
            'test_polygon_connection.py': 'polygon_connection_test_*.json'
        }
        
        pattern = patterns.get(script_name)
        if not pattern:
            return None
        
        result_files = list(self.results_dir.glob(pattern))
        if result_files:
            # Return the most recent file
            return max(result_files, key=lambda p: p.stat().st_mtime)
        
        return None
    
    def load_benchmark_results(self, result_file: Path) -> Optional[Dict]:
        """Load results from a benchmark JSON file"""
        try:
            with open(result_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load {result_file}: {e}")
            return None
    
    def run_all(self) -> Dict:
        """Run all benchmarks"""
        print("\n" + "="*80)
        print("That's My Quant - Comprehensive Benchmark Suite")
        print("="*80)
        print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Number of benchmarks: {len(self.benchmark_scripts)}")
        
        overall_start = time.time()
        
        for benchmark in self.benchmark_scripts:
            result = self.run_benchmark(benchmark)
            self.all_results[benchmark['name']] = {
                'info': benchmark,
                'result': result
            }
            
            # Brief pause between benchmarks
            time.sleep(1)
        
        total_time = time.time() - overall_start
        
        # Generate summary
        summary = self.generate_summary(total_time)
        
        # Save comprehensive results
        self.save_comprehensive_results(summary)
        
        # Print summary
        self.print_summary(summary)
        
        return summary
    
    def generate_summary(self, total_time: float) -> Dict:
        """Generate benchmark summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_execution_time': total_time,
            'benchmarks': {},
            'overall_status': 'success',
            'key_metrics': {}
        }
        
        successful = 0
        failed = 0
        
        for name, data in self.all_results.items():
            result = data['result']
            status = result['status']
            
            summary['benchmarks'][name] = {
                'status': status,
                'execution_time': result.get('execution_time', 0)
            }
            
            if status == 'success':
                successful += 1
                
                # Load actual benchmark results if available
                if result.get('result_file'):
                    result_path = Path(result['result_file'])
                    if result_path.exists():
                        benchmark_data = self.load_benchmark_results(result_path)
                        if benchmark_data:
                            summary['benchmarks'][name]['data'] = benchmark_data
                            
                            # Extract key metrics
                            self.extract_key_metrics(name, benchmark_data, summary['key_metrics'])
            else:
                failed += 1
                summary['benchmarks'][name]['error'] = result.get('error', result.get('stderr', 'Unknown error'))
                summary['overall_status'] = 'partial'
        
        summary['summary_stats'] = {
            'total': len(self.benchmark_scripts),
            'successful': successful,
            'failed': failed
        }
        
        if failed == len(self.benchmark_scripts):
            summary['overall_status'] = 'failed'
        
        return summary
    
    def extract_key_metrics(self, benchmark_name: str, data: Dict, key_metrics: Dict):
        """Extract key metrics from benchmark data"""
        if benchmark_name == 'Hardware Performance':
            if 'cpu_performance' in data:
                key_metrics['cpu_gflops'] = data['cpu_performance'].get('gflops', 0)
            if 'memory_performance' in data:
                key_metrics['memory_bandwidth_gb_s'] = data['memory_performance'].get('bandwidth_gb_s', 0)
        
        elif benchmark_name == 'VectorBT Performance':
            if 'results' in data and 'minute_data_1y' in data['results']:
                key_metrics['vectorbt_1y_backtest_time'] = data['results']['minute_data_1y'].get('backtest_time', 0)
        
        elif benchmark_name == 'I/O Performance':
            # Extract parquet read speed for 100k rows
            if 'size_100000' in data:
                parquet_speed = data['size_100000']['formats']['parquet'].get('read_speed_mb_s', 0)
                key_metrics['parquet_read_speed_mb_s'] = parquet_speed
    
    def print_summary(self, summary: Dict):
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("Benchmark Summary")
        print("="*80)
        
        stats = summary['summary_stats']
        print(f"Total benchmarks: {stats['total']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Overall status: {summary['overall_status'].upper()}")
        print(f"Total time: {summary['total_execution_time']:.1f} seconds")
        
        # Individual benchmark results
        print("\nIndividual Results:")
        print("-" * 60)
        for name, data in summary['benchmarks'].items():
            status = data['status']
            time_str = f"{data['execution_time']:.1f}s" if 'execution_time' in data else "N/A"
            status_emoji = "✅" if status == 'success' else "❌"
            print(f"{status_emoji} {name:<30} {status:<10} {time_str:>10}")
        
        # Key metrics
        if summary['key_metrics']:
            print("\nKey Performance Metrics:")
            print("-" * 60)
            metrics = summary['key_metrics']
            
            if 'cpu_gflops' in metrics:
                print(f"CPU Performance: {metrics['cpu_gflops']:.1f} GFLOPS")
            
            if 'memory_bandwidth_gb_s' in metrics:
                print(f"Memory Bandwidth: {metrics['memory_bandwidth_gb_s']:.1f} GB/s")
            
            if 'vectorbt_1y_backtest_time' in metrics:
                print(f"VectorBT 1Y Backtest: {metrics['vectorbt_1y_backtest_time']:.3f} seconds")
            
            if 'parquet_read_speed_mb_s' in metrics:
                print(f"Parquet Read Speed: {metrics['parquet_read_speed_mb_s']:.1f} MB/s")
    
    def save_comprehensive_results(self, summary: Dict):
        """Save comprehensive benchmark results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.results_dir / f"all_benchmarks_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Comprehensive results saved to: {output_file}")
        print(f"\nResults saved to: {output_file}")


def main():
    """Run all benchmarks"""
    runner = BenchmarkRunner()
    
    try:
        summary = runner.run_all()
        
        # Return appropriate exit code
        if summary['overall_status'] == 'failed':
            sys.exit(1)
        elif summary['overall_status'] == 'partial':
            sys.exit(2)
        else:
            sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n\nBenchmark suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error running benchmarks: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()