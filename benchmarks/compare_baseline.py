"""
Compare Benchmark Results Against Baseline
Tracks performance over time and identifies regressions
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BaselineComparator:
    """Compare benchmark results against baseline metrics"""
    
    def __init__(self):
        """Initialize comparator"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.benchmarks_dir = Path(__file__).parent
        self.results_dir = self.benchmarks_dir / "results"
        self.baseline_file = self.benchmarks_dir / "baseline.json"
        
        # Performance targets from documentation
        self.performance_targets = {
            'vectorbt_1y_backtest_time': {
                'target': 5.0,  # seconds
                'unit': 'seconds',
                'lower_is_better': True,
                'description': '1 year minute data backtest'
            },
            'memory_bandwidth_gb_s': {
                'target': 20.0,  # GB/s
                'unit': 'GB/s',
                'lower_is_better': False,
                'description': 'Memory bandwidth'
            },
            'parquet_read_speed_mb_s': {
                'target': 100.0,  # MB/s
                'unit': 'MB/s',
                'lower_is_better': False,
                'description': 'Parquet read speed'
            },
            'cpu_gflops': {
                'target': 100.0,  # GFLOPS
                'unit': 'GFLOPS',
                'lower_is_better': False,
                'description': 'CPU performance'
            }
        }
        
        # Acceptable variance (percentage)
        self.variance_threshold = 10.0
    
    def load_baseline(self) -> Optional[Dict]:
        """Load baseline metrics"""
        if not self.baseline_file.exists():
            self.logger.warning("No baseline file found")
            return None
        
        try:
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load baseline: {e}")
            return None
    
    def save_baseline(self, metrics: Dict):
        """Save baseline metrics"""
        baseline_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'targets': self.performance_targets
        }
        
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        self.logger.info(f"Baseline saved to: {self.baseline_file}")
    
    def find_latest_comprehensive_results(self) -> Optional[Path]:
        """Find the most recent comprehensive benchmark results"""
        result_files = list(self.results_dir.glob("all_benchmarks_*.json"))
        if result_files:
            return max(result_files, key=lambda p: p.stat().st_mtime)
        return None
    
    def load_results(self, file_path: Path) -> Optional[Dict]:
        """Load benchmark results from file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load results: {e}")
            return None
    
    def extract_metrics(self, results: Dict) -> Dict[str, float]:
        """Extract key metrics from results"""
        return results.get('key_metrics', {})
    
    def compare_metrics(self, current: Dict[str, float], baseline: Dict[str, float]) -> Dict:
        """Compare current metrics against baseline"""
        comparison = {}
        
        for metric, value in current.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                
                # Calculate percentage change
                if baseline_value != 0:
                    change_pct = ((value - baseline_value) / baseline_value) * 100
                else:
                    change_pct = float('inf') if value > 0 else 0
                
                # Determine if improvement or regression
                target_info = self.performance_targets.get(metric, {})
                lower_is_better = target_info.get('lower_is_better', True)
                
                if lower_is_better:
                    is_improvement = value < baseline_value
                else:
                    is_improvement = value > baseline_value
                
                comparison[metric] = {
                    'current': value,
                    'baseline': baseline_value,
                    'change_pct': change_pct,
                    'is_improvement': is_improvement,
                    'within_variance': abs(change_pct) <= self.variance_threshold
                }
            else:
                # New metric not in baseline
                comparison[metric] = {
                    'current': value,
                    'baseline': None,
                    'change_pct': None,
                    'is_improvement': None,
                    'within_variance': True
                }
        
        return comparison
    
    def check_against_targets(self, metrics: Dict[str, float]) -> Dict:
        """Check metrics against performance targets"""
        target_results = {}
        
        for metric, value in metrics.items():
            if metric in self.performance_targets:
                target_info = self.performance_targets[metric]
                target = target_info['target']
                lower_is_better = target_info['lower_is_better']
                
                if lower_is_better:
                    meets_target = value <= target
                else:
                    meets_target = value >= target
                
                target_results[metric] = {
                    'value': value,
                    'target': target,
                    'meets_target': meets_target,
                    'unit': target_info['unit'],
                    'description': target_info['description']
                }
        
        return target_results
    
    def get_historical_trends(self, metric: str, limit: int = 10) -> List[Tuple[datetime, float]]:
        """Get historical values for a metric"""
        trends = []
        
        # Find all comprehensive result files
        result_files = sorted(
            self.results_dir.glob("all_benchmarks_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]
        
        for file_path in result_files:
            results = self.load_results(file_path)
            if results and 'key_metrics' in results:
                metrics = results['key_metrics']
                if metric in metrics:
                    timestamp = datetime.fromisoformat(results['timestamp'])
                    trends.append((timestamp, metrics[metric]))
        
        return sorted(trends, key=lambda x: x[0])
    
    def print_comparison_report(self, comparison: Dict, target_results: Dict, current_metrics: Dict):
        """Print detailed comparison report"""
        print("\n" + "="*80)
        print("Performance Comparison Report")
        print("="*80)
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Baseline comparison
        if comparison:
            print("\n📊 Baseline Comparison:")
            print("-" * 60)
            
            improvements = 0
            regressions = 0
            
            for metric, data in comparison.items():
                if data['baseline'] is not None:
                    current = data['current']
                    baseline = data['baseline']
                    change = data['change_pct']
                    
                    # Determine symbol
                    if data['is_improvement']:
                        symbol = "↗️" if change > 0 else "↘️"
                        improvements += 1
                    else:
                        symbol = "↘️" if change > 0 else "↗️"
                        regressions += 1
                    
                    # Color coding for terminal
                    if data['within_variance']:
                        status = "✓"
                    else:
                        status = "⚠️"
                    
                    print(f"{metric:<30} {current:>10.2f} vs {baseline:>10.2f} "
                          f"({change:+6.1f}%) {symbol} {status}")
                else:
                    print(f"{metric:<30} {data['current']:>10.2f} (new metric)")
            
            print(f"\nSummary: {improvements} improvements, {regressions} regressions")
        
        # Target comparison
        print("\n🎯 Performance Targets:")
        print("-" * 60)
        
        all_targets_met = True
        
        for metric, data in target_results.items():
            value = data['value']
            target = data['target']
            unit = data['unit']
            meets = data['meets_target']
            
            status = "✅" if meets else "❌"
            all_targets_met &= meets
            
            print(f"{data['description']:<30} {value:>10.2f} {unit:<6} "
                  f"(target: {target} {unit}) {status}")
        
        # Overall assessment
        print("\n" + "="*60)
        print("Overall Assessment:")
        print("="*60)
        
        if all_targets_met:
            print("✅ All performance targets met!")
        else:
            print("⚠️  Some performance targets not met")
        
        if comparison:
            regression_count = sum(1 for d in comparison.values() 
                                 if d['baseline'] is not None and 
                                 not d['is_improvement'] and 
                                 not d['within_variance'])
            
            if regression_count == 0:
                print("✅ No significant regressions detected")
            else:
                print(f"⚠️  {regression_count} significant regressions detected")
    
    def generate_trend_report(self, metrics: List[str]):
        """Generate trend report for specified metrics"""
        print("\n📈 Historical Trends:")
        print("-" * 60)
        
        for metric in metrics:
            trends = self.get_historical_trends(metric)
            if trends:
                print(f"\n{metric}:")
                values = [v for _, v in trends]
                
                # Calculate statistics
                mean = np.mean(values)
                std = np.std(values)
                trend = "stable"
                
                if len(values) >= 3:
                    # Simple linear trend
                    x = np.arange(len(values))
                    coeffs = np.polyfit(x, values, 1)
                    if coeffs[0] > 0.01:
                        trend = "increasing"
                    elif coeffs[0] < -0.01:
                        trend = "decreasing"
                
                print(f"  Latest: {values[-1]:.2f}")
                print(f"  Mean: {mean:.2f} ± {std:.2f}")
                print(f"  Trend: {trend}")
                print(f"  History: {' -> '.join(f'{v:.1f}' for v in values[-5:])}")
    
    def run_comparison(self, results_file: Optional[Path] = None) -> bool:
        """Run full comparison analysis"""
        # Load current results
        if results_file is None:
            results_file = self.find_latest_comprehensive_results()
        
        if not results_file:
            print("❌ No benchmark results found. Run benchmarks first.")
            return False
        
        print(f"Loading results from: {results_file}")
        current_results = self.load_results(results_file)
        
        if not current_results:
            print("❌ Failed to load benchmark results")
            return False
        
        current_metrics = self.extract_metrics(current_results)
        
        if not current_metrics:
            print("❌ No metrics found in results")
            return False
        
        # Load baseline
        baseline = self.load_baseline()
        
        # Compare against baseline
        comparison = {}
        if baseline and 'metrics' in baseline:
            comparison = self.compare_metrics(current_metrics, baseline['metrics'])
        else:
            print("ℹ️  No baseline found. Current results will become the baseline.")
        
        # Check against targets
        target_results = self.check_against_targets(current_metrics)
        
        # Print report
        self.print_comparison_report(comparison, target_results, current_metrics)
        
        # Generate trend report
        if current_metrics:
            self.generate_trend_report(list(current_metrics.keys()))
        
        # Ask to update baseline
        if comparison:
            print("\n" + "-"*60)
            response = input("Update baseline with current results? (y/n): ")
            if response.lower() == 'y':
                self.save_baseline(current_metrics)
                print("✅ Baseline updated")
        else:
            # No baseline exists, save current as baseline
            self.save_baseline(current_metrics)
            print("✅ Baseline created")
        
        # Return success/failure based on targets and regressions
        all_targets_met = all(data['meets_target'] for data in target_results.values())
        no_regressions = not comparison or all(
            data['within_variance'] or data['baseline'] is None 
            for data in comparison.values()
        )
        
        return all_targets_met and no_regressions


def main():
    """Run baseline comparison"""
    comparator = BaselineComparator()
    
    # Check for command line arguments
    results_file = None
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
        if not results_file.exists():
            print(f"❌ Results file not found: {results_file}")
            sys.exit(1)
    
    success = comparator.run_comparison(results_file)
    
    if success:
        print("\n✅ All checks passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()