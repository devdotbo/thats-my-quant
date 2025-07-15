# Performance Tracking and Monitoring

## Overview

This document defines the performance metrics, tracking methods, and monitoring procedures to ensure the backtesting system maintains optimal performance throughout development and production use.

## Key Performance Indicators (KPIs)

### Speed Metrics
| Metric | Target | Critical Threshold | Measurement Method |
|--------|--------|-------------------|-------------------|
| 1-year minute backtest | <5 seconds | >10 seconds | `benchmark_backtest_speed()` |
| Data loading (100MB) | <1 second | >3 seconds | `benchmark_data_loading()` |
| Feature calculation | >1M rows/sec | <500K rows/sec | `benchmark_features()` |
| Parameter optimization (1000) | <30 minutes | >60 minutes | `benchmark_optimization()` |

### Resource Metrics
| Metric | Target | Critical Threshold | Measurement Method |
|--------|--------|-------------------|-------------------|
| Memory usage (typical) | <32GB | >64GB | `monitor_memory()` |
| Memory usage (peak) | <64GB | >96GB | `monitor_peak_memory()` |
| Disk usage | <100GB | >100GB | `monitor_disk_usage()` |
| CPU utilization | 60-80% | >95% sustained | `monitor_cpu()` |

### Quality Metrics
| Metric | Target | Critical Threshold | Measurement Method |
|--------|--------|-------------------|-------------------|
| Test coverage | >80% | <70% | `pytest --cov` |
| Type coverage | 100% | <95% | `mypy --strict` |
| Code complexity | <10 | >15 | `radon cc` |
| Documentation | 100% public APIs | <90% | `interrogate` |

## Performance Monitoring Implementation

### Base Performance Monitor
```python
# src/monitoring/performance_monitor.py
import time
import psutil
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json

class PerformanceMonitor:
    """Monitor and track system performance metrics"""
    
    def __init__(self, log_dir: Path = Path("logs/performance")):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.baseline = self.load_baseline()
        
    def measure_backtest_performance(self, 
                                   strategy_name: str,
                                   data_size: str) -> Dict[str, float]:
        """Measure backtest execution performance"""
        process = psutil.Process()
        
        # Memory before
        memory_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        
        # CPU time before
        cpu_before = process.cpu_times()
        
        # Wall time
        start_time = time.perf_counter()
        
        # Run backtest (placeholder - replace with actual)
        result = self._run_standard_backtest(strategy_name, data_size)
        
        # Measurements
        elapsed_time = time.perf_counter() - start_time
        cpu_after = process.cpu_times()
        memory_after = process.memory_info().rss / 1024 / 1024 / 1024
        
        metrics = {
            'strategy': strategy_name,
            'data_size': data_size,
            'wall_time_seconds': elapsed_time,
            'cpu_time_seconds': (cpu_after.user - cpu_before.user + 
                               cpu_after.system - cpu_before.system),
            'memory_used_gb': memory_after - memory_before,
            'peak_memory_gb': memory_after,
            'trades_per_second': result.get('total_trades', 0) / elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_metrics(metrics)
        return metrics
        
    def benchmark_data_loading(self, file_path: Path) -> Dict[str, float]:
        """Benchmark data loading performance"""
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        
        # Parquet loading
        start = time.perf_counter()
        df = pd.read_parquet(file_path)
        parquet_time = time.perf_counter() - start
        
        metrics = {
            'file_path': str(file_path),
            'file_size_mb': file_size_mb,
            'load_time_seconds': parquet_time,
            'throughput_mbps': file_size_mb / parquet_time,
            'rows_loaded': len(df),
            'rows_per_second': len(df) / parquet_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_metrics(metrics, 'data_loading')
        return metrics
        
    def monitor_resource_usage(self) -> Dict[str, float]:
        """Monitor current resource usage"""
        process = psutil.Process()
        
        # Memory
        memory = psutil.virtual_memory()
        process_memory = process.memory_info()
        
        # CPU
        cpu_percent = process.cpu_percent(interval=1)
        
        # Disk
        disk = psutil.disk_usage('/')
        
        metrics = {
            'system_memory_percent': memory.percent,
            'system_memory_available_gb': memory.available / 1024**3,
            'process_memory_rss_gb': process_memory.rss / 1024**3,
            'process_memory_vms_gb': process_memory.vms / 1024**3,
            'cpu_percent': cpu_percent,
            'cpu_count': psutil.cpu_count(),
            'disk_usage_percent': disk.percent,
            'disk_free_gb': disk.free / 1024**3,
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_metrics(metrics, 'resources')
        return metrics
        
    def log_metrics(self, metrics: Dict[str, Any], 
                   category: str = 'performance') -> None:
        """Log metrics to file"""
        log_file = self.log_dir / f"{category}_{datetime.now():%Y%m%d}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
            
    def load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline performance metrics"""
        baseline_file = self.log_dir / 'baseline.json'
        if baseline_file.exists():
            with open(baseline_file) as f:
                return json.load(f)
        return None
        
    def compare_to_baseline(self, current: Dict[str, float]) -> Dict[str, float]:
        """Compare current metrics to baseline"""
        if not self.baseline:
            return {'status': 'no_baseline'}
            
        comparison = {}
        for key in ['wall_time_seconds', 'memory_used_gb', 'cpu_time_seconds']:
            if key in current and key in self.baseline:
                baseline_val = self.baseline[key]
                current_val = current[key]
                comparison[f'{key}_change_pct'] = (
                    (current_val - baseline_val) / baseline_val * 100
                )
                
        return comparison
```

### Continuous Monitoring Service
```python
# src/monitoring/continuous_monitor.py
import threading
import time
from typing import Callable, List

class ContinuousMonitor:
    """Background monitoring service"""
    
    def __init__(self, interval_seconds: int = 60):
        self.interval = interval_seconds
        self.monitor = PerformanceMonitor()
        self.alerts = []
        self.running = False
        self.thread = None
        
    def start(self) -> None:
        """Start monitoring in background"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self) -> None:
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                metrics = self.monitor.monitor_resource_usage()
                
                # Check thresholds
                self._check_thresholds(metrics)
                
                # Sleep
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Monitor error: {e}")
                
    def _check_thresholds(self, metrics: Dict[str, float]) -> None:
        """Check metrics against thresholds"""
        # Memory threshold
        if metrics['process_memory_rss_gb'] > 64:
            self.alert(f"High memory usage: {metrics['process_memory_rss_gb']:.1f}GB")
            
        # CPU threshold
        if metrics['cpu_percent'] > 95:
            self.alert(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
            
        # Disk threshold
        if metrics['disk_free_gb'] < 10:
            self.alert(f"Low disk space: {metrics['disk_free_gb']:.1f}GB free")
            
    def alert(self, message: str) -> None:
        """Send alert"""
        alert = {
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.alerts.append(alert)
        print(f"ALERT: {message}")
```

## Performance Testing Suite

### Standard Benchmark Tests
```python
# benchmarks/standard_tests.py
from typing import Dict, List
import pandas as pd
import numpy as np

class StandardBenchmarks:
    """Standard performance benchmark tests"""
    
    @staticmethod
    def get_standard_datasets() -> Dict[str, pd.DataFrame]:
        """Generate standard test datasets"""
        datasets = {}
        
        # Small: 1 month minute data, 10 symbols
        datasets['small'] = generate_ohlcv_data(
            n_symbols=10,
            n_periods=8000  # ~1 month of minute data
        )
        
        # Medium: 1 year minute data, 10 symbols
        datasets['medium'] = generate_ohlcv_data(
            n_symbols=10,
            n_periods=78840  # ~1 year of minute data
        )
        
        # Large: 1 year minute data, 100 symbols
        datasets['large'] = generate_ohlcv_data(
            n_symbols=100,
            n_periods=78840
        )
        
        return datasets
        
    @staticmethod
    def get_standard_strategies() -> List[str]:
        """List of standard test strategies"""
        return [
            'MovingAverageCrossover',
            'BollingerBands',
            'MomentumStrategy',
            'MeanReversion'
        ]
        
    def run_all_benchmarks(self) -> pd.DataFrame:
        """Run complete benchmark suite"""
        results = []
        monitor = PerformanceMonitor()
        
        datasets = self.get_standard_datasets()
        strategies = self.get_standard_strategies()
        
        for data_name, data in datasets.items():
            for strategy_name in strategies:
                print(f"Benchmarking {strategy_name} on {data_name} dataset...")
                
                metrics = monitor.measure_backtest_performance(
                    strategy_name, 
                    data_name
                )
                
                results.append({
                    'dataset': data_name,
                    'strategy': strategy_name,
                    **metrics
                })
                
        return pd.DataFrame(results)
```

## Performance Regression Detection

### Automated Performance Tests
```python
# tests/test_performance.py
import pytest
from benchmarks.standard_tests import StandardBenchmarks

class TestPerformance:
    """Performance regression tests"""
    
    @pytest.fixture
    def benchmarks(self):
        return StandardBenchmarks()
        
    def test_backtest_speed_small(self, benchmarks):
        """Test small dataset performance"""
        monitor = PerformanceMonitor()
        
        metrics = monitor.measure_backtest_performance(
            'MovingAverageCrossover',
            'small'
        )
        
        # Should complete in <1 second
        assert metrics['wall_time_seconds'] < 1.0
        
    def test_backtest_speed_medium(self, benchmarks):
        """Test medium dataset performance"""
        monitor = PerformanceMonitor()
        
        metrics = monitor.measure_backtest_performance(
            'MovingAverageCrossover',
            'medium'
        )
        
        # Should complete in <5 seconds
        assert metrics['wall_time_seconds'] < 5.0
        
    def test_memory_usage(self, benchmarks):
        """Test memory usage stays reasonable"""
        monitor = PerformanceMonitor()
        
        initial = monitor.monitor_resource_usage()
        
        # Run backtest
        metrics = monitor.measure_backtest_performance(
            'MovingAverageCrossover',
            'medium'
        )
        
        # Memory increase should be <16GB
        assert metrics['memory_used_gb'] < 16.0
        
    @pytest.mark.slow
    def test_optimization_performance(self, benchmarks):
        """Test parameter optimization performance"""
        import time
        
        start = time.time()
        
        # Run optimization with 100 parameter combinations
        # (Implementation depends on optimization framework)
        
        elapsed = time.time() - start
        
        # Should complete in <5 minutes
        assert elapsed < 300
```

## Performance Dashboards

### Real-time Dashboard
```python
# src/monitoring/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        
    def run(self):
        """Run Streamlit dashboard"""
        st.title("Backtesting Performance Monitor")
        
        # Sidebar
        st.sidebar.header("Settings")
        refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 5, 60, 10)
        time_window = st.sidebar.selectbox(
            "Time window",
            ["1 hour", "24 hours", "7 days", "30 days"]
        )
        
        # Auto-refresh
        st.empty()
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Backtest Time", "3.2s", "-0.5s")
            
        with col2:
            st.metric("Memory Usage", "28.5 GB", "+2.1 GB")
            
        with col3:
            st.metric("CPU Usage", "72%", "-5%")
            
        with col4:
            st.metric("Disk Free", "52 GB", "-3 GB")
            
        # Performance over time
        st.header("Performance Trends")
        
        # Load recent data
        df = self.load_recent_metrics(time_window)
        
        # Backtest speed chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['wall_time_seconds'],
            mode='lines+markers',
            name='Backtest Time'
        ))
        fig.add_hline(y=5, line_dash="dash", line_color="red",
                     annotation_text="Target: 5s")
        fig.update_layout(title="Backtest Execution Time")
        st.plotly_chart(fig)
        
        # Memory usage chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['memory_used_gb'],
            mode='lines+markers',
            name='Memory Usage'
        ))
        fig2.add_hline(y=32, line_dash="dash", line_color="orange",
                      annotation_text="Warning: 32GB")
        fig2.add_hline(y=64, line_dash="dash", line_color="red",
                      annotation_text="Critical: 64GB")
        fig2.update_layout(title="Memory Usage")
        st.plotly_chart(fig2)
        
        # Recent alerts
        st.header("Recent Alerts")
        alerts = self.load_recent_alerts()
        if alerts:
            for alert in alerts[-5:]:
                st.warning(f"{alert['timestamp']}: {alert['message']}")
        else:
            st.success("No recent alerts")
            
    def load_recent_metrics(self, time_window: str) -> pd.DataFrame:
        """Load metrics for specified time window"""
        # Parse time window
        window_map = {
            "1 hour": timedelta(hours=1),
            "24 hours": timedelta(days=1),
            "7 days": timedelta(days=7),
            "30 days": timedelta(days=30)
        }
        
        cutoff = datetime.now() - window_map[time_window]
        
        # Load and filter data
        all_metrics = []
        for file in self.log_dir.glob("performance_*.jsonl"):
            df = pd.read_json(file, lines=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[df['timestamp'] > cutoff]
            all_metrics.append(df)
            
        return pd.concat(all_metrics, ignore_index=True)
```

## Performance Reports

### Weekly Performance Report
```python
# src/monitoring/reports.py
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceReporter:
    """Generate performance reports"""
    
    def generate_weekly_report(self) -> None:
        """Generate weekly performance report"""
        # Load data
        monitor = PerformanceMonitor()
        df = self.load_week_metrics()
        
        # Create report
        report = {
            'period': f"{datetime.now() - timedelta(days=7)} to {datetime.now()}",
            'summary': self.calculate_summary(df),
            'trends': self.analyze_trends(df),
            'alerts': self.get_alerts(df),
            'recommendations': self.generate_recommendations(df)
        }
        
        # Generate PDF
        self.create_pdf_report(report)
        
    def calculate_summary(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics"""
        return {
            'total_backtests': len(df),
            'avg_execution_time': df['wall_time_seconds'].mean(),
            'max_execution_time': df['wall_time_seconds'].max(),
            'avg_memory_usage': df['memory_used_gb'].mean(),
            'max_memory_usage': df['memory_used_gb'].max(),
            'performance_vs_baseline': self.compare_to_baseline(df)
        }
        
    def analyze_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze performance trends"""
        # Group by day
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily = df.groupby('date').agg({
            'wall_time_seconds': 'mean',
            'memory_used_gb': 'mean',
            'cpu_time_seconds': 'mean'
        })
        
        # Calculate trends
        trends = {}
        for col in daily.columns:
            # Simple linear regression
            x = np.arange(len(daily))
            y = daily[col].values
            slope, intercept = np.polyfit(x, y, 1)
            
            trends[col] = {
                'slope': slope,
                'direction': 'increasing' if slope > 0 else 'decreasing',
                'change_per_day': slope
            }
            
        return trends
        
    def generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Check execution time
        if df['wall_time_seconds'].mean() > 4:
            recommendations.append(
                "Average execution time approaching limit. "
                "Consider optimizing data loading or strategy calculations."
            )
            
        # Check memory usage
        if df['memory_used_gb'].max() > 48:
            recommendations.append(
                "High memory usage detected. "
                "Implement chunked processing or reduce data retention."
            )
            
        # Check trend
        trends = self.analyze_trends(df)
        if trends['wall_time_seconds']['slope'] > 0.1:
            recommendations.append(
                "Performance degradation detected. "
                "Review recent code changes for inefficiencies."
            )
            
        return recommendations
```

## Integration with CI/CD

### GitHub Actions Performance Check
```yaml
# .github/workflows/performance.yml
name: Performance Tests

on:
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * *'  # Daily

jobs:
  performance:
    runs-on: macos-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        pip install uv
        uv pip install -r requirements.txt
        
    - name: Run performance benchmarks
      run: |
        python benchmarks/run_all_benchmarks.py
        
    - name: Check performance regression
      run: |
        python benchmarks/check_regression.py
        
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: logs/performance/
        
    - name: Comment PR
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const results = JSON.parse(
            fs.readFileSync('logs/performance/summary.json', 'utf8')
          );
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: `## Performance Test Results\n\n${results.summary}`
          });
```

## Best Practices

1. **Establish Baseline Early**
   - Run benchmarks on Day 1
   - Save as reference point
   - Update after major changes

2. **Monitor Continuously**
   - Run background monitor during development
   - Check metrics after each feature
   - Set up alerts for degradation

3. **Track Trends**
   - Look for gradual degradation
   - Identify performance cliffs
   - Correlate with code changes

4. **Optimize Based on Data**
   - Profile before optimizing
   - Focus on bottlenecks
   - Verify improvements

5. **Document Performance**
   - Keep performance log
   - Document optimization decisions
   - Share findings with team