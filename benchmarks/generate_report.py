"""
Generate Benchmark Performance Report
Creates detailed HTML/Markdown reports from benchmark results
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ReportGenerator:
    """Generate comprehensive benchmark reports"""
    
    def __init__(self):
        """Initialize report generator"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.benchmarks_dir = Path(__file__).parent
        self.results_dir = self.benchmarks_dir / "results"
        self.reports_dir = self.benchmarks_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
    
    def find_all_results(self, pattern: str = "all_benchmarks_*.json") -> List[Path]:
        """Find all benchmark result files"""
        return sorted(
            self.results_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
    
    def load_results(self, file_path: Path) -> Optional[Dict]:
        """Load benchmark results from file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def generate_markdown_report(self, results: Dict) -> str:
        """Generate markdown report from results"""
        timestamp = results.get('timestamp', 'Unknown')
        
        report = []
        report.append("# That's My Quant - Performance Report")
        report.append(f"\nGenerated: {timestamp}")
        report.append("\n---\n")
        
        # Executive Summary
        report.append("## Executive Summary")
        
        summary_stats = results.get('summary_stats', {})
        overall_status = results.get('overall_status', 'unknown')
        
        report.append(f"\n- **Overall Status**: {overall_status.upper()}")
        report.append(f"- **Total Benchmarks**: {summary_stats.get('total', 0)}")
        report.append(f"- **Successful**: {summary_stats.get('successful', 0)}")
        report.append(f"- **Failed**: {summary_stats.get('failed', 0)}")
        report.append(f"- **Total Execution Time**: {results.get('total_execution_time', 0):.1f} seconds")
        
        # Key Metrics
        key_metrics = results.get('key_metrics', {})
        if key_metrics:
            report.append("\n## Key Performance Metrics")
            report.append("\n| Metric | Value | Unit |")
            report.append("|--------|-------|------|")
            
            metric_info = {
                'cpu_gflops': ('CPU Performance', 'GFLOPS'),
                'memory_bandwidth_gb_s': ('Memory Bandwidth', 'GB/s'),
                'vectorbt_1y_backtest_time': ('1Y Backtest Time', 'seconds'),
                'parquet_read_speed_mb_s': ('Parquet Read Speed', 'MB/s')
            }
            
            for metric, value in key_metrics.items():
                if metric in metric_info:
                    name, unit = metric_info[metric]
                    report.append(f"| {name} | {value:.2f} | {unit} |")
        
        # Individual Benchmark Results
        report.append("\n## Individual Benchmark Results")
        
        benchmarks = results.get('benchmarks', {})
        for name, data in benchmarks.items():
            report.append(f"\n### {name}")
            
            status = data.get('status', 'unknown')
            exec_time = data.get('execution_time', 0)
            
            status_emoji = "✅" if status == 'success' else "❌"
            report.append(f"\n- **Status**: {status_emoji} {status}")
            report.append(f"- **Execution Time**: {exec_time:.1f} seconds")
            
            # Add specific benchmark data if available
            if 'data' in data and status == 'success':
                bench_data = data['data']
                
                # Hardware Performance
                if name == 'Hardware Performance':
                    if 'cpu_performance' in bench_data:
                        cpu = bench_data['cpu_performance']
                        report.append(f"\n#### CPU Performance")
                        report.append(f"- Matrix Operations: {cpu.get('gflops', 0):.1f} GFLOPS")
                        report.append(f"- Execution Time: {cpu.get('execution_time', 0):.3f} seconds")
                    
                    if 'memory_performance' in bench_data:
                        mem = bench_data['memory_performance']
                        report.append(f"\n#### Memory Performance")
                        report.append(f"- Bandwidth: {mem.get('bandwidth_gb_s', 0):.1f} GB/s")
                        report.append(f"- Array Size: {mem.get('array_size_mb', 0):.1f} MB")
                
                # VectorBT Performance
                elif name == 'VectorBT Performance':
                    if 'results' in bench_data:
                        report.append(f"\n#### Backtest Performance")
                        for test_name, test_data in bench_data['results'].items():
                            report.append(f"\n**{test_name}**:")
                            report.append(f"- Backtest Time: {test_data.get('backtest_time', 0):.3f} seconds")
                            report.append(f"- Total Return: {test_data.get('total_return', 0):.2%}")
                            report.append(f"- Sharpe Ratio: {test_data.get('sharpe_ratio', 0):.2f}")
                            report.append(f"- Max Drawdown: {test_data.get('max_drawdown', 0):.2%}")
                
                # I/O Performance
                elif name == 'I/O Performance':
                    report.append(f"\n#### Data I/O Performance")
                    
                    # Find 100k row results
                    if 'size_100000' in bench_data:
                        size_data = bench_data['size_100000']
                        formats = size_data.get('formats', {})
                        
                        report.append(f"\n| Format | File Size | Write Speed | Read Speed |")
                        report.append("|--------|-----------|-------------|------------|")
                        
                        for fmt, perf in formats.items():
                            file_size = perf.get('file_size_mb', 0)
                            write_speed = perf.get('write_speed_mb_s', 0)
                            read_speed = perf.get('read_speed_mb_s', 0)
                            report.append(f"| {fmt.upper()} | {file_size:.1f} MB | "
                                        f"{write_speed:.1f} MB/s | {read_speed:.1f} MB/s |")
            
            elif status == 'failed' and 'error' in data:
                report.append(f"\n**Error**: {data['error']}")
        
        # System Information
        report.append("\n## System Information")
        
        if benchmarks and 'Hardware Performance' in benchmarks:
            hw_data = benchmarks['Hardware Performance'].get('data', {})
            sys_info = hw_data.get('system_info', {})
            
            if sys_info:
                report.append(f"\n- **Platform**: {sys_info.get('platform', 'Unknown')}")
                report.append(f"- **CPU Count**: {sys_info.get('cpu_count', 0)}")
                report.append(f"- **Total Memory**: {sys_info.get('total_memory_gb', 0):.1f} GB")
                report.append(f"- **Python Version**: {sys_info.get('python_version', 'Unknown')}")
        
        # Performance Validation
        report.append("\n## Performance Validation")
        
        targets = {
            'vectorbt_1y_backtest_time': ('1Y Backtest < 5s', 5.0, True),
            'memory_bandwidth_gb_s': ('Memory > 20 GB/s', 20.0, False),
            'parquet_read_speed_mb_s': ('Parquet > 100 MB/s', 100.0, False),
            'cpu_gflops': ('CPU > 100 GFLOPS', 100.0, False)
        }
        
        report.append("\n| Target | Requirement | Actual | Status |")
        report.append("|--------|-------------|--------|--------|")
        
        for metric, (desc, target, lower_is_better) in targets.items():
            if metric in key_metrics:
                value = key_metrics[metric]
                if lower_is_better:
                    meets = value <= target
                else:
                    meets = value >= target
                
                status = "✅ PASS" if meets else "❌ FAIL"
                report.append(f"| {desc} | {target} | {value:.2f} | {status} |")
        
        return "\n".join(report)
    
    def generate_html_report(self, markdown_content: str) -> str:
        """Convert markdown report to HTML"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>That's My Quant - Performance Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        .pass {{
            color: #27ae60;
            font-weight: bold;
        }}
        .fail {{
            color: #e74c3c;
            font-weight: bold;
        }}
        hr {{
            border: 0;
            height: 1px;
            background: #ddd;
            margin: 30px 0;
        }}
        ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        ul li:before {{
            content: "▸ ";
            color: #3498db;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    {content}
</body>
</html>
"""
        
        # Simple markdown to HTML conversion
        # In production, use a proper markdown parser like markdown2
        html_content = markdown_content
        
        # Convert headers
        html_content = html_content.replace('# ', '<h1>').replace('\n## ', '</h1>\n<h2>')
        html_content = html_content.replace('\n### ', '</h2>\n<h3>')
        html_content = html_content.replace('\n#### ', '</h3>\n<h4>')
        
        # Convert lists
        lines = html_content.split('\n')
        in_list = False
        converted_lines = []
        
        for line in lines:
            if line.startswith('- '):
                if not in_list:
                    converted_lines.append('<ul>')
                    in_list = True
                converted_lines.append(f'<li>{line[2:]}</li>')
            else:
                if in_list and not line.startswith('- '):
                    converted_lines.append('</ul>')
                    in_list = False
                converted_lines.append(line)
        
        if in_list:
            converted_lines.append('</ul>')
        
        html_content = '\n'.join(converted_lines)
        
        # Convert tables (simple approach)
        html_content = html_content.replace('|--------|', '</tr><tr>')
        html_content = html_content.replace('| ', '<td>').replace(' |', '</td>')
        
        # Add status colors
        html_content = html_content.replace('✅ PASS', '<span class="pass">✅ PASS</span>')
        html_content = html_content.replace('❌ FAIL', '<span class="fail">❌ FAIL</span>')
        
        return html_template.format(content=html_content)
    
    def collect_historical_data(self, limit: int = 30) -> pd.DataFrame:
        """Collect historical benchmark data for trends"""
        result_files = self.find_all_results()[:limit]
        
        data = []
        for file_path in result_files:
            results = self.load_results(file_path)
            if results and 'key_metrics' in results:
                row = {
                    'timestamp': pd.to_datetime(results['timestamp']),
                    'file': file_path.name
                }
                row.update(results['key_metrics'])
                data.append(row)
        
        if data:
            return pd.DataFrame(data).set_index('timestamp').sort_index()
        else:
            return pd.DataFrame()
    
    def generate_trend_charts_data(self, df: pd.DataFrame) -> Dict:
        """Generate data for trend charts"""
        charts = {}
        
        metrics = [
            ('cpu_gflops', 'CPU Performance (GFLOPS)'),
            ('memory_bandwidth_gb_s', 'Memory Bandwidth (GB/s)'),
            ('vectorbt_1y_backtest_time', '1Y Backtest Time (seconds)'),
            ('parquet_read_speed_mb_s', 'Parquet Read Speed (MB/s)')
        ]
        
        for metric, title in metrics:
            if metric in df.columns:
                series = df[metric].dropna()
                if len(series) > 0:
                    charts[metric] = {
                        'title': title,
                        'dates': series.index.strftime('%Y-%m-%d %H:%M').tolist(),
                        'values': series.values.tolist(),
                        'mean': float(series.mean()),
                        'std': float(series.std()),
                        'trend': 'stable'  # Could add trend detection
                    }
        
        return charts
    
    def generate_comprehensive_report(self, latest_results: Optional[Path] = None):
        """Generate comprehensive performance report"""
        # Find latest results if not specified
        if latest_results is None:
            result_files = self.find_all_results()
            if not result_files:
                print("❌ No benchmark results found")
                return None
            latest_results = result_files[0]
        
        print(f"Loading results from: {latest_results}")
        results = self.load_results(latest_results)
        
        if not results:
            print("❌ Failed to load results")
            return None
        
        # Generate markdown report
        markdown_report = self.generate_markdown_report(results)
        
        # Save markdown report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_file = self.reports_dir / f"performance_report_{timestamp}.md"
        
        with open(md_file, 'w') as f:
            f.write(markdown_report)
        
        print(f"✅ Markdown report saved to: {md_file}")
        
        # Generate HTML report
        html_report = self.generate_html_report(markdown_report)
        html_file = self.reports_dir / f"performance_report_{timestamp}.html"
        
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        print(f"✅ HTML report saved to: {html_file}")
        
        # Generate trend data
        historical_df = self.collect_historical_data()
        if not historical_df.empty:
            trend_data = self.generate_trend_charts_data(historical_df)
            
            # Save trend data
            trend_file = self.reports_dir / f"performance_trends_{timestamp}.json"
            with open(trend_file, 'w') as f:
                json.dump(trend_data, f, indent=2)
            
            print(f"✅ Trend data saved to: {trend_file}")
        
        return {
            'markdown': md_file,
            'html': html_file,
            'trends': trend_file if not historical_df.empty else None
        }


def main():
    """Generate performance report"""
    generator = ReportGenerator()
    
    # Check for command line arguments
    results_file = None
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
        if not results_file.exists():
            print(f"❌ Results file not found: {results_file}")
            sys.exit(1)
    
    report_files = generator.generate_comprehensive_report(results_file)
    
    if report_files:
        print("\n✅ Reports generated successfully!")
        print("\nYou can view the reports at:")
        for report_type, file_path in report_files.items():
            if file_path:
                print(f"  - {report_type}: {file_path}")
    else:
        print("\n❌ Failed to generate reports")
        sys.exit(1)


if __name__ == "__main__":
    main()