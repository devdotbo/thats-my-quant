# Examples

This directory contains example scripts demonstrating various features of the quantitative trading backtesting system.

## Available Examples

### 1. Performance Comparison Demo (`performance_comparison_demo.py`)
Demonstrates the complete performance comparison framework:
- Running multiple strategies on the same data
- Statistical significance testing between strategies
- Generating performance visualizations
- Creating professional HTML reports

**Usage:**
```bash
python examples/performance_comparison_demo.py
```

**Output:**
- `output/equity_curves.png` - Multi-strategy equity curve comparison
- `output/performance_heatmap.png` - Performance metrics heatmap
- `output/strategy_comparison_report.html` - Complete HTML report
- `output/strategy_comparison_summary.json` - JSON summary of results

## Prerequisites

Before running the examples, ensure you have:
1. Downloaded sample data (at least AAPL for January 2024)
2. Activated your conda environment
3. Installed all dependencies

## Notes

- Examples use minute data from the `data/raw/minute_aggs/by_symbol/` directory
- Output files are saved to the `examples/output/` directory
- Feel free to modify the examples to test your own strategies