# That's My Quant

> "That's my quant." - Jared Vennett, The Big Short

![That's My Quant](quant-quantitative.gif)

A comprehensive quantitative trading backtesting system for educational purposes, inspired by the world of quantitative finance portrayed in popular culture.

## Educational Disclaimer

**This project is for educational purposes only.** It is designed to help you learn about:
- Quantitative trading strategies
- Backtesting methodologies
- Risk management in algorithmic trading
- Market microstructure and transaction costs

**This is not financial advice.** Any trading strategies developed or tested with this system should not be used with real money without thorough understanding of the risks involved. Financial markets are complex and risky.

## Overview

That's My Quant is a Python-based backtesting system that allows you to test trading strategies against historical market data. Built with performance and realism in mind, it provides a robust framework for learning about algorithmic trading while avoiding common pitfalls like overfitting.

### Key Features

- **Real Market Data**: Integration with Polygon.io for high-quality historical data
- **Dual Backtesting Engines**: VectorBT for speed, Backtrader for complex strategies
- **Overfitting Prevention**: Walk-forward optimization and Monte Carlo validation
- **Realistic Cost Modeling**: Comprehensive transaction costs including spread, slippage, and market impact
- **Apple Silicon Optimized**: Leverages M-series processors for maximum performance
- **Modular Architecture**: Easy to extend with new strategies and data sources

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │   Jupyter   │  │     CLI      │  │    Analysis      │    │
│  │  Notebooks  │  │  Interface   │  │    Dashboard     │    │
│  └─────────────┘  └──────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                            │
│         Strategy Engine │ Backtesting │ Risk Management          │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                                 │
│         Polygon.io │ Cache Manager │ Data Preprocessing          │
└─────────────────────────────────────────────────────────────────┘
```

## Getting Started

### Prerequisites

- Python 3.11+
- Anaconda (recommended for environment management)
- uv (for fast package management)
- 100GB available disk space for market data
- Polygon.io API credentials

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/thats-my-quant.git
cd thats-my-quant
```

2. Create and activate the conda environment:
```bash
conda create -n quant-m3 python=3.11
conda activate quant-m3
```

3. Install Apple Silicon optimized packages:
```bash
conda install numpy "blas=*=*accelerate*"
conda install scipy pandas
```

4. Install remaining dependencies:
```bash
pip install uv
uv pip install -r requirements.txt
```

5. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your Polygon.io credentials
```

6. Run initial benchmarks:
```bash
python benchmarks/hardware_test.py
```

### Quick Example

```python
from src.strategies import MovingAverageCrossover
from src.backtesting import VectorBTEngine
from src.data import PolygonDataLoader

# Load data
loader = PolygonDataLoader()
data = loader.get_minute_bars('AAPL', start='2023-01-01', end='2023-12-31')

# Create strategy
strategy = MovingAverageCrossover(fast_period=20, slow_period=50)

# Run backtest
engine = VectorBTEngine()
results = engine.run(strategy, data)

# Analyze results
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
```

## Documentation

- [Technical Architecture](plan.md) - System design and components
- [Data Architecture](data_architecture.md) - Data pipeline and storage
- [Overfitting Prevention](overfitting_prevention.md) - Validation techniques
- [Transaction Costs](transaction_costs.md) - Realistic cost modeling
- [Strategy Development](strategies/README.md) - Creating trading strategies
- [Claude AI Instructions](claude.md) - Development guidelines

## Performance

Optimized for Apple Silicon M-series processors:
- Backtest 1 year of minute data in <5 seconds
- Support for 1000+ parameter combinations
- Memory efficient with datasets up to 100GB

## Risk Warning

Trading financial instruments carries a high level of risk and may not be suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade, you should carefully consider your investment objectives, level of experience, and risk appetite.

This software is provided as-is for educational purposes. The authors and contributors are not responsible for any financial losses incurred through the use of this software.

## Contributing

Contributions are welcome! Please read our contributing guidelines and ensure all code includes proper tests and documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by quantitative finance concepts and the portrayal of quants in popular culture
- Built on the shoulders of giants: VectorBT, Backtrader, and the Python scientific computing ecosystem
- Special thanks to the open-source community for making projects like this possible

---

*Remember: The best trading strategy is often the simplest one. Start small, test thoroughly, and never risk money you can't afford to lose.*