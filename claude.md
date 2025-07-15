# Claude AI Assistant Instructions for Quantitative Trading Backtesting Project

## Project Overview
This is a quantitative trading backtesting system designed to test various trading strategies on historical market data from Polygon.io. The system is built with Python and emphasizes robustness, flexibility, and realistic simulation of trading conditions.

## Environment Setup

### Python Environment
- **Package Manager**: Use `uv` for fast dependency resolution and package management
- **Environment Manager**: Use Anaconda for managing Python environments
- **Python Version**: 3.11+ recommended for compatibility with latest libraries

### Key Commands
```bash
# Create new conda environment
conda create -n quant-backtest python=3.11

# Activate environment
conda activate quant-backtest

# Install uv
pip install uv

# Install dependencies with uv
uv pip install -r requirements.txt

# Sync dependencies
uv pip sync
```

## Development Workflow

### Git Best Practices
1. **ALWAYS commit after completing each task**
   - Use descriptive commit messages
   - Follow conventional commits format: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`

2. **Before EVERY commit**:
   ```bash
   # Dry run to check what will be committed
   git add --dry-run .
   
   # Check status
   git status
   
   # Review changes
   git diff --cached
   ```

3. **NEVER commit**:
   - API keys or secrets
   - Large data files
   - Personal information
   - Cache files or temporary data

### .gitignore Management
Update `.gitignore` proactively when:
- Adding new data directories
- Creating cache folders
- Generating output files
- Adding environment-specific configs

Essential entries:
```
# Secrets
.env
*.key
*.pem

# Data
data/
*.csv
*.parquet
*.h5

# Cache
__pycache__/
.cache/
*.pyc

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store
```

## MCP Servers Usage

### Research Phase
- **Tavily MCP Server** (`mcp__tavily__`): Use for researching trading concepts, market mechanics, and general quantitative finance topics

### Implementation Phase
- **GitHub MCP Server** (`mcp__github__`): Use for exploring library source code, finding examples, and understanding implementation details
- **Context7 MCP Server** (`mcp__context7__`): Use for specific library documentation and API references when implementing features

## Code Standards

### Testing Requirements
Before marking any feature as complete:
1. Run unit tests: `pytest tests/`
2. Run type checking: `mypy src/`
3. Run linting: `ruff check .`
4. Run formatting: `ruff format .`

### Code Quality
- Write type hints for all functions
- Document all public APIs with docstrings
- Keep functions focused and under 50 lines
- Use descriptive variable names
- Follow PEP 8 style guide

## Project Structure
```
thats-my-quant/
├── claude.md                    # This file
├── research.md                  # Library research and recommendations
├── plan.md                      # Technical architecture
├── data_architecture.md         # Data pipeline design
├── overfitting_prevention.md    # Validation framework
├── transaction_costs.md         # Cost modeling documentation
├── strategies/                  # Strategy implementations
│   └── README.md               # Strategy development guide
├── src/                        # Source code
│   ├── data/                   # Data loading and processing
│   ├── strategies/             # Strategy implementations
│   ├── backtesting/            # Backtesting engine
│   ├── analysis/               # Results analysis
│   └── utils/                  # Utility functions
├── tests/                      # Test suite
├── notebooks/                  # Jupyter notebooks for research
├── data/                       # Local data cache (gitignored)
└── results/                    # Backtest results (gitignored)
```

## Key Libraries

### Primary Stack
- **VectorBT**: Core backtesting engine for vectorized operations
- **Backtrader**: Alternative for complex event-driven strategies
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Matplotlib/Plotly**: Visualization

### Data Access
- **boto3**: S3 access for Polygon.io flat files
- **pyarrow**: Parquet file handling
- **requests**: API interactions

## Polygon.io Integration

### Environment Variables
Required in `.env`:
```
polygon_io_api_key=<key>
polygon_io_s3_access_key_id=<key>
polygon_io_s3_access_secret=<secret>
polygon_io_s3_endpoint=https://files.polygon.io
polygon_io_s3_bucket=flatfiles
```

### Data Types Available
- Trades
- Quotes  
- Minute aggregates
- Day aggregates

## Strategy Development Guidelines

1. **Start Simple**: Implement basic strategies first (MA crossover, momentum)
2. **Test Incrementally**: Validate each component before combining
3. **Document Assumptions**: Clearly state market assumptions
4. **Parameter Ranges**: Define reasonable parameter bounds
5. **Risk Management**: Always include stop-loss and position sizing

## Common Pitfalls to Avoid

1. **Look-ahead Bias**: Never use future data in calculations
2. **Survivorship Bias**: Account for delisted securities
3. **Overfitting**: Use walk-forward optimization
4. **Unrealistic Costs**: Include all transaction costs
5. **Data Snooping**: Limit strategy iterations on same dataset

## Jupyter Notebook Guidelines

- Use notebooks for exploration and visualization
- Convert proven strategies to Python modules
- Clear outputs before committing
- Name notebooks descriptively with dates

## Performance Considerations

- Use vectorized operations wherever possible
- Cache downloaded data locally
- Implement incremental data updates
- Profile code for bottlenecks
- Consider parallel processing for multiple strategies

## Remember

1. **Commit frequently** - After each completed feature or fix
2. **Test thoroughly** - No code is complete without tests  
3. **Document clearly** - Future you will thank present you
4. **Stay realistic** - Model real trading conditions accurately
5. **Iterate carefully** - Avoid overfitting through excessive optimization