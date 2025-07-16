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
# Create Apple Silicon optimized environment
conda create -n quant-m3 python=3.11

# Activate environment
conda activate quant-m3

# Install Accelerate-optimized NumPy/SciPy
conda install numpy "blas=*=*accelerate*"
conda install scipy pandas

# Install uv
pip install uv

# Install dependencies with uv
uv pip install -r requirements.txt

# Run initial benchmarks
python benchmarks/hardware_test.py
```

## Development Workflow

### Python File Permissions
- **NEVER use chmod on Python files** - Python files don't need execute permissions
- Python files are executed with `python filename.py`, not `./filename.py`
- Keep Python files with standard 644 permissions (rw-r--r--)

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

### Implementation Workflow
For EVERY feature implementation:

1. **Write Tests FIRST**
```python
def test_new_feature():
    """Test must exist before implementation"""
    expected = calculate_expected_result()
    actual = new_feature(test_data)
    assert actual == expected
```

2. **Implement Feature**
- Start with minimal working version
- NO TODOs in code - complete each function
- Include proper error handling

3. **Run All Validations**
```bash
# Run tests
pytest tests/test_new_feature.py -v

# Check types
mypy src/module.py

# Lint code
ruff check src/module.py

# Run feature with real data
python -m src.module --test

# Check performance
python benchmarks/test_performance.py
```

4. **Document and Commit**
```bash
# Verify no secrets
git diff | grep -E '(api_key|secret|password)'

# Dry run
git add --dry-run .

# Commit with descriptive message
git commit -m "feat: implement feature with full tests"
```

### Validation Gates
Critical checkpoints that MUST pass:

#### Day 1 Gate
- [ ] Polygon.io credentials verified
- [ ] Initial benchmark completed
- [ ] VectorBT installed and working
- [ ] First data download successful

#### Feature Completion Gate
- [ ] Unit tests written and passing (>80% coverage)
- [ ] Type hints on all functions
- [ ] Docstrings with examples
- [ ] Performance within targets
- [ ] NO TODOs in code
- [ ] Integration test passing

#### Weekly Gate
- [ ] All tests passing
- [ ] Benchmarks within 10% of baseline
- [ ] Memory usage <32GB
- [ ] No hardcoded values
- [ ] Documentation updated

### Testing Requirements
Before marking ANY feature as complete:

1. **Unit Tests**: `pytest tests/ -v --cov=src --cov-report=html`
   - Minimum 80% coverage
   - Test edge cases
   - **NEVER mock external dependencies** - use real services

2. **Type Checking**: `mypy src/ --strict`
   - No type errors allowed
   - Explicit types for all public APIs

### Testing Philosophy
- **NEVER mock external services in tests** - Use real connections and real data
- If a service is unavailable, skip the test with appropriate markers
- Integration tests should test actual integration, not mocked behavior
- Use test data files and fixtures, but connect to real services
- Real tests catch real problems that mocks would miss

3. **Code Quality**: `ruff check . && ruff format .`
   - Must pass all linting rules
   - Properly formatted code

4. **Performance**: `python benchmarks/run_benchmark.py`
   - Backtest time <5s for standard test
   - Memory usage <32GB

5. **Integration**: Run actual backtest
   - Download real data
   - Execute strategy
   - Verify results reasonable

### NO TODOs Policy
```python
# NEVER DO THIS:
def calculate_sharpe():
    # TODO: implement this
    pass

# ALWAYS DO THIS:
def calculate_sharpe(returns: pd.Series, 
                    risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Annualized Sharpe ratio
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> sharpe = calculate_sharpe(returns)
        >>> assert 0 < sharpe < 3  # Reasonable range
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 returns")
        
    excess_returns = returns - risk_free_rate/252
    
    if excess_returns.std() == 0:
        return 0.0
        
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
```

### Code Quality Standards
- Write type hints for ALL functions
- Document ALL public APIs with docstrings including examples
- Keep functions focused and under 50 lines
- Use descriptive variable names
- Follow PEP 8 style guide
- Handle errors explicitly

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

### Exploring Bucket Structure
```bash
# List all files (WARNING: Very large output ~900k+ files)
rclone ls s3polygon:flatfiles > data/polygon_bucket_structure.txt

# List directory structure only
rclone lsd s3polygon:flatfiles -R > data/polygon_bucket_dirs.txt

# Find specific data types
rclone ls s3polygon:flatfiles --include "**/SPY.csv.gz" --max-depth 4

# Explore specific paths
rclone ls s3polygon:flatfiles/us_stocks_sip/ --max-depth 2
```

### Important Discovery
**S3 Credentials**: Use `polygon_io_api_key` as the S3 secret access key (not `polygon_io_s3_access_secret`). This was discovered through rclone testing and is critical for successful connections.

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

## Benchmarking Requirements

### Initial Benchmarks (Day 1)
Must complete before any implementation:
```bash
# Run hardware capability tests
python benchmarks/hardware_test.py

# Test VectorBT performance
python benchmarks/vectorbt_performance.py

# Verify data pipeline speed
python benchmarks/io_performance.py
```

### Continuous Performance Tracking
After each major feature:
```bash
# Run full benchmark suite
python benchmarks/run_all_benchmarks.py

# Compare with baseline
python benchmarks/compare_baseline.py

# Generate performance report
python benchmarks/generate_report.py
```

### Performance Targets
- **Backtest Speed**: 1 year minute data <5 seconds
- **Memory Usage**: <32GB typical, <64GB peak
- **Data Loading**: >100 MB/s from disk
- **Optimization**: 1000 parameter combinations <30 minutes

## Data Management Guidelines

### Initial Data Strategy
Start small and expand progressively:
1. **Week 1**: 1 year of minute data for 10 stocks (~5GB)
2. **Week 3**: Expand to 5 years (~25GB)
3. **Week 5**: Add more symbols as needed (~50GB)
4. **Reserve**: Keep 50GB free for cache/results

### Storage Best Practices
- Use Parquet format with snappy compression
- Implement LRU cache with automatic cleanup
- Monitor disk usage continuously
- Never exceed 100GB total allocation

## Remember

1. **Commit frequently** - After each completed feature or fix
2. **Test thoroughly** - No code is complete without tests  
3. **Document clearly** - Future you will thank present you
4. **Stay realistic** - Model real trading conditions accurately
5. **Iterate carefully** - Avoid overfitting through excessive optimization
6. **Benchmark regularly** - Track performance degradation
7. **No TODOs** - Complete every function fully
8. **Run actual tests** - Always validate with real data

## Documentation Management Rules

1. **Single Source of Truth**: 
   - `implementation_status.md` - Current project state and progress
   - No duplicate status tracking in other files

2. **Session Work**:
   - Do NOT create session-specific MD files (SESSION_SUMMARY, HANDOFF, etc.)
   - Update implementation_status.md with completed work
   - Use git commits for detailed session history

3. **File Organization**:
   - `README.md` - User-facing overview
   - Technical docs - Specific design/architecture topics
   - `implementation_status.md` - Progress and next steps
   - No "summary" or "handoff" files

4. **When Adding Documentation**:
   - Check if content already exists elsewhere
   - Update existing files rather than creating new ones
   - Keep documentation DRY (Don't Repeat Yourself)
   - If unsure, add to implementation_status.md

