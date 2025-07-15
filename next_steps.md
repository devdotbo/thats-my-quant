# Next Steps Guide - Fresh Context

## Immediate Actions (First 30 minutes)

### 1. Commit Pending Changes
```bash
# Add utility modules and tests
git add src/utils/config.py src/utils/logging.py
git add tests/test_utils/
git add pytest.ini
git add progress_summary.md implementation_status.md next_steps.md

# Commit with comprehensive message
git commit -m "feat: implement configuration and logging utilities with comprehensive tests

- Add type-safe configuration loader with env override support
- Create structured logging system with JSON formatting
- Implement specialized QuantLogger for trading events
- Add comprehensive test suites (33 tests total)
- Update documentation for context reset
- Configuration tests: 18/18 passing
- Logging tests: 15/16 passing (1 skipped)"
```

### 2. Environment Setup & Verification
```bash
# Create and activate environment
conda create -n quant-m3 python=3.11
conda activate quant-m3

# Install Apple Silicon optimized packages
conda install numpy "blas=*=*accelerate*"
conda install scipy pandas

# Install dependencies
pip install uv
uv pip install -r requirements.txt

# Verify installation
python -c "import vectorbt; print(f'VectorBT {vectorbt.__version__} installed')"
python -c "import numpy; numpy.show_config()"
```

### 3. Run Benchmarks
```bash
# Test hardware performance
python benchmarks/hardware_test.py

# Test VectorBT performance
python benchmarks/vectorbt_benchmark.py

# Test Polygon.io connection (ensure .env has credentials)
python benchmarks/test_polygon_connection.py
```

### 4. Verify Tests Pass
```bash
# Run all tests
python -m pytest tests/ -v

# Should see:
# - 18 passed in test_config.py
# - 15 passed, 1 skipped in test_logging.py
```

## Phase 1 Implementation: Data Pipeline

### Day 2: Polygon Data Downloader

#### 1. Create Test First (tests/test_data/test_downloader.py)
```python
import pytest
from unittest.mock import Mock, patch
from src.data.downloader import PolygonDownloader

class TestPolygonDownloader:
    def test_initialization(self, mock_polygon_credentials):
        downloader = PolygonDownloader()
        assert downloader.s3_client is not None
    
    def test_download_minute_aggregates(self, mock_s3_client):
        # Test downloading minute data
        pass
    
    def test_retry_on_failure(self):
        # Test retry logic
        pass
    
    def test_progress_tracking(self):
        # Test tqdm progress
        pass
```

#### 2. Implement Downloader (src/data/downloader.py)
```python
from typing import Optional, List, Dict, Any
import boto3
from datetime import datetime, date
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.utils.config import get_config
from src.utils.logging import get_logger, QuantLogger

class PolygonDownloader:
    """Downloads market data from Polygon.io S3 flat files"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.quant_logger = QuantLogger('data')
        self._setup_s3_client()
    
    def download_minute_aggregates(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """Download minute aggregate data for symbol"""
        # Implementation here
        pass
```

### Day 3: Cache Management System

#### 1. Test First (tests/test_data/test_cache.py)
```python
class TestCacheManager:
    def test_cache_size_limit(self):
        # Test 100GB limit enforcement
        pass
    
    def test_lru_eviction(self):
        # Test least recently used eviction
        pass
    
    def test_cache_statistics(self):
        # Test hit/miss tracking
        pass
```

#### 2. Implement Cache (src/data/cache.py)
```python
class CacheManager:
    """Manages local data cache with LRU eviction"""
    
    def __init__(self, max_size_gb: float = 100):
        self.max_size_bytes = max_size_gb * 1024**3
        self._init_cache_db()
    
    def get_or_download(self, key: str, download_func):
        """Get from cache or download if missing"""
        pass
```

### Day 4: Data Preprocessor

#### Key Implementation Points:
1. Always write tests first
2. Use type hints on all functions
3. Add comprehensive docstrings with examples
4. Log important operations
5. Handle errors gracefully
6. NO TODOs - complete each function

### Day 5: VectorBT Engine Integration

#### Critical Implementation:
```python
from vectorbt import Portfolio
from src.strategies.base import BaseStrategy

class VectorBTEngine:
    """Wrapper for VectorBT backtesting engine"""
    
    def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float = 100000
    ) -> BacktestResult:
        """Run backtest with strategy on data"""
        # 1. Generate signals
        signals = strategy.generate_signals(data)
        
        # 2. Calculate positions
        positions = strategy.calculate_positions(
            signals, initial_capital
        )
        
        # 3. Run VectorBT portfolio
        portfolio = Portfolio.from_orders(...)
        
        # 4. Calculate metrics
        return BacktestResult(...)
```

## Testing Guidelines

### For Every New Module:
1. Create test file first
2. Write test cases covering:
   - Happy path
   - Edge cases
   - Error conditions
   - Performance requirements
3. Run tests in watch mode: `pytest-watch`
4. Aim for >80% coverage

### Integration Testing:
After each module, create integration test:
```python
def test_full_data_pipeline():
    # Download data
    downloader = PolygonDownloader()
    data = downloader.download_minute_aggregates('SPY', ...)
    
    # Cache it
    cache = CacheManager()
    cache.store(data)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.clean(data)
    
    # Verify end-to-end
    assert len(clean_data) > 0
```

## Performance Monitoring

Track these metrics continuously:
```python
from src.utils.logging import log_performance

# Monitor download speed
with log_performance(logger, 'polygon_download'):
    data = downloader.download(...)
    mb_per_sec = len(data) / elapsed_time / 1024 / 1024
    logger.info(f"Download speed: {mb_per_sec:.2f} MB/s")

# Monitor backtest performance
with log_performance(logger, 'vectorbt_backtest'):
    result = engine.run_backtest(...)
    logger.info(f"Backtest completed in {elapsed:.2f}s")
```

## Common Pitfalls to Avoid

1. **Don't Skip Tests**: Every function needs tests
2. **Don't Hardcode Paths**: Use config system
3. **Don't Ignore Errors**: Log and handle gracefully
4. **Don't Mix Concerns**: Keep modules focused
5. **Don't Optimize Early**: Get it working first

## Daily Checklist

- [ ] Read previous day's progress
- [ ] Run all tests to ensure clean state
- [ ] Implement one module completely
- [ ] Write comprehensive tests
- [ ] Update documentation
- [ ] Commit with descriptive message
- [ ] Note any blockers or decisions

## Resources

- VectorBT Docs: https://vectorbt.dev/
- Polygon.io Flat Files: https://polygon.io/flat-files
- Pytest Best Practices: https://docs.pytest.org/en/stable/goodpractices.html
- Type Hints: https://docs.python.org/3/library/typing.html

## Questions for User

Before implementing major features, clarify:
1. Preferred data format for cache (Parquet confirmed?)
2. Strategy parameter ranges for optimization
3. Specific performance metrics to track
4. Live trading integration plans (future phase?)
5. ML strategy priority (based on GPU benchmarks)