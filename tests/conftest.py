"""
Pytest configuration and fixtures for That's My Quant
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import os


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="thatsmyquant_test_")
    yield Path(temp_dir)
    # Cleanup after all tests
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_config():
    """Sample configuration for testing"""
    return {
        'data': {
            'cache_dir': './test_cache',
            'max_cache_size_gb': 1
        },
        'backtesting': {
            'initial_capital': 100000,
            'costs': {
                'commission_rate': 0.001,
                'slippage_bps': 5
            }
        },
        'optimization': {
            'walk_forward': {
                'train_periods': 252,
                'test_periods': 63,
                'step_size': 21
            }
        }
    }


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    n_days = 500
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    
    # Generate realistic price movement
    np.random.seed(42)
    returns = np.random.randn(n_days) * 0.02
    price = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    noise = np.random.randn(n_days)
    data = pd.DataFrame({
        'open': price * (1 + noise * 0.001),
        'high': price * (1 + np.abs(noise) * 0.005),
        'low': price * (1 - np.abs(noise) * 0.005),
        'close': price,
        'volume': np.random.randint(1000000, 10000000, n_days)
    }, index=dates)
    
    # Ensure high >= low and high >= open, close
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def sample_multi_symbol_data():
    """Generate multi-symbol OHLCV data"""
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    n_days = 252
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    
    data = {}
    np.random.seed(42)
    
    for symbol in symbols:
        returns = np.random.randn(n_days) * 0.02
        price = 100 * np.exp(np.cumsum(returns))
        noise = np.random.randn(n_days)
        
        data[('open', symbol)] = price * (1 + noise * 0.001)
        data[('high', symbol)] = price * (1 + np.abs(noise) * 0.005)
        data[('low', symbol)] = price * (1 - np.abs(noise) * 0.005)
        data[('close', symbol)] = price
        data[('volume', symbol)] = np.random.randint(1000000, 10000000, n_days)
    
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    
    return df


@pytest.fixture
def sample_minute_data():
    """Generate sample minute-frequency data"""
    # One day of minute data
    dates = pd.date_range('2023-01-03 09:30:00', '2023-01-03 16:00:00', freq='1min')
    n_periods = len(dates)
    
    np.random.seed(42)
    returns = np.random.randn(n_periods) * 0.0001  # Smaller volatility for minute data
    price = 100 * np.exp(np.cumsum(returns))
    
    noise = np.random.randn(n_periods)
    data = pd.DataFrame({
        'open': price * (1 + noise * 0.0001),
        'high': price * (1 + np.abs(noise) * 0.0002),
        'low': price * (1 - np.abs(noise) * 0.0002),
        'close': price,
        'volume': np.random.randint(10000, 100000, n_periods)
    }, index=dates)
    
    # Ensure high >= low and high >= open, close
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def mock_polygon_credentials(monkeypatch):
    """Mock Polygon.io credentials for testing"""
    monkeypatch.setenv('polygon_io_api_key', 'test_api_key')
    monkeypatch.setenv('polygon_io_s3_access_key_id', 'test_access_key')
    monkeypatch.setenv('polygon_io_s3_access_secret', 'test_secret')
    monkeypatch.setenv('polygon_io_s3_endpoint', 'https://test.endpoint.com')
    monkeypatch.setenv('polygon_io_s3_bucket', 'test-bucket')


@pytest.fixture
def sample_strategy_params():
    """Sample strategy parameters for testing"""
    return {
        'ma_crossover': {
            'fast_period': 20,
            'slow_period': 50
        },
        'orb': {
            'opening_minutes': 5,
            'stop_loss_atr': 2.0,
            'take_profit_ratio': 2.0
        },
        'momentum': {
            'lookback_period': 20,
            'entry_threshold': 0.02,
            'exit_threshold': -0.01
        }
    }


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup any test files created during tests"""
    yield
    # Cleanup patterns
    patterns = [
        'test_*.csv',
        'test_*.parquet',
        'test_*.json',
        'temp_*'
    ]
    
    for pattern in patterns:
        for file in Path('.').glob(pattern):
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)


@pytest.fixture
def performance_timer():
    """Simple performance timer for benchmarking"""
    import time
    
    class Timer:
        def __init__(self):
            self.times = {}
        
        def start(self, name: str):
            self.times[name] = time.perf_counter()
        
        def stop(self, name: str) -> float:
            if name not in self.times:
                raise ValueError(f"Timer '{name}' not started")
            elapsed = time.perf_counter() - self.times[name]
            del self.times[name]
            return elapsed
    
    return Timer()


# Markers for conditional test execution
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_polygon: marks tests that require Polygon.io credentials"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU/MPS support"
    )


# Skip tests based on environment
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment"""
    
    # Check if Polygon credentials are available
    has_polygon_creds = all([
        os.getenv('polygon_io_api_key'),
        os.getenv('polygon_io_s3_access_key_id'),
        os.getenv('polygon_io_s3_access_secret')
    ])
    
    # Check if GPU is available
    try:
        import torch
        has_gpu = torch.backends.mps.is_available()
    except ImportError:
        has_gpu = False
    
    skip_polygon = pytest.mark.skip(reason="Polygon.io credentials not available")
    skip_gpu = pytest.mark.skip(reason="GPU/MPS not available")
    
    for item in items:
        if "requires_polygon" in item.keywords and not has_polygon_creds:
            item.add_marker(skip_polygon)
        if "requires_gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)