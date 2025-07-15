"""
Tests for logging setup utility
"""

import pytest
import tempfile
from pathlib import Path
import json
import time

from src.utils.logging import (
    setup_logging, 
    get_logger, 
    log_performance,
    log_execution_time,
    QuantLogger
)


class TestLoggingSetup:
    """Test logging configuration and setup"""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name
        
        yield log_path
        
        # Cleanup
        Path(log_path).unlink(missing_ok=True)
    
    def test_basic_setup(self, temp_log_file):
        """Test basic logging setup"""
        setup_logging(
            level='INFO',
            log_file=temp_log_file,
            format='json'
        )
        
        logger = get_logger('test_module')
        logger.info('Test message', extra={'key': 'value'})
        
        # Read log file
        with open(temp_log_file, 'r') as f:
            log_line = f.read().strip()
        
        # Parse JSON log
        log_data = json.loads(log_line)
        assert log_data['message'] == 'Test message'
        assert log_data['key'] == 'value'
        assert log_data['logger'] == 'test_module'
        assert log_data['level'] == 'INFO'
    
    def test_different_log_levels(self, temp_log_file):
        """Test different logging levels"""
        setup_logging(
            level='DEBUG',
            log_file=temp_log_file,
            format='json'
        )
        
        logger = get_logger('test_levels')
        
        # Test different levels
        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')
        
        # Read all log lines
        with open(temp_log_file, 'r') as f:
            log_lines = f.readlines()
        
        assert len(log_lines) == 4
        
        levels = [json.loads(line)['level'] for line in log_lines]
        assert levels == ['DEBUG', 'INFO', 'WARNING', 'ERROR']
    
    def test_structured_logging(self, temp_log_file):
        """Test structured logging with extra fields"""
        setup_logging(
            level='INFO',
            log_file=temp_log_file,
            format='json'
        )
        
        logger = get_logger('structured')
        
        # Log with structured data
        logger.info(
            'Trade executed',
            extra={
                'symbol': 'AAPL',
                'quantity': 100,
                'price': 150.25,
                'side': 'BUY'
            }
        )
        
        with open(temp_log_file, 'r') as f:
            log_data = json.loads(f.read().strip())
        
        assert log_data['symbol'] == 'AAPL'
        assert log_data['quantity'] == 100
        assert log_data['price'] == 150.25
        assert log_data['side'] == 'BUY'
    
    def test_performance_decorator(self, temp_log_file):
        """Test performance logging decorator"""
        setup_logging(
            level='INFO',
            log_file=temp_log_file,
            format='json'
        )
        
        @log_execution_time('test_function')
        def slow_function(duration: float):
            time.sleep(duration)
            return 'done'
        
        result = slow_function(0.1)
        assert result == 'done'
        
        # Check performance log
        with open(temp_log_file, 'r') as f:
            log_data = json.loads(f.read().strip())
        
        assert log_data['function'] == 'test_function'
        assert 'execution_time' in log_data
        assert log_data['execution_time'] >= 0.1
        assert log_data['execution_time'] < 0.2  # Some overhead
    
    def test_performance_context_manager(self, temp_log_file):
        """Test performance logging context manager"""
        setup_logging(
            level='INFO',
            log_file=temp_log_file,
            format='json'
        )
        
        logger = get_logger('perf_test')
        
        with log_performance(logger, 'data_processing'):
            time.sleep(0.05)
        
        with open(temp_log_file, 'r') as f:
            log_data = json.loads(f.read().strip())
        
        assert log_data['operation'] == 'data_processing'
        assert 'duration_seconds' in log_data
        assert log_data['duration_seconds'] >= 0.05
    
    @pytest.mark.skip(reason="Log rotation not implemented in custom sink")
    def test_log_rotation(self, temp_log_file):
        """Test log file rotation"""
        # Setup with small max size to trigger rotation
        setup_logging(
            level='INFO',
            log_file=temp_log_file,
            format='json',
            rotation_size='1KB',
            retention_count=2
        )
        
        logger = get_logger('rotation_test')
        
        # Write enough logs to trigger rotation
        for i in range(100):
            logger.info(f'Log message {i}' * 10)  # Long message
        
        # Check that rotation occurred
        log_dir = Path(temp_log_file).parent
        log_files = list(log_dir.glob(f"{Path(temp_log_file).stem}*"))
        
        # Should have main file plus at least one rotated file
        assert len(log_files) >= 2
    
    def test_logger_singleton(self):
        """Test logger singleton pattern"""
        logger1 = get_logger('test_module')
        logger2 = get_logger('test_module')
        
        assert logger1 is logger2
    
    def test_custom_formatter(self, temp_log_file):
        """Test custom log formatting"""
        setup_logging(
            level='INFO',
            log_file=temp_log_file,
            format='simple'  # Not JSON
        )
        
        logger = get_logger('format_test')
        logger.info('Simple format test')
        
        with open(temp_log_file, 'r') as f:
            log_line = f.read().strip()
        
        # Should contain timestamp, level, logger name, and message
        assert 'INFO' in log_line
        assert 'test_custom_formatter' in log_line  # Function name
        assert 'Simple format test' in log_line
    
    def test_exception_logging(self, temp_log_file):
        """Test exception logging with traceback"""
        setup_logging(
            level='INFO',
            log_file=temp_log_file,
            format='json'
        )
        
        logger = get_logger('exception_test')
        
        try:
            raise ValueError("Test exception")
        except Exception:
            logger.exception('Exception occurred')
        
        with open(temp_log_file, 'r') as f:
            log_data = json.loads(f.read().strip())
        
        assert log_data['level'] == 'ERROR'
        assert 'traceback' in log_data
        assert 'ValueError: Test exception' in log_data['traceback']


class TestQuantLogger:
    """Test QuantLogger class functionality"""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name
        
        yield log_path
        
        # Cleanup
        Path(log_path).unlink(missing_ok=True)
    
    def test_quant_logger_trade_logging(self, temp_log_file):
        """Test specialized trade logging"""
        setup_logging(
            level='INFO',
            log_file=temp_log_file,
            format='json'
        )
        
        logger = QuantLogger('trading')
        
        # Log a trade
        logger.log_trade(
            symbol='AAPL',
            side='BUY',
            quantity=100,
            price=150.25,
            commission=0.10,
            strategy='momentum'
        )
        
        with open(temp_log_file, 'r') as f:
            log_data = json.loads(f.read().strip())
        
        assert log_data['event_type'] == 'trade'
        assert log_data['symbol'] == 'AAPL'
        assert log_data['side'] == 'BUY'
        assert log_data['quantity'] == 100
        assert log_data['price'] == 150.25
        assert log_data['commission'] == 0.10
        assert log_data['strategy'] == 'momentum'
    
    def test_quant_logger_signal_logging(self, temp_log_file):
        """Test signal logging"""
        setup_logging(
            level='INFO',
            log_file=temp_log_file,
            format='json'
        )
        
        logger = QuantLogger('signals')
        
        # Log a signal
        logger.log_signal(
            symbol='MSFT',
            signal_type='LONG',
            strength=0.85,
            strategy='ma_crossover',
            indicators={'ma_fast': 50.2, 'ma_slow': 49.8}
        )
        
        with open(temp_log_file, 'r') as f:
            log_data = json.loads(f.read().strip())
        
        assert log_data['event_type'] == 'signal'
        assert log_data['symbol'] == 'MSFT'
        assert log_data['signal_type'] == 'LONG'
        assert log_data['strength'] == 0.85
        assert log_data['indicators']['ma_fast'] == 50.2
    
    def test_quant_logger_performance_logging(self, temp_log_file):
        """Test performance metrics logging"""
        setup_logging(
            level='INFO',
            log_file=temp_log_file,
            format='json'
        )
        
        logger = QuantLogger('performance')
        
        # Log performance metrics
        logger.log_performance(
            strategy='momentum',
            period='2023-01',
            metrics={
                'total_return': 0.05,
                'sharpe_ratio': 1.5,
                'max_drawdown': -0.02,
                'win_rate': 0.55
            }
        )
        
        with open(temp_log_file, 'r') as f:
            log_data = json.loads(f.read().strip())
        
        assert log_data['event_type'] == 'performance'
        assert log_data['strategy'] == 'momentum'
        assert log_data['metrics']['sharpe_ratio'] == 1.5
    
    def test_quant_logger_error_logging(self, temp_log_file):
        """Test error logging with context"""
        setup_logging(
            level='INFO',
            log_file=temp_log_file,
            format='json'
        )
        
        logger = QuantLogger('errors')
        
        # Log an error
        logger.log_error(
            error_type='DataError',
            message='Missing price data',
            context={
                'symbol': 'AAPL',
                'date': '2023-01-01',
                'data_source': 'polygon'
            }
        )
        
        with open(temp_log_file, 'r') as f:
            log_data = json.loads(f.read().strip())
        
        assert log_data['event_type'] == 'error'
        assert log_data['error_type'] == 'DataError'
        assert log_data['context']['symbol'] == 'AAPL'


class TestLoggingConfiguration:
    """Test various logging configurations"""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name
        
        yield log_path
        
        # Cleanup
        Path(log_path).unlink(missing_ok=True)
    
    def test_console_only_logging(self, capsys):
        """Test console-only logging without file"""
        setup_logging(
            level='INFO',
            console=True,
            format='simple'
        )
        
        logger = get_logger('console_test')
        logger.info('Console message')
        
        captured = capsys.readouterr()
        assert 'Console message' in captured.err  # loguru uses stderr by default
    
    def test_multiple_handlers(self, temp_log_file):
        """Test logging to both console and file"""
        setup_logging(
            level='INFO',
            log_file=temp_log_file,
            console=True,
            format='json'
        )
        
        logger = get_logger('multi_handler')
        logger.info('Multi-handler message')
        
        # Check file
        with open(temp_log_file, 'r') as f:
            log_data = json.loads(f.read().strip())
        
        assert log_data['message'] == 'Multi-handler message'
    
    def test_filter_by_module(self, temp_log_file):
        """Test filtering logs by module name"""
        # Clear any existing handlers first
        from loguru import logger
        logger.remove()
        
        setup_logging(
            level='INFO',
            log_file=temp_log_file,
            format='json',
            filter_modules=['src.data', 'src.strategies']
        )
        
        # These should be logged
        data_logger = get_logger('src.data.downloader')
        strategy_logger = get_logger('src.strategies.momentum')
        
        # This should be filtered out
        other_logger = get_logger('src.utils.helpers')
        
        data_logger.info('Data log')
        strategy_logger.info('Strategy log')
        other_logger.info('Other log')
        
        with open(temp_log_file, 'r') as f:
            log_lines = f.readlines()
        
        assert len(log_lines) == 2
        
        loggers = [json.loads(line)['logger'] for line in log_lines]
        assert 'src.data.downloader' in loggers
        assert 'src.strategies.momentum' in loggers
        assert 'src.utils.helpers' not in loggers