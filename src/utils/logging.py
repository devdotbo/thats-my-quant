"""
Logging setup and utilities for That's My Quant
Provides structured logging with performance tracking
"""

import sys
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, Callable
from functools import wraps
from contextlib import contextmanager
from datetime import datetime

from loguru import logger


# Global logger instances cache
_logger_instances: Dict[str, Any] = {}

# Track open file handles for rotation
_log_file_handles: Dict[str, Any] = {}


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    format: str = 'json',
    rotation_size: str = '100 MB',
    retention_count: int = 10,
    filter_modules: Optional[List[str]] = None
) -> None:
    """
    Setup logging configuration for the application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        console: Whether to log to console
        format: Log format ('json' or 'simple')
        rotation_size: Size for log rotation (e.g., '100 MB', '1 GB')
        retention_count: Number of rotated logs to keep
        filter_modules: List of module prefixes to include (e.g., ['src.data'])
    """
    # Remove default handler
    logger.remove()
    
    # Configure format
    if format == 'json':
        # Add console handler with JSON serialization
        if console:
            logger.add(
                sys.stderr,
                serialize=True,
                level=level,
                filter=create_module_filter(filter_modules) if filter_modules else None
            )
        
        # Add file handler with custom JSON formatting
        if log_file:
            # Create a custom sink
            def json_sink(message):
                record = message.record
                json_str = json_formatter(record)
                with open(log_file, 'a') as f:
                    f.write(json_str + '\n')
            
            logger.add(
                json_sink,
                level=level,
                filter=create_module_filter(filter_modules) if filter_modules else None
            )
    else:
        formatter = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
        
        # Add console handler
        if console:
            logger.add(
                sys.stderr,
                format=formatter,
                level=level,
                filter=create_module_filter(filter_modules) if filter_modules else None
            )
        
        # Add file handler
        if log_file:
            logger.add(
                log_file,
                format=formatter,
                level=level,
                rotation=rotation_size,
                retention=retention_count,
                compression="gz",
                filter=create_module_filter(filter_modules) if filter_modules else None
            )


def json_formatter(record):
    """Custom JSON formatter for loguru"""
    # Build log data
    log_data = {
        'timestamp': record['time'].isoformat(),
        'level': record['level'].name,
        'logger': record.get('extra', {}).get('logger', record['name']),
        'message': record['message'],
        'function': record['function'],
        'line': record['line']
    }
    
    # Add extra fields
    extra = record.get('extra', {})
    if extra:
        # Handle nested extra from logger.bind()
        for key, value in extra.items():
            if key == 'extra' and isinstance(value, dict):
                # Merge nested extra dict
                log_data.update(value)
            elif key != 'logger':  # Skip logger since we already have it
                log_data[key] = value
    
    # Add exception info if present
    if record.get('exception'):
        exc_info = record['exception']
        if exc_info:
            log_data['traceback'] = ''.join(
                traceback.format_exception(
                    type(exc_info.value),
                    exc_info.value,
                    exc_info.traceback
                )
            )
    
    # Return the JSON string without newline (loguru adds it)
    return json.dumps(log_data)


def create_module_filter(module_prefixes: List[str]) -> Callable:
    """
    Create a filter function for module-based filtering
    
    Args:
        module_prefixes: List of module prefixes to include
        
    Returns:
        Filter function
    """
    def filter_func(record: Dict[str, Any]) -> bool:
        # Check the logger name from extra data first (from logger.bind)
        logger_name = record.get('extra', {}).get('logger', record['name'])
        return any(logger_name.startswith(prefix) for prefix in module_prefixes)
    
    return filter_func


def get_logger(name: str) -> Any:
    """
    Get a logger instance (cached for singleton pattern)
    
    Args:
        name: Logger name (usually module name)
        
    Returns:
        Logger instance
    """
    if name not in _logger_instances:
        _logger_instances[name] = logger.bind(logger=name)
    
    return _logger_instances[name]


def log_execution_time(operation_name: str) -> Callable:
    """
    Decorator to log function execution time
    
    Args:
        operation_name: Name of the operation for logging
        
    Returns:
        Decorator function
    
    Example:
        @log_execution_time('data_processing')
        def process_data(df):
            # Process data
            return df
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                logger.info(
                    f"Function {operation_name} completed",
                    extra={
                        'function': operation_name,
                        'execution_time': execution_time,
                        'status': 'success'
                    }
                )
                
                return result
                
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                
                logger.error(
                    f"Function {operation_name} failed",
                    extra={
                        'function': operation_name,
                        'execution_time': execution_time,
                        'status': 'error',
                        'error': str(e)
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


@contextmanager
def log_performance(logger_instance: Any, operation: str):
    """
    Context manager for logging performance of code blocks
    
    Args:
        logger_instance: Logger instance to use
        operation: Name of the operation
        
    Example:
        with log_performance(logger, 'data_loading'):
            # Load data
            df = pd.read_csv('data.csv')
    """
    start_time = time.perf_counter()
    
    try:
        yield
        
        duration = time.perf_counter() - start_time
        logger_instance.info(
            f"Operation {operation} completed",
            extra={
                'operation': operation,
                'duration_seconds': duration,
                'status': 'success'
            }
        )
        
    except Exception as e:
        duration = time.perf_counter() - start_time
        
        logger_instance.error(
            f"Operation {operation} failed",
            extra={
                'operation': operation,
                'duration_seconds': duration,
                'status': 'error',
                'error': str(e)
            },
            exc_info=True
        )
        raise


class QuantLogger:
    """
    Specialized logger for quantitative trading operations
    """
    
    def __init__(self, name: str):
        """
        Initialize QuantLogger
        
        Args:
            name: Logger name
        """
        self.logger = get_logger(name)
    
    def log_trade(self, 
                  symbol: str,
                  side: str,
                  quantity: float,
                  price: float,
                  commission: float = 0.0,
                  strategy: Optional[str] = None,
                  **kwargs) -> None:
        """
        Log a trade execution
        
        Args:
            symbol: Trading symbol
            side: Trade side (BUY/SELL)
            quantity: Trade quantity
            price: Execution price
            commission: Commission paid
            strategy: Strategy that generated the trade
            **kwargs: Additional trade metadata
        """
        self.logger.info(
            f"Trade executed: {side} {quantity} {symbol} @ {price}",
            extra={
                'event_type': 'trade',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'strategy': strategy,
                'timestamp': datetime.now().isoformat(),
                **kwargs
            }
        )
    
    def log_signal(self,
                   symbol: str,
                   signal_type: str,
                   strength: float,
                   strategy: str,
                   indicators: Optional[Dict[str, Any]] = None,
                   **kwargs) -> None:
        """
        Log a trading signal
        
        Args:
            symbol: Trading symbol
            signal_type: Signal type (LONG/SHORT/NEUTRAL)
            strength: Signal strength (0-1)
            strategy: Strategy that generated the signal
            indicators: Indicator values used
            **kwargs: Additional signal metadata
        """
        self.logger.info(
            f"Signal generated: {signal_type} {symbol} (strength: {strength})",
            extra={
                'event_type': 'signal',
                'symbol': symbol,
                'signal_type': signal_type,
                'strength': strength,
                'strategy': strategy,
                'indicators': indicators or {},
                'timestamp': datetime.now().isoformat(),
                **kwargs
            }
        )
    
    def log_performance(self,
                       strategy: str,
                       period: str,
                       metrics: Dict[str, Any],
                       **kwargs) -> None:
        """
        Log performance metrics
        
        Args:
            strategy: Strategy name
            period: Time period
            metrics: Performance metrics dictionary
            **kwargs: Additional metadata
        """
        self.logger.info(
            f"Performance metrics for {strategy} ({period})",
            extra={
                'event_type': 'performance',
                'strategy': strategy,
                'period': period,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                **kwargs
            }
        )
    
    def log_error(self,
                  error_type: str,
                  message: str,
                  context: Optional[Dict[str, Any]] = None,
                  **kwargs) -> None:
        """
        Log an error with context
        
        Args:
            error_type: Type of error
            message: Error message
            context: Error context
            **kwargs: Additional metadata
        """
        self.logger.error(
            f"{error_type}: {message}",
            extra={
                'event_type': 'error',
                'error_type': error_type,
                'message': message,
                'context': context or {},
                'timestamp': datetime.now().isoformat(),
                **kwargs
            },
            exc_info=True
        )
    
    def log_data_event(self,
                      event_type: str,
                      symbol: str,
                      data_type: str,
                      records: int,
                      time_range: Optional[Dict[str, str]] = None,
                      **kwargs) -> None:
        """
        Log data-related events
        
        Args:
            event_type: Type of event (download, cache_hit, etc.)
            symbol: Trading symbol
            data_type: Type of data (trades, quotes, etc.)
            records: Number of records
            time_range: Time range of data
            **kwargs: Additional metadata
        """
        self.logger.info(
            f"Data event: {event_type} {records} {data_type} records for {symbol}",
            extra={
                'event_type': f'data_{event_type}',
                'symbol': symbol,
                'data_type': data_type,
                'records': records,
                'time_range': time_range,
                'timestamp': datetime.now().isoformat(),
                **kwargs
            }
        )