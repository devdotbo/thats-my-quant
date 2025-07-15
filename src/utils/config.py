"""
Configuration management for That's My Quant
Provides type-safe configuration loading with environment variable override support
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, TypeVar, cast
from dataclasses import dataclass
import copy


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    pass


class Config:
    """
    Configuration object with attribute and dictionary access
    
    Allows both config.data.cache_dir and config['data']['cache_dir'] syntax
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration with dictionary"""
        if config_dict is None:
            config_dict = {}
        
        self._data: Dict[str, Any] = {}
        self._update_from_dict(config_dict)
    
    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Recursively update configuration from dictionary"""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._data[key] = Config(value)
            else:
                self._data[key] = value
    
    def __getattr__(self, name: str) -> Any:
        """Get configuration value as attribute"""
        if name.startswith('_'):
            # Don't interfere with private attributes
            return object.__getattribute__(self, name)
        
        if name in self._data:
            return self._data[name]
        
        raise AttributeError(f"Configuration has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set configuration value as attribute"""
        if name.startswith('_'):
            # Set private attributes normally
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value as dictionary item"""
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value as dictionary item"""
        self._data[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration"""
        return key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return self._data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Config({self._data})"


class ConfigLoader:
    """Loads and validates configuration"""
    
    _instance: Optional[Config] = None
    _config_path: Optional[str] = None
    
    # Required configuration structure
    REQUIRED_FIELDS = {
        'data': {
            'cache_dir': str,
            'max_cache_size_gb': (int, float)
        },
        'backtesting': {
            'initial_capital': (int, float),
            'costs': {
                'commission_rate': (int, float),
                'slippage_bps': (int, float)
            }
        }
    }
    
    @classmethod
    def load_from_file(cls, 
                      config_path: Union[str, Path],
                      allow_env_override: bool = True) -> Config:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to configuration file
            allow_env_override: Whether to allow environment variable overrides
            
        Returns:
            Config object
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if allow_env_override:
            config_dict = cls._apply_env_overrides(config_dict)
        
        return cls.load_from_dict(config_dict)
    
    @classmethod
    def load_from_dict(cls, config_dict: Dict[str, Any]) -> Config:
        """
        Load configuration from dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config object
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        # Validate configuration
        cls._validate_config(config_dict)
        
        # Create Config object
        config = Config(config_dict)
        
        return config
    
    @classmethod
    def _validate_config(cls, config_dict: Dict[str, Any]) -> None:
        """
        Validate configuration against required fields and types
        
        Raises:
            ConfigValidationError: If validation fails
        """
        # Check required fields
        cls._check_required_fields(config_dict, cls.REQUIRED_FIELDS, '')
        
        # Validate specific constraints
        cls._validate_constraints(config_dict)
    
    @classmethod
    def _check_required_fields(cls, 
                              config: Dict[str, Any],
                              required: Dict[str, Any],
                              path: str) -> None:
        """Recursively check required fields exist"""
        for key, value in required.items():
            full_path = f"{path}.{key}" if path else key
            
            if key not in config:
                raise ConfigValidationError(
                    f"Missing required configuration field: {full_path}"
                )
            
            if isinstance(value, dict):
                if not isinstance(config[key], dict):
                    raise ConfigValidationError(
                        f"Configuration field {full_path} must be a dictionary"
                    )
                cls._check_required_fields(config[key], value, full_path)
            
            elif isinstance(value, (type, tuple)):
                # Check type
                expected_types = value if isinstance(value, tuple) else (value,)
                if not isinstance(config[key], expected_types):
                    type_names = ', '.join(t.__name__ for t in expected_types)
                    raise ConfigValidationError(
                        f"Configuration field {full_path} must be of type {type_names}, "
                        f"got {type(config[key]).__name__}"
                    )
    
    @classmethod
    def _validate_constraints(cls, config_dict: Dict[str, Any]) -> None:
        """Validate configuration value constraints"""
        # Data constraints
        if 'data' in config_dict:
            max_cache = config_dict['data'].get('max_cache_size_gb', 0)
            if not isinstance(max_cache, (int, float)) or max_cache <= 0:
                raise ConfigValidationError(
                    "max_cache_size_gb must be a positive number"
                )
        
        # Backtesting constraints
        if 'backtesting' in config_dict:
            capital = config_dict['backtesting'].get('initial_capital', 0)
            if not isinstance(capital, (int, float)) or capital <= 0:
                raise ConfigValidationError(
                    "initial_capital must be positive"
                )
            
            if 'costs' in config_dict['backtesting']:
                costs = config_dict['backtesting']['costs']
                
                commission = costs.get('commission_rate', 0)
                if not isinstance(commission, (int, float)) or commission < 0 or commission >= 1:
                    raise ConfigValidationError(
                        "commission_rate must be between 0 and 1"
                    )
                
                slippage = costs.get('slippage_bps', 0)
                if not isinstance(slippage, (int, float)) or slippage < 0:
                    raise ConfigValidationError(
                        "slippage_bps must be non-negative"
                    )
    
    @classmethod
    def _apply_env_overrides(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides
        
        Environment variables should be prefixed with QUANT_ and use underscores
        for nested values. For example:
        - QUANT_DATA_CACHE_DIR overrides data.cache_dir
        - QUANT_BACKTESTING_INITIAL_CAPITAL overrides backtesting.initial_capital
        """
        config_copy = copy.deepcopy(config_dict)
        
        # Define mapping of environment variables to config paths
        env_mappings = {
            'QUANT_DATA_CACHE_DIR': ['data', 'cache_dir'],
            'QUANT_DATA_MAX_CACHE_SIZE_GB': ['data', 'max_cache_size_gb'],
            'QUANT_BACKTESTING_INITIAL_CAPITAL': ['backtesting', 'initial_capital'],
            'QUANT_BACKTESTING_COSTS_COMMISSION_RATE': ['backtesting', 'costs', 'commission_rate'],
            'QUANT_BACKTESTING_COSTS_SLIPPAGE_BPS': ['backtesting', 'costs', 'slippage_bps'],
        }
        
        for env_key, config_path in env_mappings.items():
            if env_key not in os.environ:
                continue
            
            env_value = os.environ[env_key]
            
            # Navigate to the right location in config
            current = config_copy
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value
            final_key = config_path[-1]
            
            # Try to convert to appropriate type
            try:
                # Try float first (also handles int)
                current[final_key] = float(env_value)
                if current[final_key].is_integer():
                    current[final_key] = int(current[final_key])
            except ValueError:
                # Keep as string
                current[final_key] = env_value
        
        return config_copy
    
    @classmethod
    def get_config(cls, config_path: Optional[Union[str, Path]] = None) -> Config:
        """
        Get configuration (singleton pattern)
        
        Args:
            config_path: Path to configuration file. If not provided and no
                        config is loaded, will look for configs/config.yaml
                        
        Returns:
            Config object
        """
        if cls._instance is None:
            if config_path is None:
                # Look for default config
                default_path = Path('configs/config.yaml')
                if default_path.exists():
                    config_path = default_path
                else:
                    raise ConfigValidationError(
                        "No configuration loaded and no default config found"
                    )
            
            cls._instance = cls.load_from_file(config_path)
            cls._config_path = str(config_path)
        
        return cls._instance
    
    @classmethod
    def reload_config(cls, config_path: Optional[Union[str, Path]] = None) -> Config:
        """
        Reload configuration from file
        
        Args:
            config_path: Path to configuration file. If not provided,
                        uses the last loaded path
                        
        Returns:
            Config object
        """
        if config_path is None and cls._config_path is not None:
            config_path = cls._config_path
        elif config_path is None:
            raise ConfigValidationError("No configuration path available for reload")
        
        cls._instance = cls.load_from_file(config_path)
        cls._config_path = str(config_path)
        
        return cls._instance
    
    @classmethod
    def merge_configs(cls, base: Config, override: Dict[str, Any]) -> Config:
        """
        Merge override configuration into base configuration
        
        Args:
            base: Base configuration
            override: Override values
            
        Returns:
            Merged configuration
        """
        base_dict = base.to_dict()
        merged_dict = cls._deep_merge(base_dict, override)
        return cls.load_from_dict(merged_dict)
    
    @classmethod
    def _deep_merge(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result


def get_config() -> Config:
    """Convenience function to get global configuration"""
    return ConfigLoader.get_config()