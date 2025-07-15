"""
Tests for configuration loader utility
"""

import pytest
import os
import tempfile
from pathlib import Path
import yaml
from typing import Dict, Any

from src.utils.config import Config, ConfigLoader, ConfigValidationError


class TestConfigLoader:
    """Test configuration loading and validation"""
    
    @pytest.fixture
    def sample_config_dict(self) -> Dict[str, Any]:
        """Sample configuration dictionary"""
        return {
            'data': {
                'cache_dir': './data/cache',
                'max_cache_size_gb': 100
            },
            'backtesting': {
                'initial_capital': 100000.0,
                'costs': {
                    'commission_rate': 0.001,
                    'slippage_bps': 5
                }
            },
            'system': {
                'n_jobs': -1,
                'chunk_size': 10000
            }
        }
    
    @pytest.fixture
    def config_file(self, sample_config_dict):
        """Create temporary config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config_dict, f)
            config_path = f.name
        
        yield config_path
        
        # Cleanup
        os.unlink(config_path)
    
    def test_load_from_file(self, config_file):
        """Test loading configuration from file"""
        config = ConfigLoader.load_from_file(config_file)
        
        assert isinstance(config, Config)
        assert config.data.cache_dir == './data/cache'
        assert config.data.max_cache_size_gb == 100
        assert config.backtesting.initial_capital == 100000.0
        assert config.backtesting.costs.commission_rate == 0.001
    
    def test_load_from_dict(self, sample_config_dict):
        """Test loading configuration from dictionary"""
        config = ConfigLoader.load_from_dict(sample_config_dict)
        
        assert isinstance(config, Config)
        assert config.data.cache_dir == './data/cache'
        assert config.system.n_jobs == -1
    
    def test_environment_override(self, config_file, monkeypatch):
        """Test environment variable override"""
        # Set environment variables
        monkeypatch.setenv('QUANT_DATA_CACHE_DIR', '/tmp/override_cache')
        monkeypatch.setenv('QUANT_BACKTESTING_INITIAL_CAPITAL', '200000')
        
        config = ConfigLoader.load_from_file(config_file, allow_env_override=True)
        
        assert config.data.cache_dir == '/tmp/override_cache'
        assert config.backtesting.initial_capital == 200000.0
    
    def test_validation_missing_required(self):
        """Test validation of missing required fields"""
        incomplete_config = {
            'data': {
                'cache_dir': './data/cache'
                # Missing max_cache_size_gb
            }
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load_from_dict(incomplete_config)
        
        assert 'max_cache_size_gb' in str(exc_info.value)
    
    def test_validation_invalid_type(self):
        """Test validation of invalid types"""
        invalid_config = {
            'data': {
                'cache_dir': './data/cache',
                'max_cache_size_gb': 'invalid'  # Should be numeric
            },
            'backtesting': {
                'initial_capital': 100000.0,
                'costs': {
                    'commission_rate': 0.001,
                    'slippage_bps': 5
                }
            }
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load_from_dict(invalid_config)
        
        assert 'must be of type' in str(exc_info.value) and 'int, float' in str(exc_info.value)
    
    def test_validation_invalid_range(self):
        """Test validation of values outside valid range"""
        invalid_config = {
            'data': {
                'cache_dir': './data/cache',
                'max_cache_size_gb': -10  # Should be positive
            },
            'backtesting': {
                'initial_capital': -1000,  # Should be positive
                'costs': {
                    'commission_rate': 2.0,  # Should be < 1
                    'slippage_bps': -5  # Should be positive
                }
            }
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load_from_dict(invalid_config)
        
        error_msg = str(exc_info.value)
        assert 'positive' in error_msg or 'range' in error_msg
    
    def test_singleton_pattern(self, config_file):
        """Test singleton pattern for global config"""
        config1 = ConfigLoader.get_config(config_file)
        config2 = ConfigLoader.get_config()
        
        assert config1 is config2
        assert id(config1) == id(config2)
    
    def test_reload_config(self, config_file, sample_config_dict):
        """Test reloading configuration"""
        # Load initial config
        config1 = ConfigLoader.get_config(config_file)
        initial_capital = config1.backtesting.initial_capital
        
        # Modify config file
        sample_config_dict['backtesting']['initial_capital'] = 200000.0
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        # Reload config
        config2 = ConfigLoader.reload_config(config_file)
        
        assert config2.backtesting.initial_capital == 200000.0
        assert config2.backtesting.initial_capital != initial_capital
    
    def test_nested_access(self, sample_config_dict):
        """Test nested configuration access"""
        config = ConfigLoader.load_from_dict(sample_config_dict)
        
        # Test nested access
        assert config.backtesting.costs.commission_rate == 0.001
        assert config.backtesting.costs.slippage_bps == 5
        
        # Test get with default
        assert config.get('nonexistent', 'default') == 'default'
        assert config.backtesting.costs.get('nonexistent', 0) == 0
    
    def test_config_to_dict(self, sample_config_dict):
        """Test converting config back to dictionary"""
        config = ConfigLoader.load_from_dict(sample_config_dict)
        config_dict = config.to_dict()
        
        assert config_dict == sample_config_dict
    
    def test_default_config_path(self, monkeypatch):
        """Test loading from default config path"""
        # Clear singleton instance
        ConfigLoader._instance = None
        ConfigLoader._config_path = None
        
        # Create a temporary default config
        default_config = {
            'data': {
                'cache_dir': './default_cache',
                'max_cache_size_gb': 50
            },
            'backtesting': {
                'initial_capital': 50000.0,
                'costs': {
                    'commission_rate': 0.002,
                    'slippage_bps': 10
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'configs' / 'config.yaml'
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f)
            
            # Change working directory
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                config = ConfigLoader.get_config()
                assert config.data.cache_dir == './default_cache'
                assert config.backtesting.initial_capital == 50000.0
            finally:
                os.chdir(original_cwd)
    
    def test_merge_configs(self):
        """Test merging multiple configurations"""
        base_config = {
            'data': {
                'cache_dir': './data/cache',
                'max_cache_size_gb': 100
            },
            'backtesting': {
                'initial_capital': 100000.0,
                'costs': {
                    'commission_rate': 0.001,
                    'slippage_bps': 5
                }
            }
        }
        
        override_config = {
            'data': {
                'cache_dir': './override/cache'  # Override this
            },
            'backtesting': {
                'costs': {
                    'commission_rate': 0.002  # Override this
                }
            },
            'system': {  # Add new section
                'n_jobs': 4
            }
        }
        
        config = ConfigLoader.load_from_dict(base_config)
        config = ConfigLoader.merge_configs(config, override_config)
        
        assert config.data.cache_dir == './override/cache'
        assert config.data.max_cache_size_gb == 100  # Not overridden
        assert config.backtesting.costs.commission_rate == 0.002
        assert config.backtesting.costs.slippage_bps == 5  # Not overridden
        assert config.system.n_jobs == 4


class TestConfig:
    """Test Config class functionality"""
    
    def test_attribute_access(self):
        """Test attribute-style access"""
        config = Config({
            'level1': {
                'level2': {
                    'value': 42
                }
            }
        })
        
        assert config.level1.level2.value == 42
    
    def test_dictionary_access(self):
        """Test dictionary-style access"""
        config = Config({
            'level1': {
                'level2': {
                    'value': 42
                }
            }
        })
        
        assert config['level1']['level2']['value'] == 42
    
    def test_mixed_access(self):
        """Test mixed attribute and dictionary access"""
        config = Config({
            'level1': {
                'level2': {
                    'value': 42
                }
            }
        })
        
        assert config.level1['level2'].value == 42
        assert config['level1'].level2['value'] == 42
    
    def test_update_value(self):
        """Test updating configuration values"""
        config = Config({
            'data': {
                'cache_dir': './original'
            }
        })
        
        config.data.cache_dir = './updated'
        assert config.data.cache_dir == './updated'
        
        config['data']['cache_dir'] = './updated_again'
        assert config.data.cache_dir == './updated_again'
    
    def test_add_new_key(self):
        """Test adding new configuration keys"""
        config = Config({})
        
        config.new_section = Config({'key': 'value'})
        assert config.new_section.key == 'value'
    
    def test_contains(self):
        """Test 'in' operator"""
        config = Config({
            'data': {
                'cache_dir': './cache'
            }
        })
        
        assert 'data' in config
        assert 'cache_dir' in config.data
        assert 'nonexistent' not in config