"""
Tests for base strategy interface
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.strategies.base import BaseStrategy, SignalType, StrategyMetadata


class ConcreteTestStrategy(BaseStrategy):
    """Concrete implementation for testing"""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Simple MA crossover signals for testing"""
        fast_ma = data['close'].rolling(10).mean()
        slow_ma = data['close'].rolling(20).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        
        return signals
    
    def calculate_positions(self, signals: pd.Series, capital: float,
                          current_positions=None, risk_params=None) -> pd.Series:
        """Simple position calculation"""
        risk_fraction = 0.1 if risk_params is None else risk_params.get('risk_fraction', 0.1)
        position_value = capital * risk_fraction
        
        # Assume constant price of 100 for simplicity
        shares = position_value / 100
        return signals * shares
    
    @property
    def required_history(self) -> int:
        return 20
    
    @property
    def required_features(self) -> List[str]:
        return ['atr', 'rsi']
    
    def _create_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="Test Strategy",
            version="1.0",
            author="Test",
            description="Strategy for testing base class",
            parameters=self.parameters,
            required_history=20,
            required_features=['atr', 'rsi'],
            supports_short=True,
            supports_intraday=True
        )


class MinimalTestStrategy(BaseStrategy):
    """Minimal implementation with custom validation"""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=data.index)
    
    def calculate_positions(self, signals: pd.Series, capital: float,
                          current_positions=None, risk_params=None) -> pd.Series:
        return signals * 100
    
    @property
    def required_history(self) -> int:
        return 1
    
    @property
    def required_features(self) -> List[str]:
        return []
    
    def _create_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="Minimal Strategy",
            version="1.0",
            author="Test",
            description="Minimal strategy",
            parameters=self.parameters,
            required_history=1,
            required_features=[]
        )
    
    def _validate_parameters(self) -> None:
        """Custom parameter validation"""
        if 'threshold' in self.parameters:
            if not 0 <= self.parameters['threshold'] <= 1:
                raise ValueError("Threshold must be between 0 and 1")


class TestSignalType:
    """Test SignalType enum"""
    
    def test_signal_type_values(self):
        """Test enum values"""
        assert SignalType.LONG.value == 1
        assert SignalType.SHORT.value == -1
        assert SignalType.NEUTRAL.value == 0
    
    def test_signal_type_from_value(self):
        """Test conversion from numeric values"""
        assert SignalType.from_value(1) == SignalType.LONG
        assert SignalType.from_value(0.5) == SignalType.LONG
        assert SignalType.from_value(-1) == SignalType.SHORT
        assert SignalType.from_value(-0.5) == SignalType.SHORT
        assert SignalType.from_value(0) == SignalType.NEUTRAL
        assert SignalType.from_value(0.0) == SignalType.NEUTRAL


class TestStrategyMetadata:
    """Test StrategyMetadata dataclass"""
    
    def test_metadata_creation(self):
        """Test creating metadata"""
        metadata = StrategyMetadata(
            name="Test",
            version="1.0",
            author="Tester",
            description="Test strategy",
            parameters={'param1': 10},
            required_history=20,
            required_features=['feature1']
        )
        
        assert metadata.name == "Test"
        assert metadata.version == "1.0"
        assert metadata.author == "Tester"
        assert metadata.supports_short is True  # Default
        assert metadata.supports_intraday is True  # Default
    
    def test_metadata_custom_flags(self):
        """Test metadata with custom flags"""
        metadata = StrategyMetadata(
            name="Test",
            version="1.0",
            author="Tester",
            description="Test strategy",
            parameters={},
            required_history=1,
            required_features=[],
            supports_short=False,
            supports_intraday=False
        )
        
        assert metadata.supports_short is False
        assert metadata.supports_intraday is False


class TestBaseStrategy:
    """Test BaseStrategy abstract class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data with features"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        df = pd.DataFrame({
            'open': 100 + np.random.randn(100).cumsum(),
            'high': 101 + np.random.randn(100).cumsum(),
            'low': 99 + np.random.randn(100).cumsum(),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000, 10000, 100),
            'atr': np.random.rand(100) * 2,
            'rsi': 30 + np.random.rand(100) * 40
        }, index=dates)
        
        # Ensure high > low
        df['high'] = df[['open', 'high', 'close']].max(axis=1) + 0.1
        df['low'] = df[['open', 'low', 'close']].min(axis=1) - 0.1
        
        return df
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        strategy = ConcreteTestStrategy()
        assert strategy.parameters == {}
        
        strategy = ConcreteTestStrategy(parameters={'fast': 5, 'slow': 20})
        assert strategy.parameters['fast'] == 5
        assert strategy.parameters['slow'] == 20
    
    def test_abstract_methods_required(self):
        """Test that abstract methods must be implemented"""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseStrategy()
    
    def test_generate_signals(self, sample_data):
        """Test signal generation"""
        strategy = ConcreteTestStrategy()
        signals = strategy.generate_signals(sample_data)
        
        # Check return type and shape
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert signals.index.equals(sample_data.index)
        
        # Check signal values
        assert all(s in [-1, 0, 1] for s in signals.unique())
    
    def test_calculate_positions(self, sample_data):
        """Test position calculation"""
        strategy = ConcreteTestStrategy()
        signals = strategy.generate_signals(sample_data)
        
        positions = strategy.calculate_positions(signals, capital=10000)
        
        # Check return type
        assert isinstance(positions, pd.Series)
        assert len(positions) == len(signals)
        
        # Check that positions align with signals
        assert all((p > 0 and s > 0) or (p < 0 and s < 0) or (p == 0 and s == 0)
                  for p, s in zip(positions, signals))
    
    def test_calculate_positions_with_risk_params(self, sample_data):
        """Test position calculation with risk parameters"""
        strategy = ConcreteTestStrategy()
        signals = strategy.generate_signals(sample_data)
        
        # Different risk fractions should produce different position sizes
        positions1 = strategy.calculate_positions(
            signals, capital=10000, risk_params={'risk_fraction': 0.05}
        )
        positions2 = strategy.calculate_positions(
            signals, capital=10000, risk_params={'risk_fraction': 0.10}
        )
        
        # Higher risk fraction should produce larger positions
        assert abs(positions2[signals != 0].mean()) > abs(positions1[signals != 0].mean())
    
    def test_required_properties(self):
        """Test required properties"""
        strategy = ConcreteTestStrategy()
        
        assert strategy.required_history == 20
        assert strategy.required_features == ['atr', 'rsi']
        assert isinstance(strategy.required_history, int)
        assert isinstance(strategy.required_features, list)
    
    def test_validate_data_basic(self, sample_data):
        """Test basic data validation"""
        strategy = ConcreteTestStrategy()
        
        # Valid data should pass
        assert strategy.validate_data(sample_data) is True
        
        # Missing required columns
        bad_data = sample_data.drop(columns=['close'])
        with pytest.raises(ValueError, match="Missing required columns"):
            strategy.validate_data(bad_data)
    
    def test_validate_data_features(self, sample_data):
        """Test feature validation"""
        strategy = ConcreteTestStrategy()
        
        # Missing required features
        bad_data = sample_data.drop(columns=['atr'])
        with pytest.raises(ValueError, match="Missing required features"):
            strategy.validate_data(bad_data)
    
    def test_validate_data_history(self, sample_data):
        """Test history validation"""
        strategy = ConcreteTestStrategy()
        
        # Insufficient history
        short_data = sample_data.iloc[:10]  # Only 10 bars
        with pytest.raises(ValueError, match="Insufficient data"):
            strategy.validate_data(short_data)
    
    def test_validate_data_types(self, sample_data):
        """Test data type validation"""
        strategy = ConcreteTestStrategy()
        
        # Non-numeric column
        bad_data = sample_data.copy()
        bad_data['close'] = bad_data['close'].astype(str)
        with pytest.raises(ValueError, match="must be numeric"):
            strategy.validate_data(bad_data)
    
    def test_validate_data_nans(self, sample_data):
        """Test NaN validation"""
        strategy = ConcreteTestStrategy()
        
        # NaN in critical column
        bad_data = sample_data.copy()
        bad_data.loc[bad_data.index[10], 'close'] = np.nan
        with pytest.raises(ValueError, match="NaN values found"):
            strategy.validate_data(bad_data)
        
        # NaN in non-critical column should be OK
        ok_data = sample_data.copy()
        ok_data.loc[ok_data.index[10], 'open'] = np.nan
        assert strategy.validate_data(ok_data) is True
    
    def test_get_info(self):
        """Test strategy info retrieval"""
        strategy = ConcreteTestStrategy(parameters={'fast': 10, 'slow': 20})
        info = strategy.get_info()
        
        assert info['name'] == "Test Strategy"
        assert info['version'] == "1.0"
        assert info['author'] == "Test"
        assert info['parameters'] == {'fast': 10, 'slow': 20}
        assert info['required_history'] == 20
        assert info['required_features'] == ['atr', 'rsi']
        assert info['supports_short'] is True
        assert info['supports_intraday'] is True
    
    def test_calculate_position_size_fixed(self):
        """Test fixed position sizing"""
        strategy = ConcreteTestStrategy()
        
        # Long position
        size = strategy.calculate_position_size_fixed(
            signal=1, capital=10000, price=50, risk_fraction=0.02
        )
        assert size == 4  # 10000 * 0.02 / 50 = 4
        
        # Short position
        size = strategy.calculate_position_size_fixed(
            signal=-1, capital=10000, price=50, risk_fraction=0.02
        )
        assert size == -4
        
        # No position
        size = strategy.calculate_position_size_fixed(
            signal=0, capital=10000, price=50, risk_fraction=0.02
        )
        assert size == 0
    
    def test_calculate_position_size_volatility(self):
        """Test volatility-based position sizing"""
        strategy = ConcreteTestStrategy()
        
        # Normal volatility
        size = strategy.calculate_position_size_volatility(
            signal=1, capital=10000, price=100, volatility=0.02, target_risk=0.02
        )
        expected = int(10000 * 0.02 / (100 * 0.02))  # 100 shares
        assert size == expected
        
        # Higher volatility should mean smaller position
        size_high_vol = strategy.calculate_position_size_volatility(
            signal=1, capital=10000, price=100, volatility=0.04, target_risk=0.02
        )
        assert size_high_vol < size
        
        # Zero volatility should return zero
        size_zero_vol = strategy.calculate_position_size_volatility(
            signal=1, capital=10000, price=100, volatility=0, target_risk=0.02
        )
        assert size_zero_vol == 0
    
    def test_parameter_validation(self):
        """Test custom parameter validation"""
        # Valid parameters
        strategy = MinimalTestStrategy(parameters={'threshold': 0.5})
        assert strategy.parameters['threshold'] == 0.5
        
        # Invalid parameters
        with pytest.raises(ValueError, match="Threshold must be between"):
            MinimalTestStrategy(parameters={'threshold': 1.5})
    
    def test_string_representations(self):
        """Test string representations"""
        strategy = ConcreteTestStrategy(parameters={'fast': 10})
        
        # __repr__
        repr_str = repr(strategy)
        assert "ConcreteTestStrategy" in repr_str
        assert "parameters={'fast': 10}" in repr_str
        
        # __str__
        str_str = str(strategy)
        assert str_str == "Test Strategy v1.0"


class TestBaseStrategyIntegration:
    """Integration tests for BaseStrategy"""
    
    @pytest.fixture
    def real_market_data(self):
        """Create realistic market data"""
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        
        # Generate realistic price series
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, 252)
        prices = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, 252)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 252))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 252))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, 252).astype(int),
            'atr': pd.Series(prices).rolling(14).std() * np.sqrt(252),
            'rsi': 50 + 30 * np.tanh(returns.cumsum() / 0.1)
        }, index=dates)
        
        # Fill NaN values from rolling calculations
        df['atr'] = df['atr'].bfill()
        
        return df
    
    def test_complete_strategy_workflow(self, real_market_data):
        """Test complete strategy workflow from data to positions"""
        strategy = ConcreteTestStrategy()
        
        # Validate data
        assert strategy.validate_data(real_market_data)
        
        # Generate signals
        signals = strategy.generate_signals(real_market_data)
        assert not signals.isna().any()
        
        # Calculate positions
        positions = strategy.calculate_positions(signals, capital=100000)
        assert len(positions) == len(signals)
        
        # Verify position changes align with signal changes
        signal_changes = signals.diff().fillna(0)
        position_changes = positions.diff().fillna(0)
        
        # When signal changes, position should change too
        mask = signal_changes != 0
        assert (position_changes[mask] != 0).all()
    
    def test_strategy_with_different_parameters(self, real_market_data):
        """Test that strategy behaves differently with different parameters"""
        strategy1 = ConcreteTestStrategy(parameters={'risk_fraction': 0.01})
        strategy2 = ConcreteTestStrategy(parameters={'risk_fraction': 0.05})
        
        signals = strategy1.generate_signals(real_market_data)
        
        positions1 = strategy1.calculate_positions(
            signals, capital=100000, risk_params={'risk_fraction': 0.01}
        )
        positions2 = strategy2.calculate_positions(
            signals, capital=100000, risk_params={'risk_fraction': 0.05}
        )
        
        # Different risk fractions should produce different position sizes
        assert not positions1.equals(positions2)
        assert abs(positions2[signals != 0]).mean() > abs(positions1[signals != 0]).mean()