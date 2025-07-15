# Strategy Development Guide

## Overview

This directory contains trading strategy implementations and serves as a guide for developing robust, testable trading strategies. All strategies follow a consistent interface to ensure compatibility with our backtesting framework.

## Strategy Architecture

### Base Strategy Interface

All strategies must inherit from the `BaseStrategy` abstract class:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.validate_parameters()
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on market data
        
        Returns:
            pd.Series: Signal values (-1: short, 0: neutral, 1: long)
        """
        pass
        
    @abstractmethod
    def calculate_positions(self, 
                          signals: pd.Series,
                          capital: float,
                          risk_params: Optional[Dict] = None) -> pd.Series:
        """
        Convert signals to actual position sizes
        
        Returns:
            pd.Series: Position sizes in shares/contracts
        """
        pass
        
    @property
    @abstractmethod
    def required_history(self) -> int:
        """Minimum number of bars required before generating signals"""
        pass
        
    @property
    @abstractmethod
    def required_features(self) -> List[str]:
        """List of required data columns"""
        pass
        
    def validate_parameters(self) -> None:
        """Validate strategy parameters"""
        pass
        
    def get_info(self) -> Dict[str, Any]:
        """Return strategy information"""
        return {
            'name': self.__class__.__name__,
            'parameters': self.parameters,
            'required_history': self.required_history,
            'required_features': self.required_features
        }
```

## Strategy Categories

### 1. Trend Following Strategies

#### Moving Average Crossover
```python
class MovingAverageCrossover(BaseStrategy):
    """Classic MA crossover strategy"""
    
    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period
        }
        super().__init__(parameters)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on MA crossover"""
        fast_ma = data['close'].rolling(self.parameters['fast_period']).mean()
        slow_ma = data['close'].rolling(self.parameters['slow_period']).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        
        return signals
        
    @property
    def required_history(self) -> int:
        return self.parameters['slow_period'] + 1
        
    @property
    def required_features(self) -> List[str]:
        return ['close']
```

#### Momentum Strategy
```python
class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""
    
    def __init__(self, lookback: int = 20, holding_period: int = 5):
        parameters = {
            'lookback': lookback,
            'holding_period': holding_period
        }
        super().__init__(parameters)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on momentum"""
        returns = data['close'].pct_change(self.parameters['lookback'])
        
        # Rank momentum
        ranked = returns.rank(pct=True)
        
        signals = pd.Series(0, index=data.index)
        signals[ranked > 0.8] = 1  # Long top 20%
        signals[ranked < 0.2] = -1  # Short bottom 20%
        
        return signals
```

### 2. Mean Reversion Strategies

#### Bollinger Bands Strategy
```python
class BollingerBandsStrategy(BaseStrategy):
    """Mean reversion using Bollinger Bands"""
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        parameters = {
            'period': period,
            'num_std': num_std
        }
        super().__init__(parameters)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on Bollinger Bands"""
        close = data['close']
        
        # Calculate bands
        sma = close.rolling(self.parameters['period']).mean()
        std = close.rolling(self.parameters['period']).std()
        upper_band = sma + self.parameters['num_std'] * std
        lower_band = sma - self.parameters['num_std'] * std
        
        signals = pd.Series(0, index=data.index)
        signals[close < lower_band] = 1   # Buy when oversold
        signals[close > upper_band] = -1  # Sell when overbought
        
        return signals
```

### 3. Market Microstructure Strategies

#### Order Flow Imbalance
```python
class OrderFlowImbalance(BaseStrategy):
    """Strategy based on order flow imbalance"""
    
    def __init__(self, lookback: int = 10, threshold: float = 0.6):
        parameters = {
            'lookback': lookback,
            'threshold': threshold
        }
        super().__init__(parameters)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on order flow"""
        # Calculate order flow imbalance
        bid_volume = data['bid_volume'].rolling(self.parameters['lookback']).sum()
        ask_volume = data['ask_volume'].rolling(self.parameters['lookback']).sum()
        
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        signals = pd.Series(0, index=data.index)
        signals[imbalance > self.parameters['threshold']] = 1
        signals[imbalance < -self.parameters['threshold']] = -1
        
        return signals
        
    @property
    def required_features(self) -> List[str]:
        return ['close', 'bid_volume', 'ask_volume']
```

### 4. Machine Learning Strategies

#### ML Strategy Template
```python
class MLStrategy(BaseStrategy):
    """Template for ML-based strategies"""
    
    def __init__(self, model_path: str, features: List[str], lookback: int = 30):
        parameters = {
            'model_path': model_path,
            'features': features,
            'lookback': lookback
        }
        super().__init__(parameters)
        self.model = self.load_model()
        
    def load_model(self):
        """Load pre-trained model"""
        import joblib
        return joblib.load(self.parameters['model_path'])
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model prediction"""
        features = pd.DataFrame()
        
        for feature in self.parameters['features']:
            if feature == 'returns':
                features['returns'] = data['close'].pct_change()
            elif feature == 'volume_ratio':
                features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            # Add more feature engineering
            
        return features
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals using ML model"""
        features = self.prepare_features(data)
        
        # Get predictions
        predictions = pd.Series(
            self.model.predict(features.dropna()),
            index=features.dropna().index
        )
        
        return predictions
```

## Position Sizing Methods

### Fixed Fractional
```python
def fixed_fractional_sizing(signals: pd.Series, 
                           capital: float,
                           fraction: float = 0.02) -> pd.Series:
    """Allocate fixed fraction of capital per trade"""
    position_value = capital * fraction
    return signals * position_value
```

### Kelly Criterion
```python
def kelly_sizing(signals: pd.Series,
                win_rate: float,
                avg_win: float,
                avg_loss: float,
                capital: float,
                kelly_fraction: float = 0.25) -> pd.Series:
    """Kelly criterion position sizing"""
    # Calculate Kelly percentage
    kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    
    # Apply Kelly fraction (never use full Kelly)
    position_pct = kelly_pct * kelly_fraction
    
    # Ensure reasonable bounds
    position_pct = np.clip(position_pct, 0.01, 0.1)
    
    return signals * capital * position_pct
```

### Volatility-Based Sizing
```python
def volatility_sizing(signals: pd.Series,
                     data: pd.DataFrame,
                     capital: float,
                     target_vol: float = 0.02) -> pd.Series:
    """Size positions based on volatility"""
    # Calculate rolling volatility
    returns = data['close'].pct_change()
    volatility = returns.rolling(20).std()
    
    # Calculate position size
    position_pct = target_vol / volatility
    position_pct = position_pct.clip(0.01, 0.1)
    
    return signals * capital * position_pct
```

## Risk Management Integration

### Stop Loss Implementation
```python
class StopLossManager:
    """Manage stop losses for positions"""
    
    def __init__(self, stop_pct: float = 0.02):
        self.stop_pct = stop_pct
        self.stop_prices = {}
        
    def update_stops(self, positions: Dict[str, Position], 
                    current_prices: Dict[str, float]):
        """Update stop loss levels"""
        for symbol, position in positions.items():
            if symbol not in self.stop_prices:
                # Set initial stop
                if position.side == 'long':
                    self.stop_prices[symbol] = position.entry_price * (1 - self.stop_pct)
                else:
                    self.stop_prices[symbol] = position.entry_price * (1 + self.stop_pct)
                    
    def check_stops(self, positions: Dict[str, Position],
                   current_prices: Dict[str, float]) -> List[str]:
        """Check if any stops are triggered"""
        triggered = []
        
        for symbol, position in positions.items():
            price = current_prices[symbol]
            stop = self.stop_prices.get(symbol)
            
            if position.side == 'long' and price <= stop:
                triggered.append(symbol)
            elif position.side == 'short' and price >= stop:
                triggered.append(symbol)
                
        return triggered
```

## Strategy Development Workflow

### 1. Research Phase
```python
# Use Jupyter notebook for exploration
# File: notebooks/strategy_research.ipynb

# Load data
data = load_market_data('SPY', '2020-01-01', '2023-12-31')

# Test simple idea
fast_ma = data['close'].rolling(10).mean()
slow_ma = data['close'].rolling(30).mean()

# Visualize
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['close'], label='Price')
plt.plot(data.index, fast_ma, label='Fast MA')
plt.plot(data.index, slow_ma, label='Slow MA')
plt.legend()
plt.show()

# Quick backtest
signals = (fast_ma > slow_ma).astype(int) - (fast_ma < slow_ma).astype(int)
returns = data['close'].pct_change() * signals.shift(1)
cumulative_returns = (1 + returns).cumprod()
```

### 2. Implementation Phase
```python
# Convert to strategy class
# File: strategies/ma_crossover.py

class MACrossoverStrategy(BaseStrategy):
    # Implementation as shown above
    pass
```

### 3. Testing Phase
```python
# File: tests/test_ma_crossover.py

def test_ma_crossover_signals():
    """Test signal generation"""
    strategy = MACrossoverStrategy(fast_period=10, slow_period=30)
    
    # Create test data
    test_data = create_test_data()
    
    # Generate signals
    signals = strategy.generate_signals(test_data)
    
    # Assertions
    assert len(signals) == len(test_data)
    assert signals.isin([-1, 0, 1]).all()
```

### 4. Validation Phase
```python
# Run comprehensive validation
validator = StrategyValidator(strategy)

# Walk-forward optimization
wfo_results = validator.walk_forward_optimization(
    data, 
    param_grid={'fast_period': [10, 20, 50], 
                'slow_period': [50, 100, 200]}
)

# Monte Carlo simulation
mc_results = validator.monte_carlo_simulation(
    strategy_results,
    n_simulations=1000
)

# Generate report
validator.generate_report('reports/ma_crossover_validation.html')
```

## Integration with Research Paper

Based on the paper "Can Day Trading Really Be Profitable", we can implement the entry/exit strategies described:

```python
class DayTradingStrategy(BaseStrategy):
    """Implementation of day trading strategy from research paper"""
    
    def __init__(self, 
                 entry_threshold: float = 0.02,
                 exit_threshold: float = 0.01,
                 stop_loss: float = 0.015):
        parameters = {
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'stop_loss': stop_loss
        }
        super().__init__(parameters)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on paper's methodology"""
        # Implement specific entry/exit logic from paper
        # This is a placeholder - actual implementation would follow paper
        
        signals = pd.Series(0, index=data.index)
        
        # Long entry conditions
        breakout = data['high'] > data['high'].rolling(20).max().shift(1)
        volume_surge = data['volume'] > data['volume'].rolling(20).mean() * 1.5
        
        signals[breakout & volume_surge] = 1
        
        return signals
```

## Best Practices

1. **Start Simple**: Begin with basic strategies before adding complexity
2. **Document Everything**: Clear documentation of strategy logic
3. **Test Thoroughly**: Unit tests for all strategy components
4. **Validate Robustness**: Use walk-forward and Monte Carlo methods
5. **Monitor Performance**: Track live vs backtest performance
6. **Version Control**: Track all parameter changes
7. **Risk First**: Always implement risk management

## Strategy Checklist

Before deploying any strategy:

- [ ] Unit tests pass
- [ ] Walk-forward optimization completed
- [ ] Monte Carlo simulation shows robustness
- [ ] Transaction costs included
- [ ] Risk limits defined
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] Paper trading tested
- [ ] Live monitoring setup
- [ ] Drawdown limits set

## Directory Structure

```
strategies/
├── README.md                 # This file
├── base.py                   # Base strategy class
├── trend_following/
│   ├── __init__.py
│   ├── ma_crossover.py
│   └── momentum.py
├── mean_reversion/
│   ├── __init__.py
│   ├── bollinger_bands.py
│   └── pairs_trading.py
├── microstructure/
│   ├── __init__.py
│   └── order_flow.py
├── ml_strategies/
│   ├── __init__.py
│   └── rf_classifier.py
└── utils/
    ├── __init__.py
    ├── position_sizing.py
    └── risk_management.py
```