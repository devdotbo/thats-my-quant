# Machine Learning and GPU Integration Guide

## Overview

This guide provides comprehensive instructions for integrating machine learning models and leveraging Apple Silicon GPU acceleration in the quantitative trading backtesting system.

## Apple Silicon Capabilities

### Hardware Specifications (M3 Max)
- **Neural Engine**: 16-core for ML inference
- **GPU**: 40-core GPU with 128GB unified memory access
- **Memory Bandwidth**: ~400 GB/s
- **Unified Memory**: Direct GPU access to full 128GB

### Framework Support Status

| Framework | Apple Silicon Support | Performance | Recommendation |
|-----------|---------------------|-------------|----------------|
| PyTorch + MPS | ✅ Good | Variable | Primary choice |
| TensorFlow + Metal | ✅ Good | Good | Alternative |
| JAX + Metal | ⚠️ Beta | Experimental | Research only |
| XGBoost | ✅ CPU only | Excellent | For tree models |
| LightGBM | ✅ CPU only | Excellent | For tree models |
| scikit-learn | ✅ CPU only | Good | For simple models |

## Setup and Configuration

### PyTorch with MPS (Metal Performance Shaders)

```bash
# Install PyTorch with MPS support
uv pip install torch torchvision torchaudio

# Verify MPS availability
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

```python
# src/ml/pytorch_config.py
import torch
import os

class PyTorchConfig:
    """Configuration for PyTorch on Apple Silicon"""
    
    @staticmethod
    def get_device() -> torch.device:
        """Get optimal device for PyTorch"""
        if torch.backends.mps.is_available():
            # Use MPS but with fallback for unsupported ops
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            return torch.device("mps")
        else:
            return torch.device("cpu")
            
    @staticmethod
    def optimize_memory() -> None:
        """Optimize memory usage on M3"""
        # Empty cache periodically
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        # Set memory fraction
        torch.mps.set_per_process_memory_fraction(0.8)  # Use 80% max
```

### TensorFlow with Metal Plugin

```bash
# Install TensorFlow with Metal acceleration
uv pip install tensorflow tensorflow-metal

# Verify GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

```python
# src/ml/tensorflow_config.py
import tensorflow as tf

class TensorFlowConfig:
    """Configuration for TensorFlow on Apple Silicon"""
    
    @staticmethod
    def configure_gpu() -> None:
        """Configure TensorFlow for Metal GPU"""
        # List GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                # Set memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                # Limit memory if needed
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=80000  # 80GB limit
                    )]
                )
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
```

## ML Strategy Development

### Base ML Strategy Class

```python
# src/strategies/ml/base_ml_strategy.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import torch

class BaseMLStrategy(BaseStrategy):
    """Base class for ML-based trading strategies"""
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 feature_config: Dict[str, Any],
                 device: Optional[str] = None):
        super().__init__(model_config)
        self.feature_config = feature_config
        self.device = self._setup_device(device)
        self.model = None
        self.scaler = None
        
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup computation device"""
        if device:
            return torch.device(device)
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
            
    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw data"""
        pass
        
    @abstractmethod
    def train_model(self, 
                   train_data: pd.DataFrame,
                   val_data: Optional[pd.DataFrame] = None) -> None:
        """Train the ML model"""
        pass
        
    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        pass
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals using ML model"""
        # Create features
        features = self.create_features(data)
        
        # Generate predictions
        predictions = self.predict(features)
        
        # Convert to signals
        signals = self.predictions_to_signals(predictions)
        
        return signals
        
    def predictions_to_signals(self, predictions: np.ndarray) -> pd.Series:
        """Convert model predictions to trading signals"""
        # Default: threshold at 0
        signals = pd.Series(0, index=range(len(predictions)))
        signals[predictions > 0.5] = 1
        signals[predictions < -0.5] = -1
        
        return signals
```

### Feature Engineering Pipeline

```python
# src/ml/features.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import talib

class FeatureEngineering:
    """Comprehensive feature engineering for ML strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_names = []
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Returns
        for period in [1, 5, 10, 20]:
            features[f'returns_{period}'] = df['close'].pct_change(period)
            features[f'log_returns_{period}'] = np.log(
                df['close'] / df['close'].shift(period)
            )
            
        # Moving averages
        for period in [10, 20, 50, 200]:
            features[f'ma_{period}'] = df['close'].rolling(period).mean()
            features[f'ma_ratio_{period}'] = df['close'] / features[f'ma_{period}']
            
        # Volatility
        for period in [10, 20, 50]:
            features[f'volatility_{period}'] = (
                df['close'].pct_change().rolling(period).std()
            )
            
        # Price position
        for period in [20, 50]:
            rolling_high = df['high'].rolling(period).max()
            rolling_low = df['low'].rolling(period).min()
            features[f'price_position_{period}'] = (
                (df['close'] - rolling_low) / (rolling_high - rolling_low)
            )
            
        return features
        
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Volume metrics
        features['volume_ma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma_20']
        
        # Dollar volume
        features['dollar_volume'] = df['close'] * df['volume']
        features['dollar_volume_ma'] = features['dollar_volume'].rolling(20).mean()
        
        # Volume-price correlation
        features['volume_price_corr'] = (
            df['close'].pct_change()
            .rolling(20)
            .corr(df['volume'].pct_change())
        )
        
        return features
        
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators using TA-Lib"""
        features = pd.DataFrame(index=df.index)
        
        # RSI
        features['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
        
        # MACD
        macd, signal, hist = talib.MACD(
            df['close'].values,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['close'].values,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2
        )
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = (upper - lower) / middle
        features['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        # ATR
        features['atr_14'] = talib.ATR(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            timeperiod=14
        )
        
        return features
        
    def create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        features = pd.DataFrame(index=df.index)
        
        # Spread metrics (if available)
        if 'bid' in df.columns and 'ask' in df.columns:
            features['spread'] = df['ask'] - df['bid']
            features['spread_pct'] = features['spread'] / df['mid_price']
            features['spread_ma'] = features['spread'].rolling(20).mean()
            
        # High-low range
        features['hl_range'] = df['high'] - df['low']
        features['hl_range_pct'] = features['hl_range'] / df['close']
        
        # Close position in range
        features['close_position'] = (
            (df['close'] - df['low']) / (df['high'] - df['low'])
        )
        
        return features
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features"""
        features = pd.concat([
            self.create_price_features(df),
            self.create_volume_features(df),
            self.create_technical_indicators(df),
            self.create_microstructure_features(df)
        ], axis=1)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        return features
```

### PyTorch LSTM Strategy Example

```python
# src/strategies/ml/lstm_strategy.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take last output
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

class LSTMStrategy(BaseMLStrategy):
    """LSTM-based trading strategy"""
    
    def __init__(self, 
                 lookback: int = 30,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 50):
        
        model_config = {
            'lookback': lookback,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs
        }
        
        feature_config = {
            'price_features': True,
            'volume_features': True,
            'technical_indicators': True
        }
        
        super().__init__(model_config, feature_config)
        
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineering(feature_config)
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for LSTM"""
        return self.feature_engineer.create_all_features(data)
        
    def prepare_sequences(self, 
                         features: pd.DataFrame,
                         labels: Optional[pd.Series] = None) -> Tuple:
        """Prepare sequences for LSTM"""
        # Scale features
        scaled_features = self.scaler.fit_transform(features.fillna(0))
        
        # Create sequences
        X, y = [], []
        lookback = self.parameters['lookback']
        
        for i in range(lookback, len(scaled_features)):
            X.append(scaled_features[i-lookback:i])
            if labels is not None:
                y.append(labels.iloc[i])
                
        X = np.array(X)
        y = np.array(y) if labels is not None else None
        
        return X, y
        
    def train_model(self, 
                   train_data: pd.DataFrame,
                   val_data: Optional[pd.DataFrame] = None) -> None:
        """Train LSTM model"""
        # Create features
        train_features = self.create_features(train_data)
        
        # Create labels (next day returns)
        train_labels = train_data['close'].pct_change().shift(-1)
        train_labels = np.sign(train_labels)  # Convert to signals
        
        # Prepare sequences
        X_train, y_train = self.prepare_sequences(train_features, train_labels)
        
        # Create model
        input_size = X_train.shape[2]
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.parameters['hidden_size'],
            num_layers=self.parameters['num_layers']
        ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.parameters['learning_rate']
        )
        criterion = nn.MSELoss()
        
        # Create DataLoader
        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        loader = DataLoader(
            dataset,
            batch_size=self.parameters['batch_size'],
            shuffle=True
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.parameters['epochs']):
            total_loss = 0
            
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")
                
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions using LSTM"""
        # Prepare sequences
        X, _ = self.prepare_sequences(features)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Generate predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().squeeze()
            
        # Pad predictions to match original length
        padded_predictions = np.zeros(len(features))
        padded_predictions[self.parameters['lookback']:] = predictions
        
        return padded_predictions
```

### XGBoost Strategy Example

```python
# src/strategies/ml/xgboost_strategy.py
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class XGBoostStrategy(BaseMLStrategy):
    """XGBoost-based trading strategy"""
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 5,
                 learning_rate: float = 0.1,
                 feature_lookback: int = 20):
        
        model_config = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'feature_lookback': feature_lookback
        }
        
        super().__init__(model_config, {})
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for XGBoost"""
        features = pd.DataFrame(index=data.index)
        
        # Lagged returns
        for i in range(1, self.parameters['feature_lookback'] + 1):
            features[f'return_lag_{i}'] = data['close'].pct_change(i)
            
        # Rolling statistics
        for window in [5, 10, 20]:
            returns = data['close'].pct_change()
            features[f'mean_return_{window}'] = returns.rolling(window).mean()
            features[f'std_return_{window}'] = returns.rolling(window).std()
            features[f'skew_return_{window}'] = returns.rolling(window).skew()
            
        # Technical indicators
        features['rsi'] = self.calculate_rsi(data['close'])
        features['macd'] = self.calculate_macd(data['close'])
        
        # Volume features
        features['volume_ratio'] = (
            data['volume'] / data['volume'].rolling(20).mean()
        )
        
        return features
        
    def train_model(self,
                   train_data: pd.DataFrame,
                   val_data: Optional[pd.DataFrame] = None) -> None:
        """Train XGBoost model"""
        # Create features and labels
        features = self.create_features(train_data).dropna()
        labels = np.sign(train_data['close'].pct_change().shift(-1))
        labels = labels.loc[features.index]
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Find best parameters using CV
        best_score = -np.inf
        best_params = None
        
        for train_idx, val_idx in tscv.split(features):
            X_train = features.iloc[train_idx]
            y_train = labels.iloc[train_idx]
            X_val = features.iloc[val_idx]
            y_val = labels.iloc[val_idx]
            
            # Train model
            model = xgb.XGBClassifier(
                n_estimators=self.parameters['n_estimators'],
                max_depth=self.parameters['max_depth'],
                learning_rate=self.parameters['learning_rate'],
                tree_method='hist',  # Fast histogram method
                n_jobs=-1  # Use all CPU cores
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            score = model.score(X_val, y_val)
            if score > best_score:
                best_score = score
                self.model = model
                
        print(f"Best validation score: {best_score:.4f}")
        
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions using XGBoost"""
        # Handle missing values
        features_clean = features.fillna(0)
        
        # Get probability predictions
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(features_clean)
            # Convert to directional signal
            predictions = probas[:, 2] - probas[:, 0]  # Long prob - Short prob
        else:
            predictions = self.model.predict(features_clean)
            
        return predictions
```

## Performance Optimization

### GPU Utilization Monitoring

```python
# src/ml/gpu_monitor.py
import torch
import subprocess
import psutil

class GPUMonitor:
    """Monitor GPU usage on Apple Silicon"""
    
    @staticmethod
    def get_mps_memory_usage() -> Dict[str, float]:
        """Get MPS memory usage"""
        if torch.backends.mps.is_available():
            # Get current allocation
            current = torch.mps.current_allocated_memory() / 1024**3
            
            # Get driver allocation
            driver = torch.mps.driver_allocated_memory() / 1024**3
            
            return {
                'current_allocated_gb': current,
                'driver_allocated_gb': driver,
                'available_gb': 128 - driver  # M3 Max has 128GB
            }
        return {}
        
    @staticmethod
    def optimize_batch_size(model: nn.Module,
                          input_shape: Tuple,
                          max_batch_size: int = 1024) -> int:
        """Find optimal batch size for model"""
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
        
        # Binary search for optimal batch size
        low, high = 1, max_batch_size
        optimal = 1
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Try forward pass
                dummy_input = torch.randn(mid, *input_shape).to(device)
                with torch.no_grad():
                    _ = model(dummy_input)
                    
                # If successful, try larger
                optimal = mid
                low = mid + 1
                
                # Clear cache
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    high = mid - 1
                else:
                    raise e
                    
        return optimal
```

### Performance Benchmarking

```python
# benchmarks/ml_benchmarks.py
def benchmark_ml_strategies() -> pd.DataFrame:
    """Benchmark ML strategy performance"""
    results = []
    
    # Test data
    data = generate_test_data(n_symbols=10, n_periods=10000)
    
    strategies = {
        'LSTM': LSTMStrategy(epochs=10),
        'XGBoost': XGBoostStrategy(n_estimators=100),
        'RandomForest': RandomForestStrategy(n_estimators=100)
    }
    
    for name, strategy in strategies.items():
        # Training time
        train_start = time.time()
        strategy.train_model(data.iloc[:8000])
        train_time = time.time() - train_start
        
        # Inference time
        inference_start = time.time()
        signals = strategy.generate_signals(data.iloc[8000:])
        inference_time = time.time() - inference_start
        
        # Memory usage
        if torch.backends.mps.is_available():
            memory = GPUMonitor.get_mps_memory_usage()
        else:
            memory = {'current_allocated_gb': psutil.Process().memory_info().rss / 1024**3}
            
        results.append({
            'strategy': name,
            'train_time': train_time,
            'inference_time': inference_time,
            'signals_per_second': len(signals) / inference_time,
            'memory_gb': memory['current_allocated_gb']
        })
        
    return pd.DataFrame(results)
```

## Best Practices

### 1. Start Simple
- Begin with CPU-based models (XGBoost, RandomForest)
- Move to GPU only if demonstrable speedup
- Profile before optimizing

### 2. Data Pipeline Optimization
```python
# Efficient data loading
def create_data_loader(data: pd.DataFrame, 
                      batch_size: int,
                      num_workers: int = 4) -> DataLoader:
    """Create efficient data loader"""
    # Convert to tensors once
    tensor_data = torch.FloatTensor(data.values)
    
    # Use memory pinning for faster GPU transfer
    return DataLoader(
        TensorDataset(tensor_data),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.backends.mps.is_available() else False
    )
```

### 3. Memory Management
```python
# Regular cleanup
def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
```

### 4. Fallback Strategy
```python
# Graceful degradation
try:
    model = model.to("mps")
    results = model(data)
except Exception as e:
    print(f"MPS failed: {e}, falling back to CPU")
    model = model.to("cpu")
    results = model(data)
```

### 5. Monitoring and Alerts
- Track GPU memory usage
- Monitor training/inference times
- Alert on performance degradation
- Compare GPU vs CPU regularly

## Troubleshooting

### Common Issues

1. **MPS Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Clear cache regularly

2. **Unsupported Operations**
   - Set PYTORCH_ENABLE_MPS_FALLBACK=1
   - Check PyTorch MPS documentation
   - Consider CPU fallback

3. **Slow Performance**
   - Profile with PyTorch profiler
   - Check data transfer overhead
   - Verify operations are on GPU

4. **Numerical Instability**
   - Use mixed precision carefully
   - Validate results against CPU
   - Check for NaN/Inf values