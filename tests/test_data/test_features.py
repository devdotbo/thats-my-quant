"""
Tests for feature engineering module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.features import FeatureEngine


class TestFeatureEngine:
    """Test feature engineering functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range('2024-01-02 09:30:00', periods=390, freq='1min', tz='America/New_York')
        
        # Generate realistic price data
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(390) * 0.1)
        
        data = pd.DataFrame({
            'open': close_prices + np.random.randn(390) * 0.05,
            'high': close_prices + np.abs(np.random.randn(390) * 0.1),
            'low': close_prices - np.abs(np.random.randn(390) * 0.1),
            'close': close_prices,
            'volume': np.random.randint(10000, 100000, 390)
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    @pytest.fixture
    def feature_engine(self):
        """Create FeatureEngine instance"""
        return FeatureEngine()
    
    def test_add_moving_averages(self, feature_engine, sample_data):
        """Test moving average calculations"""
        # Add moving averages
        result = feature_engine.add_moving_averages(
            sample_data, 
            periods=[20, 50],
            ma_types=['sma', 'ema']
        )
        
        # Check columns added
        assert 'sma_20' in result.columns
        assert 'sma_50' in result.columns
        assert 'ema_20' in result.columns
        assert 'ema_50' in result.columns
        
        # Verify SMA calculation
        manual_sma_20 = sample_data['close'].rolling(20).mean()
        pd.testing.assert_series_equal(
            result['sma_20'].dropna(),
            manual_sma_20.dropna(),
            check_names=False
        )
        
        # Check no NaN after warmup period
        assert result['sma_20'].iloc[20:].notna().all()
        assert result['sma_50'].iloc[50:].notna().all()
    
    def test_add_rsi(self, feature_engine, sample_data):
        """Test RSI calculation"""
        result = feature_engine.add_momentum_indicators(
            sample_data,
            indicators=['rsi'],
            rsi_period=14
        )
        
        assert 'rsi_14' in result.columns
        
        # RSI should be between 0 and 100
        rsi_values = result['rsi_14'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
        
        # Check extreme values
        # If all price changes are positive, RSI should be high
        # If all price changes are negative, RSI should be low
    
    def test_add_macd(self, feature_engine, sample_data):
        """Test MACD calculation"""
        result = feature_engine.add_momentum_indicators(
            sample_data,
            indicators=['macd'],
            macd_fast=12,
            macd_slow=26,
            macd_signal=9
        )
        
        # Check all MACD components added
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_histogram' in result.columns
        
        # Verify MACD calculation
        ema_12 = sample_data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = sample_data['close'].ewm(span=26, adjust=False).mean()
        expected_macd = ema_12 - ema_26
        
        pd.testing.assert_series_equal(
            result['macd'].dropna(),
            expected_macd.dropna(),
            check_names=False,
            rtol=1e-5
        )
    
    def test_add_bollinger_bands(self, feature_engine, sample_data):
        """Test Bollinger Bands calculation"""
        result = feature_engine.add_volatility_indicators(
            sample_data,
            indicators=['bollinger'],
            bb_period=20,
            bb_std=2
        )
        
        # Check all BB components
        assert 'bb_upper' in result.columns
        assert 'bb_middle' in result.columns
        assert 'bb_lower' in result.columns
        assert 'bb_width' in result.columns
        assert 'bb_percent' in result.columns
        
        # Verify calculations
        sma = sample_data['close'].rolling(20).mean()
        std = sample_data['close'].rolling(20).std()
        
        expected_upper = sma + (2 * std)
        expected_lower = sma - (2 * std)
        
        pd.testing.assert_series_equal(
            result['bb_upper'].dropna(),
            expected_upper.dropna(),
            check_names=False,
            rtol=1e-5
        )
    
    def test_add_atr(self, feature_engine, sample_data):
        """Test Average True Range calculation"""
        result = feature_engine.add_volatility_indicators(
            sample_data,
            indicators=['atr'],
            atr_period=14
        )
        
        assert 'atr_14' in result.columns
        
        # ATR should always be positive
        atr_values = result['atr_14'].dropna()
        assert (atr_values > 0).all()
        
        # ATR should be reasonable relative to price
        avg_price = sample_data['close'].mean()
        avg_atr = atr_values.mean()
        assert avg_atr < avg_price * 0.1  # ATR typically < 10% of price
    
    def test_add_volume_analytics(self, feature_engine, sample_data):
        """Test volume-based indicators"""
        result = feature_engine.add_volume_analytics(
            sample_data,
            indicators=['vwap', 'obv'],
            vwap_period=None  # Session VWAP
        )
        
        # Check VWAP
        assert 'vwap' in result.columns
        vwap_values = result['vwap'].dropna()
        
        # VWAP should be within price range
        assert (vwap_values >= sample_data['low'].min()).all()
        assert (vwap_values <= sample_data['high'].max()).all()
        
        # Check OBV
        assert 'obv' in result.columns
        
        # OBV changes should match volume direction
        price_changes = sample_data['close'].diff()
        obv_changes = result['obv'].diff()
        
        # When price goes up, OBV should increase by volume
        # When price goes down, OBV should decrease by volume
    
    def test_add_microstructure_features(self, feature_engine, sample_data):
        """Test market microstructure features"""
        result = feature_engine.add_microstructure_features(
            sample_data,
            features=['spread_proxy', 'high_low_ratio']
        )
        
        # Check spread proxy (high-low spread)
        assert 'spread_proxy' in result.columns
        spread = result['spread_proxy']
        assert (spread >= 0).all()
        
        # Check high-low ratio
        assert 'high_low_ratio' in result.columns
        ratio = result['high_low_ratio']
        assert (ratio >= 1).all()  # High should always be >= Low
    
    def test_add_rolling_statistics(self, feature_engine, sample_data):
        """Test rolling statistics calculations"""
        result = feature_engine.add_rolling_statistics(
            sample_data,
            windows=[20, 50],
            stats=['volatility', 'skew', 'kurtosis']
        )
        
        # Check volatility
        assert 'volatility_20' in result.columns
        assert 'volatility_50' in result.columns
        
        # Volatility should be positive
        vol_20 = result['volatility_20'].dropna()
        assert (vol_20 > 0).all()
        
        # Check skew and kurtosis
        assert 'skew_20' in result.columns
        assert 'kurtosis_20' in result.columns
    
    def test_add_all_features(self, feature_engine, sample_data):
        """Test convenience method for adding all common features"""
        result = feature_engine.add_all_features(sample_data)
        
        # Should have many more columns than original
        assert len(result.columns) > len(sample_data.columns) + 20
        
        # Check some key features are present
        expected_features = [
            'sma_20', 'ema_50', 'rsi_14', 'macd', 
            'bb_upper', 'atr_14', 'vwap', 'volatility_20'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
        
        # No infinite or extreme values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not result[col].isin([np.inf, -np.inf]).any()
    
    def test_feature_caching(self, feature_engine, sample_data, tmp_path):
        """Test caching of computed features"""
        # Set cache directory
        cache_dir = tmp_path / "feature_cache"
        feature_engine_with_cache = FeatureEngine(cache_dir=cache_dir)
        
        # Add features
        result1 = feature_engine_with_cache.add_all_features(
            sample_data,
            cache_key="test_features"
        )
        
        # Load from cache
        result2 = feature_engine_with_cache.load_cached_features("test_features")
        
        # Should be identical (check_freq=False to ignore timezone metadata differences)
        pd.testing.assert_frame_equal(result1, result2, check_freq=False)
    
    def test_handle_missing_data(self, feature_engine):
        """Test handling of missing data in input"""
        # Create data with gaps
        dates = pd.date_range('2024-01-02 09:30:00', periods=100, freq='1min', tz='America/New_York')
        data = pd.DataFrame({
            'open': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'close': [100] * 100,
            'volume': [50000] * 100
        }, index=dates)
        
        # Introduce gaps
        data.iloc[20:25] = np.nan
        
        # Should handle gracefully
        result = feature_engine.add_moving_averages(data, periods=[10])
        
        # Should propagate NaN appropriately
        # After 5 NaN values (20-24), the next 10 values will also be NaN due to 10-period window
        assert result['sma_10'].iloc[20:30].isna().all()  # NaN and affected by NaN
        assert result['sma_10'].iloc[34:].notna().all()   # Should recover after gap + window


@pytest.mark.performance
class TestFeatureEnginePerformance:
    """Test performance of feature engineering"""
    
    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing"""
        # One year of minute data
        dates = pd.date_range('2024-01-02 09:30:00', periods=98280, freq='1min', tz='America/New_York')
        
        # Filter to market hours only
        dates = dates[
            (dates.time >= pd.Timestamp('09:30').time()) & 
            (dates.time <= pd.Timestamp('16:00').time()) &
            (dates.weekday < 5)
        ]
        
        # Generate data
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
        
        return pd.DataFrame({
            'open': close_prices + np.random.randn(len(dates)) * 0.05,
            'high': close_prices + np.abs(np.random.randn(len(dates)) * 0.1),
            'low': close_prices - np.abs(np.random.randn(len(dates)) * 0.1),
            'close': close_prices,
            'volume': np.random.randint(10000, 100000, len(dates))
        }, index=dates)
    
    def test_performance_all_features(self, large_dataset):
        """Test that all features can be computed in <2 seconds"""
        import time
        
        feature_engine = FeatureEngine()
        
        start_time = time.perf_counter()
        result = feature_engine.add_all_features(large_dataset)
        end_time = time.perf_counter()
        
        processing_time = end_time - start_time
        
        print(f"\nFeature Engineering Performance:")
        print(f"- Processed {len(large_dataset)} bars")
        print(f"- Added {len(result.columns) - len(large_dataset.columns)} features")
        print(f"- Time: {processing_time:.2f} seconds")
        print(f"- Rate: {len(large_dataset) / processing_time:.0f} bars/second")
        
        # Should complete in under 2 seconds
        assert processing_time < 2.0, f"Feature engineering took {processing_time:.2f}s, target is <2s"