"""
Feature Engineering Module
Adds technical indicators and market microstructure features to market data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

from src.utils.logging import get_logger
from src.data.cache import CacheManager
import ephem  # Astronomical calculations for moon phases


class FeatureEngine:
    """
    Adds technical indicators and features to market data
    
    Features:
    - Moving averages (SMA, EMA)
    - Momentum indicators (RSI, MACD, ROC)
    - Volatility indicators (ATR, Bollinger Bands)
    - Volume analytics (VWAP, OBV)
    - Market microstructure features
    - Rolling statistics
    """
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the feature engine
        
        Args:
            cache_dir: Optional directory for caching computed features
        """
        self.logger = get_logger("features")
        
        # Initialize cache if directory provided
        self.cache_manager = None
        if cache_dir:
            self.cache_manager = CacheManager(cache_dir=Path(cache_dir))
    
    def add_moving_averages(self, 
                          df: pd.DataFrame,
                          periods: List[int] = [20, 50, 200],
                          ma_types: List[str] = ['sma', 'ema'],
                          price_col: str = 'close') -> pd.DataFrame:
        """
        Add moving averages to the dataframe
        
        Args:
            df: Input dataframe with OHLCV data
            periods: List of periods for moving averages
            ma_types: Types of moving averages ('sma', 'ema')
            price_col: Column to calculate MA on
            
        Returns:
            DataFrame with added moving average columns
        """
        result = df.copy()
        
        for period in periods:
            if 'sma' in ma_types:
                col_name = f'sma_{period}'
                result[col_name] = result[price_col].rolling(window=period).mean()
                self.logger.debug(f"Added {col_name}")
            
            if 'ema' in ma_types:
                col_name = f'ema_{period}'
                result[col_name] = result[price_col].ewm(span=period, adjust=False).mean()
                self.logger.debug(f"Added {col_name}")
        
        return result
    
    def add_momentum_indicators(self,
                              df: pd.DataFrame,
                              indicators: List[str] = ['rsi', 'macd', 'roc'],
                              rsi_period: int = 14,
                              macd_fast: int = 12,
                              macd_slow: int = 26,
                              macd_signal: int = 9,
                              roc_period: int = 10) -> pd.DataFrame:
        """
        Add momentum indicators to the dataframe
        
        Args:
            df: Input dataframe with OHLCV data
            indicators: List of indicators to add
            rsi_period: Period for RSI calculation
            macd_fast: Fast EMA period for MACD
            macd_slow: Slow EMA period for MACD
            macd_signal: Signal line EMA period for MACD
            roc_period: Period for Rate of Change
            
        Returns:
            DataFrame with added momentum indicators
        """
        result = df.copy()
        
        if 'rsi' in indicators:
            result[f'rsi_{rsi_period}'] = self._calculate_rsi(result['close'], rsi_period)
            self.logger.debug(f"Added RSI({rsi_period})")
        
        if 'macd' in indicators:
            macd, signal, histogram = self._calculate_macd(
                result['close'], macd_fast, macd_slow, macd_signal
            )
            result['macd'] = macd
            result['macd_signal'] = signal
            result['macd_histogram'] = histogram
            self.logger.debug(f"Added MACD({macd_fast},{macd_slow},{macd_signal})")
        
        if 'roc' in indicators:
            result[f'roc_{roc_period}'] = self._calculate_roc(result['close'], roc_period)
            self.logger.debug(f"Added ROC({roc_period})")
        
        return result
    
    def add_volatility_indicators(self,
                                df: pd.DataFrame,
                                indicators: List[str] = ['atr', 'bollinger'],
                                atr_period: int = 14,
                                bb_period: int = 20,
                                bb_std: float = 2.0) -> pd.DataFrame:
        """
        Add volatility indicators to the dataframe
        
        Args:
            df: Input dataframe with OHLCV data
            indicators: List of indicators to add
            atr_period: Period for ATR calculation
            bb_period: Period for Bollinger Bands
            bb_std: Number of standard deviations for bands
            
        Returns:
            DataFrame with added volatility indicators
        """
        result = df.copy()
        
        if 'atr' in indicators:
            result[f'atr_{atr_period}'] = self._calculate_atr(
                result['high'], result['low'], result['close'], atr_period
            )
            self.logger.debug(f"Added ATR({atr_period})")
        
        if 'bollinger' in indicators:
            upper, middle, lower, width, percent = self._calculate_bollinger_bands(
                result['close'], bb_period, bb_std
            )
            result['bb_upper'] = upper
            result['bb_middle'] = middle
            result['bb_lower'] = lower
            result['bb_width'] = width
            result['bb_percent'] = percent
            self.logger.debug(f"Added Bollinger Bands({bb_period},{bb_std})")
        
        return result
    
    def add_volume_analytics(self,
                           df: pd.DataFrame,
                           indicators: List[str] = ['vwap', 'obv'],
                           vwap_period: Optional[int] = None) -> pd.DataFrame:
        """
        Add volume-based indicators to the dataframe
        
        Args:
            df: Input dataframe with OHLCV data
            indicators: List of indicators to add
            vwap_period: Period for VWAP (None for session VWAP)
            
        Returns:
            DataFrame with added volume indicators
        """
        result = df.copy()
        
        if 'vwap' in indicators:
            result['vwap'] = self._calculate_vwap(
                result['high'], result['low'], result['close'], 
                result['volume'], vwap_period
            )
            self.logger.debug("Added VWAP")
        
        if 'obv' in indicators:
            result['obv'] = self._calculate_obv(result['close'], result['volume'])
            self.logger.debug("Added OBV")
        
        return result
    
    def add_microstructure_features(self,
                                  df: pd.DataFrame,
                                  features: List[str] = ['spread_proxy', 'high_low_ratio']) -> pd.DataFrame:
        """
        Add market microstructure features
        
        Args:
            df: Input dataframe with OHLCV data
            features: List of features to add
            
        Returns:
            DataFrame with added microstructure features
        """
        result = df.copy()
        
        if 'spread_proxy' in features:
            # High-low spread as proxy for bid-ask spread
            result['spread_proxy'] = result['high'] - result['low']
            self.logger.debug("Added spread proxy")
        
        if 'high_low_ratio' in features:
            # Ratio of high to low prices
            result['high_low_ratio'] = result['high'] / result['low']
            self.logger.debug("Added high-low ratio")
        
        if 'price_efficiency' in features:
            # Garman-Klass volatility estimator
            result['price_efficiency'] = self._calculate_garman_klass(
                result['high'], result['low'], result['close'], result['open']
            )
            self.logger.debug("Added price efficiency")
        
        return result
    
    # ------------------------------------------------------------------
    # Lunar features
    # ------------------------------------------------------------------
    def add_lunar_features(self, df: pd.DataFrame,
                           columns: List[str] = ['moon_age', 'is_full_moon', 'is_new_moon']
                           ) -> pd.DataFrame:
        """
        Append lunar-cycle features computed from the timestamp index.
        The calculation is done once per day and forward-filled intraday.

        Features:
        - moon_age: age of the moon in days (0-29.53)
        - is_full_moon: 1 on full-moon day else 0
        - is_new_moon:  1 on new-moon day  else 0
        """
        if 'moon_age' in df.columns:
            # Already computed
            return df

        self.logger.debug("Calculating lunar features")
        # Work on daily frequency to avoid millions of ephem calls
        daily_idx = pd.to_datetime(df.index.date).unique()
        moon_data = {}
        for day in daily_idx:
            obs_date = ephem.Date(day.strftime('%Y/%m/%d 00:00'))
            moon = ephem.Moon(obs_date)
            moon_age = moon.age  # days since last new moon
            # Full moon ~14.77 days; allow ±0.5-day tolerance
            is_full = int(abs(moon_age - 14.77) < 0.5)
            is_new = int(moon_age < 0.5 or moon_age > 29)
            moon_data[day] = {'moon_age': moon_age,
                              'is_full_moon': is_full,
                              'is_new_moon': is_new}

        moon_df = pd.DataFrame.from_dict(moon_data, orient='index')
        moon_df.index = pd.to_datetime(moon_df.index)
        # Merge and forward-fill to minute level
        result = df.join(moon_df, on=pd.to_datetime(df.index.date), how='left')
        result[columns] = result[columns].fillna(method='ffill')
        return result

    def add_rolling_statistics(self,
                             df: pd.DataFrame,
                             windows: List[int] = [20, 50],
                             stats: List[str] = ['volatility', 'skew', 'kurtosis']) -> pd.DataFrame:
        """
        Add rolling statistical measures
        
        Args:
            df: Input dataframe with OHLCV data
            windows: Rolling window sizes
            stats: Statistics to calculate
            
        Returns:
            DataFrame with added rolling statistics
        """
        result = df.copy()
        
        # Calculate returns for statistics
        returns = result['close'].pct_change()
        
        for window in windows:
            if 'volatility' in stats:
                # Annualized volatility
                result[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252 * 390)
                self.logger.debug(f"Added volatility({window})")
            
            if 'skew' in stats:
                result[f'skew_{window}'] = returns.rolling(window).skew()
                self.logger.debug(f"Added skew({window})")
            
            if 'kurtosis' in stats:
                result[f'kurtosis_{window}'] = returns.rolling(window).kurt()
                self.logger.debug(f"Added kurtosis({window})")
        
        return result
    
    def add_all_features(self, 
                        df: pd.DataFrame,
                        cache_key: Optional[str] = None) -> pd.DataFrame:
        """
        Add all common features to the dataframe
        
        Args:
            df: Input dataframe with OHLCV data
            cache_key: Optional key for caching results
            
        Returns:
            DataFrame with all features added
        """
        # Check cache first
        if cache_key and self.cache_manager:
            cached_data = self.load_cached_features(cache_key)
            if cached_data is not None:
                self.logger.info(f"Loaded features from cache: {cache_key}")
                return cached_data
        
        self.logger.info("Adding all features to dataframe")
        
        # Start with copy
        result = df.copy()
        
        # Add moving averages
        result = self.add_moving_averages(result, periods=[10, 20, 50], ma_types=['sma', 'ema'])
        
        # Add momentum indicators
        result = self.add_momentum_indicators(result, indicators=['rsi', 'macd'])
        
        # Add volatility indicators
        result = self.add_volatility_indicators(result, indicators=['atr', 'bollinger'])
        
        # Add volume analytics
        result = self.add_volume_analytics(result, indicators=['vwap', 'obv'])
        
        # Add microstructure features
        result = self.add_microstructure_features(result, features=['spread_proxy', 'high_low_ratio'])
        
        # Add rolling statistics
        result = self.add_rolling_statistics(result, windows=[20], stats=['volatility'])

        # Add lunar-cycle features (used by crypto strategies)
        result = self.add_lunar_features(result)
        
        self.logger.info(f"Added {len(result.columns) - len(df.columns)} features")
        
        # Cache if requested
        if cache_key and self.cache_manager:
            self._cache_features(result, cache_key)
        
        return result
    
    # Helper methods
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast: int, slow: int, signal: int) -> tuple:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change"""
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _calculate_bollinger_bands(self, prices: pd.Series, 
                                  period: int, num_std: float) -> tuple:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        width = upper - lower
        
        # Bollinger %B
        percent = (prices - lower) / (upper - lower)
        
        return upper, middle, lower, width, percent
    
    def _calculate_vwap(self, high: pd.Series, low: pd.Series, 
                       close: pd.Series, volume: pd.Series, 
                       period: Optional[int] = None) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        
        if period is None:
            # Session VWAP - reset each day
            dates = pd.Series(typical_price.index.date, index=typical_price.index)
            cumulative_tpv = (typical_price * volume).groupby(dates).cumsum()
            cumulative_volume = volume.groupby(dates).cumsum()
        else:
            # Rolling VWAP
            cumulative_tpv = (typical_price * volume).rolling(window=period).sum()
            cumulative_volume = volume.rolling(window=period).sum()
        
        vwap = cumulative_tpv / cumulative_volume
        return vwap
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On Balance Volume"""
        price_diff = close.diff()
        
        # Create volume direction series
        volume_direction = pd.Series(0, index=close.index)
        volume_direction[price_diff > 0] = volume[price_diff > 0]
        volume_direction[price_diff < 0] = -volume[price_diff < 0]
        
        obv = volume_direction.cumsum()
        return obv
    
    def _calculate_garman_klass(self, high: pd.Series, low: pd.Series,
                               close: pd.Series, open_: pd.Series) -> pd.Series:
        """Calculate Garman-Klass volatility estimator"""
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        
        gk = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
        return gk
    
    def _cache_features(self, df: pd.DataFrame, cache_key: str) -> None:
        """Cache computed features"""
        if self.cache_manager:
            # Save as parquet
            cache_file = Path(f"features_{cache_key}.parquet")
            df.to_parquet(cache_file, compression='snappy')
            
            self.cache_manager.cache_file(
                cache_file,
                f"features/{cache_key}",
                category="computed_features"
            )
            
            # Clean up temp file
            cache_file.unlink()
            
            self.logger.info(f"Cached features with key: {cache_key}")
    
    def load_cached_features(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load previously cached features"""
        if self.cache_manager:
            cached_path = self.cache_manager.get_cached_file(f"features/{cache_key}")
            if cached_path and cached_path.exists():
                return pd.read_parquet(cached_path)
        
        return None
