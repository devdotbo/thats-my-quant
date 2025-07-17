"""
Lunar Features Calculator
Calculates moon phases and lunar-related features for trading strategies
"""

import ephem
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class LunarEvent:
    """Container for lunar event information"""
    timestamp: datetime
    event_type: str  # 'new_moon', 'full_moon', 'first_quarter', 'last_quarter'
    phase: float  # 0.0 = new moon, 0.5 = full moon
    distance: float  # Earth-moon distance in km
    illumination: float  # Fraction illuminated (0.0 to 1.0)


class LunarCalculator:
    """
    Calculate moon phases and lunar features for trading
    
    Uses PyEphem for astronomical calculations
    Phase convention: 0.0 = new moon, 0.5 = full moon
    """
    
    def __init__(self):
        """Initialize lunar calculator"""
        self.moon = ephem.Moon()
        
    def get_moon_phase(self, date: Union[datetime, str]) -> float:
        """
        Calculate moon phase for a given date
        
        Args:
            date: Date to calculate phase for
            
        Returns:
            Phase as float (0.0 = new moon, 0.5 = full moon)
        """
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
            
        # Convert to ephem date
        ephem_date = ephem.Date(date)
        
        # Get previous and next new moon
        prev_new = ephem.previous_new_moon(ephem_date)
        next_new = ephem.next_new_moon(ephem_date)
        
        # Calculate phase as fraction of lunar cycle
        lunation = (ephem_date - prev_new) / (next_new - prev_new)
        
        return float(lunation)
    
    def get_moon_illumination(self, date: Union[datetime, str]) -> float:
        """
        Calculate moon illumination (lit fraction)
        
        Args:
            date: Date to calculate for
            
        Returns:
            Illumination fraction (0.0 to 1.0)
        """
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
            
        self.moon.compute(date)
        return float(self.moon.moon_phase)
    
    def get_moon_distance(self, date: Union[datetime, str]) -> Dict[str, float]:
        """
        Calculate Earth-Moon distance and related metrics
        
        Args:
            date: Date to calculate for
            
        Returns:
            Dict with distance metrics
        """
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
            
        self.moon.compute(date)
        
        # Distance in AU, convert to km
        distance_km = self.moon.earth_distance * ephem.meters_per_au / 1000
        
        # Calculate if near apogee or perigee
        # Average distance ~384,400 km
        avg_distance = 384400
        distance_ratio = distance_km / avg_distance
        
        return {
            'distance_km': float(distance_km),
            'distance_ratio': float(distance_ratio),
            'is_apogee': distance_ratio > 1.03,  # >3% above average
            'is_perigee': distance_ratio < 0.97   # >3% below average
        }
    
    def get_lunar_features(self, date: Union[datetime, str]) -> Dict[str, float]:
        """
        Calculate comprehensive lunar features for a date
        
        Args:
            date: Date to calculate for
            
        Returns:
            Dict with all lunar features
        """
        phase = self.get_moon_phase(date)
        illumination = self.get_moon_illumination(date)
        distance_info = self.get_moon_distance(date)
        
        # Determine moon phase category
        if phase < 0.125:
            phase_name = 'new_moon'
        elif 0.125 <= phase < 0.375:
            phase_name = 'waxing'
        elif 0.375 <= phase < 0.625:
            phase_name = 'full_moon'
        else:
            phase_name = 'waning'
            
        # Calculate days since key events
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        ephem_date = ephem.Date(date)
        
        days_since_new = float(ephem_date - ephem.previous_new_moon(ephem_date))
        days_since_full = float(ephem_date - ephem.previous_full_moon(ephem_date))
        days_until_new = float(ephem.next_new_moon(ephem_date) - ephem_date)
        days_until_full = float(ephem.next_full_moon(ephem_date) - ephem_date)
        
        features = {
            'phase': phase,
            'illumination': illumination,
            'phase_name': phase_name,
            'is_new_moon': phase < 0.05 or phase > 0.95,
            'is_full_moon': 0.45 < phase < 0.55,
            'is_first_quarter': 0.20 < phase < 0.30,
            'is_last_quarter': 0.70 < phase < 0.80,
            'days_since_new': days_since_new,
            'days_since_full': days_since_full,
            'days_until_new': days_until_new,
            'days_until_full': days_until_full,
            **distance_info
        }
        
        return features
    
    def calculate_lunar_series(self, 
                             dates: pd.DatetimeIndex,
                             features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate lunar features for a series of dates
        
        Args:
            dates: Pandas DatetimeIndex
            features: List of features to calculate (None = all)
            
        Returns:
            DataFrame with lunar features
        """
        results = []
        
        for date in dates:
            lunar_data = self.get_lunar_features(date)
            lunar_data['date'] = date
            results.append(lunar_data)
            
        df = pd.DataFrame(results)
        df.set_index('date', inplace=True)
        
        # Filter features if requested
        if features:
            available = [f for f in features if f in df.columns]
            df = df[available]
            
        return df
    
    def find_lunar_events(self, 
                         start_date: Union[datetime, str],
                         end_date: Union[datetime, str]) -> List[LunarEvent]:
        """
        Find all lunar events (new/full moons, quarters) in date range
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of LunarEvent objects
        """
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
            
        events = []
        current = ephem.Date(start_date)
        end = ephem.Date(end_date)
        
        # Find new moons
        while current < end:
            next_new = ephem.next_new_moon(current)
            if next_new > end:
                break
                
            date = next_new.datetime()
            features = self.get_lunar_features(date)
            
            event = LunarEvent(
                timestamp=date,
                event_type='new_moon',
                phase=0.0,
                distance=features['distance_km'],
                illumination=0.0
            )
            events.append(event)
            current = next_new + 1
            
        # Find full moons
        current = ephem.Date(start_date)
        while current < end:
            next_full = ephem.next_full_moon(current)
            if next_full > end:
                break
                
            date = next_full.datetime()
            features = self.get_lunar_features(date)
            
            event = LunarEvent(
                timestamp=date,
                event_type='full_moon',
                phase=0.5,
                distance=features['distance_km'],
                illumination=1.0
            )
            events.append(event)
            current = next_full + 1
            
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        return events
    
    def add_lunar_features_to_data(self,
                                 data: pd.DataFrame,
                                 features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Add lunar features to existing market data
        
        Args:
            data: DataFrame with DatetimeIndex
            features: Specific features to add (None = all)
            
        Returns:
            DataFrame with lunar features added
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
            
        # Calculate lunar features
        lunar_df = self.calculate_lunar_series(data.index, features)
        
        # Merge with original data
        result = pd.concat([data, lunar_df], axis=1)
        
        return result


def create_lunar_trading_signals(data: pd.DataFrame,
                               strategy: str = 'classic') -> pd.Series:
    """
    Create trading signals based on lunar cycles
    
    Args:
        data: DataFrame with lunar features
        strategy: Trading strategy type
            - 'classic': Buy new moon, sell full moon
            - 'momentum': Buy after full moon
            - 'reversal': Trade phase extremes
            
    Returns:
        Series of trading signals (-1, 0, 1)
    """
    if 'phase' not in data.columns:
        raise ValueError("Data must contain 'phase' column")
        
    signals = pd.Series(0, index=data.index)
    
    if strategy == 'classic':
        # Buy on new moon, sell on full moon
        signals[data['is_new_moon']] = 1
        signals[data['is_full_moon']] = -1
        
    elif strategy == 'momentum':
        # Buy 1-3 days after full moon
        full_moon_days = data[data['is_full_moon']].index
        for fm_date in full_moon_days:
            # Buy for next 3 days
            buy_dates = pd.date_range(
                start=fm_date + timedelta(days=1),
                end=fm_date + timedelta(days=3),
                freq='D'
            )
            buy_dates = buy_dates.intersection(data.index)
            signals.loc[buy_dates] = 1
            
    elif strategy == 'reversal':
        # Trade when phase is at extremes
        signals[data['phase'] < 0.1] = 1  # Near new moon
        signals[data['phase'] > 0.9] = 1  # Also near new moon
        signals[(data['phase'] > 0.4) & (data['phase'] < 0.6)] = -1  # Near full moon
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    return signals


# Example usage
if __name__ == "__main__":
    # Create calculator
    calc = LunarCalculator()
    
    # Test single date
    date = datetime(2024, 1, 15)
    features = calc.get_lunar_features(date)
    print(f"Lunar features for {date}:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # Test date range
    dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
    lunar_df = calc.calculate_lunar_series(dates)
    print(f"\nLunar features for January 2024:")
    print(lunar_df.head())
    
    # Find lunar events
    events = calc.find_lunar_events('2024-01-01', '2024-12-31')
    print(f"\nLunar events in 2024: {len(events)} events")
    for event in events[:5]:
        print(f"  {event.timestamp}: {event.event_type}")