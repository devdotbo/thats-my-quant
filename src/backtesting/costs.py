"""
Transaction Cost Models
Comprehensive cost modeling for realistic backtesting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.utils.logging import get_logger


@dataclass
class Order:
    """Represents a trading order"""
    symbol: str
    size: float
    price: float
    order_type: str = 'market'  # 'market', 'limit', etc.
    side: str = 'buy'  # 'buy' or 'sell'


@dataclass
class Trade:
    """Represents an executed trade"""
    symbol: str
    timestamp: pd.Timestamp
    side: str
    quantity: float
    price: float
    value: float


@dataclass
class MarketConditions:
    """Market conditions at time of trade"""
    timestamp: pd.Timestamp
    volatility: float
    avg_volume: float
    bid_ask_spread: float
    is_opening_30min: bool = False
    is_closing_30min: bool = False


@dataclass
class MarketState:
    """Complete market state for cost calculation"""
    timestamp: pd.Timestamp
    symbol: str
    avg_volume: float
    volatility: float
    bid: float
    ask: float
    conditions: MarketConditions


@dataclass
class CostBreakdown:
    """Detailed breakdown of transaction costs"""
    commission: float
    spread: float
    impact: float
    slippage: float
    total: float
    cost_bps: float  # Total cost in basis points


class CommissionModel:
    """
    Models various commission structures
    
    Supports:
    - Per-share pricing
    - Per-trade flat fee
    - Percentage of trade value
    - Tiered pricing structures
    """
    
    def __init__(self,
                 commission_type: str = 'per_share',
                 rate: float = 0.005,
                 minimum: float = 0.0,
                 maximum: float = float('inf'),
                 tiers: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize commission model
        
        Args:
            commission_type: Type of commission ('per_share', 'per_trade', 'percentage', 'tiered')
            rate: Base commission rate
            minimum: Minimum commission per trade
            maximum: Maximum commission per trade
            tiers: List of (threshold, rate) tuples for tiered pricing
        """
        self.commission_type = commission_type
        self.rate = rate
        self.minimum = minimum
        self.maximum = maximum
        self.tiers = tiers
        self.logger = get_logger("commission_model")
    
    def calculate(self, quantity: float, price: float, value: float) -> float:
        """
        Calculate commission for a trade
        
        Args:
            quantity: Number of shares
            price: Price per share
            value: Total trade value
            
        Returns:
            Commission amount
        """
        if self.commission_type == 'per_share':
            commission = quantity * self.rate
            
        elif self.commission_type == 'per_trade':
            commission = self.rate
            
        elif self.commission_type == 'percentage':
            commission = value * self.rate
            
        elif self.commission_type == 'tiered':
            commission = self._calculate_tiered(quantity)
            
        else:
            raise ValueError(f"Unknown commission type: {self.commission_type}")
        
        # Apply minimum and maximum
        commission = max(self.minimum, min(self.maximum, commission))
        
        return commission
    
    def _calculate_tiered(self, quantity: float) -> float:
        """Calculate commission using tiered structure"""
        if not self.tiers:
            return quantity * self.rate
        
        # Find applicable tier
        for threshold, rate in self.tiers:
            if quantity < threshold:
                return quantity * rate
        
        # If beyond all thresholds, use last rate
        return quantity * self.tiers[-1][1]


class SpreadModel:
    """
    Models bid-ask spread costs
    
    Uses actual spread data when available, otherwise estimates
    based on symbol liquidity and market conditions
    """
    
    def __init__(self, spread_data: Optional[pd.DataFrame] = None):
        """
        Initialize spread model
        
        Args:
            spread_data: DataFrame with columns [timestamp, symbol, bid, ask, spread]
        """
        self.spread_data = spread_data
        self.default_spreads = {
            'SPY': 0.01,    # $0.01 for ultra-liquid ETFs
            'QQQ': 0.01,
            'IWM': 0.02,
            'AAPL': 0.02,   # $0.02 for large-cap stocks
            'MSFT': 0.02,
            'GOOGL': 0.05,
            'DEFAULT': 0.05  # $0.05 for others
        }
        self.logger = get_logger("spread_model")
    
    def get_spread(self, symbol: str, timestamp: pd.Timestamp, price: float) -> float:
        """
        Get bid-ask spread at specific time
        
        Args:
            symbol: Trading symbol
            timestamp: Time of trade
            price: Current price (for estimation)
            
        Returns:
            Spread amount
        """
        # Try to get actual spread from data
        if self.spread_data is not None:
            mask = (self.spread_data['symbol'] == symbol) & \
                   (self.spread_data['timestamp'] == timestamp)
            
            matching_data = self.spread_data[mask]
            if len(matching_data) > 0:
                return matching_data['spread'].iloc[0]
        
        # Use default spreads
        if symbol in self.default_spreads:
            return self.default_spreads[symbol]
        else:
            # Estimate based on price (2 basis points minimum $0.01)
            return max(0.01, price * 0.0002)
    
    def calculate_spread_cost(self,
                            symbol: str,
                            timestamp: pd.Timestamp,
                            price: float,
                            quantity: float) -> float:
        """
        Calculate total spread cost
        
        Args:
            symbol: Trading symbol
            timestamp: Time of trade
            price: Trade price
            quantity: Number of shares
            
        Returns:
            Spread cost (assuming crossing half the spread)
        """
        spread = self.get_spread(symbol, timestamp, price)
        
        # Assume crossing half the spread
        return (spread / 2) * quantity


class MarketImpactModel:
    """
    Models price impact of large orders
    
    Supports multiple impact models:
    - Linear: Impact proportional to participation rate
    - Square-root: More realistic, impact proportional to sqrt(participation)
    - Power law: Flexible power relationship
    """
    
    def __init__(self,
                 impact_type: str = 'square_root',
                 base_coefficient: float = 10,
                 power: float = 0.6):
        """
        Initialize market impact model
        
        Args:
            impact_type: Type of impact model ('linear', 'square_root', 'power_law')
            base_coefficient: Base impact coefficient in basis points
            power: Power for power law model
        """
        self.impact_type = impact_type
        self.base_coefficient = base_coefficient
        self.power = power
        self.logger = get_logger("market_impact_model")
    
    def calculate_impact(self,
                        order_size: float,
                        avg_volume: float,
                        volatility: float,
                        price: float) -> float:
        """
        Calculate market impact in dollars
        
        Args:
            order_size: Number of shares in order
            avg_volume: Average daily volume
            volatility: Current volatility
            price: Current price
            
        Returns:
            Market impact cost in dollars
        """
        # Participation rate (% of average volume)
        participation = order_size / avg_volume
        
        if self.impact_type == 'linear':
            # Simple linear model
            impact_bps = self.base_coefficient * participation
            
        elif self.impact_type == 'square_root':
            # Square-root model (more realistic)
            impact_bps = self.base_coefficient * np.sqrt(participation)
            
        elif self.impact_type == 'power_law':
            # Power law model
            impact_bps = self.base_coefficient * (participation ** self.power)
            
        else:
            raise ValueError(f"Unknown impact type: {self.impact_type}")
        
        # Adjust for volatility (normalized to 2% vol)
        impact_bps *= (volatility / 0.02)
        
        # Convert to dollar amount
        return (impact_bps / 10000) * price * order_size


class SlippageModel:
    """
    Advanced slippage modeling
    
    Accounts for:
    - Order size
    - Market volatility
    - Time of day
    - Order type
    """
    
    def __init__(self, base_slippage: float = 0.0001):
        """
        Initialize slippage model
        
        Args:
            base_slippage: Base slippage rate (e.g., 0.0001 = 1 basis point)
        """
        self.base_slippage = base_slippage
        self.logger = get_logger("slippage_model")
    
    def calculate_slippage(self,
                          order: Order,
                          market_conditions: MarketConditions) -> float:
        """
        Calculate expected slippage
        
        Args:
            order: Order details
            market_conditions: Current market conditions
            
        Returns:
            Slippage cost in dollars
        """
        slippage = self.base_slippage
        
        # Adjust for order size
        size_factor = min(2.0, 1 + order.size / market_conditions.avg_volume)
        slippage *= size_factor
        
        # Adjust for volatility (normalized to 2%)
        vol_factor = market_conditions.volatility / 0.02
        slippage *= vol_factor
        
        # Adjust for time of day
        if market_conditions.is_opening_30min:
            slippage *= 1.5
        elif market_conditions.is_closing_30min:
            slippage *= 1.3
        
        # Adjust for order type
        if order.order_type == 'market':
            slippage *= 1.0
        elif order.order_type == 'limit':
            slippage *= 0.5  # Less slippage but may not fill
        
        return slippage * order.price * order.size


class TransactionCostEngine:
    """
    Complete transaction cost calculation engine
    
    Combines all cost components:
    - Commission
    - Spread
    - Market impact
    - Slippage
    """
    
    def __init__(self,
                 commission_model: CommissionModel,
                 spread_model: SpreadModel,
                 impact_model: MarketImpactModel,
                 slippage_model: SlippageModel,
                 safety_factor: float = 1.0):
        """
        Initialize transaction cost engine
        
        Args:
            commission_model: Commission calculation model
            spread_model: Spread cost model
            impact_model: Market impact model
            slippage_model: Slippage model
            safety_factor: Multiplier for conservative estimation
        """
        self.commission_model = commission_model
        self.spread_model = spread_model
        self.impact_model = impact_model
        self.slippage_model = slippage_model
        self.safety_factor = safety_factor
        self.logger = get_logger("transaction_cost_engine")
    
    def calculate_total_cost(self,
                           trade: Trade,
                           market_state: MarketState) -> CostBreakdown:
        """
        Calculate all components of transaction cost
        
        Args:
            trade: Trade details
            market_state: Current market state
            
        Returns:
            Detailed cost breakdown
        """
        # Commission
        commission = self.commission_model.calculate(
            trade.quantity,
            trade.price,
            trade.value
        )
        
        # Spread cost
        spread_cost = self.spread_model.calculate_spread_cost(
            trade.symbol,
            trade.timestamp,
            trade.price,
            trade.quantity
        )
        
        # Market impact
        impact_cost = self.impact_model.calculate_impact(
            trade.quantity,
            market_state.avg_volume,
            market_state.volatility,
            trade.price
        )
        
        # Slippage
        order = Order(
            symbol=trade.symbol,
            size=trade.quantity,
            price=trade.price,
            order_type='market',
            side=trade.side
        )
        
        slippage_cost = self.slippage_model.calculate_slippage(
            order,
            market_state.conditions
        )
        
        # Apply safety factor to variable costs (not commission)
        spread_cost *= self.safety_factor
        impact_cost *= self.safety_factor
        slippage_cost *= self.safety_factor
        
        # Total cost
        total_cost = commission + spread_cost + impact_cost + slippage_cost
        
        # Cost in basis points
        cost_bps = total_cost / trade.value * 10000
        
        return CostBreakdown(
            commission=commission,
            spread=spread_cost,
            impact=impact_cost,
            slippage=slippage_cost,
            total=total_cost,
            cost_bps=cost_bps
        )


# Predefined cost profiles for different asset classes
EQUITY_COST_PROFILES = {
    'large_cap_liquid': {
        'commission': 0.005,      # $0.005/share
        'spread_bps': 1,          # 1 basis point
        'impact_coefficient': 5,   # Low impact
        'base_slippage': 0.0001   # 1 basis point
    },
    'mid_cap': {
        'commission': 0.005,
        'spread_bps': 5,          # 5 basis points
        'impact_coefficient': 10,
        'base_slippage': 0.0003
    },
    'small_cap': {
        'commission': 0.005,
        'spread_bps': 20,         # 20 basis points
        'impact_coefficient': 20,
        'base_slippage': 0.001
    }
}

ETF_COST_PROFILES = {
    'spy_qqq': {  # Ultra-liquid ETFs
        'commission': 0.0,         # Often commission-free
        'spread_bps': 0.5,        # 0.5 basis points
        'impact_coefficient': 2,
        'base_slippage': 0.00005
    },
    'sector_etf': {
        'commission': 0.0,
        'spread_bps': 3,
        'impact_coefficient': 8,
        'base_slippage': 0.0002
    },
    'international_etf': {
        'commission': 0.005,
        'spread_bps': 10,
        'impact_coefficient': 15,
        'base_slippage': 0.0005
    }
}

OPTIONS_COST_PROFILES = {
    'spy_options': {
        'commission': 0.65,        # Per contract
        'spread_bps': 10,         # Wider spreads
        'impact_coefficient': 20,
        'base_slippage': 0.001
    },
    'equity_options': {
        'commission': 0.65,
        'spread_bps': 50,         # Much wider
        'impact_coefficient': 30,
        'base_slippage': 0.003
    }
}


def create_cost_engine_from_profile(asset_class: str,
                                   liquidity_tier: str,
                                   safety_factor: float = 1.2) -> TransactionCostEngine:
    """
    Create a transaction cost engine from predefined profiles
    
    Args:
        asset_class: 'equity', 'etf', or 'options'
        liquidity_tier: Specific tier within asset class
        safety_factor: Conservative estimation multiplier
        
    Returns:
        Configured TransactionCostEngine
    """
    # Get appropriate profile
    if asset_class == 'equity':
        profile = EQUITY_COST_PROFILES.get(liquidity_tier, EQUITY_COST_PROFILES['mid_cap'])
    elif asset_class == 'etf':
        profile = ETF_COST_PROFILES.get(liquidity_tier, ETF_COST_PROFILES['sector_etf'])
    elif asset_class == 'options':
        profile = OPTIONS_COST_PROFILES.get(liquidity_tier, OPTIONS_COST_PROFILES['equity_options'])
    else:
        raise ValueError(f"Unknown asset class: {asset_class}")
    
    # Create models from profile
    commission_model = CommissionModel(
        commission_type='per_share' if asset_class != 'options' else 'per_contract',
        rate=profile['commission']
    )
    
    spread_model = SpreadModel()  # Could enhance with profile data
    
    impact_model = MarketImpactModel(
        impact_type='square_root',
        base_coefficient=profile['impact_coefficient']
    )
    
    slippage_model = SlippageModel(
        base_slippage=profile['base_slippage']
    )
    
    return TransactionCostEngine(
        commission_model=commission_model,
        spread_model=spread_model,
        impact_model=impact_model,
        slippage_model=slippage_model,
        safety_factor=safety_factor
    )


def adjust_costs_by_time(base_costs: Dict[str, float], 
                        timestamp: pd.Timestamp) -> Dict[str, float]:
    """
    Adjust costs based on time of day
    
    Args:
        base_costs: Base cost parameters
        timestamp: Time of trade
        
    Returns:
        Adjusted cost parameters
    """
    hour = timestamp.hour
    minute = timestamp.minute
    
    # Market open (first 30 minutes)
    if hour == 9 and minute < 30:
        multiplier = 1.5
    # Market close (last 30 minutes)
    elif hour == 15 and minute >= 30:
        multiplier = 1.3
    # Lunch time (lower liquidity)
    elif 12 <= hour <= 13:
        multiplier = 1.2
    # Normal trading hours
    else:
        multiplier = 1.0
    
    adjusted_costs = base_costs.copy()
    if 'spread_bps' in adjusted_costs:
        adjusted_costs['spread_bps'] *= multiplier
    if 'base_slippage' in adjusted_costs:
        adjusted_costs['base_slippage'] *= multiplier
    
    return adjusted_costs


def adjust_costs_by_volatility(base_costs: Dict[str, float],
                             current_vol: float,
                             normal_vol: float = 0.02) -> Dict[str, float]:
    """
    Adjust costs based on volatility regime
    
    Args:
        base_costs: Base cost parameters
        current_vol: Current volatility
        normal_vol: Normal volatility baseline
        
    Returns:
        Adjusted cost parameters
    """
    vol_multiplier = current_vol / normal_vol
    
    adjusted_costs = base_costs.copy()
    if 'spread_bps' in adjusted_costs:
        adjusted_costs['spread_bps'] *= vol_multiplier
    if 'impact_coefficient' in adjusted_costs:
        adjusted_costs['impact_coefficient'] *= np.sqrt(vol_multiplier)
    if 'base_slippage' in adjusted_costs:
        adjusted_costs['base_slippage'] *= vol_multiplier
    
    return adjusted_costs