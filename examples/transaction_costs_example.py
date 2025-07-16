"""
Example: Using Transaction Cost Models
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.backtesting.costs import (
    CommissionModel, SpreadModel, MarketImpactModel, SlippageModel,
    TransactionCostEngine, Order, Trade, MarketState, MarketConditions,
    create_cost_engine_from_profile, EQUITY_COST_PROFILES, ETF_COST_PROFILES
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def demonstrate_commission_models():
    """Show different commission structures"""
    print("\n" + "="*60)
    print("COMMISSION MODELS")
    print("="*60)
    
    # Per-share commission
    per_share = CommissionModel(commission_type='per_share', rate=0.005)
    commission1 = per_share.calculate(quantity=1000, price=50.0, value=50000.0)
    print(f"\nPer-share commission (1000 shares @ $0.005): ${commission1:.2f}")
    
    # Percentage-based commission
    percentage = CommissionModel(commission_type='percentage', rate=0.001)
    commission2 = percentage.calculate(quantity=1000, price=50.0, value=50000.0)
    print(f"Percentage commission (0.1% of $50,000): ${commission2:.2f}")
    
    # Tiered commission
    tiers = [
        (1000, 0.01),      # <1000 shares: $0.01/share
        (10000, 0.005),    # 1000-10000 shares: $0.005/share
        (float('inf'), 0.003)  # >10000 shares: $0.003/share
    ]
    tiered = CommissionModel(commission_type='tiered', tiers=tiers)
    
    for qty in [500, 5000, 20000]:
        comm = tiered.calculate(quantity=qty, price=50.0, value=qty*50.0)
        print(f"Tiered commission ({qty} shares): ${comm:.2f}")


def demonstrate_spread_models():
    """Show spread cost calculations"""
    print("\n" + "="*60)
    print("SPREAD MODELS")
    print("="*60)
    
    spread_model = SpreadModel()
    
    # Different symbols have different spreads
    symbols = ['SPY', 'AAPL', 'UNKNOWN_STOCK']
    for symbol in symbols:
        spread = spread_model.get_spread(symbol, pd.Timestamp.now(), 100.0)
        cost = spread_model.calculate_spread_cost(symbol, pd.Timestamp.now(), 100.0, 1000)
        print(f"\n{symbol}:")
        print(f"  Spread: ${spread:.3f}")
        print(f"  Cost for 1000 shares: ${cost:.2f}")


def demonstrate_market_impact():
    """Show market impact calculations"""
    print("\n" + "="*60)
    print("MARKET IMPACT MODELS")
    print("="*60)
    
    # Create different impact models
    linear_model = MarketImpactModel(impact_type='linear', base_coefficient=10)
    sqrt_model = MarketImpactModel(impact_type='square_root', base_coefficient=10)
    
    # Test with different order sizes
    avg_volume = 1000000  # 1M average daily volume
    price = 100.0
    volatility = 0.02
    
    print("\nMarket Impact for Different Order Sizes (ADV = 1M shares):")
    print("Order Size | % of ADV | Linear Impact | Square-Root Impact")
    print("-" * 60)
    
    for order_size in [1000, 10000, 50000, 100000]:
        participation = order_size / avg_volume * 100
        
        linear_impact = linear_model.calculate_impact(
            order_size, avg_volume, volatility, price
        )
        sqrt_impact = sqrt_model.calculate_impact(
            order_size, avg_volume, volatility, price
        )
        
        print(f"{order_size:>10} | {participation:>7.1f}% | ${linear_impact:>12.2f} | ${sqrt_impact:>17.2f}")


def demonstrate_complete_cost_calculation():
    """Show complete cost breakdown for a trade"""
    print("\n" + "="*60)
    print("COMPLETE COST CALCULATION")
    print("="*60)
    
    # Create cost engine
    engine = TransactionCostEngine(
        commission_model=CommissionModel(commission_type='per_share', rate=0.005),
        spread_model=SpreadModel(),
        impact_model=MarketImpactModel(impact_type='square_root', base_coefficient=10),
        slippage_model=SlippageModel(base_slippage=0.0001),
        safety_factor=1.2  # 20% conservative buffer
    )
    
    # Create a sample trade
    trade = Trade(
        symbol='AAPL',
        timestamp=pd.Timestamp('2024-01-02 10:30:00'),
        side='buy',
        quantity=5000,
        price=150.0,
        value=750000.0
    )
    
    # Market conditions
    market_state = MarketState(
        timestamp=trade.timestamp,
        symbol='AAPL',
        avg_volume=50000000,  # 50M ADV for AAPL
        volatility=0.025,     # 2.5% volatility
        bid=149.99,
        ask=150.01,
        conditions=MarketConditions(
            timestamp=trade.timestamp,
            volatility=0.025,
            avg_volume=50000000,
            bid_ask_spread=0.02,
            is_opening_30min=False,
            is_closing_30min=False
        )
    )
    
    # Calculate costs
    cost_breakdown = engine.calculate_total_cost(trade, market_state)
    
    print(f"\nTrade: Buy {trade.quantity} shares of {trade.symbol} @ ${trade.price}")
    print(f"Trade Value: ${trade.value:,.2f}")
    print("\nCost Breakdown:")
    print(f"  Commission:    ${cost_breakdown.commission:>10.2f}")
    print(f"  Spread Cost:   ${cost_breakdown.spread:>10.2f}")
    print(f"  Market Impact: ${cost_breakdown.impact:>10.2f}")
    print(f"  Slippage:      ${cost_breakdown.slippage:>10.2f}")
    print(f"  " + "-"*30)
    print(f"  Total Cost:    ${cost_breakdown.total:>10.2f}")
    print(f"  Cost (bps):    {cost_breakdown.cost_bps:>10.2f}")


def demonstrate_cost_profiles():
    """Show predefined cost profiles"""
    print("\n" + "="*60)
    print("PREDEFINED COST PROFILES")
    print("="*60)
    
    print("\nEquity Cost Profiles:")
    for tier, profile in EQUITY_COST_PROFILES.items():
        print(f"\n{tier}:")
        print(f"  Commission: ${profile['commission']}/share")
        print(f"  Spread: {profile['spread_bps']} bps")
        print(f"  Impact coefficient: {profile['impact_coefficient']}")
        print(f"  Base slippage: {profile['base_slippage']*10000:.1f} bps")
    
    print("\n\nETF Cost Profiles:")
    for tier, profile in ETF_COST_PROFILES.items():
        print(f"\n{tier}:")
        print(f"  Commission: ${profile['commission']}/share")
        print(f"  Spread: {profile['spread_bps']} bps")
        print(f"  Impact coefficient: {profile['impact_coefficient']}")
        print(f"  Base slippage: {profile['base_slippage']*10000:.1f} bps")


def demonstrate_time_of_day_effects():
    """Show how costs vary by time of day"""
    print("\n" + "="*60)
    print("TIME OF DAY EFFECTS")
    print("="*60)
    
    # Create engine
    engine = create_cost_engine_from_profile('equity', 'large_cap_liquid')
    
    # Same trade at different times
    trade = Trade(
        symbol='SPY',
        timestamp=pd.Timestamp('2024-01-02 09:30:00'),
        side='buy',
        quantity=10000,
        price=400.0,
        value=4000000.0
    )
    
    times = [
        ('09:35:00', True, False, "Opening (high volatility)"),
        ('10:30:00', False, False, "Mid-morning (normal)"),
        ('12:30:00', False, False, "Lunch (lower liquidity)"),
        ('15:45:00', False, True, "Closing (high activity)")
    ]
    
    print("\nCost variation by time of day (10,000 SPY shares):")
    print("Time      | Market Phase         | Total Cost | Cost (bps)")
    print("-" * 60)
    
    for time_str, is_open, is_close, phase in times:
        trade.timestamp = pd.Timestamp(f'2024-01-02 {time_str}')
        
        market_state = MarketState(
            timestamp=trade.timestamp,
            symbol='SPY',
            avg_volume=100000000,  # 100M ADV
            volatility=0.015,
            bid=399.99,
            ask=400.01,
            conditions=MarketConditions(
                timestamp=trade.timestamp,
                volatility=0.015,
                avg_volume=100000000,
                bid_ask_spread=0.02,
                is_opening_30min=is_open,
                is_closing_30min=is_close
            )
        )
        
        costs = engine.calculate_total_cost(trade, market_state)
        print(f"{time_str} | {phase:<20} | ${costs.total:>9.2f} | {costs.cost_bps:>9.2f}")


def main():
    """Run all demonstrations"""
    logger.info("Starting transaction cost examples...")
    
    demonstrate_commission_models()
    demonstrate_spread_models()
    demonstrate_market_impact()
    demonstrate_complete_cost_calculation()
    demonstrate_cost_profiles()
    demonstrate_time_of_day_effects()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("\n1. Transaction costs can significantly impact returns")
    print("2. Large orders have disproportionate market impact")
    print("3. Costs vary by time of day and market conditions")
    print("4. Different asset classes have different cost structures")
    print("5. Always use conservative estimates in backtesting")
    
    logger.info("Transaction cost examples completed")


if __name__ == "__main__":
    main()