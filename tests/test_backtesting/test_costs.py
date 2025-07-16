"""
Tests for transaction cost models
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time

from src.backtesting.costs import (
    CommissionModel, SpreadModel, MarketImpactModel, 
    SlippageModel, TransactionCostEngine, CostBreakdown,
    MarketConditions, Order, Trade, MarketState
)


class TestCommissionModel:
    """Test commission calculation models"""
    
    def test_per_share_commission(self):
        """Test per-share commission model"""
        model = CommissionModel(commission_type='per_share', rate=0.005)
        
        # Test basic calculation
        commission = model.calculate(quantity=100, price=50.0, value=5000.0)
        assert commission == 0.5  # 100 shares * $0.005
        
        # Test with larger quantity
        commission = model.calculate(quantity=1000, price=50.0, value=50000.0)
        assert commission == 5.0  # 1000 shares * $0.005
    
    def test_per_trade_commission(self):
        """Test per-trade commission model"""
        model = CommissionModel(commission_type='per_trade', rate=1.0)
        
        # Should be same regardless of quantity
        commission1 = model.calculate(quantity=100, price=50.0, value=5000.0)
        commission2 = model.calculate(quantity=1000, price=50.0, value=50000.0)
        
        assert commission1 == 1.0
        assert commission2 == 1.0
    
    def test_percentage_commission(self):
        """Test percentage-based commission model"""
        model = CommissionModel(commission_type='percentage', rate=0.001)  # 0.1%
        
        commission = model.calculate(quantity=100, price=50.0, value=5000.0)
        assert commission == 5.0  # 0.1% of $5000
        
        commission = model.calculate(quantity=1000, price=50.0, value=50000.0)
        assert commission == 50.0  # 0.1% of $50000
    
    def test_tiered_commission(self):
        """Test tiered commission structure"""
        tiers = [
            (1000, 0.01),     # <1000 shares: $0.01/share
            (10000, 0.005),   # 1000-10000 shares: $0.005/share
            (float('inf'), 0.003)  # >10000 shares: $0.003/share
        ]
        model = CommissionModel(commission_type='tiered', tiers=tiers)
        
        # Test each tier
        commission = model.calculate(quantity=500, price=50.0, value=25000.0)
        assert commission == 5.0  # 500 * 0.01
        
        commission = model.calculate(quantity=5000, price=50.0, value=250000.0)
        assert commission == 25.0  # 5000 * 0.005
        
        commission = model.calculate(quantity=20000, price=50.0, value=1000000.0)
        assert commission == 60.0  # 20000 * 0.003
    
    def test_minimum_maximum_commission(self):
        """Test minimum and maximum commission limits"""
        model = CommissionModel(
            commission_type='per_share', 
            rate=0.005,
            minimum=1.0,
            maximum=5.0
        )
        
        # Test minimum
        commission = model.calculate(quantity=10, price=50.0, value=500.0)
        assert commission == 1.0  # Would be $0.05, but minimum is $1
        
        # Test maximum
        commission = model.calculate(quantity=2000, price=50.0, value=100000.0)
        assert commission == 5.0  # Would be $10, but maximum is $5
        
        # Test normal case
        commission = model.calculate(quantity=500, price=50.0, value=25000.0)
        assert commission == 2.5  # 500 * 0.005 = $2.50


class TestSpreadModel:
    """Test bid-ask spread cost models"""
    
    @pytest.fixture
    def spread_data(self):
        """Create sample spread data"""
        dates = pd.date_range('2024-01-02 09:30', periods=100, freq='1min')
        data = []
        for i, date in enumerate(dates):
            data.append({
                'timestamp': date,
                'symbol': 'SPY',
                'bid': 400.0 + i * 0.01,
                'ask': 400.02 + i * 0.01,
                'spread': 0.02
            })
        return pd.DataFrame(data)
    
    def test_default_spreads(self):
        """Test default spread estimation"""
        model = SpreadModel()
        
        # Test known symbols
        assert model.get_spread('SPY', pd.Timestamp('2024-01-02'), 400.0) == 0.01
        assert model.get_spread('AAPL', pd.Timestamp('2024-01-02'), 150.0) == 0.02
        
        # Test unknown symbol - should use price-based estimate
        spread = model.get_spread('XYZ', pd.Timestamp('2024-01-02'), 50.0)
        assert spread == max(0.01, 50.0 * 0.0002)  # 2 basis points or $0.01
    
    def test_spread_with_data(self, spread_data):
        """Test spread calculation with actual data"""
        model = SpreadModel(spread_data=spread_data)
        
        # Test exact match
        timestamp = pd.Timestamp('2024-01-02 09:30:00')
        spread = model.get_spread('SPY', timestamp, 400.0)
        assert spread == 0.02
        
        # Test missing data - should fall back to default
        spread = model.get_spread('AAPL', timestamp, 150.0)
        assert spread == 0.02  # Default for AAPL
    
    def test_spread_cost_calculation(self):
        """Test total spread cost calculation"""
        model = SpreadModel()
        
        # Test spread cost (assuming crossing half the spread)
        cost = model.calculate_spread_cost('SPY', pd.Timestamp('2024-01-02'), 400.0, 100)
        assert cost == (0.01 / 2) * 100  # Half spread * quantity
        
        cost = model.calculate_spread_cost('AAPL', pd.Timestamp('2024-01-02'), 150.0, 1000)
        assert cost == (0.02 / 2) * 1000  # Half spread * quantity
    
    def test_time_varying_spreads(self, spread_data):
        """Test that spreads can vary by time"""
        # Add time-based variation to spread data
        spread_data.loc[spread_data.index[:10], 'spread'] = 0.05  # Wider at open
        spread_data.loc[spread_data.index[-10:], 'spread'] = 0.03  # Wider at close
        
        model = SpreadModel(spread_data=spread_data)
        
        # Test opening spread
        open_spread = model.get_spread('SPY', spread_data.iloc[0]['timestamp'], 400.0)
        assert open_spread == 0.05
        
        # Test closing spread
        close_spread = model.get_spread('SPY', spread_data.iloc[-1]['timestamp'], 400.0)
        assert close_spread == 0.03


class TestMarketImpactModel:
    """Test market impact models"""
    
    def test_linear_impact(self):
        """Test linear market impact model"""
        model = MarketImpactModel(impact_type='linear', base_coefficient=10)
        
        # 1% of volume = 10 bps impact
        impact = model.calculate_impact(
            order_size=10000,
            avg_volume=1000000,
            volatility=0.02,
            price=100.0
        )
        
        participation = 10000 / 1000000  # 1%
        expected_bps = 10 * participation  # 10 bps
        expected_cost = (expected_bps / 10000) * 100.0 * 10000
        
        assert impact == expected_cost
    
    def test_square_root_impact(self):
        """Test square-root market impact model"""
        model = MarketImpactModel(impact_type='square_root', base_coefficient=10)
        
        # Test with 1% participation
        impact = model.calculate_impact(
            order_size=10000,
            avg_volume=1000000,
            volatility=0.02,
            price=100.0
        )
        
        participation = 0.01
        expected_bps = 10 * np.sqrt(participation)  # 10 * sqrt(0.01) = 1 bp
        expected_cost = (expected_bps / 10000) * 100.0 * 10000
        
        assert abs(impact - expected_cost) < 0.01
    
    def test_power_law_impact(self):
        """Test power law market impact model"""
        model = MarketImpactModel(impact_type='power_law', base_coefficient=5, power=0.6)
        
        impact = model.calculate_impact(
            order_size=10000,
            avg_volume=1000000,
            volatility=0.02,
            price=100.0
        )
        
        participation = 0.01
        expected_bps = 5 * (participation ** 0.6)
        expected_cost = (expected_bps / 10000) * 100.0 * 10000
        
        assert abs(impact - expected_cost) < 0.01
    
    def test_volatility_adjustment(self):
        """Test that impact adjusts for volatility"""
        model = MarketImpactModel(impact_type='linear', base_coefficient=10)
        
        # Normal volatility (2%)
        impact_normal = model.calculate_impact(
            order_size=10000,
            avg_volume=1000000,
            volatility=0.02,
            price=100.0
        )
        
        # High volatility (4%)
        impact_high = model.calculate_impact(
            order_size=10000,
            avg_volume=1000000,
            volatility=0.04,
            price=100.0
        )
        
        # Impact should be higher with higher volatility
        assert impact_high == impact_normal * 2  # 4% / 2% = 2x
    
    def test_large_order_impact(self):
        """Test impact for large orders"""
        model = MarketImpactModel(impact_type='square_root', base_coefficient=10)
        
        # 10% of daily volume - should have significant impact
        impact = model.calculate_impact(
            order_size=100000,
            avg_volume=1000000,
            volatility=0.02,
            price=100.0
        )
        
        participation = 0.1
        expected_bps = 10 * np.sqrt(participation)  # ~3.16 bps
        expected_cost = (expected_bps / 10000) * 100.0 * 100000
        
        assert impact > 0  # Should have measurable impact
        assert abs(impact - expected_cost) < 1.0


class TestSlippageModel:
    """Test slippage models"""
    
    @pytest.fixture
    def market_conditions(self):
        """Create sample market conditions"""
        return MarketConditions(
            timestamp=pd.Timestamp('2024-01-02 10:00:00'),
            volatility=0.02,
            avg_volume=1000000,
            bid_ask_spread=0.01,
            is_opening_30min=False,
            is_closing_30min=False
        )
    
    @pytest.fixture
    def order(self):
        """Create sample order"""
        return Order(
            symbol='SPY',
            size=1000,
            price=400.0,
            order_type='market',
            side='buy'
        )
    
    def test_base_slippage(self, order, market_conditions):
        """Test basic slippage calculation"""
        model = SlippageModel(base_slippage=0.0001)  # 1 basis point
        
        slippage = model.calculate_slippage(order, market_conditions)
        
        # The calculation includes size factor adjustment
        # size_factor = min(2.0, 1 + order.size / market_conditions.avg_volume)
        # size_factor = min(2.0, 1 + 1000 / 1000000) = min(2.0, 1.001) = 1.001
        size_factor = min(2.0, 1 + 1000 / 1000000)
        vol_factor = 0.02 / 0.02  # 1.0 (normalized to itself)
        
        expected = 0.0001 * size_factor * vol_factor * 400.0 * 1000
        
        assert abs(slippage - expected) < 0.01  # Allow small numerical difference
    
    def test_size_adjustment(self, order, market_conditions):
        """Test slippage adjustment for order size"""
        model = SlippageModel(base_slippage=0.0001)
        
        # Large order - 10% of daily volume
        large_order = Order(
            symbol='SPY',
            size=100000,
            price=400.0,
            order_type='market',
            side='buy'
        )
        
        slippage_small = model.calculate_slippage(order, market_conditions)
        slippage_large = model.calculate_slippage(large_order, market_conditions)
        
        # Large order should have more slippage
        assert slippage_large > slippage_small
    
    def test_volatility_adjustment(self, order):
        """Test slippage adjustment for volatility"""
        model = SlippageModel(base_slippage=0.0001)
        
        # Normal volatility
        normal_conditions = MarketConditions(
            timestamp=pd.Timestamp('2024-01-02 10:00:00'),
            volatility=0.02,
            avg_volume=1000000,
            bid_ask_spread=0.01,
            is_opening_30min=False,
            is_closing_30min=False
        )
        
        # High volatility
        high_vol_conditions = MarketConditions(
            timestamp=pd.Timestamp('2024-01-02 10:00:00'),
            volatility=0.04,
            avg_volume=1000000,
            bid_ask_spread=0.01,
            is_opening_30min=False,
            is_closing_30min=False
        )
        
        slippage_normal = model.calculate_slippage(order, normal_conditions)
        slippage_high = model.calculate_slippage(order, high_vol_conditions)
        
        # Higher volatility should mean more slippage
        assert slippage_high > slippage_normal
    
    def test_time_of_day_adjustment(self, order, market_conditions):
        """Test slippage adjustment for time of day"""
        model = SlippageModel(base_slippage=0.0001)
        
        # Opening conditions
        opening_conditions = MarketConditions(
            timestamp=pd.Timestamp('2024-01-02 09:35:00'),
            volatility=0.02,
            avg_volume=1000000,
            bid_ask_spread=0.01,
            is_opening_30min=True,
            is_closing_30min=False
        )
        
        # Closing conditions
        closing_conditions = MarketConditions(
            timestamp=pd.Timestamp('2024-01-02 15:45:00'),
            volatility=0.02,
            avg_volume=1000000,
            bid_ask_spread=0.01,
            is_opening_30min=False,
            is_closing_30min=True
        )
        
        slippage_normal = model.calculate_slippage(order, market_conditions)
        slippage_opening = model.calculate_slippage(order, opening_conditions)
        slippage_closing = model.calculate_slippage(order, closing_conditions)
        
        # Opening and closing should have higher slippage
        assert slippage_opening > slippage_normal
        assert slippage_closing > slippage_normal
        assert slippage_opening > slippage_closing  # Opening typically worse
    
    def test_order_type_adjustment(self, market_conditions):
        """Test slippage differences by order type"""
        model = SlippageModel(base_slippage=0.0001)
        
        # Market order
        market_order = Order(
            symbol='SPY',
            size=1000,
            price=400.0,
            order_type='market',
            side='buy'
        )
        
        # Limit order
        limit_order = Order(
            symbol='SPY',
            size=1000,
            price=400.0,
            order_type='limit',
            side='buy'
        )
        
        slippage_market = model.calculate_slippage(market_order, market_conditions)
        slippage_limit = model.calculate_slippage(limit_order, market_conditions)
        
        # Limit orders should have less slippage
        assert slippage_limit < slippage_market


class TestTransactionCostEngine:
    """Test the complete transaction cost engine"""
    
    @pytest.fixture
    def cost_engine(self):
        """Create a transaction cost engine with all components"""
        commission_model = CommissionModel(commission_type='per_share', rate=0.005)
        spread_model = SpreadModel()
        impact_model = MarketImpactModel(impact_type='square_root', base_coefficient=10)
        slippage_model = SlippageModel(base_slippage=0.0001)
        
        return TransactionCostEngine(
            commission_model=commission_model,
            spread_model=spread_model,
            impact_model=impact_model,
            slippage_model=slippage_model
        )
    
    @pytest.fixture
    def trade(self):
        """Create a sample trade"""
        return Trade(
            symbol='SPY',
            timestamp=pd.Timestamp('2024-01-02 10:00:00'),
            side='buy',
            quantity=1000,
            price=400.0,
            value=400000.0
        )
    
    @pytest.fixture
    def market_state(self):
        """Create sample market state"""
        return MarketState(
            timestamp=pd.Timestamp('2024-01-02 10:00:00'),
            symbol='SPY',
            avg_volume=10000000,  # 10M shares average daily volume
            volatility=0.02,
            bid=399.99,
            ask=400.01,
            conditions=MarketConditions(
                timestamp=pd.Timestamp('2024-01-02 10:00:00'),
                volatility=0.02,
                avg_volume=10000000,
                bid_ask_spread=0.02,
                is_opening_30min=False,
                is_closing_30min=False
            )
        )
    
    def test_total_cost_calculation(self, cost_engine, trade, market_state):
        """Test that all cost components are calculated correctly"""
        cost_breakdown = cost_engine.calculate_total_cost(trade, market_state)
        
        # Check all components are present
        assert hasattr(cost_breakdown, 'commission')
        assert hasattr(cost_breakdown, 'spread')
        assert hasattr(cost_breakdown, 'impact')
        assert hasattr(cost_breakdown, 'slippage')
        assert hasattr(cost_breakdown, 'total')
        assert hasattr(cost_breakdown, 'cost_bps')
        
        # Commission: 1000 shares * $0.005 = $5
        assert cost_breakdown.commission == 5.0
        
        # Spread: Half spread * quantity = 0.005 * 1000 = $5
        assert cost_breakdown.spread == 5.0
        
        # Market impact should be positive (small order relative to ADV)
        assert cost_breakdown.impact > 0
        
        # Slippage should be positive
        assert cost_breakdown.slippage > 0
        
        # Total should be sum of all components
        expected_total = (cost_breakdown.commission + cost_breakdown.spread + 
                         cost_breakdown.impact + cost_breakdown.slippage)
        assert abs(cost_breakdown.total - expected_total) < 0.01
        
        # Cost in basis points
        expected_bps = cost_breakdown.total / trade.value * 10000
        assert abs(cost_breakdown.cost_bps - expected_bps) < 0.01
    
    def test_large_order_costs(self, cost_engine, market_state):
        """Test that large orders have higher costs"""
        # Small order
        small_trade = Trade(
            symbol='SPY',
            timestamp=pd.Timestamp('2024-01-02 10:00:00'),
            side='buy',
            quantity=100,
            price=400.0,
            value=40000.0
        )
        
        # Large order (1% of ADV)
        large_trade = Trade(
            symbol='SPY',
            timestamp=pd.Timestamp('2024-01-02 10:00:00'),
            side='buy',
            quantity=100000,
            price=400.0,
            value=40000000.0
        )
        
        small_costs = cost_engine.calculate_total_cost(small_trade, market_state)
        large_costs = cost_engine.calculate_total_cost(large_trade, market_state)
        
        # Large order should have higher cost in basis points
        assert large_costs.cost_bps > small_costs.cost_bps
        
        # Market impact should be significantly higher for large order
        assert large_costs.impact > small_costs.impact * 10
    
    def test_volatile_market_costs(self, cost_engine, trade):
        """Test that costs increase in volatile markets"""
        # Normal market
        normal_state = MarketState(
            timestamp=pd.Timestamp('2024-01-02 10:00:00'),
            symbol='SPY',
            avg_volume=10000000,
            volatility=0.02,
            bid=399.99,
            ask=400.01,
            conditions=MarketConditions(
                timestamp=pd.Timestamp('2024-01-02 10:00:00'),
                volatility=0.02,
                avg_volume=10000000,
                bid_ask_spread=0.02,
                is_opening_30min=False,
                is_closing_30min=False
            )
        )
        
        # Volatile market
        volatile_state = MarketState(
            timestamp=pd.Timestamp('2024-01-02 10:00:00'),
            symbol='SPY',
            avg_volume=10000000,
            volatility=0.05,  # 2.5x normal volatility
            bid=399.95,
            ask=400.05,
            conditions=MarketConditions(
                timestamp=pd.Timestamp('2024-01-02 10:00:00'),
                volatility=0.05,
                avg_volume=10000000,
                bid_ask_spread=0.10,  # Wider spread
                is_opening_30min=False,
                is_closing_30min=False
            )
        )
        
        normal_costs = cost_engine.calculate_total_cost(trade, normal_state)
        volatile_costs = cost_engine.calculate_total_cost(trade, volatile_state)
        
        # Costs should be higher in volatile market
        assert volatile_costs.total > normal_costs.total
        assert volatile_costs.cost_bps > normal_costs.cost_bps
    
    def test_safety_factor(self, trade, market_state):
        """Test that safety factor increases cost estimates"""
        # Engine without safety factor
        engine_normal = TransactionCostEngine(
            commission_model=CommissionModel(commission_type='per_share', rate=0.005),
            spread_model=SpreadModel(),
            impact_model=MarketImpactModel(impact_type='square_root'),
            slippage_model=SlippageModel(base_slippage=0.0001),
            safety_factor=1.0
        )
        
        # Engine with safety factor
        engine_conservative = TransactionCostEngine(
            commission_model=CommissionModel(commission_type='per_share', rate=0.005),
            spread_model=SpreadModel(),
            impact_model=MarketImpactModel(impact_type='square_root'),
            slippage_model=SlippageModel(base_slippage=0.0001),
            safety_factor=1.2  # 20% safety margin
        )
        
        costs_normal = engine_normal.calculate_total_cost(trade, market_state)
        costs_conservative = engine_conservative.calculate_total_cost(trade, market_state)
        
        # Safety factor applies only to variable costs (spread, impact, slippage), not commission
        # Commission stays the same
        assert costs_conservative.commission == costs_normal.commission
        
        # Variable costs should be 20% higher
        assert abs(costs_conservative.spread - costs_normal.spread * 1.2) < 0.01
        assert abs(costs_conservative.impact - costs_normal.impact * 1.2) < 0.01
        assert abs(costs_conservative.slippage - costs_normal.slippage * 1.2) < 0.01
        
        # Total should reflect the safety factor on variable costs
        variable_costs_normal = costs_normal.total - costs_normal.commission
        variable_costs_conservative = costs_conservative.total - costs_conservative.commission
        assert abs(variable_costs_conservative - variable_costs_normal * 1.2) < 0.01


class TestCostProfiles:
    """Test predefined cost profiles for different asset classes"""
    
    def test_equity_profiles(self):
        """Test equity market cost profiles"""
        from src.backtesting.costs import EQUITY_COST_PROFILES
        
        # Large cap should have lowest costs
        assert EQUITY_COST_PROFILES['large_cap_liquid']['spread_bps'] < \
               EQUITY_COST_PROFILES['mid_cap']['spread_bps']
        
        assert EQUITY_COST_PROFILES['mid_cap']['spread_bps'] < \
               EQUITY_COST_PROFILES['small_cap']['spread_bps']
        
        # Impact coefficients should increase with decreasing liquidity
        assert EQUITY_COST_PROFILES['large_cap_liquid']['impact_coefficient'] < \
               EQUITY_COST_PROFILES['small_cap']['impact_coefficient']
    
    def test_etf_profiles(self):
        """Test ETF market cost profiles"""
        from src.backtesting.costs import ETF_COST_PROFILES
        
        # SPY/QQQ should have lowest costs
        assert ETF_COST_PROFILES['spy_qqq']['spread_bps'] <= 1
        assert ETF_COST_PROFILES['spy_qqq']['commission'] == 0  # Often commission-free
        
        # International ETFs should have higher costs
        assert ETF_COST_PROFILES['international_etf']['spread_bps'] > \
               ETF_COST_PROFILES['spy_qqq']['spread_bps']


@pytest.mark.performance
class TestCostCalculationPerformance:
    """Test performance of cost calculations"""
    
    def test_bulk_cost_calculation(self):
        """Test that cost calculations are fast for many trades"""
        import time
        
        # Create cost engine
        engine = TransactionCostEngine(
            commission_model=CommissionModel(commission_type='per_share', rate=0.005),
            spread_model=SpreadModel(),
            impact_model=MarketImpactModel(impact_type='square_root'),
            slippage_model=SlippageModel(base_slippage=0.0001)
        )
        
        # Create 10,000 trades
        trades = []
        for i in range(10000):
            trades.append(Trade(
                symbol='SPY',
                timestamp=pd.Timestamp('2024-01-02') + pd.Timedelta(minutes=i),
                side='buy' if i % 2 == 0 else 'sell',
                quantity=100 + i % 1000,
                price=400.0 + (i % 10) * 0.1,
                value=(100 + i % 1000) * (400.0 + (i % 10) * 0.1)
            ))
        
        # Market state
        market_state = MarketState(
            timestamp=pd.Timestamp('2024-01-02 10:00:00'),
            symbol='SPY',
            avg_volume=10000000,
            volatility=0.02,
            bid=399.99,
            ask=400.01,
            conditions=MarketConditions(
                timestamp=pd.Timestamp('2024-01-02 10:00:00'),
                volatility=0.02,
                avg_volume=10000000,
                bid_ask_spread=0.02,
                is_opening_30min=False,
                is_closing_30min=False
            )
        )
        
        # Time the calculations
        start_time = time.perf_counter()
        
        for trade in trades:
            _ = engine.calculate_total_cost(trade, market_state)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        print(f"\nCost Calculation Performance:")
        print(f"- Trades: {len(trades)}")
        print(f"- Time: {elapsed:.2f} seconds")
        print(f"- Rate: {len(trades) / elapsed:.0f} trades/second")
        
        # Should process at least 1000 trades per second
        assert len(trades) / elapsed > 1000