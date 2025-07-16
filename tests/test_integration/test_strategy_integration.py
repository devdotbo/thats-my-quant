"""
Strategy Integration Tests
Tests strategy execution, optimization, and multi-asset portfolios
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.strategies.examples.moving_average import MovingAverageCrossover
from src.strategies.examples.orb import OpeningRangeBreakout
from src.backtesting.engines.vectorbt_engine import VectorBTEngine
from src.backtesting.costs import TransactionCostEngine, CommissionModel
from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngine


@pytest.mark.integration
class TestStrategyIntegration:
    """Test strategy execution in realistic scenarios"""
    
    @pytest.fixture
    def engine(self):
        """Create backtesting engine"""
        return VectorBTEngine()
    
    @pytest.fixture
    def test_data(self):
        """Load test data with features"""
        # Load AAPL data
        data_path = Path("data/raw/minute_aggs/by_symbol/AAPL/AAPL_2024_01.csv.gz")
        if not data_path.exists():
            pytest.skip("Test data not available")
        
        # Process data
        preprocessor = DataPreprocessor(
            raw_data_dir=Path("data/raw/minute_aggs/by_symbol"),
            processed_data_dir=Path("data/processed"),
            cache_dir=Path("data/cache")
        )
        processed = preprocessor.process('AAPL', ['2024_01'])
        
        # Add features
        feature_eng = FeatureEngine()
        # Add all features
        with_features = feature_eng.add_all_features(processed)
        
        return with_features
    
    def test_ma_crossover_with_costs(self, engine, test_data):
        """Test MA crossover strategy with realistic transaction costs"""
        strategy = MovingAverageCrossover({
            'fast_period': 20,
            'slow_period': 50,
            'position_size': 0.95,
            'ma_type': 'sma'
        })
        
        # Define realistic costs
        cost_engine = TransactionCostEngine(
            commission_model=CommissionModel(
                maker_fee=0.0016,  # 0.16% Interactive Brokers
                taker_fee=0.0016,
                min_fee=1.0
            )
        )
        
        # Run backtest with costs
        result = engine.run_backtest(
            strategy=strategy,
            data=test_data,
            initial_capital=100000,
            commission=0.001,  # $1 per trade
            slippage=0.0005    # 5 bps
        )
        
        # Verify results
        assert result.metrics['total_return'] is not None
        assert result.metrics['sharpe_ratio'] is not None
        assert result.metrics['trades_count'] >= 0
        
        # Check impact of costs
        if result.metrics['trades_count'] > 0:
            # Average cost per trade should be reasonable
            total_cost = abs(result.metrics.get('total_commission', 0) + 
                           result.metrics.get('total_slippage', 0))
            if result.metrics['trades_count'] > 0:
                avg_cost_per_trade = total_cost / result.metrics['trades_count']
                assert avg_cost_per_trade < 100, f"Cost per trade too high: ${avg_cost_per_trade:.2f}"
        
        print(f"\nMA Crossover Results (with costs):")
        print(f"  Total Return: {result.metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {result.metrics['max_drawdown']:.2%}")
        print(f"  Total Trades: {result.metrics['trades_count']}")
        if result.metrics['trades_count'] > 0:
            print(f"  Win Rate: {result.metrics.get('win_rate', 0):.1%}")
    
    def test_orb_intraday_strategy(self, engine, test_data):
        """Test Opening Range Breakout strategy with intraday data"""
        # Filter to market hours only
        market_data = test_data.between_time('09:30', '16:00')
        
        strategy = OpeningRangeBreakout({
            'opening_minutes': 30,
            'buffer_percent': 0.1,
            'stop_type': 'atr',
            'stop_atr_multiplier': 2.0,
            'profit_target_r': 3.0,
            'exit_at_close': True
        })
        
        # Run backtest
        result = engine.run_backtest(
            strategy=strategy,
            data=market_data,
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005
        )
        
        # Verify results
        assert result.metrics['total_return'] is not None
        
        # ORB should exit all positions by close
        if len(result.positions) > 0:
            # Check no overnight positions
            for date in result.positions.index.normalize().unique():
                day_positions = result.positions[result.positions.index.normalize() == date]
                if len(day_positions) > 0:
                    last_position = day_positions.iloc[-1]
                    # Should be flat at end of day
                    end_of_day = day_positions.index[-1].replace(hour=16, minute=0)
                    if day_positions.index[-1] >= end_of_day:
                        assert last_position == 0, f"Position not closed at EOD: {date}"
        
        print(f"\nORB Strategy Results:")
        print(f"  Total Return: {result.metrics['total_return']:.2%}")
        print(f"  Total Trades: {result.metrics['trades_count']}")
        print(f"  Avg Trade Duration: {result.metrics.get('avg_trade_duration', 0):.1f} bars")
    
    def test_multi_strategy_portfolio(self, engine, test_data):
        """Test running multiple strategies on same data"""
        strategies = {
            'MA_Fast': MovingAverageCrossover({
                'fast_period': 10,
                'slow_period': 30,
                'position_size': 0.5  # 50% allocation
            }),
            'MA_Slow': MovingAverageCrossover({
                'fast_period': 20,
                'slow_period': 50,
                'position_size': 0.5  # 50% allocation
            })
        }
        
        portfolio_results = {}
        combined_positions = pd.DataFrame()
        
        # Run each strategy
        for name, strategy in strategies.items():
            result = engine.run_backtest(
                strategy=strategy,
                data=test_data,
                initial_capital=50000  # Split capital
            )
            portfolio_results[name] = result
            
            # Combine positions
            if name not in combined_positions.columns:
                combined_positions[name] = result.positions
        
        # Calculate combined metrics
        total_capital = 100000
        combined_returns = pd.Series(0.0, index=test_data.index)
        
        for name, result in portfolio_results.items():
            weight = 0.5  # Equal weight
            strategy_returns = result.equity_curve.pct_change().fillna(0)
            combined_returns += strategy_returns * weight
        
        # Calculate portfolio metrics
        total_return = (1 + combined_returns).prod() - 1
        sharpe_ratio = np.sqrt(252 * 390) * combined_returns.mean() / combined_returns.std() if combined_returns.std() > 0 else 0
        
        print(f"\nMulti-Strategy Portfolio Results:")
        for name, result in portfolio_results.items():
            print(f"\n{name}:")
            print(f"  Return: {result.metrics['total_return']:.2%}")
            print(f"  Sharpe: {result.metrics['sharpe_ratio']:.2f}")
            print(f"  Trades: {result.metrics['trades_count']}")
        
        print(f"\nCombined Portfolio:")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    
    def test_parameter_optimization_grid(self, engine):
        """Test parameter optimization with grid search"""
        # Use smaller dataset for faster optimization
        data_path = Path("data/raw/minute_aggs/by_symbol/MSFT/MSFT_2024_01.csv.gz")
        if not data_path.exists():
            pytest.skip("MSFT data not available")
        
        preprocessor = DataPreprocessor(
            raw_data_dir=Path("data/raw/minute_aggs/by_symbol"),
            processed_data_dir=Path("data/processed"),
            cache_dir=Path("data/cache")
        )
        processed = preprocessor.process('MSFT', ['2024_01'])
        # Trim to first week only
        processed = processed.head(2000)
        
        # Add features
        feature_eng = FeatureEngine()
        data_with_features = feature_eng.add_moving_averages(
            processed,
            periods=[5, 10, 20, 30, 50]
        )
        
        # Define parameter grid
        param_grid = {
            'fast_period': [5, 10, 20],
            'slow_period': [20, 30, 50],
            'position_size': [0.8, 0.95],
            'ma_type': ['sma', 'ema']
        }
        
        strategy = MovingAverageCrossover()
        
        # Run optimization
        result = engine.optimize_parameters(
            strategy=strategy,
            data=data_with_features,
            parameter_grid=param_grid,
            metric='sharpe_ratio',
            initial_capital=100000
        )
        
        # Verify optimization results
        assert result.best_params is not None
        assert result.best_metric is not None
        assert len(result.results_df) > 0
        
        # Best parameters should be valid
        assert result.best_params['fast_period'] < result.best_params['slow_period']
        
        # Check distribution of results
        sharpe_ratios = result.results_df['sharpe_ratio']
        print(f"\nOptimization Results:")
        print(f"  Parameters tested: {len(result.results_df)}")
        print(f"  Best Sharpe Ratio: {result.best_metric:.2f}")
        print(f"  Best Parameters: {result.best_params}")
        print(f"  Sharpe Ratio Range: [{sharpe_ratios.min():.2f}, {sharpe_ratios.max():.2f}]")
        print(f"  Median Sharpe: {sharpe_ratios.median():.2f}")
    
    def test_position_sizing_methods(self, engine, test_data):
        """Test different position sizing methods"""
        position_methods = {
            'fixed': {'position_size': 0.95, 'position_sizing_method': 'fixed'},
            'volatility': {'position_size': 0.02, 'position_sizing_method': 'volatility'},
            'kelly': {'position_size': 0.25, 'position_sizing_method': 'kelly'}  # 1/4 Kelly
        }
        
        results = {}
        
        for method, params in position_methods.items():
            strategy = MovingAverageCrossover({
                'fast_period': 20,
                'slow_period': 50,
                **params
            })
            
            result = engine.run_backtest(
                strategy=strategy,
                data=test_data,
                initial_capital=100000
            )
            
            results[method] = result
            
            # Check position sizes are reasonable
            if len(result.positions) > 0:
                max_position = result.positions.abs().max()
                assert max_position > 0, f"No positions taken with {method} sizing"
                
                # Volatility sizing should adjust position sizes
                if method == 'volatility':
                    position_values = result.positions * test_data['close']
                    position_pcts = position_values / 100000
                    # Should target consistent risk
                    assert position_pcts.abs().std() < 0.5, "Volatility sizing not working"
        
        print(f"\nPosition Sizing Comparison:")
        for method, result in results.items():
            print(f"\n{method.title()} Sizing:")
            print(f"  Total Return: {result.metrics['total_return']:.2%}")
            print(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {result.metrics['max_drawdown']:.2%}")
            if len(result.positions) > 0:
                print(f"  Avg Position Size: {result.positions.abs().mean():.0f} shares")
    
    def test_risk_management_stops(self, engine, test_data):
        """Test stop loss and take profit functionality"""
        # Test different stop configurations
        stop_configs = [
            {'stop_loss_pct': 0.02, 'take_profit_pct': 0.05},  # 2% stop, 5% target
            {'stop_loss_pct': 0.01, 'take_profit_pct': 0.03},  # Tight stops
            {'stop_loss_pct': 0.05, 'take_profit_pct': 0.10},  # Wide stops
        ]
        
        results = []
        
        for config in stop_configs:
            strategy = MovingAverageCrossover({
                'fast_period': 20,
                'slow_period': 50,
                **config
            })
            
            result = engine.run_backtest(
                strategy=strategy,
                data=test_data,
                initial_capital=100000
            )
            
            results.append({
                'config': config,
                'return': result.metrics['total_return'],
                'sharpe': result.metrics['sharpe_ratio'],
                'max_dd': result.metrics['max_drawdown'],
                'trades': result.metrics['trades_count'],
                'win_rate': result.metrics.get('win_rate', 0)
            })
        
        print(f"\nRisk Management Comparison:")
        print(f"{'Stop Loss':>10} {'Take Profit':>12} {'Return':>10} {'Sharpe':>8} {'Win Rate':>10}")
        print("-" * 60)
        
        for r in results:
            stop = r['config']['stop_loss_pct']
            target = r['config']['take_profit_pct']
            print(f"{stop:>10.1%} {target:>12.1%} {r['return']:>10.2%} "
                  f"{r['sharpe']:>8.2f} {r['win_rate']:>10.1%}")
    
    def test_multi_timeframe_strategy(self, engine):
        """Test strategy using multiple timeframes"""
        # Load minute data
        minute_path = Path("data/raw/minute_aggs/by_symbol/SPY/SPY_2024_01.csv.gz")
        if not minute_path.exists():
            pytest.skip("SPY minute data not available")
        
        preprocessor = DataPreprocessor(
            raw_data_dir=Path("data/raw/minute_aggs/by_symbol"),
            processed_data_dir=Path("data/processed"),
            cache_dir=Path("data/cache")
        )
        minute_data = preprocessor.process('SPY', ['2024_01'])
        
        # Create multiple timeframes
        timeframes = {
            '5min': minute_data.resample('5min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna(),
            '15min': minute_data.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna(),
            '1H': minute_data.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        }
        
        # Add features to each timeframe
        feature_eng = FeatureEngine()
        for tf_name, tf_data in timeframes.items():
            timeframes[tf_name] = feature_eng.add_moving_averages(tf_data, [20, 50])
        
        # Simple multi-timeframe logic:
        # Trade on 5min, but only when 15min and 1H trend agree
        base_tf = timeframes['5min']
        
        # Get trend from higher timeframes
        tf_15min_trend = (timeframes['15min']['sma_20'] > timeframes['15min']['sma_50']).astype(int) * 2 - 1
        tf_1h_trend = (timeframes['1H']['sma_20'] > timeframes['1H']['sma_50']).astype(int) * 2 - 1
        
        # Reindex to 5min
        tf_15min_trend = tf_15min_trend.reindex(base_tf.index, method='ffill')
        tf_1h_trend = tf_1h_trend.reindex(base_tf.index, method='ffill')
        
        # Add to base timeframe
        base_tf['tf_15min_trend'] = tf_15min_trend
        base_tf['tf_1h_trend'] = tf_1h_trend
        
        print(f"\nMulti-Timeframe Analysis:")
        print(f"  5min bars: {len(timeframes['5min'])}")
        print(f"  15min bars: {len(timeframes['15min'])}")
        print(f"  1H bars: {len(timeframes['1H'])}")
        
        # Could implement a custom strategy that uses these trends
        # For now, just verify the data structure
        assert len(base_tf) > 0
        assert 'tf_15min_trend' in base_tf.columns
        assert 'tf_1h_trend' in base_tf.columns