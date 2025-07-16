"""
Tests for VectorBT backtesting engine wrapper
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock

from src.backtesting.engines.vectorbt_engine import VectorBTEngine
from src.strategies.base import BaseStrategy, SignalType, StrategyMetadata


class SimpleTestStrategy(BaseStrategy):
    """Simple strategy for testing"""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate simple MA crossover signals"""
        fast_ma = data['close'].rolling(10).mean()
        slow_ma = data['close'].rolling(20).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        
        return signals
    
    def calculate_positions(self, signals: pd.Series, capital: float,
                          current_positions=None, risk_params=None) -> pd.Series:
        """Convert signals to positions"""
        # Simple fixed position sizing
        position_value = capital * 0.1  # 10% per position
        prices = signals.index.to_series().map(lambda x: 100)  # Dummy price
        shares = position_value / prices
        
        return signals * shares
    
    @property
    def required_history(self) -> int:
        return 20
    
    @property
    def required_features(self) -> list:
        return []
    
    def _create_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="Simple Test Strategy",
            version="1.0",
            author="Test",
            description="Simple MA crossover for testing",
            parameters={"fast": 10, "slow": 20},
            required_history=20,
            required_features=[]
        )


class TestVectorBTEngine:
    """Test VectorBT engine functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data"""
        dates = pd.date_range('2024-01-02 09:30:00', periods=390, freq='1min', tz='America/New_York')
        
        # Generate trending data
        np.random.seed(42)
        trend = np.linspace(100, 110, 390)
        noise = np.random.randn(390) * 0.5
        close_prices = trend + noise
        
        return pd.DataFrame({
            'open': close_prices + np.random.randn(390) * 0.1,
            'high': close_prices + np.abs(np.random.randn(390) * 0.2),
            'low': close_prices - np.abs(np.random.randn(390) * 0.2),
            'close': close_prices,
            'volume': np.random.randint(50000, 150000, 390)
        }, index=dates)
    
    @pytest.fixture
    def test_strategy(self):
        """Create test strategy instance"""
        return SimpleTestStrategy()
    
    @pytest.fixture
    def engine(self):
        """Create VectorBT engine instance"""
        return VectorBTEngine()
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert hasattr(engine, 'run_backtest')
        assert hasattr(engine, 'calculate_metrics')
        assert hasattr(engine, 'optimize_parameters')
    
    def test_basic_backtest(self, engine, sample_data, test_strategy):
        """Test basic backtest execution"""
        result = engine.run_backtest(
            strategy=test_strategy,
            data=sample_data,
            initial_capital=10000,
            commission=0.001
        )
        
        # Check result structure
        assert hasattr(result, 'portfolio')
        assert hasattr(result, 'trades')
        assert hasattr(result, 'metrics')
        
        # Check portfolio has expected attributes
        portfolio = result.portfolio
        assert hasattr(portfolio, 'total_return')  # This is a method
        assert hasattr(result, 'equity_curve')  # This is on result
        assert hasattr(result, 'positions')  # This is on result
    
    def test_backtest_with_costs(self, engine, sample_data, test_strategy):
        """Test backtest with transaction costs"""
        # Run with different commission levels
        result_low = engine.run_backtest(
            strategy=test_strategy,
            data=sample_data,
            initial_capital=10000,
            commission=0.0001  # 1 basis point
        )
        
        result_high = engine.run_backtest(
            strategy=test_strategy,
            data=sample_data,
            initial_capital=10000,
            commission=0.01  # 100 basis points
        )
        
        # Higher costs should reduce returns
        assert result_low.metrics['total_return'] > result_high.metrics['total_return']
        assert result_low.metrics['sharpe_ratio'] > result_high.metrics['sharpe_ratio']
    
    def test_calculate_metrics(self, engine, sample_data, test_strategy):
        """Test metrics calculation"""
        result = engine.run_backtest(
            strategy=test_strategy,
            data=sample_data,
            initial_capital=10000
        )
        
        metrics = result.metrics
        
        # Check all expected metrics are present
        expected_metrics = [
            'total_return', 'annualized_return', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'win_rate',
            'profit_factor', 'trades_count', 'avg_trade_return'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert metrics[metric] is not None
        
        # Validate metric ranges
        assert -1 <= metrics['total_return'] <= 10  # Reasonable return range
        assert 0 <= metrics['max_drawdown'] <= 1  # Drawdown is positive
        assert 0 <= metrics['win_rate'] <= 1  # Win rate is percentage
    
    def test_multi_asset_backtest(self, engine):
        """Test backtesting with multiple assets"""
        # Create multi-asset data
        dates = pd.date_range('2024-01-02', periods=100, freq='D')
        assets = ['SPY', 'QQQ', 'IWM']
        
        data = {}
        for i, asset in enumerate(assets):
            base_price = 100 + i * 50
            prices = base_price + np.cumsum(np.random.randn(100) * 1)
            
            data[asset] = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, 100)
            }, index=dates)
        
        # Create multi-asset strategy signals
        signals = pd.DataFrame({
            asset: pd.Series(np.random.choice([-1, 0, 1], 100), index=dates)
            for asset in assets
        })
        
        result = engine.run_multi_asset_backtest(
            data=data,
            signals=signals,
            initial_capital=100000,
            weights={'SPY': 0.5, 'QQQ': 0.3, 'IWM': 0.2}
        )
        
        # Check multi-asset results
        assert hasattr(result, 'portfolio')
        assert hasattr(result, 'asset_returns')
        assert len(result.asset_returns) == 3
    
    def test_parameter_optimization(self, engine, sample_data):
        """Test parameter optimization functionality"""
        # Define parameter grid
        param_grid = {
            'fast_period': [5, 10, 15],
            'slow_period': [20, 30, 40]
        }
        
        # Run optimization
        optimization_result = engine.optimize_parameters(
            strategy_class=SimpleTestStrategy,
            data=sample_data,
            param_grid=param_grid,
            metric='sharpe_ratio',
            initial_capital=10000
        )
        
        # Check optimization results
        assert hasattr(optimization_result, 'best_params')
        assert hasattr(optimization_result, 'results_df')
        assert hasattr(optimization_result, 'best_metric')
        
        # Best params should be from the grid
        assert optimization_result.best_params['fast_period'] in param_grid['fast_period']
        assert optimization_result.best_params['slow_period'] in param_grid['slow_period']
        
        # Results dataframe should have all combinations
        expected_combinations = len(param_grid['fast_period']) * len(param_grid['slow_period'])
        assert len(optimization_result.results_df) == expected_combinations
    
    def test_position_sizing_methods(self, engine, sample_data):
        """Test different position sizing methods"""
        strategy = SimpleTestStrategy()
        
        # Test fixed position sizing
        result_fixed = engine.run_backtest(
            strategy=strategy,
            data=sample_data,
            initial_capital=10000,
            position_size='fixed',
            position_size_params={'size_pct': 0.1}
        )
        
        # Test volatility-based sizing
        result_vol = engine.run_backtest(
            strategy=strategy,
            data=sample_data,
            initial_capital=10000,
            position_size='volatility',
            position_size_params={'target_vol': 0.02}
        )
        
        # Both should produce valid results
        assert result_fixed.metrics['trades_count'] > 0
        assert result_vol.metrics['trades_count'] > 0
    
    def test_equity_curve_analysis(self, engine, sample_data, test_strategy):
        """Test equity curve and drawdown analysis"""
        result = engine.run_backtest(
            strategy=test_strategy,
            data=sample_data,
            initial_capital=10000
        )
        
        equity_curve = result.equity_curve
        drawdown_series = result.portfolio.drawdown_series
        
        # Equity curve validations
        assert len(equity_curve) == len(sample_data)
        assert equity_curve.iloc[0] == 10000  # Should start at initial capital
        assert (equity_curve >= 0).all()  # No negative equity
        
        # Drawdown validations
        assert len(drawdown_series) == len(sample_data)
        assert (drawdown_series <= 0).all()  # Drawdowns are negative
        assert drawdown_series.iloc[0] == 0  # No drawdown at start
    
    def test_trade_analysis(self, engine, sample_data, test_strategy):
        """Test trade-level analysis"""
        result = engine.run_backtest(
            strategy=test_strategy,
            data=sample_data,
            initial_capital=10000
        )
        
        trades = result.trades
        
        if len(trades) > 0:
            # Check trade structure - only check columns we actually create
            expected_columns = [
                'entry_time', 'exit_time', 'symbol', 'side',
                'size', 'pnl', 'return_pct', 'commission'
            ]
            
            for col in expected_columns:
                assert col in trades.columns
            
            # Validate trade data
            assert (trades['size'] != 0).all()  # No zero-size trades
            assert trades['entry_time'].lt(trades['exit_time']).all()  # Entry before exit
    
    def test_generate_report(self, engine, sample_data, test_strategy):
        """Test report generation"""
        result = engine.run_backtest(
            strategy=test_strategy,
            data=sample_data,
            initial_capital=10000
        )
        
        # Generate text report
        report = engine.generate_report(result, format='text')
        
        # Check report contains key information
        assert 'Total Return' in report
        assert 'Sharpe Ratio' in report
        assert 'Max Drawdown' in report
        assert 'Number of Trades' in report
        
        # Generate dict report
        report_dict = engine.generate_report(result, format='dict')
        assert isinstance(report_dict, dict)
        assert 'metrics' in report_dict
        assert 'summary' in report_dict


@pytest.mark.performance
class TestVectorBTEnginePerformance:
    """Test performance of VectorBT engine"""
    
    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing"""
        # One year of minute data
        dates = pd.date_range('2024-01-02 09:30:00', periods=98280, freq='1min', tz='America/New_York')
        
        # Filter to market hours
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
            'volume': np.random.randint(50000, 150000, len(dates))
        }, index=dates)
    
    def test_backtest_performance(self, large_dataset):
        """Test that backtest completes in reasonable time"""
        import time
        
        engine = VectorBTEngine()
        strategy = SimpleTestStrategy()
        
        start_time = time.perf_counter()
        
        result = engine.run_backtest(
            strategy=strategy,
            data=large_dataset,
            initial_capital=100000
        )
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        print(f"\nBacktest Performance:")
        print(f"- Data points: {len(large_dataset)}")
        print(f"- Time: {elapsed_time:.2f} seconds")
        print(f"- Rate: {len(large_dataset) / elapsed_time:.0f} bars/second")
        
        # Should complete within 5 seconds
        assert elapsed_time < 5.0, f"Backtest took {elapsed_time:.2f}s, target is <5s"
        
        # Should produce valid results
        assert result.metrics['trades_count'] > 0
    
    def test_optimization_performance(self):
        """Test parameter optimization performance"""
        import time
        
        # Create smaller dataset for optimization
        dates = pd.date_range('2024-01-02', periods=252, freq='D')
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(252) * 1),
            'volume': np.random.randint(1000000, 5000000, 252)
        }, index=dates)
        
        engine = VectorBTEngine()
        
        # 1000 parameter combinations
        param_grid = {
            'fast_period': list(range(5, 20, 2)),  # 8 values
            'slow_period': list(range(20, 100, 5)),  # 16 values
            'threshold': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]  # 8 values
        }
        # Total: 8 * 16 * 8 = 1024 combinations
        
        start_time = time.perf_counter()
        
        result = engine.optimize_parameters(
            strategy_class=SimpleTestStrategy,
            data=data,
            param_grid=param_grid,
            metric='sharpe_ratio',
            initial_capital=100000
        )
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        total_combinations = 8 * 16 * 8
        print(f"\nOptimization Performance:")
        print(f"- Parameter combinations: {total_combinations}")
        print(f"- Time: {elapsed_time:.2f} seconds")
        print(f"- Rate: {total_combinations / elapsed_time:.1f} combinations/second")
        
        # Should complete within 30 seconds
        assert elapsed_time < 30.0, f"Optimization took {elapsed_time:.2f}s, target is <30s"