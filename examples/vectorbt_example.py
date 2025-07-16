"""
Example: Running a backtest with VectorBT engine
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngine
from src.backtesting.engines.vectorbt_engine import VectorBTEngine
from src.strategies.base import BaseStrategy, SignalType, StrategyMetadata
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SimpleMovingAverageCrossover(BaseStrategy):
    """Simple MA crossover strategy for demonstration"""
    
    def __init__(self, parameters=None):
        """Initialize with default parameters"""
        default_params = {
            'fast_period': 10,
            'slow_period': 30,
            'stop_loss': 0.02,  # 2% stop loss
            'take_profit': 0.05  # 5% take profit
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on MA crossover"""
        # Calculate moving averages
        fast_ma = data['close'].rolling(self.parameters['fast_period']).mean()
        slow_ma = data['close'].rolling(self.parameters['slow_period']).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Long when fast MA crosses above slow MA
        signals[(fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))] = 1
        
        # Short when fast MA crosses below slow MA
        signals[(fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))] = -1
        
        # Forward fill signals to maintain position
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        return signals
    
    def calculate_positions(self, signals, capital, current_positions=None, risk_params=None):
        """Calculate position sizes"""
        # Simple fixed position sizing - 20% of capital per trade
        position_size = capital * 0.2
        return signals * position_size / 100  # Assuming $100 price for simplicity
    
    @property
    def required_history(self):
        return self.parameters['slow_period']
    
    @property
    def required_features(self):
        return []
    
    def _create_metadata(self):
        return StrategyMetadata(
            name="Simple MA Crossover",
            version="1.0",
            author="Example",
            description="Moving average crossover strategy",
            parameters=self.parameters,
            required_history=self.required_history,
            required_features=self.required_features
        )


def main():
    """Example of running a backtest with VectorBT"""
    
    # 1. Load and prepare data
    logger.info("Loading market data...")
    preprocessor = DataPreprocessor(
        raw_data_dir=Path("data/raw/minute_aggs/by_symbol"),
        processed_data_dir=Path("data/processed"),
        cache_dir=Path("data/cache")
    )
    
    # Load SPY data for January 2024
    data = preprocessor.load_processed("SPY", "2024_01")
    if data is None:
        logger.info("Processing SPY data...")
        data = preprocessor.process("SPY", months=["2024_01"])
    
    # 2. Add technical indicators
    logger.info("Adding technical indicators...")
    feature_engine = FeatureEngine()
    data_with_features = feature_engine.add_all_features(data)
    
    # 3. Initialize strategy
    strategy = SimpleMovingAverageCrossover(parameters={
        'fast_period': 10,
        'slow_period': 30
    })
    
    # 4. Initialize backtesting engine
    engine = VectorBTEngine(freq='1min')
    
    # 5. Run backtest
    logger.info("Running backtest...")
    result = engine.run_backtest(
        strategy=strategy,
        data=data_with_features,
        initial_capital=10000,
        commission=0.001,  # 0.1% commission
        slippage=0.0001   # 1 basis point slippage
    )
    
    # 6. Display results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    # Print performance report
    report = engine.generate_report(result, format='text')
    print(report)
    
    # Additional statistics
    print("\nTrade Analysis:")
    if len(result.trades) > 0:
        print(f"  First Trade: {result.trades.iloc[0]['entry_time']}")
        print(f"  Last Trade: {result.trades.iloc[-1]['exit_time']}")
        print(f"  Best Trade: {result.trades['return_pct'].max():.2%}")
        print(f"  Worst Trade: {result.trades['return_pct'].min():.2%}")
        print(f"  Average Trade Duration: {(result.trades['exit_time'] - result.trades['entry_time']).mean()}")
    
    # 7. Parameter optimization example
    logger.info("\nRunning parameter optimization...")
    param_grid = {
        'fast_period': [5, 10, 15, 20],
        'slow_period': [20, 30, 40, 50]
    }
    
    optimization_result = engine.optimize_parameters(
        strategy_class=SimpleMovingAverageCrossover,
        data=data_with_features,
        param_grid=param_grid,
        metric='sharpe_ratio',
        initial_capital=10000
    )
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best Parameters: {optimization_result.best_params}")
    print(f"Best Sharpe Ratio: {optimization_result.best_metric:.2f}")
    
    # Show top 5 parameter combinations
    print("\nTop 5 Parameter Combinations:")
    top_5 = optimization_result.results_df.nlargest(5, 'sharpe_ratio')
    print(top_5[['fast_period', 'slow_period', 'sharpe_ratio', 'total_return']].to_string())
    
    # 8. Run backtest with optimized parameters
    logger.info("\nRunning backtest with optimized parameters...")
    optimized_strategy = SimpleMovingAverageCrossover(parameters=optimization_result.best_params)
    
    optimized_result = engine.run_backtest(
        strategy=optimized_strategy,
        data=data_with_features,
        initial_capital=10000,
        commission=0.001
    )
    
    print("\n" + "="*60)
    print("OPTIMIZED STRATEGY RESULTS")
    print("="*60)
    print(engine.generate_report(optimized_result, format='text'))
    
    # Save results
    output_dir = Path("results/backtests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save equity curve
    equity_curve_path = output_dir / "ma_crossover_equity_curve.csv"
    optimized_result.equity_curve.to_csv(equity_curve_path)
    print(f"\nEquity curve saved to: {equity_curve_path}")
    
    # Save trades
    if len(optimized_result.trades) > 0:
        trades_path = output_dir / "ma_crossover_trades.csv"
        optimized_result.trades.to_csv(trades_path, index=False)
        print(f"Trades saved to: {trades_path}")


if __name__ == "__main__":
    main()