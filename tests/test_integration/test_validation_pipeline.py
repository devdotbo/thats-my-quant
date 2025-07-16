"""
Validation Pipeline Integration Tests
Tests walk-forward and Monte Carlo validation workflows
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

from src.validation.walk_forward import WalkForwardValidator, WindowType
from src.validation.monte_carlo import MonteCarloValidator, ResamplingMethod
from src.strategies.examples.moving_average import MovingAverageCrossover
from src.backtesting.engines.vectorbt_engine import VectorBTEngine
from src.data.preprocessor import DataPreprocessor
from src.data.features import FeatureEngine


@pytest.mark.integration
class TestValidationPipeline:
    """Test validation frameworks with real data and strategies"""
    
    @pytest.fixture
    def prepared_data(self):
        """Load and prepare test data"""
        data_path = Path("data/raw/minute_aggs/by_symbol/AAPL/AAPL_2024_01.csv.gz")
        if not data_path.exists():
            pytest.skip("Test data not available")
        
        raw_data = pd.read_csv(data_path, compression='gzip')
        
        preprocessor = DataPreprocessor(
            raw_data_dir=Path("data/raw"),
            processed_data_dir=Path("data/processed"),
            cache_dir=Path("data/cache")
        )
        processed = preprocessor.process(raw_data, 'AAPL')
        
        # Add necessary features
        feature_eng = FeatureEngineer()
        with_features = feature_eng.add_all_features(
            processed,
            features=['sma_20', 'sma_50', 'rsi_14', 'atr_14']
        )
        
        return with_features
    
    @pytest.fixture
    def strategy(self):
        """Create test strategy"""
        return MovingAverageCrossover({
            'fast_period': 20,
            'slow_period': 50,
            'position_size': 0.95
        })
    
    @pytest.fixture
    def engine(self):
        """Create backtesting engine"""
        return VectorBTEngine()
    
    def test_walk_forward_complete_workflow(self, prepared_data, strategy, engine):
        """Test complete walk-forward validation workflow"""
        # Create validator
        validator = WalkForwardValidator(
            engine=engine,
            train_periods=10,  # 10 days training
            test_periods=5,    # 5 days testing
            step_size=5,       # 5 days step
            window_type=WindowType.ROLLING
        )
        
        # Create windows
        windows = validator.create_windows(
            prepared_data,
            start_date=prepared_data.index[0],
            end_date=prepared_data.index[-1]
        )
        
        print(f"Created {len(windows)} walk-forward windows")
        assert len(windows) >= 2, "Need at least 2 windows for meaningful validation"
        
        # Define parameter ranges
        param_ranges = {
            'fast_period': [10, 15, 20],
            'slow_period': [40, 50, 60]
        }
        
        # Run validation
        results = validator.run_validation(
            strategy=strategy,
            data=prepared_data,
            windows=windows[:3],  # Just first 3 windows for speed
            optimization_metric='sharpe_ratio',
            parameter_ranges=param_ranges,
            initial_capital=100000
        )
        
        # Verify results structure
        assert results.summary_stats is not None
        assert len(results.window_results) == 3
        assert results.summary_stats['total_windows'] == 3
        
        # Check for overfitting
        overfitting_analysis = validator.analyze_overfitting(results)
        
        print("\nWalk-Forward Results:")
        print(f"  Windows Tested: {results.summary_stats['total_windows']}")
        print(f"  Avg IS Sharpe: {results.summary_stats['avg_in_sample_sharpe']:.2f}")
        print(f"  Avg OOS Sharpe: {results.summary_stats['avg_out_sample_sharpe']:.2f}")
        print(f"  Overfitting Score: {overfitting_analysis['overfitting_score']:.2f}")
        print(f"  Parameter Stability: {overfitting_analysis['parameter_stability']:.2f}")
        
        # Parameter stability should be reasonable
        assert overfitting_analysis['parameter_stability'] >= 0, "Invalid stability score"
        
        # Export results
        export_path = Path("test_wf_results.json")
        validator.export_results(results, export_path)
        assert export_path.exists()
        
        # Load and verify exported results
        with open(export_path, 'r') as f:
            exported = json.load(f)
        assert 'summary_stats' in exported
        assert 'window_results' in exported
        
        # Cleanup
        export_path.unlink()
    
    def test_monte_carlo_complete_workflow(self, prepared_data, strategy, engine):
        """Test complete Monte Carlo validation workflow"""
        # First run a backtest to get results
        backtest_result = engine.run_backtest(
            strategy=strategy,
            data=prepared_data,
            initial_capital=100000,
            commission=0.001
        )
        
        # Create validator
        validator = MonteCarloValidator(
            n_simulations=500,
            confidence_level=0.95,
            random_seed=42
        )
        
        # Test different resampling methods
        methods = [
            ResamplingMethod.BOOTSTRAP,
            ResamplingMethod.BLOCK_BOOTSTRAP,
            ResamplingMethod.STATIONARY_BOOTSTRAP
        ]
        
        results = {}
        
        for method in methods:
            print(f"\nTesting {method.value} resampling...")
            
            mc_result = validator.run_simulation(
                backtest_result=backtest_result,
                resampling_method=method,
                block_size=20 if method == ResamplingMethod.BLOCK_BOOTSTRAP else None,
                parallel=True,
                n_jobs=4
            )
            
            results[method.value] = mc_result
            
            # Verify results
            assert mc_result.n_simulations == 500
            assert len(mc_result.simulation_metrics) == 500
            assert 'sharpe_ratio' in mc_result.confidence_intervals
            assert mc_result.risk_metrics['risk_of_ruin'] >= 0
            
            # Print summary
            print(f"  Sharpe CI: [{mc_result.confidence_intervals['sharpe_ratio']['lower']:.2f}, "
                  f"{mc_result.confidence_intervals['sharpe_ratio']['upper']:.2f}]")
            print(f"  Risk of Ruin: {mc_result.risk_metrics['risk_of_ruin']:.1%}")
            print(f"  95% Max DD: {mc_result.risk_metrics['max_drawdown_95_percentile']:.2%}")
        
        # Compare methods
        print("\nResampling Method Comparison:")
        print(f"{'Method':<20} {'Sharpe Mean':>12} {'Sharpe Std':>12} {'RoR':>8}")
        print("-" * 55)
        
        for method, result in results.items():
            sharpe_mean = result.summary_statistics['mean']['sharpe_ratio']
            sharpe_std = result.summary_statistics['std']['sharpe_ratio']
            ror = result.risk_metrics['risk_of_ruin']
            print(f"{method:<20} {sharpe_mean:>12.2f} {sharpe_std:>12.2f} {ror:>8.1%}")
    
    def test_walk_forward_anchored_windows(self, prepared_data, strategy, engine):
        """Test walk-forward with anchored (expanding) windows"""
        validator = WalkForwardValidator(
            engine=engine,
            train_periods=15,
            test_periods=5,
            step_size=5,
            window_type=WindowType.ANCHORED
        )
        
        windows = validator.create_windows(prepared_data)
        
        # Verify anchored windows expand
        for i in range(1, len(windows)):
            assert windows[i].train_start == windows[0].train_start, "Anchored window should have same start"
            assert windows[i].train_end > windows[i-1].train_end, "Training period should expand"
        
        # Run on subset
        results = validator.run_validation(
            strategy=strategy,
            data=prepared_data,
            windows=windows[:2],
            optimization_metric='total_return',
            parameter_ranges={'fast_period': [15, 20], 'slow_period': [45, 50]}
        )
        
        print(f"\nAnchored Walk-Forward:")
        print(f"  Window 1 train size: {(windows[0].train_end - windows[0].train_start).days} days")
        print(f"  Window 2 train size: {(windows[1].train_end - windows[1].train_start).days} days")
    
    def test_monte_carlo_statistical_tests(self, prepared_data, strategy, engine):
        """Test Monte Carlo statistical significance testing"""
        # Run two different strategies
        strategy1 = MovingAverageCrossover({'fast_period': 20, 'slow_period': 50})
        strategy2 = MovingAverageCrossover({'fast_period': 10, 'slow_period': 30})
        
        result1 = engine.run_backtest(strategy1, prepared_data, initial_capital=100000)
        result2 = engine.run_backtest(strategy2, prepared_data, initial_capital=100000)
        
        # Run Monte Carlo on both
        validator = MonteCarloValidator(n_simulations=200, random_seed=42)
        
        mc1 = validator.run_simulation(result1, ResamplingMethod.BOOTSTRAP)
        mc2 = validator.run_simulation(result2, ResamplingMethod.BOOTSTRAP)
        
        # Test statistical significance
        p_value = validator.test_significance(
            mc1.simulation_metrics,
            mc2.simulation_metrics,
            metric='sharpe_ratio'
        )
        
        print(f"\nStatistical Significance Test:")
        print(f"  Strategy 1 Sharpe: {mc1.summary_statistics['mean']['sharpe_ratio']:.2f}")
        print(f"  Strategy 2 Sharpe: {mc2.summary_statistics['mean']['sharpe_ratio']:.2f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")
        
        assert 0 <= p_value <= 1, "Invalid p-value"
    
    def test_validation_with_parameter_stability(self, prepared_data, strategy, engine):
        """Test parameter stability across walk-forward windows"""
        validator = WalkForwardValidator(
            engine=engine,
            train_periods=10,
            test_periods=5,
            step_size=5,
            window_type=WindowType.ROLLING
        )
        
        # Create windows
        windows = validator.create_windows(prepared_data)[:5]  # First 5 windows
        
        # Run with tracking parameter changes
        param_ranges = {
            'fast_period': list(range(10, 25, 5)),
            'slow_period': list(range(40, 60, 5))
        }
        
        results = validator.run_validation(
            strategy=strategy,
            data=prepared_data,
            windows=windows,
            optimization_metric='sharpe_ratio',
            parameter_ranges=param_ranges
        )
        
        # Analyze parameter stability
        stability_analysis = validator.analyze_parameter_stability(results)
        
        print(f"\nParameter Stability Analysis:")
        for param, stability in stability_analysis.items():
            print(f"  {param}:")
            print(f"    Changes: {stability['changes']}")
            print(f"    Unique Values: {stability['unique_values']}")
            print(f"    Stability Score: {stability['stability_score']:.2f}")
        
        # Check that we tracked parameters correctly
        assert 'fast_period' in stability_analysis
        assert 'slow_period' in stability_analysis
    
    def test_monte_carlo_with_custom_metrics(self, prepared_data, strategy, engine):
        """Test Monte Carlo with custom performance metrics"""
        # Run backtest
        result = engine.run_backtest(
            strategy=strategy,
            data=prepared_data,
            initial_capital=100000
        )
        
        # Define custom metrics function
        def custom_metrics(equity_curve: pd.Series, trades: pd.DataFrame) -> dict:
            """Calculate custom metrics"""
            returns = equity_curve.pct_change().dropna()
            
            # Calmar ratio
            if returns.std() > 0:
                annual_return = (1 + returns.mean()) ** 252 - 1
                max_dd = (equity_curve / equity_curve.cummax() - 1).min()
                calmar = annual_return / abs(max_dd) if max_dd < 0 else 0
            else:
                calmar = 0
            
            # Omega ratio
            threshold = 0
            gains = returns[returns > threshold] - threshold
            losses = threshold - returns[returns <= threshold]
            omega = gains.sum() / losses.sum() if losses.sum() > 0 else np.inf
            
            return {
                'calmar_ratio': calmar,
                'omega_ratio': min(omega, 10),  # Cap at 10 for stability
                'positive_days': (returns > 0).mean()
            }
        
        # Run Monte Carlo with custom metrics
        validator = MonteCarloValidator(
            n_simulations=100,
            confidence_level=0.95,
            random_seed=42,
            custom_metrics=custom_metrics
        )
        
        mc_result = validator.run_simulation(
            result,
            ResamplingMethod.BOOTSTRAP,
            parallel=True
        )
        
        # Verify custom metrics were calculated
        assert 'calmar_ratio' in mc_result.confidence_intervals
        assert 'omega_ratio' in mc_result.confidence_intervals
        assert 'positive_days' in mc_result.confidence_intervals
        
        print(f"\nCustom Metrics Results:")
        for metric in ['calmar_ratio', 'omega_ratio', 'positive_days']:
            ci = mc_result.confidence_intervals[metric]
            print(f"  {metric}: [{ci['lower']:.2f}, {ci['upper']:.2f}]")
    
    def test_export_import_validation_results(self, prepared_data, strategy, engine, tmp_path):
        """Test exporting and importing validation results"""
        # Run walk-forward
        wf_validator = WalkForwardValidator(
            engine=engine,
            train_periods=10,
            test_periods=5,
            step_size=5
        )
        
        windows = wf_validator.create_windows(prepared_data)[:2]
        wf_results = wf_validator.run_validation(
            strategy=strategy,
            data=prepared_data,
            windows=windows,
            optimization_metric='sharpe_ratio',
            parameter_ranges={'fast_period': [15, 20], 'slow_period': [45, 50]}
        )
        
        # Export walk-forward results
        wf_export_path = tmp_path / "wf_results.json"
        wf_validator.export_results(wf_results, wf_export_path)
        
        # Run Monte Carlo
        backtest_result = engine.run_backtest(strategy, prepared_data, 100000)
        mc_validator = MonteCarloValidator(n_simulations=50)
        mc_results = mc_validator.run_simulation(backtest_result, ResamplingMethod.BOOTSTRAP)
        
        # Export Monte Carlo results
        mc_export_path = tmp_path / "mc_results.json"
        mc_validator.export_results(mc_results, mc_export_path)
        
        # Verify files exist
        assert wf_export_path.exists()
        assert mc_export_path.exists()
        
        # Load and verify walk-forward results
        with open(wf_export_path, 'r') as f:
            wf_loaded = json.load(f)
        
        assert 'summary_stats' in wf_loaded
        assert 'window_results' in wf_loaded
        assert wf_loaded['summary_stats']['total_windows'] == 2
        
        # Load and verify Monte Carlo results
        with open(mc_export_path, 'r') as f:
            mc_loaded = json.load(f)
        
        assert 'n_simulations' in mc_loaded
        assert 'confidence_intervals' in mc_loaded
        assert mc_loaded['n_simulations'] == 50
        
        print(f"\nExport/Import Test:")
        print(f"  WF results size: {wf_export_path.stat().st_size / 1024:.1f} KB")
        print(f"  MC results size: {mc_export_path.stat().st_size / 1024:.1f} KB")
    
    def test_combined_validation_workflow(self, prepared_data, strategy, engine):
        """Test combining walk-forward and Monte Carlo validation"""
        # Step 1: Walk-forward to find robust parameters
        wf_validator = WalkForwardValidator(
            engine=engine,
            train_periods=10,
            test_periods=5,
            step_size=5
        )
        
        windows = wf_validator.create_windows(prepared_data)[:3]
        wf_results = wf_validator.run_validation(
            strategy=strategy,
            data=prepared_data,
            windows=windows,
            optimization_metric='sharpe_ratio',
            parameter_ranges={
                'fast_period': [10, 15, 20],
                'slow_period': [40, 50, 60]
            }
        )
        
        # Get most stable parameters
        stability = wf_validator.analyze_parameter_stability(wf_results)
        
        # Step 2: Use stable parameters for final backtest
        # Find most common parameter values
        best_params = {}
        for window_result in wf_results.window_results:
            for param, value in window_result['best_params'].items():
                if param not in best_params:
                    best_params[param] = []
                best_params[param].append(value)
        
        # Use mode (most common value)
        from statistics import mode
        stable_params = {
            param: mode(values) for param, values in best_params.items()
        }
        
        stable_strategy = MovingAverageCrossover(stable_params)
        final_result = engine.run_backtest(
            stable_strategy,
            prepared_data,
            initial_capital=100000
        )
        
        # Step 3: Monte Carlo on final strategy
        mc_validator = MonteCarloValidator(
            n_simulations=200,
            confidence_level=0.95
        )
        
        mc_results = mc_validator.run_simulation(
            final_result,
            ResamplingMethod.STATIONARY_BOOTSTRAP
        )
        
        print(f"\nCombined Validation Results:")
        print(f"  Walk-Forward Windows: {len(windows)}")
        print(f"  Stable Parameters: {stable_params}")
        print(f"  Final Sharpe: {final_result.metrics['sharpe_ratio']:.2f}")
        print(f"  MC 95% CI: [{mc_results.confidence_intervals['sharpe_ratio']['lower']:.2f}, "
              f"{mc_results.confidence_intervals['sharpe_ratio']['upper']:.2f}]")
        print(f"  Risk of Ruin: {mc_results.risk_metrics['risk_of_ruin']:.1%}")