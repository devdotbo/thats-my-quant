#!/usr/bin/env python3
"""
Bitcoin Lunar Trading Analysis
Downloads Bitcoin data and analyzes lunar cycle correlations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.data.bitcoin_downloader import BitcoinDownloader
from src.features.lunar_features import LunarCalculator, create_lunar_trading_signals
from src.strategies.crypto.lunar_btc import LunarBitcoinStrategy
from src.backtesting.engines.vectorbt_engine import VectorBTEngine


def download_bitcoin_data(period: str = "2y", interval: str = "1h") -> pd.DataFrame:
    """Download Bitcoin data using yfinance"""
    print("Downloading Bitcoin data from Yahoo Finance...")
    print(f"Period: {period}, Interval: {interval}")
    
    downloader = BitcoinDownloader()
    btc_data = downloader.download(period=period, interval=interval)
    
    print(f"Downloaded {len(btc_data)} rows")
    print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
    print(f"Price range: ${btc_data['close'].min():.2f} - ${btc_data['close'].max():.2f}")
    
    return btc_data


def add_lunar_features(btc_data: pd.DataFrame) -> pd.DataFrame:
    """Add lunar features to Bitcoin data"""
    print("\nCalculating lunar features...")
    
    lunar_calc = LunarCalculator()
    btc_data = lunar_calc.add_lunar_features_to_data(btc_data)
    
    # Show lunar events in the data
    lunar_events = btc_data[btc_data['is_new_moon'] | btc_data['is_full_moon']]
    print(f"Found {len(lunar_events)} lunar events (new/full moons)")
    
    return btc_data


def analyze_lunar_correlation(btc_data: pd.DataFrame) -> pd.DataFrame:
    """Analyze correlation between Bitcoin returns and lunar phases"""
    print("\nAnalyzing Bitcoin-Lunar Correlations...")
    
    # Calculate returns if not present
    if 'returns' not in btc_data.columns:
        btc_data['returns'] = btc_data['close'].pct_change()
    
    # Group by moon phase
    phase_analysis = btc_data.groupby('phase_name').agg({
        'returns': ['mean', 'std', 'count'],
        'volatility': 'mean',
        'volume': 'mean'
    })
    
    # Convert to percentage
    phase_analysis[('returns', 'mean')] *= 100
    phase_analysis[('returns', 'std')] *= 100
    
    print("\nAverage Bitcoin returns by moon phase:")
    print(phase_analysis)
    
    # Analyze days around full moon
    full_moon_analysis = []
    full_moon_days = btc_data[btc_data['is_full_moon']].index
    
    for fm_date in full_moon_days:
        # Get returns for -3 to +5 days around full moon
        fm_idx = btc_data.index.get_loc(fm_date)
        
        for offset in range(-3, 6):
            if 0 <= fm_idx + offset < len(btc_data):
                row = btc_data.iloc[fm_idx + offset]
                full_moon_analysis.append({
                    'days_from_full_moon': offset,
                    'returns': row['returns'],
                    'volatility': row['volatility']
                })
    
    fm_df = pd.DataFrame(full_moon_analysis)
    fm_summary = fm_df.groupby('days_from_full_moon').agg({
        'returns': ['mean', 'std'],
        'volatility': 'mean'
    })
    
    print("\nReturns around full moon (day 0 = full moon):")
    print(fm_summary)
    
    return phase_analysis


def backtest_lunar_strategies(btc_data: pd.DataFrame) -> dict:
    """Backtest different lunar trading strategies"""
    print("\nBacktesting Lunar Trading Strategies...")
    
    # Initialize backtesting engine
    engine = VectorBTEngine()
    
    # Test different strategy types
    strategy_types = ['classic', 'momentum', 'distance', 'combined']
    results = {}
    
    for strategy_type in strategy_types:
        print(f"\nTesting {strategy_type} strategy...")
        
        # Create strategy
        strategy = LunarBitcoinStrategy({
            'strategy_type': strategy_type,
            'hold_days': 3,
            'stop_loss': 0.03,  # 3% stop loss for crypto
            'min_volume_percentile': 20
        })
        
        try:
            # Run backtest
            result = engine.run_backtest(
                strategy, 
                btc_data,
                initial_capital=10000,
                commission=0.001  # 0.1% for crypto exchanges
            )
            
            results[strategy_type] = result
            
            print(f"  Total Return: {result.metrics['total_return']:.1%}")
            print(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {result.metrics['max_drawdown']:.1%}")
            print(f"  Win Rate: {result.metrics['win_rate']:.1%}")
            print(f"  Number of Trades: {result.metrics['num_trades']}")
            
        except Exception as e:
            print(f"  Error: {e}")
            
    return results


def create_visualizations(btc_data: pd.DataFrame, results: dict):
    """Create visualization of lunar cycles and Bitcoin performance"""
    print("\nCreating visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Plot 1: Bitcoin price with lunar events
    ax1 = axes[0]
    ax1.plot(btc_data.index, btc_data['close'], 'b-', alpha=0.7, linewidth=1)
    
    # Mark full moons
    full_moons = btc_data[btc_data['is_full_moon']]
    ax1.scatter(full_moons.index, full_moons['close'], 
               color='gold', s=80, marker='o', label='Full Moon', 
               edgecolors='orange', linewidth=2, zorder=5)
    
    # Mark new moons
    new_moons = btc_data[btc_data['is_new_moon']]
    ax1.scatter(new_moons.index, new_moons['close'], 
               color='black', s=80, marker='o', label='New Moon',
               edgecolors='gray', linewidth=2, zorder=5)
    
    ax1.set_ylabel('Bitcoin Price (USD)', fontsize=10)
    ax1.set_title('Bitcoin Price and Lunar Cycles', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moon phase
    ax2 = axes[1]
    ax2.plot(btc_data.index, btc_data['phase'], 'purple', alpha=0.7)
    ax2.fill_between(btc_data.index, 0, btc_data['phase'], alpha=0.3, color='purple')
    ax2.axhline(y=0.5, color='gold', linestyle='--', alpha=0.5, label='Full Moon Line')
    ax2.set_ylabel('Moon Phase', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_title('Lunar Phase (0 = New Moon, 0.5 = Full Moon)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bitcoin volatility
    ax3 = axes[2]
    ax3.plot(btc_data.index, btc_data['volatility'] * 100, 'red', alpha=0.7)
    ax3.set_ylabel('Volatility (%)', fontsize=10)
    ax3.set_title('Bitcoin Volatility', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Strategy performance comparison
    ax4 = axes[3]
    if results:
        # Calculate cumulative returns for best strategy
        best_strategy = max(results.items(), 
                          key=lambda x: x[1].metrics.get('sharpe_ratio', -999))
        strategy_name, best_result = best_strategy
        
        if hasattr(best_result, 'portfolio'):
            cumret = best_result.portfolio.cumulative_returns()
            ax4.plot(cumret.index, cumret.values * 100, 
                    label=f'{strategy_name.title()} Strategy', 
                    color='green', linewidth=2)
            
        # Add buy & hold
        btc_data['buy_hold_cumret'] = (1 + btc_data['returns']).cumprod() - 1
        ax4.plot(btc_data.index, btc_data['buy_hold_cumret'] * 100, 
                label='Buy & Hold', color='gray', alpha=0.7)
        
        ax4.set_ylabel('Cumulative Return (%)', fontsize=10)
        ax4.set_xlabel('Date', fontsize=10)
        ax4.set_title('Strategy Performance', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("results/lunar_bitcoin")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "bitcoin_lunar_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_dir / 'bitcoin_lunar_analysis.png'}")
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    
    # Select lunar and market features for correlation
    corr_features = ['phase', 'illumination', 'distance_ratio', 
                    'days_since_new', 'days_since_full',
                    'returns', 'volatility', 'volume']
    
    # Filter available features
    available_features = [f for f in corr_features if f in btc_data.columns]
    corr_matrix = btc_data[available_features].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.3f', square=True, linewidths=1)
    plt.title('Correlation: Lunar Features vs Bitcoin Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "lunar_bitcoin_correlation.png", dpi=150)
    print(f"Saved correlation heatmap to {output_dir / 'lunar_bitcoin_correlation.png'}")
    
    plt.show()


def main():
    """Main analysis function"""
    print("=" * 60)
    print("Bitcoin Lunar Trading Analysis")
    print("=" * 60)
    
    # Download Bitcoin data
    btc_data = download_bitcoin_data(period="2y", interval="1h")
    
    # Add lunar features
    btc_data = add_lunar_features(btc_data)
    
    # Analyze correlations
    phase_analysis = analyze_lunar_correlation(btc_data)
    
    # Run backtests
    results = backtest_lunar_strategies(btc_data)
    
    # Create visualizations
    try:
        create_visualizations(btc_data, results)
    except Exception as e:
        print(f"\nCouldn't create visualizations: {e}")
    
    # Save results
    output_dir = Path("results/lunar_bitcoin")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data with lunar features
    btc_data.to_parquet(output_dir / "bitcoin_with_lunar_features.parquet")
    print(f"\nSaved data to {output_dir / 'bitcoin_with_lunar_features.parquet'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    if results:
        # Find best strategy
        best_strategy = max(results.items(), 
                          key=lambda x: x[1].metrics.get('sharpe_ratio', -999))
        print(f"\nBest Strategy: {best_strategy[0].title()}")
        print(f"Sharpe Ratio: {best_strategy[1].metrics['sharpe_ratio']:.2f}")
        print(f"Total Return: {best_strategy[1].metrics['total_return']:.1%}")
    
    print("\nNext Steps:")
    print("1. Optimize parameters using Bayesian optimization")
    print("2. Test on different time periods")
    print("3. Add technical indicators for confirmation")
    print("4. Implement live trading with paper account")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()