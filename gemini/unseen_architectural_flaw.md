# Critical Architectural Flaw Analysis

**To:** `claude-code`  
**From:** Gemini Agent  
**Date:** 2025-07-17  
**Subject:** Identification of a Critical, Undocumented Architectural Flaw in the Backtesting Workflow

---

### 1. Executive Summary

After a deep re-analysis of the source code, I have identified a critical architectural flaw that was not mentioned in the existing `claude/` documentation. The current backtesting process introduces significant **look-ahead bias** by calculating position sizes based on information that would not have been available at the time of the trade.

Specifically, the `VectorBTEngine` calculates position sizes using the `close` price of the *same bar* on which the entry signal is generated. In a live scenario, the decision to enter a trade and the calculation of how many shares to buy must be made based on the price at the *moment of the decision* (e.g., the open of the bar), not on a price that is only known after the bar has completed.

This flaw, while subtle, could be a primary contributor to the poor performance of the existing strategies and will invalidate the results of any strategy tested on this framework until it is corrected.

### 2. The Flaw Explained

The issue is located in the `VectorBTEngine.run_backtest` method and its helpers:

```python
# In src/backtesting/engines/vectorbt_engine.py

def run_backtest(self, ...):
    # ...
    signals = strategy.generate_signals(data)

    # FLAW IS HERE:
    # The position size is calculated using the full 'data' DataFrame,
    # which includes the 'close' price of the current bar.
    positions = self._calculate_fixed_positions(
        signals, data['close'], initial_capital, ...
    )

    portfolio = vbt.Portfolio.from_signals(
        close=data['close'], # VectorBT uses this close price to execute the trade
        entries=signals > 0,
        exits=signals < 0,
        size=np.abs(positions), # The size was calculated with future knowledge
        # ...
    )
```

**The sequence of events is incorrect:**

1.  A signal is generated for a specific time `t` (e.g., based on the close of bar `t-1`).
2.  The position size for the trade at time `t` is calculated using the `close` price of bar `t`.
3.  The backtester then executes the trade at the `close` price of bar `t`.

This is a classic form of look-ahead bias. The strategy is, in effect, deciding how many shares to buy based on a price it won't know until the trading period is over.

### 3. Why This Is Critical

*   **Invalidates Backtest Results:** All performance metrics (Sharpe ratio, returns, etc.) are unreliable because the trades were sized with information from the future.
*   **Masks True Strategy Performance:** A strategy might appear to perform poorly when, in fact, the flawed execution logic is the root cause. For example, if the close is significantly higher than the open, the position size would be artificially smaller for a long trade, systematically reducing both risk and potential profit.
*   **Impacts Volatility-Based Sizing:** The flaw is even more pronounced in volatility-based position sizing, where both the price and the volatility (which might be calculated using the current bar's high/low/close) are sourced from the future.

### 4. The Correct Approach: Shifting the Data

The solution is to ensure that all data used for decision-making (signal generation and position sizing) is available *before* the execution price is known. The standard professional practice is to make decisions based on the previous bar's data and execute on the current bar's `open` price.

**The corrected workflow should be:**

1.  **Generate Signals:** Signals for time `t` are generated using data up to and including `t-1`.
2.  **Calculate Position Size:** The position size for the trade at time `t` is calculated using the `open` price of bar `t` (or the `close` of `t-1`).
3.  **Execute Trade:** The backtester executes the trade at the `open` price of bar `t`.

### 5. Recommended Implementation Changes

I propose the following refactoring of the `VectorBTEngine`:

1.  **Shift the Price/Signal Relationship:** The `from_signals` method in VectorBT has parameters to handle this. We should execute trades on the `open` of the next bar.

2.  **Refactor `run_backtest`:**

    ```python
    # In src/backtesting/engines/vectorbt_engine.py

    def run_backtest(self, ...):
        # ...
        signals = strategy.generate_signals(data)

        # --- FIX --- 
        # Use the OPEN price of the bar for sizing and execution.
        # We must shift the signals by 1 to ensure we use the previous bar's signal
        # to trade on the current bar's open.
        
        # The price used for sizing should be the open price.
        position_prices = data['open']

        positions = self._calculate_fixed_positions(
            signals.shift(1), position_prices, initial_capital, ...
        )

        portfolio = vbt.Portfolio.from_signals(
            close=data['open'],  # <--- EXECUTE ON OPEN
            entries=(signals.shift(1) > 0),
            exits=(signals.shift(1) < 0),
            size=np.abs(positions),
            # ...
        )
        # ...
    ```

3.  **Update All Position Sizing Methods:** The helper methods (`_calculate_fixed_positions`, `_calculate_volatility_positions`) and the base strategy's `calculate_positions` must all be updated to use the shifted price series for their calculations.

### 6. Conclusion

This look-ahead bias is a subtle but fundamental issue that undermines the core value of the backtesting framework. Correcting it should be the **absolute highest priority** before any further strategy research or optimization is conducted.

Once this is fixed, the backtest results for the existing MA and ORB strategies should be re-run. It is possible they will perform differently (either better or worse), but the new results will be a true and valid baseline from which to build.

This discovery does not diminish the quality of the project's architecture; it highlights the extreme difficulty of building a correct backtesting system and provides a clear, actionable path to making it even more robust.
