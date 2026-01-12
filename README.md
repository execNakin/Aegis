# 🛡️ Aegis: SMC + XGBoost Crypto Trading Bot

An advanced cryptocurrency trading bot that combines **Smart Money Concepts (SMC)** with **XGBoost Machine Learning** to identify and execute high-probability trade setups with robust risk management.

## 🌟 Key Features

-   **Multi-Timeframe Analysis:** Analyzes the **1-Hour** timeframe for trend bias and the **15-Minute** timeframe for trade execution.
-   **Smart Money Concepts (SMC):** Implements automated detection of:
    -   Fair Value Gaps (FVG)
    -   Order Blocks (OB)
    -   Change of Character (CHoCH)
-   **Advanced Machine Learning:** Uses **XGBoost Classifier** to predict the success probability of trade signals based on SMC and technical indicators.
-   **Comprehensive Risk Management:**
    -   **Dynamic Position Sizing:** Based on ATR volatility and signal confidence.
    -   **Kelly Criterion:** Optional fractional Kelly for optimal capital allocation.
    -   **Drawdown Protection:** Automatically reduces risk or stops trading during periods of significant drawdown.
-   **Live Monitoring Dashboard:** Streamlit-based UI for tracking open trades and system logs.
-   **Walk-Forward Validation:** Prevents overfitting by validating model performance across rolling time windows.
-   **Hyperparameter Tuning:** Optimized using **Optuna** for both model parameters and strategy thresholds.

## 🛠️ Project Structure

```text
├── src/
│   ├── smc_analysis.py       # Core SMC and Technical Analysis logic
│   ├── ml_model.py           # XGBoost model wrapper and feature engineering
│   ├── data_fetcher.py       # Binance OHLCV data acquisition via ccxt
│   ├── risk_manager.py       # Position sizing and drawdown protection
│   ├── advanced_strategy.py  # Walk-forward validation and Short signal model
│   ├── hyperparameter_tuner.py # Parameter optimization using Optuna
│   └── executor.py           # Trade execution logic
├── models/                   # Saved model files (.joblib)
├── data/                     # Cached CSV historical data
├── main.py                   # Main entry point (Dry Run mode)
├── test.py                   # Real-time advanced testing script
├── train.py                  # Model training script
├── backtest.py               # Historical performance evaluation
├── dashboard.py              # Streamlit dashboard
└── opening.json              # Tracking file for open positions
```

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install pandas numpy xgboost ccxt joblib streamlit optuna scikit-learn
```

### 2. Training the Model
To train the model on historical data (2022 - 2023):
```bash
python train.py
```
This will generate `models/xgboost_smc.joblib`.

### 3. Running Backtests
Evaluate the model's performance on unseen data (July 2023 onwards):
```bash
python backtest.py
```

### 4. Live Testing (Real-time)
Run the advanced real-time testing script to monitor signals for BTC, XRP, and SOL:
```bash
python test.py
```

### 5. Launch Dashboard
Visualize open trades and logs:
```bash
streamlit run dashboard.py
```

## 📊 Strategy Logic

1.  **Bias Filter:** Determines trend direction on the 1H timeframe.
2.  **SMC Setup:** Looks for price tapping into an Order Block or FVG on the 15m timeframe.
3.  **ML Confirmation:** The XGBoost model calculates the probability of the trade hitting a 2:1 Take Profit (TP) before a 1.5% Stop Loss (SL).
4.  **Signal Quality:** Signals are graded (⭐ to ⭐⭐⭐) based on confidence, trend alignment, and momentum.
5.  **Execution:** Calculates the optimal dollar amount to risk based on current market volatility (ATR).

## ⚠️ Disclaimer
This bot is for educational and testing purposes only. Trading cryptocurrency involves significant risk. Always use a dry run or demo account before trading with real capital.
