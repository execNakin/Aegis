import pandas as pd
import joblib
import os
from src.data_fetcher import DataFetcher
from src.smc_analysis import SMCAnalysis
from src.ml_model import MLModel
from train import get_data_cached

def run_backtest():
    print("Initializing Backtest (2023.07 to 2025.12)...")
    if not os.path.exists('models/xgboost_smc.joblib'):
        print("Error: No trained model found. Please run train.py first.")
        return

    fetcher = DataFetcher()
    analyzer = SMCAnalysis()
    
    # Load model and metadata
    model_data = joblib.load('models/xgboost_smc.joblib')
    if isinstance(model_data, dict):
        model_booster = model_data['model']
        trained_features = model_data['features']
    else:
        model_booster = model_data
        trained_features = ['fvg_bull', 'fvg_bear', 'fvg_size', 'is_uptrend', 'is_bull_ob', 'is_bear_ob', 'choch_bull', 'is_rsi_bull', 'is_high_vol']
    
    symbols = ['BTC/USDT', 'XRP/USDT', 'SOL/USDT']
    
    print("\n--- Backtest Results (2023.07 onwards) ---")
    backtest_start = pd.to_datetime("2023-07-01 00:00:00")
    
    total_wins = 0
    total_trades = 0
    total_profit = 0
    
    for sym in symbols:
        df = get_data_cached(fetcher, sym, "2023-07-01T00:00:00Z", "2025-12-31T23:59:59Z")
        if df is None: continue
        
        # FILTER: Only use data from 2023-07 onwards
        df = df[df['timestamp'] >= backtest_start].reset_index(drop=True)
            
        print(f"  Analyzing {len(df)} candles for {sym}...")
        df = analyzer.apply_all(df, symbol=sym)
        ml = MLModel()
        df = ml.prepare_labels(df, sl_pct=1.5, tp_pct=2.0)
        df = df.dropna()
        
        if len(df) == 0: continue

        # Use trained features
        available_features = [f for f in trained_features if f in df.columns]
        X = df[available_features].fillna(0)
        probs = model_booster.predict_proba(X)[:, 1]
        
        # Enhanced Signal Filters for Higher Win Rate
        # Base conditions that must be met
        base_prob = probs > 0.62  # Higher base probability threshold
        
        # Trend confirmation (at least one must be true)
        trend_ok = (
            (df['is_uptrend'] == 1) | 
            (df['ema_aligned_bull'] == 1) |
            (df['is_trending'] == 1)
        )
        
        # Momentum confirmation (at least one must be true)
        momentum_ok = (
            (df['macd_bull'] == 1) | 
            (df['momentum_bull'] == 1) |
            (df['rsi_rising'] == 1)
        )
        
        # Entry trigger (at least one SMC signal)
        entry_trigger = (
            (df['is_bull_ob'] == 1) | 
            (df['fvg_bull'] == 1) | 
            (df['choch_bull'] == 1)
        )
        
        # Avoid bad conditions
        avoid_conditions = (
            (df['bb_overbought'] == 0) &  # Not overbought
            (df['near_resistance'] == 0)   # Not near resistance
        )
        
        # Symbol-specific adjustments
        if "SOL" in sym:
            # SOL needs extra trend confirmation due to high volatility
            signals = base_prob & trend_ok & momentum_ok & entry_trigger & avoid_conditions & (df['is_low_vol'] == 1)
        elif "XRP" in sym:
            # XRP can be slightly relaxed
            signals = (probs > 0.58) & trend_ok & entry_trigger & avoid_conditions
        else:
            # BTC: Standard high-quality filter
            signals = base_prob & trend_ok & momentum_ok & entry_trigger & avoid_conditions
            
        trades = df[signals]
        
        wins = trades['target'].sum()
        losses = len(trades) - wins
        win_rate = (wins / len(trades) * 100) if len(trades) > 0 else 0
        
        # Risk:Reward based profit calculation (SL=1.5%, TP=2%)
        profit_per_trade = (wins * 2.0) - (losses * 1.5)
        
        total_wins += wins
        total_trades += len(trades)
        total_profit += profit_per_trade
        
        print(f"Symbol: {sym}")
        print(f"  Total Signals: {len(trades)}")
        print(f"  Wins: {int(wins)}, Losses: {int(losses)}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Estimated Return: {profit_per_trade:.2f}%")
        print("-" * 30)
    
    # Overall Summary
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    print("\n=== OVERALL SUMMARY ===")
    print(f"Total Trades: {total_trades}")
    print(f"Total Wins: {int(total_wins)}")
    print(f"Overall Win Rate: {overall_win_rate:.2f}%")
    print(f"Total Estimated Return: {total_profit:.2f}%")

if __name__ == "__main__":
    run_backtest()

