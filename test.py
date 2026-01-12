import time
import joblib
import os
import pandas as pd
import numpy as np
from src.data_fetcher import DataFetcher
from src.smc_analysis import SMCAnalysis
from src.ml_model import MLModel
from src.risk_manager import PositionSizer

def load_models():
    """Load both Long and Short models"""
    models = {}
    
    # Long model
    long_path = 'models/xgboost_smc.joblib'
    if os.path.exists(long_path):
        data = joblib.load(long_path)
        ml_long = MLModel()
        if isinstance(data, dict):
            ml_long.model = data['model']
            ml_long.trained_features = data.get('features', ml_long.features)
            models['long_threshold'] = data.get('threshold', 0.62)
        else:
            ml_long.model = data
            ml_long.trained_features = ml_long.features
            models['long_threshold'] = 0.62
        models['long'] = ml_long
    
    # Short model
    short_path = 'models/xgboost_short.joblib'
    if os.path.exists(short_path):
        data = joblib.load(short_path)
        ml_short = MLModel()
        ml_short.model = data['model']
        ml_short.trained_features = data.get('features', [])
        models['short'] = ml_short
        models['short_threshold'] = 0.62
    
    return models if 'long' in models else None

def get_signal_quality(prob, is_trending, is_aligned, has_momentum):
    """Calculate signal quality score"""
    score = 0
    if prob > 0.7:
        score += 3
    elif prob > 0.6:
        score += 2
    else:
        score += 1
    
    if is_trending:
        score += 1
    if is_aligned:
        score += 1
    if has_momentum:
        score += 1
    
    if score >= 5:
        return "⭐⭐⭐"
    elif score >= 3:
        return "⭐⭐"
    else:
        return "⭐"

def main():
    print("🚀 Initializing Real-time Bot Test (Advanced)...")
    fetcher = DataFetcher()
    analyzer = SMCAnalysis()
    models = load_models()
    position_sizer = PositionSizer()
    
    if models is None:
        print("❌ Error: Model not found at models/xgboost_smc.joblib")
        print("Please run train_advanced.py first to generate the models.")
        return
    
    has_short = 'short' in models
    print(f"✅ Long model loaded")
    if has_short:
        print(f"✅ Short model loaded")
    else:
        print("⚠️  Short model not found (run train_advanced.py)")

    symbols = ['BTC/USDT', 'XRP/USDT', 'SOL/USDT']
    portfolio = 10000  # Simulated portfolio
    
    print("\n" + "="*100)
    print(f"{'SYMBOL':<12} | {'PRICE':<10} | {'SIGNAL':<8} | {'CONF':<8} | {'QUALITY':<10} | {'LIMIT':<12} | {'TP':<12} | {'SL':<12} | {'SIZE':<8}")
    print("-" * 100)

    while True:
        try:
            for sym in symbols:
                # Fetch multi-tf data
                df_1h, df_15m = fetcher.get_multi_tf_data(sym)
                
                if df_1h is None or df_15m is None:
                    continue
                
                # Apply Analysis
                df_1h = analyzer.apply_all(df_1h, symbol=sym)
                df_15m = analyzer.apply_all(df_15m, symbol=sym)
                
                # Get last row
                last_1h = df_1h.iloc[-1]
                last_15m = df_15m.iloc[-1]
                
                # Prepare features and predict
                long_features = [f for f in models['long'].trained_features if f in df_15m.columns]
                X_long = df_15m.iloc[-1:][long_features].fillna(0)
                long_prob = models['long'].model.predict_proba(X_long)[:, 1][0]
                
                short_prob = 0.0
                if has_short:
                    short_features = [f for f in models['short'].trained_features if f in df_15m.columns]
                    X_short = df_15m.iloc[-1:][short_features].fillna(0)
                    short_prob = models['short'].model.predict_proba(X_short)[:, 1][0]
                
                # Determine signal
                signal = "WAIT"
                prob = 0.0
                
                # Check Long conditions
                long_conditions = (
                    long_prob > models['long_threshold'] and
                    (last_15m['is_uptrend'] == 1 or last_15m.get('ema_aligned_bull', 0) == 1) and
                    (last_15m.get('macd_bull', 0) == 1 or last_15m.get('momentum_bull', 0) == 1) and
                    (last_15m['is_bull_ob'] == 1 or last_15m.get('fvg_bull', 0) == 1)
                )
                
                # Check Short conditions
                short_conditions = has_short and (
                    short_prob > models.get('short_threshold', 0.62) and
                    (last_15m.get('is_downtrend', 0) == 1 or last_15m.get('ema_aligned_bear', 0) == 1) and
                    (last_15m.get('macd_bear', 0) == 1 or last_15m.get('momentum_bear', 0) == 1) and
                    (last_15m['is_bear_ob'] == 1 or last_15m.get('fvg_bear', 0) == 1)
                )
                
                if long_conditions and (not short_conditions or long_prob > short_prob):
                    signal = "🟢 LONG"
                    prob = long_prob
                elif short_conditions:
                    signal = "🔴 SHORT"
                    prob = short_prob
                
                # Current Price
                current_price = last_15m['close']
                
                # Calculate Levels
                bull_obs = df_15m[df_15m['is_bull_ob'] == 1]
                bear_obs = df_15m[df_15m['is_bear_ob'] == 1]
                
                limit_price = 0.0
                tp_price = 0.0
                sl_price = 0.0
                
                if "LONG" in signal:
                    if not bull_obs.empty:
                        limit_price = bull_obs.iloc[-1]['low']
                    else:
                        limit_price = last_15m.get('support', current_price * 0.99)
                    
                    tp_price = limit_price * 1.02  # 2% TP
                    sl_price = limit_price * 0.985  # 1.5% SL
                elif "SHORT" in signal:
                    if not bear_obs.empty:
                        limit_price = bear_obs.iloc[-1]['high']
                    else:
                        limit_price = last_15m.get('resistance', current_price * 1.01)
                        
                    tp_price = limit_price * 0.98  # 2% TP (below for short)
                    sl_price = limit_price * 1.015  # 1.5% SL (above for short)
                
                # Calculate position size
                avg_atr = df_15m['atr_pct'].mean() if 'atr_pct' in df_15m.columns else 1.0
                current_atr = last_15m.get('atr_pct', avg_atr)
                
                if signal != "WAIT":
                    pos_value, _ = position_sizer.calculate_position_size(
                        portfolio_value=portfolio,
                        probability=prob,
                        current_atr_pct=current_atr,
                        avg_atr_pct=avg_atr,
                        stop_loss_pct=1.5
                    )
                    size_str = f"${pos_value:.0f}"
                else:
                    size_str = "-"
                
                # Get signal quality
                quality = get_signal_quality(
                    prob,
                    last_15m.get('is_trending', 0) == 1,
                    last_15m.get('ema_aligned_bull', 0) == 1 or last_15m.get('ema_aligned_bear', 0) == 1,
                    last_15m.get('macd_bull', 0) == 1 or last_15m.get('macd_bear', 0) == 1
                ) if signal != "WAIT" else "-"

                # Output formatting
                conf_str = f"{prob*100:.1f}%" if prob > 0 else "-"
                price_str = f"{current_price:.4f}"
                limit_str = f"{limit_price:.4f}" if limit_price > 0 else "-"
                tp_str = f"{tp_price:.4f}" if tp_price > 0 else "-"
                sl_str = f"{sl_price:.4f}" if sl_price > 0 else "-"
                
                print(f"{sym:<12} | {price_str:<10} | {signal:<8} | {conf_str:<8} | {quality:<10} | {limit_str:<12} | {tp_str:<12} | {sl_str:<12} | {size_str:<8}")

            print("-" * 100)
            print(f"Last updated: {pd.Timestamp.now()} | Portfolio: ${portfolio:,.0f}")
            print("=" * 100)
            time.sleep(30)  # Wait 30 seconds before next update
            
        except KeyboardInterrupt:
            print("\nStopping bot...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)

if __name__ == "__main__":
    main()

