import time
import joblib
import os
from src.data_fetcher import DataFetcher
from src.smc_analysis import SMCAnalysis
from src.ml_model import MLModel
from src.executor import Executor

def main():
    print("Starting SMC + XGBoost Crypto Bot...")
    fetcher = DataFetcher()
    analyzer = SMCAnalysis()
    model = MLModel()
    executor = Executor(initial_balance=1000)
    
    symbols = ['BTC/USDT', 'XRP/USDT', 'SOL/USDT']
    
    # Load pre-trained model if exists
    if os.path.exists('models/xgboost_smc.joblib'):
        print("Loading pre-trained model...")
        model.model = joblib.load('models/xgboost_smc.joblib')
    else:
        print("No pre-trained model found. Bootstrapping with historical data...")
        # Initial Model Training (Bootstrap)
        for sym in symbols:
            df_1h, df_15m = fetcher.get_multi_tf_data(sym)
            if df_15m is not None:
                df_15m = analyzer.apply_all(df_15m)
                df_15m = model.prepare_labels(df_15m)
                model.train(df_15m)
    
    print("Bot is now running in Dry Run mode...")
    
    while True:
        try:
            for sym in symbols:
                df_1h, df_15m = fetcher.get_multi_tf_data(sym)
                if df_1h is None or df_15m is None: continue
                
                df_1h = analyzer.apply_all(df_1h)
                df_15m = analyzer.apply_all(df_15m)
                
                # Predict probability for last candle
                last_row = df_15m.iloc[-1:]
                prob = model.predict_probability(last_row)
                
                print(f"[{sym}] Probability: {prob:.2f} | Bias: {'Bullish' if df_1h.iloc[-1]['is_uptrend'] else 'Bearish'}")
                
                if executor.check_entry_conditions(sym, df_15m, df_1h, prob):
                    entry = df_15m.iloc[-1]['close']
                    sl = entry * 0.99  # 1% SL
                    tp = entry * 1.03  # 3:1 RR
                    executor.open_trade(sym, entry, sl, tp, prob)
                
            time.sleep(60) # Check every minute
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
