import pandas as pd
from src.data_fetcher import DataFetcher
from src.smc_analysis import SMCAnalysis
from src.ml_model import MLModel
import joblib
import os

def get_data_cached(fetcher, symbol, start, end):
    filename = f"data/{symbol.replace('/', '_')}_15m.csv"
    if not os.path.exists('data'):
        os.makedirs('data')
    
    if os.path.exists(filename):
        print(f"  Checking {symbol} CSV cache...")
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # requested range
        req_start = pd.to_datetime(start).tz_localize(None)
        req_end = pd.to_datetime(end).tz_localize(None)
        
        # Check if we have the range
        cache_start = df['timestamp'].min()
        cache_end = df['timestamp'].max()
        
        if cache_start <= req_start and cache_end >= req_end:
            print(f"  Found complete range in cache.")
            mask = (df['timestamp'] >= req_start) & (df['timestamp'] <= req_end)
            return df.loc[mask].reset_index(drop=True)
        else:
            print(f"  Using all available data from CSV cache ({len(df)} rows).")
            return df
            
    print(f"  Fetching {symbol} from exchange...")
    df = fetcher.get_historical_data(symbol, timeframe='15m', start_date=start, end_date=end)
    if df is not None:
        if os.path.exists(filename):
            existing = pd.read_csv(filename)
            existing['timestamp'] = pd.to_datetime(existing['timestamp'])
            df = pd.concat([existing, df]).drop_duplicates('timestamp').sort_values('timestamp')
        df.to_csv(filename, index=False)
        print(f"  Saved {symbol} to cache.")
    return df

def run_train():
    EPOCHS = 10000
    print(f"Initializing Model Training (2022 - 2023.06) with {EPOCHS} epochs...")
    fetcher = DataFetcher()
    analyzer = SMCAnalysis()
    ml = MLModel()
    
    symbols = ['BTC/USDT', 'XRP/USDT', 'SOL/USDT']
    all_data = []
    
    for sym in symbols:
        df = get_data_cached(fetcher, sym, "2022-01-01T00:00:00Z", "2023-12-31T23:59:59Z")
        if df is None: continue
        
        print(f"  Analyzing {len(df)} candles for {sym}...")
        df = analyzer.apply_all(df, symbol=sym)
        df = ml.prepare_labels(df, sl_pct=1.5, tp_pct=2.0)  # Use balanced SL/TP
        df = df.dropna()
        all_data.append(df)
    
    if not all_data:
        print("No data collected.")
        return

    combined_df = pd.concat(all_data)
    
    # FILTER: 2022-01 to 2023-06 for training
    train_cutoff = pd.to_datetime("2023-06-30 23:59:59")
    train_df = combined_df[combined_df['timestamp'] <= train_cutoff]
    
    # Show class balance
    pos_count = train_df['target'].sum()
    neg_count = len(train_df) - pos_count
    print(f"\nClass Balance: Positive={int(pos_count)}, Negative={int(neg_count)} ({pos_count/len(train_df)*100:.1f}% positive)")
    
    print(f"\nTraining on {len(train_df)} samples (Range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    ml.train(train_df, epochs=EPOCHS)
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save model with features for consistent prediction
    model_data = {
        'model': ml.model,
        'features': ml.get_trained_features()
    }
    joblib.dump(model_data, 'models/xgboost_smc.joblib')
    print("\nModel saved to models/xgboost_smc.joblib")
    print(f"Saved with {len(ml.get_trained_features())} features")

if __name__ == "__main__":
    run_train()

