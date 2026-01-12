import pandas as pd
import numpy as np
import joblib
import os
import argparse
from src.data_fetcher import DataFetcher
from src.smc_analysis import SMCAnalysis
from src.ml_model import MLModel
from src.advanced_strategy import ShortSignalModel, WalkForwardValidator
from src.hyperparameter_tuner import run_hyperparameter_tuning
from train import get_data_cached

def train_both_models(use_optuna=False, n_trials=50):
    """
    Train both Long and Short signal models
    """
    print("=" * 60)
    print("ADVANCED MODEL TRAINING")
    print("=" * 60)
    
    fetcher = DataFetcher()
    analyzer = SMCAnalysis()
    
    symbols = ['BTC/USDT', 'XRP/USDT', 'SOL/USDT']
    all_data = []
    
    print("\n--- Loading Data ---")
    for sym in symbols:
        df = get_data_cached(fetcher, sym, "2022-01-01T00:00:00Z", "2023-12-31T23:59:59Z")
        if df is None:
            continue
        
        print(f"  Analyzing {len(df)} candles for {sym}...")
        df = analyzer.apply_all(df, symbol=sym)
        all_data.append(df)
    
    if not all_data:
        print("No data collected.")
        return
    
    combined_df = pd.concat(all_data)
    
    # Filter training period
    train_cutoff = pd.to_datetime("2023-06-30 23:59:59")
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    train_df = combined_df[combined_df['timestamp'] <= train_cutoff].copy()
    
    print(f"\nTraining on {len(train_df)} samples")
    print(f"  Range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    
    # ===== LONG MODEL =====
    print("\n" + "=" * 60)
    print("TRAINING LONG MODEL")
    print("=" * 60)
    
    ml_long = MLModel()
    train_df_long = ml_long.prepare_labels(train_df, sl_pct=1.5, tp_pct=2.0)
    train_df_long = train_df_long.dropna()
    
    pos_count = train_df_long['target'].sum()
    neg_count = len(train_df_long) - pos_count
    print(f"Class Balance: Positive={int(pos_count)}, Negative={int(neg_count)} ({pos_count/len(train_df_long)*100:.1f}% positive)")
    
    if use_optuna:
        print("\nRunning Hyperparameter Tuning with Optuna...")
        result = run_hyperparameter_tuning(train_df_long, ml_long.features, n_trials=n_trials)
        long_model = result['model']
        long_features = result['features']
        long_params = result['params']
        long_threshold = result['threshold']
        
        print(f"\nOptimal Probability Threshold: {long_threshold:.3f}")
    else:
        ml_long.train(train_df_long, epochs=10000)
        long_model = ml_long.model
        long_features = ml_long.get_trained_features()
        long_params = None
        long_threshold = 0.62
    
    # Save Long model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_data = {
        'model': long_model,
        'features': long_features,
        'params': long_params,
        'threshold': long_threshold
    }
    joblib.dump(model_data, 'models/xgboost_smc.joblib')
    print(f"\nLong model saved to models/xgboost_smc.joblib")
    
    # ===== SHORT MODEL =====
    print("\n" + "=" * 60)
    print("TRAINING SHORT MODEL")
    print("=" * 60)
    
    short_model = ShortSignalModel()
    train_df_short = short_model.prepare_short_labels(train_df, sl_pct=1.5, tp_pct=2.0)
    train_df_short = train_df_short.dropna()
    
    pos_count = train_df_short['target_short'].sum()
    neg_count = len(train_df_short) - pos_count
    print(f"Class Balance: Positive={int(pos_count)}, Negative={int(neg_count)} ({pos_count/len(train_df_short)*100:.1f}% positive)")
    
    short_model.train(train_df_short, epochs=10000)
    short_model.save('models/xgboost_short.joblib')
    print(f"\nShort model saved to models/xgboost_short.joblib")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


def run_walk_forward_training():
    """
    Train using Walk-Forward Validation to test robustness
    """
    print("=" * 60)
    print("WALK-FORWARD VALIDATION TRAINING")
    print("=" * 60)
    
    fetcher = DataFetcher()
    analyzer = SMCAnalysis()
    ml = MLModel()
    
    symbols = ['BTC/USDT', 'XRP/USDT', 'SOL/USDT']
    all_data = []
    
    print("\n--- Loading Data ---")
    for sym in symbols:
        df = get_data_cached(fetcher, sym, "2022-01-01T00:00:00Z", "2025-12-31T23:59:59Z")
        if df is None:
            continue
        
        print(f"  Analyzing {len(df)} candles for {sym}...")
        df = analyzer.apply_all(df, symbol=sym)
        df = ml.prepare_labels(df, sl_pct=1.5, tp_pct=2.0)
        df = df.dropna()
        all_data.append(df)
    
    if not all_data:
        print("No data collected.")
        return
    
    combined_df = pd.concat(all_data)
    
    # Run walk-forward validation
    validator = WalkForwardValidator(
        train_months=6,
        test_months=1,
        step_months=1
    )
    
    summary, predictions = validator.run_validation(combined_df, ml.features)
    
    # Save results
    results_df = validator.get_results_df()
    results_df.to_csv('models/walk_forward_results.csv', index=False)
    print(f"\nResults saved to models/walk_forward_results.csv")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optuna', action='store_true', help='Use Optuna for hyperparameter tuning')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward validation')
    args = parser.parse_args()
    
    if args.walk_forward:
        run_walk_forward_training()
    else:
        train_both_models(use_optuna=args.optuna, n_trials=args.trials)
