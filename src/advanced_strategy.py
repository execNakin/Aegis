import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import joblib

class WalkForwardValidator:
    """
    Walk-Forward Validation System
    Prevents overfitting by using rolling window training/testing
    """
    
    def __init__(self, 
                 train_months=6,      # Training window size
                 test_months=1,       # Testing window size  
                 step_months=1):      # Step size between windows
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.results = []
        
    def generate_windows(self, df):
        """Generate train/test windows"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        
        windows = []
        current_start = start_date
        
        while True:
            train_end = current_start + pd.DateOffset(months=self.train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)
            
            if test_end > end_date:
                break
                
            windows.append({
                'train_start': current_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            current_start += pd.DateOffset(months=self.step_months)
        
        return windows
    
    def run_validation(self, df, features, model_params=None, verbose=True):
        """
        Run walk-forward validation
        Returns aggregated results across all windows
        """
        windows = self.generate_windows(df)
        
        if verbose:
            print(f"Running Walk-Forward Validation with {len(windows)} windows...")
            print(f"  Train: {self.train_months} months, Test: {self.test_months} months")
        
        self.results = []
        all_predictions = []
        
        available_features = [f for f in features if f in df.columns]
        
        for i, window in enumerate(windows):
            # Split data
            train_mask = (df['timestamp'] >= window['train_start']) & (df['timestamp'] < window['train_end'])
            test_mask = (df['timestamp'] >= window['test_start']) & (df['timestamp'] < window['test_end'])
            
            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()
            
            if len(train_df) == 0 or len(test_df) == 0:
                continue
            
            X_train = train_df[available_features].fillna(0)
            y_train = train_df['target']
            X_test = test_df[available_features].fillna(0)
            y_test = test_df['target']
            
            # Train model
            if model_params is None:
                model_params = {
                    'n_estimators': 500,
                    'max_depth': 5,
                    'learning_rate': 0.03,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
            
            model = xgb.XGBClassifier(
                **model_params,
                objective='binary:logistic',
                eval_metric='logloss',
                early_stopping_rounds=50,
                verbosity=0
            )
            
            # Split train for early stopping
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            # Predict on test set
            probs = model.predict_proba(X_test)[:, 1]
            predictions = probs > 0.62
            
            # Calculate metrics
            wins = np.sum(predictions & (y_test.values == 1))
            losses = np.sum(predictions & (y_test.values == 0))
            total = wins + losses
            
            win_rate = wins / total * 100 if total > 0 else 0
            profit = (wins * 2.0) - (losses * 1.5)
            
            result = {
                'window': i + 1,
                'train_period': f"{window['train_start'].strftime('%Y-%m')} to {window['train_end'].strftime('%Y-%m')}",
                'test_period': f"{window['test_start'].strftime('%Y-%m')} to {window['test_end'].strftime('%Y-%m')}",
                'trades': total,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'profit': profit
            }
            self.results.append(result)
            
            # Store predictions for aggregation
            test_df = test_df.copy()
            test_df['probability'] = probs
            test_df['prediction'] = predictions
            all_predictions.append(test_df)
            
            if verbose:
                print(f"  Window {i+1}: {result['test_period']} | Trades: {total} | Win Rate: {win_rate:.1f}% | Profit: {profit:.1f}%")
        
        # Aggregate results
        if self.results:
            total_trades = sum(r['trades'] for r in self.results)
            total_wins = sum(r['wins'] for r in self.results)
            total_losses = sum(r['losses'] for r in self.results)
            total_profit = sum(r['profit'] for r in self.results)
            overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
            
            self.summary = {
                'total_windows': len(self.results),
                'total_trades': total_trades,
                'total_wins': total_wins,
                'total_losses': total_losses,
                'overall_win_rate': overall_win_rate,
                'total_profit': total_profit,
                'avg_profit_per_window': total_profit / len(self.results) if self.results else 0,
                'consistency': sum(1 for r in self.results if r['profit'] > 0) / len(self.results) * 100 if self.results else 0
            }
            
            if verbose:
                print(f"\n=== Walk-Forward Summary ===")
                print(f"Total Windows: {self.summary['total_windows']}")
                print(f"Total Trades: {self.summary['total_trades']}")
                print(f"Overall Win Rate: {self.summary['overall_win_rate']:.2f}%")
                print(f"Total Profit: {self.summary['total_profit']:.2f}%")
                print(f"Consistency (profitable windows): {self.summary['consistency']:.1f}%")
        
        return self.summary, pd.concat(all_predictions) if all_predictions else None
    
    def get_results_df(self):
        """Get results as DataFrame"""
        return pd.DataFrame(self.results)


class ShortSignalModel:
    """
    Short (Sell) Signal Model
    Mirror of Long signals for bearish trades
    """
    
    def __init__(self):
        self.model = None
        self.features = None
        
    def prepare_short_labels(self, df, sl_pct=1.5, tp_pct=2.0):
        """
        Prepare labels for SHORT trades
        Win = price drops to TP before hitting SL
        """
        df = df.copy()
        df['target_short'] = 0
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        targets = np.zeros(len(df))
        
        window = 30
        
        # For shorts: SL is ABOVE entry, TP is BELOW entry
        sl_mult = 1 + (sl_pct / 100)  # Price goes UP = loss
        tp_mult = 1 - (tp_pct / 100)  # Price goes DOWN = win
        
        for i in range(len(df) - window):
            entry_price = closes[i]
            sl = entry_price * sl_mult  # Stop loss above
            tp = entry_price * tp_mult  # Take profit below
            
            fut_highs = highs[i+1 : i+1+window]
            fut_lows = lows[i+1 : i+1+window]
            
            # For shorts: TP hit when low <= tp, SL hit when high >= sl
            tp_hits = np.where(fut_lows <= tp)[0]
            sl_hits = np.where(fut_highs >= sl)[0]
            
            first_tp = tp_hits[0] if tp_hits.size > 0 else 999
            first_sl = sl_hits[0] if sl_hits.size > 0 else 999
            
            if first_tp < first_sl:
                targets[i] = 1
                
        df['target_short'] = targets
        return df
    
    def get_short_features(self):
        """Features optimized for short signals (bearish indicators)"""
        return [
            # SMC Core (bearish)
            'fvg_bear', 'fvg_bull', 'fvg_size',
            'is_bear_ob', 'is_bull_ob',
            'choch_bear', 'choch_bull',
            
            # Trend (inverted for shorts)
            'is_downtrend', 'is_trending', 'trend_strength',
            'ema_aligned_bear', 'ema_aligned_bull',
            
            # Momentum (bearish)
            'macd_bear', 'macd_bull',
            'momentum_bear', 'rsi_rising',
            'is_rsi_bear', 'rsi_overbought',
            
            # Volatility
            'atr_pct', 'is_low_vol', 'bb_width',
            'bb_overbought', 'bb_oversold',
            
            # Volume
            'is_high_vol', 'vol_ratio',
            
            # Price Action
            'is_strong_candle', 'body_ratio',
            
            # Support/Resistance (inverted for shorts)
            'near_resistance', 'near_support',
            'dist_from_resistance', 'dist_from_support'
        ]
    
    def train(self, df, epochs=5000):
        """Train short signal model"""
        self.features = self.get_short_features()
        available_features = [f for f in self.features if f in df.columns]
        
        X = df[available_features].fillna(0)
        y = df['target_short']
        
        if len(np.unique(y)) < 2:
            print("Warning: Not enough target classes for short training.")
            return None
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.model = xgb.XGBClassifier(
            n_estimators=epochs,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=100,
            verbosity=0
        )
        
        print("Training Short Signal Model...")
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        print(f"Best Iteration: {self.model.best_iteration}")
        score = self.model.score(X_val, y_val)
        print(f"Validation Accuracy: {score:.2f}")
        
        self.trained_features = available_features
        return self.model
    
    def predict(self, df):
        """Predict short probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = df[self.trained_features].fillna(0)
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'features': self.trained_features
        }, path)
    
    def load(self, path):
        """Load model"""
        data = joblib.load(path)
        self.model = data['model']
        self.trained_features = data['features']
