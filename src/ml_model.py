import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

class MLModel:
    def __init__(self):
        self.epochs = 500 # Default epochs (n_estimators)
        self.model = xgb.XGBClassifier(
            n_estimators=self.epochs,
            max_depth=5,  # Reduced to prevent overfitting
            learning_rate=0.03,  # Lower for better generalization
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=100,  # More patience
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1
        )
        
        # Enhanced feature set
        self.features = [
            # SMC Core
            'fvg_bull', 'fvg_bear', 'fvg_size', 
            'is_bull_ob', 'is_bear_ob',
            'choch_bull', 'choch_bear',
            
            # Trend
            'is_uptrend', 'is_trending', 'trend_strength',
            'ema_aligned_bull', 'ema_aligned_bear',
            
            # Momentum
            'macd_bull', 'macd_bear', 
            'momentum_bull', 'rsi_rising',
            'is_rsi_bull',
            
            # Volatility
            'atr_pct', 'is_low_vol', 'bb_width',
            'bb_oversold', 'bb_overbought',
            
            # Volume
            'is_high_vol', 'vol_ratio',
            
            # Price Action
            'is_strong_candle', 'is_bull_candle', 'body_ratio',
            
            # Support/Resistance
            'near_support', 'near_resistance',
            'dist_from_support', 'dist_from_resistance'
        ]

    def prepare_labels(self, df, sl_pct=1.5, tp_pct=2.0):
        """
        Optimized Labeling with balanced SL/TP ratio
        SL: 1.5%, TP: 2.0% (Ratio ~1:1.33 for higher win rate)
        """
        df = df.copy()
        df['target'] = 0
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        targets = np.zeros(len(df))
        
        # Lookahead window - 30 candles (7.5 hours on 15m)
        window = 30
        
        sl_mult = 1 - (sl_pct / 100)
        tp_mult = 1 + (tp_pct / 100)
        
        for i in range(len(df) - window):
            entry_price = closes[i]
            sl = entry_price * sl_mult
            tp = entry_price * tp_mult
            
            fut_highs = highs[i+1 : i+1+window]
            fut_lows = lows[i+1 : i+1+window]
            
            tp_hits = np.where(fut_highs >= tp)[0]
            sl_hits = np.where(fut_lows <= sl)[0]
            
            first_tp = tp_hits[0] if tp_hits.size > 0 else 999
            first_sl = sl_hits[0] if sl_hits.size > 0 else 999
            
            if first_tp < first_sl:
                targets[i] = 1
                
        df['target'] = targets
        return df

    def train(self, df, epochs=None):
        if epochs:
            self.model.set_params(n_estimators=epochs)

        # Filter features that exist in dataframe
        available_features = [f for f in self.features if f in df.columns]
        
        X = df[available_features]
        y = df['target']
        
        # Fill NaN with 0
        X = X.fillna(0)
        
        if len(np.unique(y)) < 2:
            print("Warning: Not enough target classes for training (need 0 and 1).")
            return

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        print(f"Training XGBoost (Max Epochs: {self.model.n_estimators})...")
        print(f"Using {len(available_features)} features: {available_features}")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Store feature list for prediction
        self.trained_features = available_features
        
        print(f"Best Iteration: {self.model.best_iteration}")
        score = self.model.score(X_val, y_val)
        print(f"Validation Accuracy: {score:.2f}")
        
        # Feature importance
        importance = self.model.feature_importances_
        feat_imp = sorted(zip(available_features, importance), key=lambda x: x[1], reverse=True)
        print("\nTop 10 Features:")
        for feat, imp in feat_imp[:10]:
            print(f"  {feat}: {imp:.3f}")

    def predict_probability(self, df_row):
        X = df_row[self.trained_features]
        return self.model.predict_proba(X)[:, 1][0]
    
    def get_trained_features(self):
        return getattr(self, 'trained_features', self.features)

