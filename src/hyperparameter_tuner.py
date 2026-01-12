import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    """
    Hyperparameter Tuning using Optuna for:
    1. XGBoost model parameters
    2. Signal thresholds (probability, SL/TP)
    3. Trading strategy parameters
    """
    
    def __init__(self, df, features, n_trials=100):
        self.df = df.copy()
        self.features = features
        self.n_trials = n_trials
        self.best_params = None
        self.study = None
        
    def objective_xgboost(self, trial):
        """Objective function for XGBoost hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        available_features = [f for f in self.features if f in self.df.columns]
        X = self.df[available_features].fillna(0)
        y = self.df['target']
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = xgb.XGBClassifier(
            **params,
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=50,
            verbosity=0
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Predict and calculate profit-based score
        probs = model.predict_proba(X_val)[:, 1]
        
        # Simulate trading with threshold tuning
        threshold = trial.suggest_float('prob_threshold', 0.55, 0.80)
        
        signals = probs > threshold
        y_val_arr = y_val.values
        
        wins = np.sum(signals & (y_val_arr == 1))
        losses = np.sum(signals & (y_val_arr == 0))
        total = wins + losses
        
        if total == 0:
            return 0
        
        win_rate = wins / total
        profit = (wins * 2.0) - (losses * 1.5)  # SL=1.5%, TP=2%
        
        # Optimize for profit while maintaining reasonable number of trades
        score = profit * (1 + np.log1p(total) / 10)  # Bonus for more trades
        
        return score
    
    def objective_strategy(self, trial, model, X_val, y_val):
        """Objective function for strategy parameters"""
        # SL/TP optimization
        sl_pct = trial.suggest_float('sl_pct', 0.5, 3.0)
        tp_pct = trial.suggest_float('tp_pct', 1.0, 5.0)
        
        # Probability threshold
        prob_threshold = trial.suggest_float('prob_threshold', 0.50, 0.80)
        
        probs = model.predict_proba(X_val)[:, 1]
        signals = probs > prob_threshold
        
        wins = np.sum(signals & (y_val == 1))
        losses = np.sum(signals & (y_val == 0))
        total = wins + losses
        
        if total == 0:
            return 0
        
        profit = (wins * tp_pct) - (losses * sl_pct)
        win_rate = wins / total
        
        # Penalize very low win rates
        if win_rate < 0.4:
            profit *= 0.5
        
        return profit
    
    def tune_xgboost(self, n_trials=None):
        """Run hyperparameter tuning for XGBoost"""
        if n_trials is None:
            n_trials = self.n_trials
        
        print(f"Starting XGBoost hyperparameter tuning ({n_trials} trials)...")
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective_xgboost, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        print(f"\nBest parameters found:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"Best score: {self.study.best_value:.2f}")
        
        return self.best_params
    
    def get_optimized_model(self):
        """Train and return model with optimized parameters"""
        if self.best_params is None:
            raise ValueError("Run tune_xgboost() first")
        
        # Separate XGBoost params from strategy params
        xgb_params = {k: v for k, v in self.best_params.items() 
                      if k not in ['prob_threshold']}
        
        available_features = [f for f in self.features if f in self.df.columns]
        X = self.df[available_features].fillna(0)
        y = self.df['target']
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = xgb.XGBClassifier(
            **xgb_params,
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=50,
            verbosity=0
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        return model, available_features, self.best_params.get('prob_threshold', 0.62)


def run_hyperparameter_tuning(df, features, n_trials=50):
    """
    Convenience function to run hyperparameter tuning
    """
    tuner = HyperparameterTuner(df, features, n_trials=n_trials)
    best_params = tuner.tune_xgboost()
    model, trained_features, threshold = tuner.get_optimized_model()
    
    return {
        'model': model,
        'features': trained_features,
        'params': best_params,
        'threshold': threshold
    }
