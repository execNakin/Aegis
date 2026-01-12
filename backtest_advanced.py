import pandas as pd
import numpy as np
import joblib
import os
from src.data_fetcher import DataFetcher
from src.smc_analysis import SMCAnalysis
from src.ml_model import MLModel
from src.risk_manager import RiskManager, PositionSizer
from src.advanced_strategy import ShortSignalModel, WalkForwardValidator
from train import get_data_cached

def run_advanced_backtest(
    initial_portfolio=10000,
    use_position_sizing=True,
    use_drawdown_protection=True,
    include_shorts=True,
    fixed_position_pct=10.0,  # Fixed 10% of portfolio per trade
    verbose=True
):
    """
    Advanced Backtest with:
    - Dynamic Position Sizing
    - Maximum Drawdown Protection
    - Long AND Short signals
    - Comprehensive metrics
    """
    print("=" * 60)
    print("ADVANCED BACKTEST (2023.07 to 2025.12)")
    print("=" * 60)
    print(f"Initial Portfolio: ${initial_portfolio:,.0f}")
    print(f"Position Size: {fixed_position_pct}% per trade")
    print(f"Position Sizing Adjustment: {'ON' if use_position_sizing else 'OFF'}")
    print(f"Drawdown Protection: {'ON' if use_drawdown_protection else 'OFF'}")
    print(f"Short Signals: {'ON' if include_shorts else 'OFF'}")
    print("=" * 60)
    
    # Load models
    if not os.path.exists('models/xgboost_smc.joblib'):
        print("Error: No trained model found. Please run train.py first.")
        return None
    
    model_data = joblib.load('models/xgboost_smc.joblib')
    if isinstance(model_data, dict):
        long_model = model_data['model']
        long_features = model_data['features']
    else:
        long_model = model_data
        long_features = ['fvg_bull', 'fvg_bear', 'fvg_size', 'is_uptrend', 'is_bull_ob', 'is_bear_ob', 'choch_bull', 'is_rsi_bull', 'is_high_vol']
    
    # Load short model if exists
    short_model = None
    short_features = None
    if include_shorts and os.path.exists('models/xgboost_short.joblib'):
        short_data = joblib.load('models/xgboost_short.joblib')
        short_model = short_data['model']
        short_features = short_data['features']
        print("Loaded Short signal model.")
    elif include_shorts:
        print("Warning: Short model not found. Run train_advanced.py first for short signals.")
        include_shorts = False
    
    fetcher = DataFetcher()
    analyzer = SMCAnalysis()
    
    symbols = ['BTC/USDT', 'XRP/USDT', 'SOL/USDT']
    backtest_start = pd.to_datetime("2023-07-01 00:00:00")
    
    # Initialize risk manager
    risk_manager = RiskManager(initial_portfolio=initial_portfolio)
    
    all_results = []
    all_trades = []
    
    for sym in symbols:
        if verbose:
            print(f"\n--- Processing {sym} ---")
        
        df = get_data_cached(fetcher, sym, "2023-07-01T00:00:00Z", "2025-12-31T23:59:59Z")
        if df is None:
            continue
        
        df = df[df['timestamp'] >= backtest_start].reset_index(drop=True)
        df = analyzer.apply_all(df, symbol=sym)
        
        # Prepare labels
        ml = MLModel()
        df = ml.prepare_labels(df, sl_pct=1.5, tp_pct=2.0)
        
        if include_shorts:
            short_model_temp = ShortSignalModel()
            df = short_model_temp.prepare_short_labels(df, sl_pct=1.5, tp_pct=2.0)
        
        df = df.dropna()
        
        if len(df) == 0:
            continue
        
        # Get predictions
        available_long_features = [f for f in long_features if f in df.columns]
        X_long = df[available_long_features].fillna(0)
        long_probs = long_model.predict_proba(X_long)[:, 1]
        
        if include_shorts and short_model is not None:
            available_short_features = [f for f in short_features if f in df.columns]
            X_short = df[available_short_features].fillna(0)
            short_probs = short_model.predict_proba(X_short)[:, 1]
        else:
            short_probs = np.zeros(len(df))
        
        # Generate signals with filters
        # Long signals
        long_signals = (
            (long_probs > 0.62) &
            ((df['is_uptrend'] == 1) | (df['ema_aligned_bull'] == 1) | (df['is_trending'] == 1)) &
            ((df['macd_bull'] == 1) | (df['momentum_bull'] == 1) | (df['rsi_rising'] == 1)) &
            ((df['is_bull_ob'] == 1) | (df['fvg_bull'] == 1) | (df['choch_bull'] == 1)) &
            (df['bb_overbought'] == 0) &
            (df['near_resistance'] == 0)
        )
        
        # Short signals
        if include_shorts:
            short_signals = (
                (short_probs > 0.62) &
                ((df['is_downtrend'] == 1) | (df['ema_aligned_bear'] == 1) | (df['is_trending'] == 1)) &
                ((df['macd_bear'] == 1) | (df['momentum_bear'] == 1)) &
                ((df['is_bear_ob'] == 1) | (df['fvg_bear'] == 1) | (df['choch_bear'] == 1)) &
                (df['bb_oversold'] == 0) &
                (df['near_support'] == 0)
            )
        else:
            short_signals = np.zeros(len(df), dtype=bool)
        
        # Simulate trades
        symbol_trades = []
        avg_atr = df['atr_pct'].mean() if 'atr_pct' in df.columns else 1.0
        
        for i in range(len(df)):
            if not (long_signals.iloc[i] or short_signals.iloc[i]):
                continue
            
            is_long = long_signals.iloc[i]
            prob = long_probs[i] if is_long else short_probs[i]
            target = df.iloc[i]['target'] if is_long else df.iloc[i].get('target_short', 0)
            
            current_atr = df.iloc[i]['atr_pct'] if 'atr_pct' in df.columns else avg_atr
            
            # Fixed position size based on initial portfolio
            base_position_value = initial_portfolio * (fixed_position_pct / 100)
            
            # Apply dynamic sizing adjustments (cap at 2x base)
            if use_position_sizing:
                # Adjust based on probability (0.5-1.5x)
                prob_multiplier = 0.5 + (prob - 0.5) * 2
                prob_multiplier = max(0.5, min(1.5, prob_multiplier))
                
                # Adjust based on volatility (inverse, 0.5-1.5x)
                if current_atr > 0 and avg_atr > 0:
                    vol_multiplier = avg_atr / current_atr
                    vol_multiplier = max(0.5, min(1.5, vol_multiplier))
                else:
                    vol_multiplier = 1.0
                
                position_value = base_position_value * prob_multiplier * vol_multiplier
                position_value = min(position_value, base_position_value * 2)  # Cap at 2x
                
                # Check drawdown protection
                dd_mult, is_stopped = risk_manager.drawdown_protection.update(risk_manager.portfolio_value)
                if is_stopped:
                    continue  # Skip trade due to drawdown protection
                
                position_value *= dd_mult
                risk_pct = (position_value / initial_portfolio) * 100
                status = "DYNAMIC" if dd_mult == 1.0 else f"REDUCED_{int(dd_mult*100)}%"
            else:
                position_value = base_position_value
                risk_pct = fixed_position_pct
                status = "FIXED"
            
            # Calculate PnL (as dollar amount)
            if target == 1:  # Win
                pnl = position_value * 0.02  # 2% TP gain on position
            else:  # Loss
                pnl = -position_value * 0.015  # 1.5% SL loss on position
            
            # Update portfolio
            risk_manager.update_portfolio(pnl)
            
            trade = {
                'symbol': sym,
                'timestamp': df.iloc[i]['timestamp'],
                'direction': 'LONG' if is_long else 'SHORT',
                'probability': prob,
                'position_size': position_value,
                'risk_pct': risk_pct,
                'status': status,
                'result': 'WIN' if target == 1 else 'LOSS',
                'pnl': pnl,
                'portfolio': risk_manager.portfolio_value
            }
            symbol_trades.append(trade)
        
        # Calculate symbol stats
        if symbol_trades:
            trades_df = pd.DataFrame(symbol_trades)
            wins = len(trades_df[trades_df['result'] == 'WIN'])
            losses = len(trades_df[trades_df['result'] == 'LOSS'])
            total = wins + losses
            win_rate = wins / total * 100 if total > 0 else 0
            total_pnl = trades_df['pnl'].sum()
            
            long_trades = len(trades_df[trades_df['direction'] == 'LONG'])
            short_trades = len(trades_df[trades_df['direction'] == 'SHORT'])
            
            result = {
                'symbol': sym,
                'total_trades': total,
                'long_trades': long_trades,
                'short_trades': short_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl
            }
            all_results.append(result)
            all_trades.extend(symbol_trades)
            
            if verbose:
                print(f"  Total Trades: {total} (Long: {long_trades}, Short: {short_trades})")
                print(f"  Wins: {wins}, Losses: {losses}")
                print(f"  Win Rate: {win_rate:.2f}%")
                print(f"  PnL: ${total_pnl:,.2f}")
    
    # Overall Summary
    stats = risk_manager.get_stats()
    
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Final Portfolio: ${risk_manager.portfolio_value:,.2f}")
    print(f"Total Return: {stats['total_return_pct']:.2f}%")
    print(f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        long_count = len(trades_df[trades_df['direction'] == 'LONG'])
        short_count = len(trades_df[trades_df['direction'] == 'SHORT'])
        print(f"Long Trades: {long_count}, Short Trades: {short_count}")
        
        # Stopped trades due to drawdown
        stopped = len(trades_df[trades_df['status'] == 'STOPPED'])
        if stopped > 0:
            print(f"Trades Blocked (Drawdown Protection): {stopped}")
    
    print("=" * 60)
    
    return {
        'final_portfolio': risk_manager.portfolio_value,
        'stats': stats,
        'results': all_results,
        'trades': all_trades
    }


def run_walk_forward_analysis():
    """Run Walk-Forward Validation analysis"""
    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)
    
    fetcher = DataFetcher()
    analyzer = SMCAnalysis()
    ml = MLModel()
    
    symbols = ['BTC/USDT', 'XRP/USDT', 'SOL/USDT']
    all_data = []
    
    for sym in symbols:
        print(f"\nLoading {sym}...")
        df = get_data_cached(fetcher, sym, "2022-01-01T00:00:00Z", "2025-12-31T23:59:59Z")
        if df is None:
            continue
        
        df = analyzer.apply_all(df, symbol=sym)
        df = ml.prepare_labels(df, sl_pct=1.5, tp_pct=2.0)
        df = df.dropna()
        all_data.append(df)
    
    if not all_data:
        print("No data available.")
        return None
    
    combined_df = pd.concat(all_data)
    
    # Run walk-forward validation
    validator = WalkForwardValidator(train_months=6, test_months=1, step_months=1)
    summary, predictions = validator.run_validation(combined_df, ml.features)
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward validation')
    parser.add_argument('--no-sizing', action='store_true', help='Disable position sizing')
    parser.add_argument('--no-protection', action='store_true', help='Disable drawdown protection')
    parser.add_argument('--no-shorts', action='store_true', help='Disable short signals')
    parser.add_argument('--portfolio', type=float, default=10000, help='Initial portfolio value')
    args = parser.parse_args()
    
    if args.walk_forward:
        run_walk_forward_analysis()
    else:
        run_advanced_backtest(
            initial_portfolio=args.portfolio,
            use_position_sizing=not args.no_sizing,
            use_drawdown_protection=not args.no_protection,
            include_shorts=not args.no_shorts
        )
