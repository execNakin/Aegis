import json
import os

class Executor:
    def __init__(self, initial_balance=1000):
        self.balance = initial_balance
        self.risk_per_trade = 0.01  # 1%
        self.open_trades_path = 'opening.json'
        self.max_trades = 2

    def load_open_trades(self):
        if os.path.exists(self.open_trades_path):
            with open(self.open_trades_path, 'r') as f:
                try:
                    return json.load(f)
                except:
                    return {}
        return {}

    def save_open_trades(self, trades):
        with open(self.open_trades_path, 'w') as f:
            json.dump(trades, f, indent=4)

    def calculate_position_size(self, symbol, entry_price, sl_price):
        risk_amount = self.balance * self.risk_per_trade
        sl_percent = abs(entry_price - sl_price) / entry_price
        if sl_percent == 0: return 0
        position_value = risk_amount / sl_percent
        return position_value / entry_price

    def check_entry_conditions(self, symbol, df_15m, df_1h, prob):
        """
        Setup: 1h Bias is bullish, 15m Price taps into 15m Bullish OB, Prob > 75%
        """
        # 1h Bias
        bias_bullish = df_1h.iloc[-1]['is_uptrend'] == 1
        
        # 15m Setup
        last_15m = df_15m.iloc[-1]
        taps_ob = last_15m['is_bull_ob'] == 1
        
        if bias_bullish and taps_ob and prob > 0.75:
            open_trades = self.load_open_trades()
            if len(open_trades) < self.max_trades:
                return True
        return False

    def open_trade(self, symbol, entry_price, sl, tp, prob):
        trades = self.load_open_trades()
        trades[symbol] = {
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'probability': prob,
            'timestamp': str(pd.Timestamp.now()),
            'size': self.calculate_position_size(symbol, entry_price, sl)
        }
        self.save_open_trades(trades)
        print(f"Opened trade for {symbol} at {entry_price} with size {trades[symbol]['size']}")

import pandas as pd # Needed for timestamp
