import pandas as pd
import numpy as np

class SMCAnalysis:
    @staticmethod
    def find_fvg(df):
        """
        Vectorized FVG detection
        """
        highs = df['high'].values
        lows = df['low'].values
        
        fvg_bull = np.zeros(len(df), dtype=int)
        fvg_bull[:-2] = (lows[2:] > highs[:-2]).astype(int)
        
        fvg_bear = np.zeros(len(df), dtype=int)
        fvg_bear[:-2] = (highs[2:] < lows[:-2]).astype(int)
        
        df['fvg_bull'] = fvg_bull
        df['fvg_bear'] = fvg_bear
        
        df['fvg_equi'] = 0.0
        mask_bull = fvg_bull.astype(bool)
        df.loc[mask_bull, 'fvg_equi'] = (highs[mask_bull] + lows[np.where(mask_bull)[0] + 2]) / 2
        
        df['fvg_size'] = 0.0
        df.loc[mask_bull, 'fvg_size'] = lows[np.where(mask_bull)[0] + 2] - highs[mask_bull]
        mask_bear = fvg_bear.astype(bool)
        df.loc[mask_bear, 'fvg_size'] = lows[mask_bear] - highs[np.where(mask_bear)[0] + 2]
        
        return df

    @staticmethod
    def calculate_atr(df, period=14):
        """Calculate Average True Range for volatility"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        df['atr'] = tr.rolling(window=period).mean()
        df['atr_pct'] = df['atr'] / close * 100  # ATR as percentage
        df['is_low_vol'] = (df['atr_pct'] < df['atr_pct'].rolling(20).mean()).astype(int)
        return df
    
    @staticmethod
    def calculate_macd(df, fast=12, slow=26, signal=9):
        """Calculate MACD for momentum"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bullish: MACD crosses above signal or histogram positive and growing
        df['macd_bull'] = ((df['macd'] > df['macd_signal']) & (df['macd_hist'] > df['macd_hist'].shift(1))).astype(int)
        df['macd_bear'] = ((df['macd'] < df['macd_signal']) & (df['macd_hist'] < df['macd_hist'].shift(1))).astype(int)
        return df
    
    @staticmethod
    def calculate_adx(df, period=14):
        """Calculate ADX for trend strength"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothed values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(window=period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Strong trend when ADX > 25
        df['is_trending'] = (df['adx'] > 25).astype(int)
        df['trend_strength'] = df['adx'] / 50  # Normalized 0-1 (capped at 50)
        df['trend_strength'] = df['trend_strength'].clip(0, 1)
        return df
    
    @staticmethod
    def calculate_bollinger(df, period=20, std_dev=2):
        """Calculate Bollinger Bands for mean reversion signals"""
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
        
        # Band width for volatility
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Price position in bands (0=lower, 1=upper)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Bullish when bouncing from lower band, bearish from upper
        df['bb_oversold'] = (df['bb_position'] < 0.2).astype(int)
        df['bb_overbought'] = (df['bb_position'] > 0.8).astype(int)
        return df

    @staticmethod
    def detect_structure(df, symbol="BTC/USDT"):
        """
        Enhanced Structure Detection with more indicators
        """
        window = 10 if "XRP" in symbol else 5
        
        # 1. Swing Detection
        df['swing_high'] = df['high'].rolling(window=window*2+1, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(window=window*2+1, center=True).min() == df['low']
        
        # 2. CHoCH (Change of Character)
        last_sh = df['high'].where(df['swing_high']).ffill()
        last_sl = df['low'].where(df['swing_low']).ffill()
        df['choch_bull'] = (df['close'] > last_sh).astype(int)
        df['choch_bear'] = (df['close'] < last_sl).astype(int)
        
        # 3. EMA Filter (Multiple timeframes)
        df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_mid'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # EMA alignment (strong uptrend)
        df['ema_aligned_bull'] = ((df['ema_fast'] > df['ema_mid']) & (df['ema_mid'] > df['ema_slow'])).astype(int)
        df['ema_aligned_bear'] = ((df['ema_fast'] < df['ema_mid']) & (df['ema_mid'] < df['ema_slow'])).astype(int)
        
        df['is_uptrend'] = (df['close'] > df['ema_mid']).astype(int)
        df['is_downtrend'] = (df['close'] < df['ema_mid']).astype(int)
        
        # EMA Crossover signals
        df['ema_cross_bull'] = ((df['ema_fast'] > df['ema_mid']) & (df['ema_fast'].shift(1) <= df['ema_mid'].shift(1))).astype(int)
        df['ema_cross_bear'] = ((df['ema_fast'] < df['ema_mid']) & (df['ema_fast'].shift(1) >= df['ema_mid'].shift(1))).astype(int)
        
        # 4. RSI with divergence zones
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI zones
        df['is_rsi_bull'] = ((df['rsi'] > 40) & (df['rsi'] < 70)).astype(int)  # Sweet spot for longs
        df['is_rsi_bear'] = ((df['rsi'] > 30) & (df['rsi'] < 60)).astype(int)  # Sweet spot for shorts
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # RSI momentum
        df['rsi_rising'] = (df['rsi'] > df['rsi'].shift(1)).astype(int)
        
        # 5. Volume Analysis
        df['vol_sma'] = df['volume'].rolling(window=20).mean()
        df['is_high_vol'] = (df['volume'] > df['vol_sma'] * 1.5).astype(int)
        df['vol_ratio'] = df['volume'] / (df['vol_sma'] + 1e-10)
        
        # 6. Momentum (Rate of Change)
        df['roc'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
        df['momentum_bull'] = (df['roc'] > 0).astype(int)
        df['momentum_bear'] = (df['roc'] < 0).astype(int)
        
        # 7. Price action score
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']
        df['body_ratio'] = df['candle_body'] / (df['candle_range'] + 1e-10)
        df['is_strong_candle'] = (df['body_ratio'] > 0.6).astype(int)
        df['is_bull_candle'] = ((df['close'] > df['open']) & df['is_strong_candle']).astype(int)
        
        return df

    @staticmethod
    def find_order_blocks(df):
        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # Enhanced OB detection with body size filter
        body_size = np.abs(closes - opens)
        avg_body = pd.Series(body_size).rolling(20).mean().values
        
        bull_ob = np.zeros(len(df), dtype=int)
        bear_ob = np.zeros(len(df), dtype=int)
        
        for i in range(1, len(df) - 1):
            # Bullish OB: Significant bearish candle followed by strong bullish move
            if closes[i-1] < opens[i-1] and closes[i] > opens[i]:
                if body_size[i] > avg_body[i] * 0.8:  # Significant body
                    bull_ob[i-1] = 1
                    
            # Bearish OB: Significant bullish candle followed by strong bearish move
            if closes[i-1] > opens[i-1] and closes[i] < opens[i]:
                if body_size[i] > avg_body[i] * 0.8:
                    bear_ob[i-1] = 1

        df['is_bull_ob'] = bull_ob
        df['is_bear_ob'] = bear_ob
        return df
    
    @staticmethod
    def calculate_support_resistance(df, window=20):
        """Identify key support and resistance levels"""
        df['resistance'] = df['high'].rolling(window=window).max()
        df['support'] = df['low'].rolling(window=window).min()
        
        # Distance from S/R as percentage
        df['dist_from_resistance'] = (df['resistance'] - df['close']) / df['close'] * 100
        df['dist_from_support'] = (df['close'] - df['support']) / df['close'] * 100
        
        # Near support/resistance
        df['near_support'] = (df['dist_from_support'] < 1).astype(int)
        df['near_resistance'] = (df['dist_from_resistance'] < 1).astype(int)
        
        return df

    @classmethod
    def apply_all(cls, df, symbol="BTC/USDT"):
        df = df.copy()
        df = cls.find_fvg(df)
        df = cls.calculate_atr(df)
        df = cls.calculate_macd(df)
        df = cls.calculate_adx(df)
        df = cls.calculate_bollinger(df)
        df = cls.detect_structure(df, symbol)
        df = cls.find_order_blocks(df)
        df = cls.calculate_support_resistance(df)
        return df
