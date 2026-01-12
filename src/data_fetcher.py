import ccxt
import pandas as pd
import time
from datetime import datetime

class DataFetcher:
    def __init__(self, exchange_id='binance'):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })

    def fetch_ohlcv(self, symbol, timeframe='15m', limit=1000, since=None):
        """
        Fetch OHLCV data from exchange
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return None

    def get_historical_data(self, symbol, timeframe='15m', start_date="2022-01-01T00:00:00Z", end_date="2023-12-31T23:59:59Z"):
        """
        Fetch historical data in chunks to handle exchange limits
        """
        since = self.exchange.parse8601(start_date)
        end_ts = self.exchange.parse8601(end_date)
        all_ohlcv = []
        
        while since < end_ts:
            ohlcv = self.fetch_ohlcv(symbol, timeframe, limit=1000, since=since)
            if ohlcv is None or len(ohlcv) == 0:
                break
            
            last_ts = int(ohlcv.iloc[-1]['timestamp'].timestamp() * 1000)
            if last_ts == since: # Prevent infinite loop
                break
            
            since = last_ts + 1
            all_ohlcv.append(ohlcv)
            time.sleep(self.exchange.rateLimit / 1000) # Respect rate limit
            
            if len(all_ohlcv) % 10 == 0:
                print(f"  Fetched {len(pd.concat(all_ohlcv))} rows so far...")

        if not all_ohlcv: return None
        df = pd.concat(all_ohlcv).drop_duplicates('timestamp').reset_index(drop=True)
        return df[df['timestamp'] <= pd.to_datetime(end_date).tz_localize(None)]

    def get_multi_tf_data(self, symbol):
        """
        Get both 1h and 15m data for a symbol (Live Mode)
        """
        df_1h = self.fetch_ohlcv(symbol, '1h', limit=500)
        df_15m = self.fetch_ohlcv(symbol, '15m', limit=1000)
        return df_1h, df_15m

if __name__ == "__main__":
    fetcher = DataFetcher()
    for sym in ['BTC/USDT', 'XRP/USDT', 'SOL/USDT']:
        h1, m15 = fetcher.get_multi_tf_data(sym)
        if h1 is not None and m15 is not None:
            print(f"Fetched {sym}: 1h({len(h1)} rows), 15m({len(m15)} rows)")
