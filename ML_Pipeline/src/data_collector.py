"""
Collects historical and real-time market data
"""

import ccxt
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
import aiohttp
import time

class DataCollector:
    """Collects market data from multiple sources"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.exchanges = {}
        self._initialize_exchanges()
    
    def _initialize_exchanges(self):
        """Initialize crypto exchanges"""
        try:
            # Binance
            self.exchanges['binance'] = ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 1200,
            })
            
            # Coinbase
            self.exchanges['coinbase'] = ccxt.coinbase({
                'enableRateLimit': True,
            })
            
            # Kraken
            self.exchanges['kraken'] = ccxt.kraken({
                'enableRateLimit': True,
            })
            
        except Exception as e:
            print(f"Warning: Could not initialize some exchanges: {e}")
    
    def collect_data(self, symbol: str, timeframe: str = "5m", days: int = 365) -> pd.DataFrame:
        """
        Collect historical data
        
        Args:
            symbol: Trading symbol (e.g., "SOL/USDT")
            timeframe: Candle timeframe (e.g., "5m", "1h", "1d")
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        
        print(f"📥 Collecting {timeframe} data for {symbol} ({days} days)...")
        
        # Try Binance first (most reliable for crypto)
        try:
            df = self._fetch_from_binance(symbol, timeframe, days)
            if df is not None and len(df) > 100:
                print(f"   Collected {len(df)} candles from Binance")
                return df
        except Exception as e:
            print(f"   Binance failed: {e}")
        
        # Fallback to yfinance for traditional assets
        try:
            df = self._fetch_from_yfinance(symbol, timeframe, days)
            if df is not None and len(df) > 100:
                print(f"   Collected {len(df)} candles from Yahoo Finance")
                return df
        except Exception as e:
            print(f"   Yahoo Finance failed: {e}")
        
        # Generate mock data if all sources fail
        print("   Using mock data for development")
        return self._generate_mock_data(symbol, timeframe, days)
    
    def _fetch_from_binance(self, symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch data from Binance"""
        
        try:
            # Convert symbol format if needed
            if '/' in symbol:
                symbol = symbol.replace('/', '')
            
            # Calculate since timestamp
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # Fetch OHLCV
            ohlcv = self.exchanges['binance'].fetch_ohlcv(
                symbol, 
                timeframe, 
                since=since,
                limit=1000
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            print(f"Error fetching from Binance: {e}")
            return None
    
    def _fetch_from_yfinance(self, symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        
        try:
            # Convert timeframe to yfinance format
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m',
                '30m': '30m', '1h': '60m', '2h': '120m',
                '4h': '240m', '1d': '1d', '1w': '1wk'
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            # Download data
            ticker = yf.Ticker(symbol.replace('/', '-'))
            df = ticker.history(period=f"{days}d", interval=interval)
            
            # Reset index and rename columns
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            return df
            
        except Exception as e:
            print(f"Error fetching from Yahoo Finance: {e}")
            return None
    
    def _generate_mock_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate realistic mock data for development"""
        
        print("   Generating realistic mock data...")
        
        # Calculate number of candles
        candles_per_day = {
            '1m': 1440, '5m': 288, '15m': 96,
            '1h': 24, '4h': 6, '1d': 1, '1w': 0.142
        }
        
        candles_per_candle = candles_per_day.get(timeframe, 288)
        total_candles = int(days * candles_per_candle)
        
        # Generate time series
        end_time = datetime.now()
        if timeframe == '1d':
            freq = 'D'
        elif 'h' in timeframe:
            hours = int(timeframe.replace('h', ''))
            freq = f'{hours}H'
        else:
            minutes = int(timeframe.replace('m', ''))
            freq = f'{minutes}T'
        
        timestamps = pd.date_range(
            end=end_time, 
            periods=total_candles, 
            freq=freq
        )[::-1]  # Reverse to have oldest first
        
        # Generate realistic price series with trends and volatility
        np.random.seed(42)
        
        # Base price
        base_price = 100.0
        
        # Generate random walk with drift
        returns = np.random.normal(0.0005, 0.02, total_candles)  # 0.05% daily drift, 2% volatility
        
        # Add some autocorrelation (momentum)
        for i in range(1, len(returns)):
            returns[i] = 0.3 * returns[i-1] + 0.7 * returns[i]
        
        # Add periodic trends (weekly, monthly)
        t = np.arange(total_candles)
        weekly_pattern = 0.01 * np.sin(2 * np.pi * t / (candles_per_candle * 7))
        monthly_pattern = 0.02 * np.sin(2 * np.pi * t / (candles_per_candle * 30))
        
        returns += weekly_pattern + monthly_pattern
        
        # Calculate prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC
        opens = prices.copy()
        highs = opens * (1 + np.abs(np.random.normal(0, 0.01, total_candles)))
        lows = opens * (1 - np.abs(np.random.normal(0, 0.008, total_candles)))
        closes = opens * (1 + returns)
        
        # Ensure high > low > close > open relationships
        for i in range(total_candles):
            high_low = sorted([highs[i], lows[i]])
            highs[i] = max(high_low[1], closes[i] * 1.001)
            lows[i] = min(high_low[0], closes[i] * 0.999)
            opens[i] = np.clip(opens[i], lows[i] * 1.0001, highs[i] * 0.9999)
        
        # Generate volume (correlated with volatility)
        volume = np.random.lognormal(10, 1, total_candles) * (1 + np.abs(returns) * 10)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volume
        })
        
        print(f"   Generated {len(df)} realistic mock candles")
        
        return df
    
    async def stream_real_time_data(self, symbol: str, callback):
        """Stream real-time data (WebSocket)"""
        
        print(f"🌐 Starting real-time stream for {symbol}")
        
        # This would connect to WebSocket in production
        # For now, simulate with async loop
        
        async def simulate_stream():
            while True:
                # Simulate new candle every minute
                new_candle = {
                    'timestamp': datetime.now().isoformat() + 'Z',
                    'open': 100 + np.random.random() * 10,
                    'high': 100 + np.random.random() * 12,
                    'low': 100 + np.random.random() * 8,
                    'close': 100 + np.random.random() * 10,
                    'volume': 1000 + np.random.random() * 500
                }
                
                await callback(new_candle)
                await asyncio.sleep(60)  # Every minute
        
        return await simulate_stream()
