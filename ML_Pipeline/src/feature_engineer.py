"""
Advanced feature engineering for trading ML
Creates 50+ technical, statistical, and derived features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import pandas_ta as ta
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Creates comprehensive feature set for trading ML"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_cache = {}
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features from OHLCV data"""
        
        print("🔧 Creating features...")
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create copy to avoid modifying original
        df_features = df.copy()
        
        # 1. Price-based features
        df_features = self._create_price_features(df_features)
        
        # 2. Technical indicators
        df_features = self._create_technical_indicators(df_features)
        
        # 3. Statistical features
        df_features = self._create_statistical_features(df_features)
        
        # 4. Lag features
        df_features = self._create_lag_features(df_features)
        
        # 5. Rolling window features
        df_features = self._create_rolling_features(df_features)
        
        # 6. Volume features
        df_features = self._create_volume_features(df_features)
        
        # 7. Market microstructure features
        df_features = self._create_microstructure_features(df_features)
        
        # 8. Time-based features
        df_features = self._create_time_features(df_features)
        
        # Remove NaN values
        df_features = df_features.dropna()
        
        print(f"✅ Created {len(df_features.columns)} features")
        print(f"   Samples: {len(df_features)}")
        
        return df_features
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price position within day range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
        
        # Normalized price (z-score)
        df['price_zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        # Price acceleration (second derivative)
        df['price_acceleration'] = df['returns'].diff()
        
        # Support/resistance levels
        df['resistance_distance'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['support_distance'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
        
        return df
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators using pandas_ta"""
        
        # Momentum indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['stoch_k'] = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3).iloc[:, 0]
        df['stoch_d'] = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3).iloc[:, 1]
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        df['awesome_oscillator'] = ta.ao(df['high'], df['low'])
        
        # Trend indicators
        macd = ta.macd(df['close'])
        df['macd'] = macd.iloc[:, 0]
        df['macd_signal'] = macd.iloc[:, 1]
        df['macd_histogram'] = macd.iloc[:, 2]
        
        df['adx'] = ta.adx(df['high'], df['low'], df['close']).iloc[:, 0]
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        df['aroon_up'] = ta.aroon(df['high'], df['low'], length=25).iloc[:, 0]
        df['aroon_down'] = ta.aroon(df['high'], df['low'], length=25).iloc[:, 1]
        df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']
        
        # Volatility indicators
        bollinger = ta.bbands(df['close'], length=20)
        df['bb_upper'] = bollinger.iloc[:, 0]
        df['bb_middle'] = bollinger.iloc[:, 1]
        df['bb_lower'] = bollinger.iloc[:, 2]
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['normalized_atr'] = df['atr'] / df['close']
        
        keltner = ta.kc(df['high'], df['low'], df['close'])
        df['kc_upper'] = keltner.iloc[:, 0]
        df['kc_lower'] = keltner.iloc[:, 1]
        
        # Volume indicators
        df['obv'] = ta.obv(df['close'], df['volume'])
        df['volume_profile'] = df['volume'] / df['volume'].rolling(20).mean()
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # Cycle indicators
        df['dpo'] = ta.dpo(df['close'])
        
        # Custom composite indicators
        df['trend_strength'] = df['adx'] / 100.0
        df['momentum_score'] = (df['rsi'] / 100.0 + (50 - df['williams_r']) / 100.0) / 2
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        
        returns = df['returns'].dropna()
        
        # Volatility
        df['volatility_5'] = returns.rolling(5).std()
        df['volatility_20'] = returns.rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # Skewness and kurtosis
        df['skewness_20'] = returns.rolling(20).skew()
        df['kurtosis_20'] = returns.rolling(20).kurt()
        
        # Autocorrelation
        df['autocorr_1'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1))
        df['autocorr_5'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=5))
        
        # Hurst exponent (market efficiency)
        df['hurst_exponent'] = returns.rolling(100).apply(self._calculate_hurst)
        
        # Variance ratio test (random walk test)
        df['variance_ratio'] = returns.rolling(20).apply(self._calculate_variance_ratio)
        
        return df
    
    def _calculate_hurst(self, series):
        """Calculate Hurst exponent"""
        if len(series) < 10:
            return np.nan
        
        lags = range(2, 10)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        
        try:
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except:
            return np.nan
    
    def _calculate_variance_ratio(self, series):
        """Calculate variance ratio test statistic"""
        if len(series) < 10:
            return np.nan
        
        n = len(series)
        mu = np.mean(series)
        var_1 = np.var(series - mu)
        var_q = np.var(series[2:] - series[:-2]) / 2
        
        if var_1 == 0:
            return np.nan
        
        return var_q / var_1
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features"""
        
        lags = self.config['features']['lag_features']['lags']
        columns = self.config['features']['lag_features']['columns']
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Create return lags
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics"""
        
        windows = self.config['features']['rolling_features']['windows']
        functions = self.config['features']['rolling_features']['functions']
        price_cols = ['close', 'volume', 'returns']
        
        for window in windows:
            for col in price_cols:
                if col in df.columns:
                    series = df[col]
                    
                    if 'mean' in functions:
                        df[f'{col}_rolling_mean_{window}'] = series.rolling(window).mean()
                    
                    if 'std' in functions:
                        df[f'{col}_rolling_std_{window}'] = series.rolling(window).std()
                    
                    if 'min' in functions:
                        df[f'{col}_rolling_min_{window}'] = series.rolling(window).min()
                    
                    if 'max' in functions:
                        df[f'{col}_rolling_max_{window}'] = series.rolling(window).max()
                    
                    if 'median' in functions:
                        df[f'{col}_rolling_median_{window}'] = series.rolling(window).median()
                    
                    # Rolling Sharpe ratio (for returns)
                    if col == 'returns':
                        df[f'sharpe_ratio_{window}'] = (
                            series.rolling(window).mean() / 
                            series.rolling(window).std().replace(0, np.nan)
                        )
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        
        # Volume price trend
        df['volume_price_trend'] = df['volume'] * df['returns'].abs()
        
        # Volume relative to average
        df['volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
        df['volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volume acceleration
        df['volume_acceleration'] = df['volume'].pct_change()
        
        # Volume correlation with price
        df['volume_price_corr_10'] = df['volume'].rolling(10).corr(df['close'])
        
        # Large volume spikes
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        
        return df
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        
        # Bid-ask spread estimation (simplified)
        df['spread_estimate'] = (df['high'] - df['low']) / df['close']
        
        # Price impact
        df['price_impact'] = df['returns'].abs() / (df['volume'] + 1)
        
        # Order flow imbalance (simplified)
        df['ofi'] = (df['close'] - df['open']) / df['atr'].replace(0, np.nan)
        
        # Realized volatility
        df['realized_volatility'] = df['returns'].rolling(5).std()
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Time of day (for intraday patterns)
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            
            # Day of week
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Week of year
            df['week_of_year'] = df['timestamp'].dt.isocalendar().week
            
            # Month
            df['month'] = df['timestamp'].dt.month
            
            # Quarter
            df['quarter'] = df['timestamp'].dt.quarter
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def select_best_features(self, df: pd.DataFrame, target: pd.Series, top_k: int = 50) -> List[str]:
        """Select best features using mutual information"""
        
        from sklearn.feature_selection import mutual_info_classif
        
        # Drop non-numeric and target columns
        X = df.select_dtypes(include=[np.number])
        X = X.drop(columns=[col for col in X.columns if 'target' in col or 'future' in col], errors='ignore')
        
        # Align with target
        common_idx = X.index.intersection(target.index)
        X = X.loc[common_idx]
        y = target.loc[common_idx]
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create feature scores DataFrame
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Select top features
        selected_features = feature_scores.head(top_k)['feature'].tolist()
        
        print(f"📊 Selected {len(selected_features)} best features")
        print(f"   Top 10 features: {selected_features[:10]}")
        
        return selected_features
