"""
Real-time processor for streaming data
Processes WebSocket, Kafka, and queue messages into features
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)


class RealTimeProcessor:
    """Processes streaming data in real-time for ML features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_buffers: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(deque)
        )
        self.feature_cache: Dict[str, Any] = {}
        
        # Processing pipelines
        self.pipelines: Dict[str, List[Callable]] = {}
        
        # Statistics
        self.stats = {
            "messages_processed": 0,
            "features_generated": 0,
            "processing_time_ms": [],
            "buffer_sizes": {},
            "errors": 0
        }
        
        # Initialize buffers based on config
        self._initialize_buffers()
    
    def _initialize_buffers(self):
        """Initialize data buffers for different timeframes"""
        buffer_configs = self.config.get("buffers", {
            "trade": {"max_size": 1000, "time_window": 300},  # 5 minutes
            "kline_1m": {"max_size": 100, "time_window": 600},  # 10 minutes
            "kline_5m": {"max_size": 50, "time_window": 3000},  # 50 minutes
            "orderbook": {"max_size": 100, "time_window": 60}   # 1 minute
        })
        
        for data_type, config in buffer_configs.items():
            self.stats["buffer_sizes"][data_type] = {
                "max_size": config["max_size"],
                "current_size": 0
            }
    
    def add_pipeline(self, pipeline_name: str, processors: List[Callable]):
        """Add processing pipeline"""
        self.pipelines[pipeline_name] = processors
        logger.info(f"Added pipeline '{pipeline_name}' with {len(processors)} processors")
    
    async def process_message(self, message: Dict[str, Any]):
        """Process incoming message from any source"""
        start_time = datetime.now()
        
        try:
            # Extract message metadata
            exchange = message.get("exchange", "unknown")
            symbol = message.get("symbol", "unknown")
            channel = message.get("channel", "unknown")
            data = message.get("data", {})
            
            # Create message key for buffering
            buffer_key = f"{exchange}:{symbol}:{channel}"
            
            # Add timestamp if not present
            if "timestamp" not in data:
                data["timestamp"] = datetime.now().isoformat()
            
            # Add to appropriate buffer
            await self._add_to_buffer(buffer_key, channel, data)
            
            # Process through pipelines
            processed_features = await self._process_through_pipelines(
                buffer_key, channel, data
            )
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.stats["messages_processed"] += 1
            self.stats["features_generated"] += len(processed_features) if processed_features else 0
            self.stats["processing_time_ms"].append(processing_time)
            
            # Keep only last 1000 processing times
            if len(self.stats["processing_time_ms"]) > 1000:
                self.stats["processing_time_ms"] = self.stats["processing_time_ms"][-1000:]
            
            # Log performance periodically
            if self.stats["messages_processed"] % 100 == 0:
                avg_time = np.mean(self.stats["processing_time_ms"]) if self.stats["processing_time_ms"] else 0
                logger.info(
                    f"Processed {self.stats['messages_processed']} messages, "
                    f"avg time: {avg_time:.2f}ms"
                )
            
            return processed_features
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.stats["errors"] += 1
            return None
    
    async def _add_to_buffer(self, buffer_key: str, channel: str, data: Dict[str, Any]):
        """Add data to appropriate buffer"""
        # Get buffer for this channel
        if channel not in self.data_buffers[buffer_key]:
            max_size = self.config.get("buffers", {}).get(channel, {}).get("max_size", 100)
            self.data_buffers[buffer_key][channel] = deque(maxlen=max_size)
        
        # Add to buffer
        self.data_buffers[buffer_key][channel].append(data)
        
        # Update buffer size stat
        if channel in self.stats["buffer_sizes"]:
            self.stats["buffer_sizes"][channel]["current_size"] = len(
                self.data_buffers[buffer_key][channel]
            )
    
    async def _process_through_pipelines(self, buffer_key: str, channel: str, 
                                        data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through all registered pipelines"""
        features = {}
        
        for pipeline_name, processors in self.pipelines.items():
            try:
                # Check if this pipeline should process this channel
                if not self._should_process_pipeline(pipeline_name, channel):
                    continue
                
                # Get buffer data for this pipeline
                buffer_data = self._get_buffer_data(buffer_key, pipeline_name)
                
                # Process through each processor in the pipeline
                pipeline_result = data.copy()
                
                for processor in processors:
                    pipeline_result = await processor(pipeline_result, buffer_data)
                    if pipeline_result is None:
                        break
                
                if pipeline_result:
                    # Add to features with pipeline name as prefix
                    for key, value in pipeline_result.items():
                        if key != "data":  # Don't include raw data in features
                            features[f"{pipeline_name}_{key}"] = value
                
            except Exception as e:
                logger.error(f"Pipeline '{pipeline_name}' error: {e}")
                continue
        
        return features
    
    def _should_process_pipeline(self, pipeline_name: str, channel: str) -> bool:
        """Check if pipeline should process this channel"""
        # Simple rule: pipelines specify which channels they handle
        pipeline_config = self.config.get("pipelines", {}).get(pipeline_name, {})
        allowed_channels = pipeline_config.get("channels", ["*"])
        
        return "*" in allowed_channels or channel in allowed_channels
    
    def _get_buffer_data(self, buffer_key: str, pipeline_name: str) -> Dict[str, List[Dict]]:
        """Get buffer data for pipeline processing"""
        pipeline_config = self.config.get("pipelines", {}).get(pipeline_name, {})
        lookback_channels = pipeline_config.get("lookback_channels", ["trade", "kline_1m"])
        
        buffer_data = {}
        
        for channel in lookback_channels:
            if channel in self.data_buffers[buffer_key]:
                buffer_data[channel] = list(self.data_buffers[buffer_key][channel])
            else:
                buffer_data[channel] = []
        
        return buffer_data
    
    async def generate_realtime_features(self, symbol: str, exchange: str = "binance") -> Dict[str, Any]:
        """Generate real-time features for ML inference"""
        buffer_key = f"{exchange}:{symbol}"
        
        # Check if we have enough data
        if buffer_key not in self.data_buffers or not self.data_buffers[buffer_key]:
            logger.warning(f"Insufficient data for {buffer_key}")
            return {}
        
        features = {}
        
        try:
            # 1. Price-based features
            price_features = await self._generate_price_features(buffer_key)
            features.update(price_features)
            
            # 2. Volume features
            volume_features = await self._generate_volume_features(buffer_key)
            features.update(volume_features)
            
            # 3. Order book features (if available)
            orderbook_features = await self._generate_orderbook_features(buffer_key)
            features.update(orderbook_features)
            
            # 4. Technical indicators (simplified real-time)
            technical_features = await self._generate_technical_features(buffer_key)
            features.update(technical_features)
            
            # 5. Market microstructure features
            microstructure_features = await self._generate_microstructure_features(buffer_key)
            features.update(microstructure_features)
            
            # Add timestamp
            features["timestamp"] = datetime.now().isoformat()
            features["symbol"] = symbol
            features["exchange"] = exchange
            
            # Cache features
            cache_key = f"{buffer_key}:{int(datetime.now().timestamp() // 60)}"  # Cache by minute
            self.feature_cache[cache_key] = {
                "features": features,
                "timestamp": datetime.now().isoformat()
            }
            
            # Clean old cache entries
            await self._clean_feature_cache()
            
            logger.debug(f"Generated {len(features)} features for {symbol}")
            return features
            
        except Exception as e:
            logger.error(f"Error generating features for {symbol}: {e}")
            return {}
    
    async def _generate_price_features(self, buffer_key: str) -> Dict[str, float]:
        """Generate price-based features"""
        features = {}
        
        try:
            # Get latest kline data
            if "kline_1m" in self.data_buffers[buffer_key]:
                klines = list(self.data_buffers[buffer_key]["kline_1m"])
                
                if len(klines) >= 2:
                    latest = klines[-1]
                    previous = klines[-2]
                    
                    # Price change
                    if "c" in latest and "c" in previous:
                        price_change = (latest["c"] - previous["c"]) / previous["c"]
                        features["price_change_1m"] = price_change
                        features["price_momentum"] = price_change * 100  # Percentage
                    
                    # Price volatility (simplified)
                    if len(klines) >= 10:
                        closes = [k.get("c", 0) for k in klines[-10:] if "c" in k]
                        if len(closes) >= 5:
                            returns = np.diff(closes) / closes[:-1]
                            features["volatility_10m"] = np.std(returns) if len(returns) > 0 else 0
            
            # Get trade data for immediate price action
            if "trade" in self.data_buffers[buffer_key]:
                trades = list(self.data_buffers[buffer_key]["trade"])
                
                if trades:
                    recent_trades = trades[-50:]  # Last 50 trades
                    
                    if recent_trades:
                        # Trade price statistics
                        prices = [t.get("p", 0) for t in recent_trades if "p" in t]
                        if prices:
                            features["avg_trade_price"] = np.mean(prices)
                            features["trade_price_std"] = np.std(prices)
                        
                        # Buy/sell pressure
                        buys = sum(1 for t in recent_trades if t.get("m", False) is False)  # m=False is buy
                        sells = len(recent_trades) - buys
                        features["buy_sell_ratio"] = buys / (sells + 1)  # Avoid division by zero
        
        except Exception as e:
            logger.error(f"Error generating price features: {e}")
        
        return features
    
    async def _generate_volume_features(self, buffer_key: str) -> Dict[str, float]:
        """Generate volume-based features"""
        features = {}
        
        try:
            if "kline_1m" in self.data_buffers[buffer_key]:
                klines = list(self.data_buffers[buffer_key]["kline_1m"])
                
                if len(klines) >= 5:
                    # Recent volume
                    volumes = [k.get("v", 0) for k in klines[-5:] if "v" in k]
                    
                    if volumes:
                        features["volume_5m"] = sum(volumes)
                        features["volume_avg_5m"] = np.mean(volumes)
                        
                        # Volume spike detection
                        if len(volumes) >= 3:
                            avg_volume = np.mean(volumes[:-1])
                            current_volume = volumes[-1]
                            features["volume_spike_ratio"] = current_volume / (avg_volume + 1)
            
            if "trade" in self.data_buffers[buffer_key]:
                trades = list(self.data_buffers[buffer_key]["trade"])
                
                if len(trades) >= 20:
                    # Trade volume statistics
                    trade_volumes = [t.get("q", 0) for t in trades[-20:] if "q" in t]
                    
                    if trade_volumes:
                        features["trade_volume_20t"] = sum(trade_volumes)
                        features["trade_volume_std"] = np.std(trade_volumes)
        
        except Exception as e:
            logger.error(f"Error generating volume features: {e}")
        
        return features
    
    async def _generate_orderbook_features(self, buffer_key: str) -> Dict[str, float]:
        """Generate order book features"""
        features = {}
        
        try:
            if "orderbook" in self.data_buffers[buffer_key]:
                orderbooks = list(self.data_buffers[buffer_key]["orderbook"])
                
                if orderbooks:
                    latest = orderbooks[-1]
                    
                    # Extract bids and asks
                    bids = latest.get("bids", [])
                    asks = latest.get("asks", [])
                    
                    if bids and asks:
                        # Best bid and ask
                        best_bid = float(bids[0][0]) if bids else 0
                        best_ask = float(asks[0][0]) if asks else 0
                        
                        if best_bid > 0 and best_ask > 0:
                            # Spread
                            spread = best_ask - best_bid
                            features["spread"] = spread
                            features["spread_percentage"] = spread / best_bid
                            
                            # Mid price
                            features["mid_price"] = (best_bid + best_ask) / 2
                            
                            # Order book imbalance
                            bid_volume = sum(float(b[1]) for b in bids[:10])  # Top 10 bids
                            ask_volume = sum(float(a[1]) for a in asks[:10])  # Top 10 asks
                            
                            total_volume = bid_volume + ask_volume
                            if total_volume > 0:
                                features["orderbook_imbalance"] = (bid_volume - ask_volume) / total_volume
        
        except Exception as e:
            logger.error(f"Error generating orderbook features: {e}")
        
        return features
    
    async def _generate_technical_features(self, buffer_key: str) -> Dict[str, float]:
        """Generate simplified technical indicators in real-time"""
        features = {}
        
        try:
            if "kline_1m" in self.data_buffers[buffer_key]:
                klines = list(self.data_buffers[buffer_key]["kline_1m"])
                
                if len(klines) >= 20:
                    closes = [k.get("c", 0) for k in klines[-20:] if "c" in k]
                    
                    if len(closes) >= 20:
                        # Simple moving averages
                        features["sma_10"] = np.mean(closes[-10:])
                        features["sma_20"] = np.mean(closes)
                        
                        # Price position relative to SMA
                        current_price = closes[-1]
                        features["price_vs_sma10"] = current_price / features["sma_10"]
                        features["price_vs_sma20"] = current_price / features["sma_20"]
                        
                        # Simple RSI (simplified)
                        gains = []
                        losses = []
                        
                        for i in range(1, len(closes)):
                            change = closes[i] - closes[i-1]
                            if change > 0:
                                gains.append(change)
                            else:
                                losses.append(abs(change))
                        
                        avg_gain = np.mean(gains) if gains else 0
                        avg_loss = np.mean(losses) if losses else 0
                        
                        if avg_loss > 0:
                            rs = avg_gain / avg_loss
                            features["rsi_simple"] = 100 - (100 / (1 + rs))
        
        except Exception as e:
            logger.error(f"Error generating technical features: {e}")
        
        return features
    
    async def _generate_microstructure_features(self, buffer_key: str) -> Dict[str, float]:
        """Generate market microstructure features"""
        features = {}
        
        try:
            if "trade" in self.data_buffers[buffer_key]:
                trades = list(self.data_buffers[buffer_key]["trade"])
                
                if len(trades) >= 10:
                    # Trade direction sequence
                    directions = []
                    for trade in trades[-10:]:
                        # m=False is buy, m=True is sell (Binance convention)
                        is_buy = not trade.get("m", False)
                        directions.append(1 if is_buy else -1)
                    
                    # Buy/sell imbalance
                    features["buy_sell_imbalance_10t"] = sum(directions) / len(directions)
                    
                    # Autocorrelation of trade signs
                    if len(directions) >= 5:
                        lag1_corr = np.corrcoef(directions[:-1], directions[1:])[0, 1]
                        features["trade_sign_autocorr"] = lag1_corr if not np.isnan(lag1_corr) else 0
            
            # Volatility clustering (simplified)
            if "kline_1m" in self.data_buffers[buffer_key]:
                klines = list(self.data_buffers[buffer_key]["kline_1m"])
                
                if len(klines) >= 10:
                    highs = [k.get("h", 0) for k in klines[-10:]]
                    lows = [k.get("l", 0) for k in klines[-10:]]
                    
                    if len(highs) == 10 and len(lows) == 10:
                        ranges = [h - l for h, l in zip(highs, lows)]
                        avg_range = np.mean(ranges)
                        features["avg_true_range_10m"] = avg_range
        
        except Exception as e:
            logger.error(f"Error generating microstructure features: {e}")
        
        return features
    
    async def _clean_feature_cache(self):
        """Clean old entries from feature cache"""
        now = datetime.now()
        keys_to_delete = []
        
        for key, value in self.feature_cache.items():
            cache_time = datetime.fromisoformat(value["timestamp"])
            if (now - cache_time).total_seconds() > 300:  # 5 minutes
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.feature_cache[key]
    
    async def get_cached_features(self, symbol: str, exchange: str = "binance", 
                                 max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
        """Get cached features if recent enough"""
        cache_key = f"{exchange}:{symbol}:{int(datetime.now().timestamp() // 60)}"
        
        if cache_key in self.feature_cache:
            cached = self.feature_cache[cache_key]
            cache_time = datetime.fromisoformat(cached["timestamp"])
            
            if (datetime.now() - cache_time).total_seconds() <= max_age_seconds:
                return cached["features"]
        
        return None
    
    async def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        stats = {}
        
        for buffer_key, channels in self.data_buffers.items():
            stats[buffer_key] = {}
            for channel, buffer in channels.items():
                stats[buffer_key][channel] = len(buffer)
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        avg_processing_time = np.mean(self.stats["processing_time_ms"]) if self.stats["processing_time_ms"] else 0
        
        return {
            **self.stats,
            "avg_processing_time_ms": avg_processing_time,
            "pipelines": len(self.pipelines),
            "feature_cache_size": len(self.feature_cache)
        }
