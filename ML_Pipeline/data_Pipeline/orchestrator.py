"""
Orchestrates the entire data pipeline
Coordinates streaming, alternative data, feature store, and validation
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import signal
import sys

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class PipelineConfig:
    """Data pipeline configuration"""
    # Streaming
    streaming_enabled: bool = True
    streaming_symbols: List[str] = None
    streaming_exchanges: List[str] = None
    
    # Alternative data
    alternative_data_enabled: bool = True
    sentiment_enabled: bool = True
    onchain_enabled: bool = True
    
    # Feature store
    feature_store_enabled: bool = True
    feature_store_backend: str = "redis"
    
    # Validation
    validation_enabled: bool = True
    anomaly_detection_enabled: bool = True
    
    # Performance
    processing_concurrency: int = 10
    batch_size: int = 100
    cache_ttl_minutes: int = 60
    
    def __post_init__(self):
        if self.streaming_symbols is None:
            self.streaming_symbols = ["SOL/USDT", "BTC/USDT", "ETH/USDT"]
        if self.streaming_exchanges is None:
            self.streaming_exchanges = ["binance", "coinbase"]


class DataPipelineOrchestrator:
    """Orchestrates the entire data pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.status = PipelineStatus.STOPPED
        
        # Initialize components
        self.components = {}
        self.message_queues = {}
        self.processing_tasks = []
        
        # Statistics
        self.stats = {
            "messages_processed": 0,
            "features_generated": 0,
            "errors": 0,
            "start_time": None,
            "uptime_seconds": 0
        }
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("Data pipeline orchestrator initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())
    
    async def initialize(self):
        """Initialize all pipeline components"""
        logger.info("Initializing data pipeline...")
        self.status = PipelineStatus.STARTING
        
        try:
            # Import components
            from data_pipeline.streaming.websocket_client import MultiExchangeWebSocketClient
            from data_pipeline.streaming.realtime_processor import RealTimeProcessor
            from data_pipeline.alternative_data.sentiment_analyzer import SentimentAnalyzer, SentimentConfig
            from data_pipeline.feature_store.feature_store import FeatureStore
            from data_pipeline.validation.data_validator import DataValidator
            
            # 1. Initialize streaming
            if self.config.streaming_enabled:
                await self._initialize_streaming()
            
            # 2. Initialize alternative data
            if self.config.alternative_data_enabled:
                await self._initialize_alternative_data()
            
            # 3. Initialize feature store
            if self.config.feature_store_enabled:
                await self._initialize_feature_store()
            
            # 4. Initialize validation
            if self.config.validation_enabled:
                await self._initialize_validation()
            
            # 5. Initialize real-time processor
            await self._initialize_realtime_processor()
            
            self.status = PipelineStatus.RUNNING
            self.stats["start_time"] = datetime.now()
            logger.info("Data pipeline initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize data pipeline: {e}")
            self.status = PipelineStatus.ERROR
            return False
    
    async def _initialize_streaming(self):
        """Initialize streaming components"""
        logger.info("Initializing streaming components...")
        
        from data_pipeline.streaming.websocket_client import MultiExchangeWebSocketClient
        from data_pipeline.streaming.message_queue import MessageQueue, QueueConfig, QueueType
        
        # Create message queue for streaming data
        queue_config = QueueConfig(
            queue_type=QueueType.REDIS,
            queue_name="streaming_data",
            max_size=10000,
            processing_concurrency=self.config.processing_concurrency
        )
        
        streaming_queue = MessageQueue(queue_config)
        await streaming_queue.connect()
        self.message_queues["streaming"] = streaming_queue
        
        # Initialize WebSocket client
        ws_config = {
            "exchanges": self.config.streaming_exchanges,
            "reconnect_attempts": 5,
            "ping_interval": 20
        }
        
        ws_client = MultiExchangeWebSocketClient(ws_config)
        
        # Add message handler to enqueue data
        async def handle_streaming_message(message):
            await streaming_queue.enqueue(message)
        
        ws_client.add_message_handler(handle_streaming_message)
        
        # Connect to exchanges
        for exchange in self.config.streaming_exchanges:
            await ws_client.connect(exchange, self.config.streaming_symbols)
        
        self.components["websocket_client"] = ws_client
        self.components["streaming_queue"] = streaming_queue
        
        logger.info(f"Streaming initialized for {len(self.config.streaming_symbols)} symbols")
    
    async def _initialize_alternative_data(self):
        """Initialize alternative data components"""
        logger.info("Initializing alternative data components...")
        
        from data_pipeline.alternative_data.sentiment_analyzer import SentimentAnalyzer, SentimentConfig
        from data_pipeline.alternative_data.onchain_analyzer import OnChainAnalyzer
        from data_pipeline.streaming.message_queue import MessageQueue, QueueConfig, QueueType
        
        # Create message queue for alternative data
        queue_config = QueueConfig(
            queue_type=QueueType.REDIS,
            queue_name="alternative_data",
            max_size=5000,
            processing_concurrency=self.config.processing_concurrency // 2
        )
        
        alt_data_queue = MessageQueue(queue_config)
        await alt_data_queue.connect()
        self.message_queues["alternative_data"] = alt_data_queue
        
        # Initialize sentiment analyzer
        if self.config.sentiment_enabled:
            sentiment_config = SentimentConfig(
                fetch_interval_minutes=5,
                cache_ttl_hours=24
            )
            
            sentiment_analyzer = SentimentAnalyzer(sentiment_config)
            await sentiment_analyzer.initialize()
            self.components["sentiment_analyzer"] = sentiment_analyzer
        
        # Initialize on-chain analyzer
        if self.config.onchain_enabled:
            onchain_config = {
                "fetch_interval_minutes": 10,
                "cache_ttl_hours": 12
            }
            
            onchain_analyzer = OnChainAnalyzer(onchain_config)
            await onchain_analyzer.initialize()
            self.components["onchain_analyzer"] = onchain_analyzer
        
        logger.info("Alternative data components initialized")
    
    async def _initialize_feature_store(self):
        """Initialize feature store"""
        logger.info("Initializing feature store...")
        
        from data_pipeline.feature_store.feature_store import FeatureStore, StorageBackend
        
        store_config = {
            "storage_backend": self.config.feature_store_backend,
            "redis_url": "redis://localhost:6379/0",
            "data_directory": "data/feature_store"
        }
        
        feature_store = FeatureStore(store_config)
        await feature_store.initialize()
        self.components["feature_store"] = feature_store
        
        logger.info("Feature store initialized")
    
    async def _initialize_validation(self):
        """Initialize validation components"""
        logger.info("Initializing validation components...")
        
        from data_pipeline.validation.data_validator import DataValidator
        
        validator_config = {
            "strict_mode": True,
            "log_errors": True
        }
        
        data_validator = DataValidator(validator_config)
        self.components["data_validator"] = data_validator
        
        logger.info("Validation components initialized")
    
    async def _initialize_realtime_processor(self):
        """Initialize real-time processor"""
        logger.info("Initializing real-time processor...")
        
        from data_pipeline.streaming.realtime_processor import RealTimeProcessor
        
        processor_config = {
            "buffers": {
                "trade": {"max_size": 1000, "time_window": 300},
                "kline_1m": {"max_size": 100, "time_window": 600},
                "orderbook": {"max_size": 100, "time_window": 60}
            },
            "pipelines": {
                "price_features": {
                    "channels": ["trade", "kline_1m"],
                    "lookback_channels": ["kline_1m"]
                },
                "volume_features": {
                    "channels": ["trade"],
                    "lookback_channels": ["trade"]
                }
            }
        }
        
        realtime_processor = RealTimeProcessor(processor_config)
        self.components["realtime_processor"] = realtime_processor
        
        logger.info("Real-time processor initialized")
    
    async def start(self):
        """Start the data pipeline"""
        if self.status == PipelineStatus.RUNNING:
            logger.warning("Pipeline already running")
            return
        
        # Initialize if not already initialized
        if self.status == PipelineStatus.STOPPED:
            success = await self.initialize()
            if not success:
                logger.error("Failed to initialize pipeline")
                return
        
        logger.info("Starting data pipeline...")
        
        try:
            # Start streaming queue processing
            if "streaming_queue" in self.components:
                streaming_queue = self.components["streaming_queue"]
                await streaming_queue.start_processing()
                
                # Add streaming processor
                async def process_streaming_data(message):
                    await self._process_streaming_message(message)
                
                streaming_queue.add_message_handler(process_streaming_data)
                logger.info("Streaming queue processing started")
            
            # Start alternative data queue processing
            if "alternative_data_queue" in self.message_queues:
                alt_data_queue = self.message_queues["alternative_data"]
                await alt_data_queue.start_processing()
                
                async def process_alternative_data(message):
                    await self._process_alternative_data(message)
                
                alt_data_queue.add_message_handler(process_alternative_data)
                logger.info("Alternative data queue processing started")
            
            # Start WebSocket client
            if "websocket_client" in self.components:
                ws_client = self.components["websocket_client"]
                
                # Start in background task
                ws_task = asyncio.create_task(ws_client.start())
                self.processing_tasks.append(ws_task)
                logger.info("WebSocket client started")
            
            # Start periodic tasks
            await self._start_periodic_tasks()
            
            self.status = PipelineStatus.RUNNING
            logger.info("Data pipeline started successfully")
            
            # Keep running
            await self._run_forever()
            
        except Exception as e:
            logger.error(f"Failed to start data pipeline: {e}")
            self.status = PipelineStatus.ERROR
            await self.stop()
    
    async def _process_streaming_message(self, message):
        """Process streaming message"""
        try:
            # Validate message
            if "data_validator" in self.components:
                validator = self.components["data_validator"]
                report = await validator.validate(message, "ohlcv")
                
                if not report.is_valid:
                    logger.warning(f"Invalid streaming message: {report.errors}")
                    return
            
            # Process with real-time processor
            if "realtime_processor" in self.components:
                processor = self.components["realtime_processor"]
                features = await processor.process_message(message)
                
                if features:
                    # Store in feature store
                    if "feature_store" in self.components:
                        feature_store = self.components["feature_store"]
                        
                        symbol = message.get("symbol", "unknown")
                        timestamp = message.get("timestamp", datetime.now().isoformat())
                        
                        await feature_store.store_features(features, symbol, timestamp)
                        
                        self.stats["features_generated"] += 1
                        self.stats["messages_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing streaming message: {e}")
            self.stats["errors"] += 1
    
    async def _process_alternative_data(self, message):
        """Process alternative data"""
        try:
            data_type = message.get("type", "unknown")
            
            if data_type == "sentiment":
                # Process sentiment data
                if "sentiment_analyzer" in self.components:
                    sentiment_analyzer = self.components["sentiment_analyzer"]
                    
                    symbol = message.get("symbol")
                    sources = message.get("sources", ["news", "onchain"])
                    
                    sentiment = await sentiment_analyzer.analyze_symbol(symbol, sources)
                    
                    # Store sentiment features
                    if "feature_store" in self.components:
                        feature_store = self.components["feature_store"]
                        
                        feature_name = f"sentiment_{symbol.replace('/', '_').lower()}"
                        features = {
                            f"{feature_name}_score": sentiment.get("sentiment_score", 0),
                            f"{feature_name}_confidence": sentiment.get("sentiment_confidence", 0),
                            f"{feature_name}_strength": sentiment.get("sentiment_strength", 0)
                        }
                        
                        await feature_store.store_features(
                            features, 
                            symbol,
                            datetime.now().isoformat()
                        )
            
            elif data_type == "onchain":
                # Process on-chain data
                if "onchain_analyzer" in self.components:
                    onchain_analyzer = self.components["onchain_analyzer"]
                    
                    symbol = message.get("symbol")
                    metrics = message.get("metrics", [])
                    
                    onchain_data = await onchain_analyzer.get_metrics(symbol, metrics)
                    
                    # Store on-chain features
                    if "feature_store" in self.components and onchain_data:
                        feature_store = self.components["feature_store"]
                        
                        features = {}
                        for metric_name, value in onchain_data.items():
                            feature_name = f"onchain_{symbol.replace('/', '_').lower()}_{metric_name}"
                            features[feature_name] = value
                        
                        await feature_store.store_features(
                            features,
                            symbol,
                            datetime.now().isoformat()
                        )
            
        except Exception as e:
            logger.error(f"Error processing alternative data: {e}")
            self.stats["errors"] += 1
    
    async def _start_periodic_tasks(self):
        """Start periodic maintenance tasks"""
        
        # Task 1: Update alternative data
        async def update_alternative_data():
            while self.status == PipelineStatus.RUNNING:
                try:
                    await self._fetch_alternative_data()
                    await asyncio.sleep(300)  # Every 5 minutes
                except Exception as e:
                    logger.error(f"Error updating alternative data: {e}")
                    await asyncio.sleep(60)
        
        # Task 2: Cleanup old data
        async def cleanup_data():
            while self.status == PipelineStatus.RUNNING:
                try:
                    await self._cleanup_old_data()
                    await asyncio.sleep(3600)  # Every hour
                except Exception as e:
                    logger.error(f"Error cleaning up data: {e}")
                    await asyncio.sleep(300)
        
        # Task 3: Update statistics
        async def update_statistics():
            while self.status == PipelineStatus.RUNNING:
                try:
                    self._update_statistics()
                    await asyncio.sleep(60)  # Every minute
                except Exception as e:
                    logger.error(f"Error updating statistics: {e}")
                    await asyncio.sleep(30)
        
        # Start tasks
        tasks = [
            asyncio.create_task(update_alternative_data()),
            asyncio.create_task(cleanup_data()),
            asyncio.create_task(update_statistics())
        ]
        
        self.processing_tasks.extend(tasks)
        logger.info(f"Started {len(tasks)} periodic tasks")
    
    async def _fetch_alternative_data(self):
        """Fetch alternative data for all symbols"""
        if not self.config.alternative_data_enabled:
            return
        
        for symbol in self.config.streaming_symbols:
            try:
                # Fetch sentiment
                if self.config.sentiment_enabled and "sentiment_analyzer" in self.components:
                    sentiment_analyzer = self.components["sentiment_analyzer"]
                    
                    sentiment = await sentiment_analyzer.analyze_symbol(symbol)
                    
                    # Enqueue for processing
                    if "alternative_data_queue" in self.message_queues:
                        queue = self.message_queues["alternative_data"]
                        
                        await queue.enqueue({
                            "type": "sentiment",
                            "symbol": symbol,
                            "data": sentiment,
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Fetch on-chain metrics
                if self.config.onchain_enabled and "onchain_analyzer" in self.components:
                    onchain_analyzer = self.components["onchain_analyzer"]
                    
                    onchain_data = await onchain_analyzer.get_metrics(symbol)
                    
                    if onchain_data:
                        # Enqueue for processing
                        if "alternative_data_queue" in self.message_queues:
                            queue = self.message_queues["alternative_data"]
                            
                            await queue.enqueue({
                                "type": "onchain",
                                "symbol": symbol,
                                "data": onchain_data,
                                "timestamp": datetime.now().isoformat()
                            })
                
            except Exception as e:
                logger.error(f"Error fetching alternative data for {symbol}: {e}")
    
    async def _cleanup_old_data(self):
        """Cleanup old data from caches and storage"""
        try:
            # Clean feature store
            if "feature_store" in self.components:
                feature_store = self.components["feature_store"]
                await feature_store.cleanup_old_features(days_to_keep=7)
            
            # Clean sentiment cache
            if "sentiment_analyzer" in self.components:
                sentiment_analyzer = self.components["sentiment_analyzer"]
                # Sentiment analyzer handles its own cache cleanup
            
            logger.debug("Old data cleanup completed")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def _update_statistics(self):
        """Update pipeline statistics"""
        if self.stats["start_time"]:
            self.stats["uptime_seconds"] = (datetime.now() - self.stats["start_time"]).total_seconds()
    
    async def _run_forever(self):
        """Keep the pipeline running"""
        while self.status == PipelineStatus.RUNNING:
            try:
                # Update statistics
                self._update_statistics()
                
                # Log status periodically
                if int(datetime.now().timestamp()) % 300 == 0:  # Every 5 minutes
                    self._log_status()
                
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)
    
    def _log_status(self):
        """Log pipeline status"""
        uptime = timedelta(seconds=int(self.stats["uptime_seconds"]))
        
        logger.info(
            f"Pipeline Status: {self.status.value} | "
            f"Uptime: {uptime} | "
            f"Messages: {self.stats['messages_processed']:,} | "
            f"Features: {self.stats['features_generated']:,} | "
            f"Errors: {self.stats['errors']}"
        )
    
    async def get_features(self, symbol: str, 
                          feature_types: List[str] = None) -> Dict[str, Any]:
        """Get features for a symbol"""
        if "feature_store" not in self.components:
            logger.error("Feature store not available")
            return {}
        
        feature_store = self.components["feature_store"]
        features = await feature_store.get_features(symbol)
        
        # Filter by feature types if specified
        if feature_types and "features" in features:
            filtered_features = {}
            for ft in feature_types:
                for key, value in features["features"].items():
                    if ft in key:
                        filtered_features[key] = value
            
            features["features"] = filtered_features
        
        return features
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get detailed pipeline status"""
        component_status = {}
        
        for name, component in self.components.items():
            try:
                if hasattr(component, 'get_stats'):
                    component_status[name] = component.get_stats()
                else:
                    component_status[name] = {"status": "unknown"}
            except Exception as e:
                component_status[name] = {"status": "error", "error": str(e)}
        
        queue_status = {}
        for name, queue in self.message_queues.items():
            try:
                if hasattr(queue, 'get_stats'):
                    queue_status[name] = queue.get_stats()
                else:
                    queue_status[name] = {"status": "unknown"}
            except Exception as e:
                queue_status[name] = {"status": "error", "error": str(e)}
        
        return {
            "pipeline_status": self.status.value,
            "uptime_seconds": self.stats["uptime_seconds"],
            "statistics": self.stats,
            "components": component_status,
            "queues": queue_status,
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat()
        }
    
    async def stop(self):
        """Stop the data pipeline gracefully"""
        if self.status == PipelineStatus.STOPPING:
            return
        
        logger.info("Stopping data pipeline...")
        self.status = PipelineStatus.STOPPING
        
        try:
            # Cancel all processing tasks
            for task in self.processing_tasks:
                if not task.done():
                    task.cancel()
            
            if self.processing_tasks:
                await asyncio.gather(*self.processing_tasks, return_exceptions=True)
                self.processing_tasks.clear()
            
            # Stop WebSocket client
            if "websocket_client" in self.components:
                ws_client = self.components["websocket_client"]
                await ws_client.stop()
            
            # Stop message queues
            for name, queue in self.message_queues.items():
                await queue.stop()
            
            # Close components
            for name, component in self.components.items():
                if hasattr(component, 'close'):
                    await component.close()
                elif hasattr(component, 'disconnect'):
                    await component.disconnect()
            
            self.components.clear()
            self.message_queues.clear()
            
            self.status = PipelineStatus.STOPPED
            logger.info("Data pipeline stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping data pipeline: {e}")
            self.status = PipelineStatus.ERROR
    
    async def restart(self):
        """Restart the data pipeline"""
        logger.info("Restarting data pipeline...")
        
        await self.stop()
        await asyncio.sleep(2)  # Brief pause
        
        await self.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.stats.copy()
