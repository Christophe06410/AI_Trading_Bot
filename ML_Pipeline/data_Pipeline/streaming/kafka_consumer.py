"""
Kafka consumer for high-throughput data streaming
For enterprise-scale data ingestion
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
from confluent_kafka import Consumer, KafkaError, KafkaException
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)


@dataclass
class KafkaConfig:
    """Kafka configuration"""
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "ml-pipeline-consumer"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    session_timeout_ms: int = 10000
    max_poll_interval_ms: int = 300000
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None


class KafkaDataConsumer:
    """Enterprise Kafka consumer for real-time data"""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.consumer = None
        self.is_running = False
        self.message_handlers: List[Callable] = []
        self.topics: List[str] = []
        
        # Statistics
        self.stats = {
            "messages_consumed": 0,
            "bytes_consumed": 0,
            "consumer_lag": 0,
            "last_message_time": None,
            "errors": 0
        }
        
        # Thread safety
        self.lock = threading.Lock()
    
    def _create_consumer_config(self) -> Dict[str, Any]:
        """Create Kafka consumer configuration"""
        config = {
            'bootstrap.servers': self.config.bootstrap_servers,
            'group.id': self.config.group_id,
            'auto.offset.reset': self.config.auto_offset_reset,
            'enable.auto.commit': self.config.enable_auto_commit,
            'session.timeout.ms': self.config.session_timeout_ms,
            'max.poll.interval.ms': self.config.max_poll_interval_ms,
        }
        
        # Add security if configured
        if self.config.security_protocol:
            config['security.protocol'] = self.config.security_protocol
        
        if self.config.sasl_mechanism and self.config.sasl_username:
            config.update({
                'sasl.mechanism': self.config.sasl_mechanism,
                'sasl.username': self.config.sasl_username,
                'sasl.password': self.config.sasl_password
            })
        
        return config
    
    async def connect(self):
        """Connect to Kafka cluster"""
        try:
            consumer_config = self._create_consumer_config()
            self.consumer = Consumer(consumer_config)
            
            logger.info(f"Kafka consumer initialized for {self.config.bootstrap_servers}")
            return True
            
        except KafkaException as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            return False
    
    def subscribe(self, topics: List[str]):
        """Subscribe to Kafka topics"""
        if not self.consumer:
            raise RuntimeError("Consumer not connected")
        
        self.topics = topics
        self.consumer.subscribe(topics)
        
        logger.info(f"Subscribed to topics: {topics}")
    
    def add_message_handler(self, handler: Callable):
        """Add message handler callback"""
        self.message_handlers.append(handler)
        logger.info(f"Added Kafka message handler: {handler.__name__}")
    
    async def start(self, batch_size: int = 100, poll_timeout: float = 1.0):
        """Start consuming messages"""
        if not self.consumer:
            raise RuntimeError("Consumer not connected")
        
        self.is_running = True
        logger.info(f"Starting Kafka consumer for topics: {self.topics}")
        
        while self.is_running:
            try:
                # Poll for messages
                messages = self.consumer.consume(
                    num_messages=batch_size,
                    timeout=poll_timeout
                )
                
                if messages:
                    await self._process_batch(messages)
                
                # Periodically log statistics
                if self.stats["messages_consumed"] % 1000 == 0:
                    self._log_statistics()
                
                # Small sleep to prevent tight loop
                await asyncio.sleep(0.001)
                
            except KeyboardInterrupt:
                logger.info("Kafka consumer interrupted")
                break
                
            except KafkaException as e:
                logger.error(f"Kafka error: {e}")
                self.stats["errors"] += 1
                
                # Wait before retry
                await asyncio.sleep(5)
            
            except Exception as e:
                logger.error(f"Unexpected error in Kafka consumer: {e}")
                self.stats["errors"] += 1
    
    async def _process_batch(self, messages):
        """Process a batch of Kafka messages"""
        processed_count = 0
        
        for msg in messages:
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition
                    continue
                else:
                    logger.error(f"Kafka message error: {msg.error()}")
                    continue
            
            try:
                # Parse message
                message_data = json.loads(msg.value().decode('utf-8'))
                
                # Add metadata
                enriched_data = {
                    "data": message_data,
                    "metadata": {
                        "topic": msg.topic(),
                        "partition": msg.partition(),
                        "offset": msg.offset(),
                        "timestamp": msg.timestamp()[1] if msg.timestamp() else None,
                        "key": msg.key().decode('utf-8') if msg.key() else None
                    }
                }
                
                # Call all registered handlers
                for handler in self.message_handlers:
                    try:
                        await handler(enriched_data)
                    except Exception as e:
                        logger.error(f"Handler {handler.__name__} failed: {e}")
                
                with self.lock:
                    self.stats["messages_consumed"] += 1
                    self.stats["bytes_consumed"] += len(msg.value())
                    self.stats["last_message_time"] = datetime.now()
                
                processed_count += 1
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Kafka message: {e}")
            except Exception as e:
                logger.error(f"Error processing Kafka message: {e}")
        
        # Log batch processing
        if processed_count > 0:
            logger.debug(f"Processed {processed_count} Kafka messages")
    
    def _log_statistics(self):
        """Log consumer statistics"""
        logger.info(
            f"Kafka Consumer Stats: "
            f"Messages={self.stats['messages_consumed']:,} "
            f"Bytes={self.stats['bytes_consumed']:,} "
            f"Errors={self.stats['errors']}"
        )
    
    async def get_consumer_lag(self) -> Dict[str, int]:
        """Get consumer lag for each topic/partition"""
        if not self.consumer:
            return {}
        
        lag_info = {}
        
        try:
            # Get assigned partitions
            assignment = self.consumer.assignment()
            
            for tp in assignment:
                # Get committed offsets
                committed = self.consumer.committed([tp], timeout=1)
                if committed[tp]:
                    committed_offset = committed[tp].offset
                    
                    # Get latest offsets
                    low, high = self.consumer.get_watermark_offsets(tp, timeout=1)
                    
                    if high >= 0 and committed_offset >= 0:
                        lag = high - committed_offset
                        key = f"{tp.topic}-{tp.partition}"
                        lag_info[key] = lag
            
            with self.lock:
                if lag_info:
                    self.stats["consumer_lag"] = sum(lag_info.values())
            
        except Exception as e:
            logger.error(f"Failed to get consumer lag: {e}")
        
        return lag_info
    
    async def commit_offsets(self):
        """Manually commit offsets"""
        if self.consumer:
            try:
                self.consumer.commit(asynchronous=False)
                logger.debug("Committed Kafka offsets")
            except KafkaException as e:
                logger.error(f"Failed to commit offsets: {e}")
    
    async def stop(self):
        """Stop consumer gracefully"""
        self.is_running = False
        
        if self.consumer:
            try:
                # Commit final offsets
                await self.commit_offsets()
                
                # Close consumer
                self.consumer.close()
                logger.info("Kafka consumer stopped")
                
            except Exception as e:
                logger.error(f"Error stopping Kafka consumer: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consumer statistics"""
        with self.lock:
            return self.stats.copy()
