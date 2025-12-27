"""
Async message queue for processing streaming data
Supports Redis, RabbitMQ, and in-memory queues
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Callable, Optional, Union
from datetime import datetime
from enum import Enum
import redis.asyncio as redis
import aio_pika
from dataclasses import dataclass
from collections import deque
import pickle

logger = logging.getLogger(__name__)


class QueueType(Enum):
    """Supported queue types"""
    REDIS = "redis"
    RABBITMQ = "rabbitmq"
    MEMORY = "memory"


@dataclass
class QueueConfig:
    """Queue configuration"""
    queue_type: QueueType = QueueType.REDIS
    redis_url: str = "redis://localhost:6379/0"
    rabbitmq_url: str = "amqp://guest:guest@localhost/"
    queue_name: str = "ml_pipeline_queue"
    max_size: int = 10000
    processing_concurrency: int = 10


class MessageQueue:
    """Enterprise message queue for streaming data"""
    
    def __init__(self, config: QueueConfig):
        self.config = config
        self.queue_type = config.queue_type
        self.queue = None
        self.consumer_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Message handlers
        self.message_handlers: List[Callable] = []
        
        # Statistics
        self.stats = {
            "messages_enqueued": 0,
            "messages_dequeued": 0,
            "processing_errors": 0,
            "queue_size": 0,
            "last_message_time": None
        }
        
        # Memory queue (fallback)
        self.memory_queue = deque(maxlen=config.max_size)
    
    async def connect(self):
        """Connect to message queue"""
        try:
            if self.queue_type == QueueType.REDIS:
                await self._connect_redis()
            elif self.queue_type == QueueType.RABBITMQ:
                await self._connect_rabbitmq()
            else:
                logger.info("Using in-memory queue")
            
            logger.info(f"Connected to {self.queue_type.value} queue")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to queue: {e}")
            return False
    
    async def _connect_redis(self):
        """Connect to Redis"""
        self.redis_client = redis.from_url(
            self.config.redis_url,
            encoding="utf-8",
            decode_responses=False
        )
        
        # Test connection
        await self.redis_client.ping()
    
    async def _connect_rabbitmq(self):
        """Connect to RabbitMQ"""
        connection = await aio_pika.connect_robust(self.config.rabbitmq_url)
        self.rabbitmq_connection = connection
        
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=self.config.processing_concurrency)
        
        # Declare queue
        self.rabbitmq_queue = await channel.declare_queue(
            self.config.queue_name,
            durable=True
        )
    
    async def enqueue(self, message: Any, priority: int = 0) -> bool:
        """Enqueue a message"""
        try:
            # Serialize message
            if isinstance(message, (dict, list)):
                serialized = json.dumps(message).encode('utf-8')
            else:
                serialized = pickle.dumps(message)
            
            # Add metadata
            message_with_meta = {
                "data": serialized,
                "timestamp": datetime.now().isoformat(),
                "priority": priority
            }
            
            # Enqueue based on type
            if self.queue_type == QueueType.REDIS:
                await self._enqueue_redis(message_with_meta)
            elif self.queue_type == QueueType.RABBITMQ:
                await self._enqueue_rabbitmq(message_with_meta)
            else:
                await self._enqueue_memory(message_with_meta)
            
            # Update stats
            self.stats["messages_enqueued"] += 1
            self.stats["queue_size"] += 1
            self.stats["last_message_time"] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to enqueue message: {e}")
            return False
    
    async def _enqueue_redis(self, message: Dict[str, Any]):
        """Enqueue to Redis"""
        serialized = pickle.dumps(message)
        
        if message["priority"] > 0:
            # Use sorted set for priority queue
            await self.redis_client.zadd(
                f"{self.config.queue_name}:priority",
                {serialized: -message["priority"]}  # Negative for descending
            )
        else:
            # Regular list for FIFO
            await self.redis_client.lpush(
                self.config.queue_name,
                serialized
            )
    
    async def _enqueue_rabbitmq(self, message: Dict[str, Any]):
        """Enqueue to RabbitMQ"""
        channel = await self.rabbitmq_connection.channel()
        
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=pickle.dumps(message),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                priority=message["priority"]
            ),
            routing_key=self.config.queue_name
        )
    
    async def _enqueue_memory(self, message: Dict[str, Any]):
        """Enqueue to memory queue"""
        if message["priority"] > 0:
            # For priority, we need to insert in order
            # Simple implementation: just append for now
            self.memory_queue.append(message)
        else:
            self.memory_queue.append(message)
    
    async def dequeue(self, timeout: float = 1.0) -> Optional[Any]:
        """Dequeue a message"""
        try:
            message = None
            
            if self.queue_type == QueueType.REDIS:
                message = await self._dequeue_redis(timeout)
            elif self.queue_type == QueueType.RABBITMQ:
                message = await self._dequeue_rabbitmq(timeout)
            else:
                message = await self._dequeue_memory(timeout)
            
            if message:
                self.stats["messages_dequeued"] += 1
                self.stats["queue_size"] = max(0, self.stats["queue_size"] - 1)
                
                # Deserialize
                if isinstance(message["data"], bytes):
                    try:
                        data = pickle.loads(message["data"])
                    except:
                        data = json.loads(message["data"].decode('utf-8'))
                    
                    return {
                        "data": data,
                        "metadata": {
                            "timestamp": message["timestamp"],
                            "priority": message["priority"]
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to dequeue message: {e}")
            return None
    
    async def _dequeue_redis(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Dequeue from Redis"""
        try:
            # Try priority queue first
            priority_result = await self.redis_client.zpopmax(
                f"{self.config.queue_name}:priority",
                count=1
            )
            
            if priority_result:
                _, serialized = priority_result[0]
                return pickle.loads(serialized)
            
            # Fall back to regular queue with timeout
            result = await self.redis_client.brpop(
                self.config.queue_name,
                timeout=timeout
            )
            
            if result:
                _, serialized = result
                return pickle.loads(serialized)
            
            return None
            
        except Exception as e:
            logger.error(f"Redis dequeue error: {e}")
            return None
    
    async def _dequeue_rabbitmq(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Dequeue from RabbitMQ"""
        try:
            async with self.rabbitmq_queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        data = pickle.loads(message.body)
                        return data
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"RabbitMQ dequeue error: {e}")
            return None
    
    async def _dequeue_memory(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Dequeue from memory queue"""
        try:
            if self.memory_queue:
                return self.memory_queue.popleft()
            
            # Wait for message if queue is empty
            await asyncio.sleep(timeout)
            return None
            
        except Exception as e:
            logger.error(f"Memory dequeue error: {e}")
            return None
    
    def add_message_handler(self, handler: Callable):
        """Add message handler"""
        self.message_handlers.append(handler)
        logger.info(f"Added queue message handler: {handler.__name__}")
    
    async def start_processing(self):
        """Start processing messages from queue"""
        self.is_running = True
        
        # Create consumer tasks
        for i in range(self.config.processing_concurrency):
            task = asyncio.create_task(self._process_messages(f"worker-{i}"))
            self.consumer_tasks.append(task)
        
        logger.info(f"Started {self.config.processing_concurrency} queue workers")
    
    async def _process_messages(self, worker_id: str):
        """Process messages from queue"""
        logger.info(f"Queue worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get message from queue
                message = await self.dequeue(timeout=0.5)
                
                if not message:
                    # No message, small sleep
                    await asyncio.sleep(0.1)
                    continue
                
                # Process with all handlers
                for handler in self.message_handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Handler {handler.__name__} failed: {e}")
                        self.stats["processing_errors"] += 1
                
                # Log progress
                if self.stats["messages_dequeued"] % 100 == 0:
                    logger.debug(
                        f"Worker {worker_id} processed {self.stats['messages_dequeued']} messages"
                    )
                
            except Exception as e:
                logger.error(f"Queue worker {worker_id} error: {e}")
                await asyncio.sleep(1)  # Backoff on error
        
        logger.info(f"Queue worker {worker_id} stopped")
    
    async def stop(self):
        """Stop queue processing"""
        self.is_running = False
        
        # Wait for tasks to complete
        if self.consumer_tasks:
            await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
            self.consumer_tasks.clear()
        
        # Close connections
        if self.queue_type == QueueType.REDIS and hasattr(self, 'redis_client'):
            await self.redis_client.close()
        
        if self.queue_type == QueueType.RABBITMQ and hasattr(self, 'rabbitmq_connection'):
            await self.rabbitmq_connection.close()
        
        logger.info("Message queue stopped")
    
    async def get_queue_size(self) -> int:
        """Get current queue size"""
        if self.queue_type == QueueType.REDIS:
            # Redis queue size
            regular_size = await self.redis_client.llen(self.config.queue_name)
            priority_size = await self.redis_client.zcard(
                f"{self.config.queue_name}:priority"
            )
            return regular_size + priority_size
        
        elif self.queue_type == QueueType.RABBITMQ:
            # RabbitMQ queue size
            channel = await self.rabbitmq_connection.channel()
            queue = await channel.declare_queue(
                self.config.queue_name,
                passive=True
            )
            return queue.declaration_result.message_count
        
        else:
            # Memory queue size
            return len(self.memory_queue)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            **self.stats,
            "queue_type": self.queue_type.value,
            "handlers": len(self.message_handlers),
            "workers": len(self.consumer_tasks)
        }
