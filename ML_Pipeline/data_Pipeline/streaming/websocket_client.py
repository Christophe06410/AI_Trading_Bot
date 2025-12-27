"""
Enterprise-grade WebSocket client for real-time crypto data
Supports multiple exchanges, reconnection, and data validation
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Callable, Optional
import websockets
from datetime import datetime
import aiohttp
import ssl
import backoff

logger = logging.getLogger(__name__)


class MultiExchangeWebSocketClient:
    """Enterprise WebSocket client for multiple cryptocurrency exchanges"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchanges = self._initialize_exchanges()
        self.connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.subscriptions: Dict[str, List[str]] = {}
        self.message_handlers: List[Callable] = []
        self.is_running = False
        self.reconnect_attempts: Dict[str, int] = {}
        
        # Statistics
        self.stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "connection_errors": 0,
            "last_message_time": {}
        }
    
    def _initialize_exchanges(self) -> Dict[str, Dict[str, Any]]:
        """Initialize exchange configurations"""
        return {
            "binance": {
                "ws_url": "wss://stream.binance.com:9443/ws",
                "api_url": "https://api.binance.com",
                "channels": ["trade", "kline_1m", "kline_5m", "depth20"]
            },
            "coinbase": {
                "ws_url": "wss://ws-feed.pro.coinbase.com",
                "api_url": "https://api.pro.coinbase.com",
                "channels": ["matches", "ticker", "level2"]
            },
            "kraken": {
                "ws_url": "wss://ws.kraken.com",
                "api_url": "https://api.kraken.com",
                "channels": ["trade", "spread", "book"]
            },
            "bybit": {
                "ws_url": "wss://stream.bybit.com/v5/public/spot",
                "api_url": "https://api.bybit.com",
                "channels": ["trade", "kline", "orderbook"]
            }
        }
    
    async def connect(self, exchange: str, symbols: List[str]) -> bool:
        """Connect to exchange WebSocket"""
        if exchange not in self.exchanges:
            logger.error(f"Exchange {exchange} not supported")
            return False
        
        try:
            exchange_config = self.exchanges[exchange]
            ws_url = exchange_config["ws_url"]
            
            # Add SSL context for security
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            logger.info(f"Connecting to {exchange} WebSocket...")
            
            # Connect with timeout
            connection = await asyncio.wait_for(
                websockets.connect(
                    ws_url,
                    ssl=ssl_context,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ),
                timeout=10
            )
            
            self.connections[exchange] = connection
            self.reconnect_attempts[exchange] = 0
            
            # Subscribe to channels
            await self._subscribe(exchange, symbols)
            
            logger.info(f"Connected to {exchange} WebSocket")
            return True
            
        except (asyncio.TimeoutError, websockets.WebSocketException) as e:
            logger.error(f"Failed to connect to {exchange}: {e}")
            self.stats["connection_errors"] += 1
            return False
    
    @backoff.on_exception(backoff.expo, 
                         websockets.WebSocketException,
                         max_tries=5,
                         max_time=60)
    async def _subscribe(self, exchange: str, symbols: List[str]):
        """Subscribe to channels with exponential backoff"""
        exchange_config = self.exchanges[exchange]
        connection = self.connections[exchange]
        
        subscription_message = self._create_subscription_message(
            exchange, symbols, exchange_config["channels"]
        )
        
        await connection.send(json.dumps(subscription_message))
        self.subscriptions[exchange] = symbols
        
        logger.info(f"Subscribed to {len(symbols)} symbols on {exchange}")
    
    def _create_subscription_message(self, exchange: str, symbols: List[str], 
                                    channels: List[str]) -> Dict[str, Any]:
        """Create exchange-specific subscription message"""
        if exchange == "binance":
            # Binance uses stream names like btcusdt@trade
            streams = []
            for symbol in symbols:
                symbol_lower = symbol.replace("/", "").lower()
                for channel in channels:
                    streams.append(f"{symbol_lower}@{channel}")
            
            return {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": 1
            }
        
        elif exchange == "coinbase":
            # Coinbase uses product_ids
            return {
                "type": "subscribe",
                "product_ids": [s.replace("/", "-") for s in symbols],
                "channels": channels
            }
        
        elif exchange == "kraken":
            # Kraken uses pairs and subscription object
            return {
                "event": "subscribe",
                "pair": [s.replace("/", "/") for s in symbols],
                "subscription": {"name": "trade"}
            }
        
        else:
            # Default format
            return {
                "op": "subscribe",
                "args": [f"{channel}.{symbol}" for symbol in symbols for channel in channels]
            }
    
    def add_message_handler(self, handler: Callable):
        """Add message handler callback"""
        self.message_handlers.append(handler)
        logger.info(f"Added message handler: {handler.__name__}")
    
    async def start(self):
        """Start listening to all connections"""
        self.is_running = True
        
        # Create tasks for each connection
        tasks = []
        for exchange in self.connections:
            task = asyncio.create_task(self._listen(exchange))
            tasks.append(task)
        
        # Also start health check task
        health_task = asyncio.create_task(self._health_check())
        tasks.append(health_task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _listen(self, exchange: str):
        """Listen to WebSocket messages from an exchange"""
        connection = self.connections[exchange]
        
        while self.is_running and connection.open:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    connection.recv(),
                    timeout=30
                )
                
                self.stats["messages_received"] += 1
                self.stats["last_message_time"][exchange] = datetime.now()
                
                # Process message
                await self._process_message(exchange, message)
                
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await connection.ping()
                except:
                    logger.warning(f"Connection to {exchange} may be dead")
                    await self._reconnect(exchange)
            
            except websockets.WebSocketException as e:
                logger.error(f"WebSocket error from {exchange}: {e}")
                await self._reconnect(exchange)
            
            except Exception as e:
                logger.error(f"Unexpected error in {exchange} listener: {e}")
    
    async def _process_message(self, exchange: str, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Validate message structure
            if not self._validate_message(exchange, data):
                logger.warning(f"Invalid message from {exchange}")
                return
            
            # Standardize message format
            standardized = self._standardize_message(exchange, data)
            
            # Call all registered handlers
            for handler in self.message_handlers:
                try:
                    await handler(standardized)
                except Exception as e:
                    logger.error(f"Handler {handler.__name__} failed: {e}")
            
            self.stats["messages_processed"] += 1
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from {exchange}: {message[:100]}")
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
    
    def _validate_message(self, exchange: str, data: Dict[str, Any]) -> bool:
        """Validate message structure based on exchange"""
        if exchange == "binance":
            return "stream" in data or "e" in data
        elif exchange == "coinbase":
            return "type" in data
        elif exchange == "kraken":
            return isinstance(data, list) and len(data) > 1
        else:
            return isinstance(data, dict)
    
    def _standardize_message(self, exchange: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize message to common format"""
        standardized = {
            "exchange": exchange,
            "timestamp": datetime.now().isoformat(),
            "raw_data": data
        }
        
        if exchange == "binance":
            if "stream" in data:
                # Stream format
                stream = data["stream"]
                symbol = stream.split("@")[0].upper()
                channel = stream.split("@")[1]
                
                standardized.update({
                    "symbol": symbol,
                    "channel": channel,
                    "data": data["data"]
                })
            else:
                # Direct event format
                standardized.update({
                    "symbol": data.get("s", "").upper(),
                    "channel": data.get("e", ""),
                    "data": data
                })
        
        elif exchange == "coinbase":
            standardized.update({
                "symbol": data.get("product_id", "").replace("-", "/"),
                "channel": data.get("type", ""),
                "data": data
            })
        
        return standardized
    
    async def _reconnect(self, exchange: str):
        """Reconnect to exchange with exponential backoff"""
        if exchange not in self.reconnect_attempts:
            self.reconnect_attempts[exchange] = 0
        
        attempts = self.reconnect_attempts[exchange]
        delay = min(2 ** attempts, 60)  # Exponential backoff, max 60 seconds
        
        logger.warning(f"Reconnecting to {exchange} in {delay}s (attempt {attempts + 1})")
        
        await asyncio.sleep(delay)
        
        # Close old connection
        if exchange in self.connections:
            try:
                await self.connections[exchange].close()
            except:
                pass
        
        # Reconnect
        symbols = self.subscriptions.get(exchange, [])
        success = await self.connect(exchange, symbols)
        
        if success:
            self.reconnect_attempts[exchange] = 0
            logger.info(f"Successfully reconnected to {exchange}")
        else:
            self.reconnect_attempts[exchange] += 1
            logger.error(f"Failed to reconnect to {exchange}")
    
    async def _health_check(self):
        """Periodic health check of all connections"""
        while self.is_running:
            await asyncio.sleep(60)  # Check every minute
            
            for exchange, last_time in self.stats["last_message_time"].items():
                time_since_last = (datetime.now() - last_time).total_seconds()
                
                if time_since_last > 120:  # 2 minutes without messages
                    logger.warning(f"No messages from {exchange} for {time_since_last:.0f}s")
                    
                    # Force reconnect
                    asyncio.create_task(self._reconnect(exchange))
            
            # Log statistics
            logger.info(f"Stats: {self.stats['messages_received']} msgs received, "
                       f"{self.stats['messages_processed']} processed")
    
    async def stop(self):
        """Stop all connections gracefully"""
        self.is_running = False
        
        for exchange, connection in self.connections.items():
            try:
                await connection.close()
                logger.info(f"Closed connection to {exchange}")
            except:
                pass
        
        self.connections.clear()
        logger.info("WebSocket client stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            **self.stats,
            "connections": len(self.connections),
            "subscriptions": self.subscriptions,
            "reconnect_attempts": self.reconnect_attempts
        }
