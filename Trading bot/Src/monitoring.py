"""
Monitoring and metrics for trading bot
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

from src.config import TradingBotConfig
from src.utils import get_logger

logger = get_logger(__name__)


class MonitoringSystem:
    """Monitors trading bot performance and health"""
    
    def __init__(self, config: TradingBotConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Metrics
        self.metrics_start_time = time.time()
        
        # Prometheus metrics
        self.trades_total = Counter(
            'trading_bot_trades_total',
            'Total trades executed',
            ['pair', 'direction', 'status']
        )
        
        self.active_positions = Gauge(
            'trading_bot_active_positions',
            'Number of active positions'
        )
        
        self.total_pnl = Gauge(
            'trading_bot_total_pnl',
            'Total profit and loss'
        )
        
        self.win_rate = Gauge(
            'trading_bot_win_rate',
            'Win rate percentage'
        )
        
        self.trade_execution_time = Histogram(
            'trading_bot_trade_execution_time_seconds',
            'Trade execution time',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
        )
        
        self.api_request_duration = Summary(
            'trading_bot_api_request_duration_seconds',
            'API request duration'
        )
        
        self.errors_total = Counter(
            'trading_bot_errors_total',
            'Total errors',
            ['type']
        )
        
        self.health_status = Gauge(
            'trading_bot_health_status',
            'Health status (1=healthy, 0=unhealthy)'
        )
        
        self.logger.info("Monitoring system initialized")
    
    async def start(self):
        """Start monitoring system"""
        if self.config.monitoring.enabled:
            try:
                # Start Prometheus metrics server
                start_http_server(self.config.monitoring.metrics_port)
                self.logger.info(
                    f"Metrics server started on port {self.config.monitoring.metrics_port}"
                )
                
                # Initial health check
                self.health_status.set(1)
                
            except Exception as e:
                self.logger.error(f"Failed to start metrics server: {e}")
                self.health_status.set(0)
    
    async def update_metrics(self, trading_metrics: Optional[Dict[str, Any]] = None):
        """Update monitoring metrics"""
        try:
            if trading_metrics:
                # Update from trading engine metrics
                self.active_positions.set(trading_metrics.get('active_positions', 0))
                self.total_pnl.set(trading_metrics.get('total_pnl', 0))
                self.win_rate.set(trading_metrics.get('win_rate', 0))
            
            # Update uptime
            uptime = time.time() - self.metrics_start_time
            Gauge('trading_bot_uptime_seconds', 'Bot uptime in seconds').set(uptime)
            
            # Health check
            self.health_status.set(1)
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")
            self.errors_total.labels(type='metrics_update').inc()
            self.health_status.set(0)
    
    def record_trade(
        self,
        pair: str,
        direction: str,
        status: str,
        execution_time: float
    ):
        """Record a trade execution"""
        try:
            self.trades_total.labels(
                pair=pair,
                direction=direction,
                status=status
            ).inc()
            
            self.trade_execution_time.observe(execution_time)
            
            self.logger.info(
                "Trade recorded",
                pair=pair,
                direction=direction,
                status=status,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Failed to record trade: {e}")
            self.errors_total.labels(type='trade_recording').inc()
    
    def record_api_request(self, endpoint: str, duration: float, status: str = "success"):
        """Record an API request"""
        try:
            self.api_request_duration.observe(duration)
            
            Counter(
                'trading_bot_api_requests_total',
                'Total API requests',
                ['endpoint', 'status']
            ).labels(endpoint=endpoint, status=status).inc()
            
        except Exception as e:
            self.logger.error(f"Failed to record API request: {e}")
            self.errors_total.labels(type='api_recording').inc()
    
    def record_error(self, error_type: str, error_message: str = ""):
        """Record an error"""
        try:
            self.errors_total.labels(type=error_type).inc()
            
            self.logger.error(
                "Error recorded",
                error_type=error_type,
                error_message=error_message
            )
            
        except Exception as e:
            self.logger.error(f"Failed to record error: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        try:
            # Collect basic metrics
            metrics = {
                "uptime_seconds": time.time() - self.metrics_start_time,
                "active_positions": self.active_positions._value.get(),
                "total_pnl": self.total_pnl._value.get(),
                "win_rate": self.win_rate._value.get(),
                "health_status": self.health_status._value.get(),
                "timestamp": datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics summary: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        try:
            # Check if metrics server is running
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', self.config.monitoring.metrics_port))
            sock.close()
            
            health["components"]["metrics_server"] = {
                "status": "healthy" if result == 0 else "unhealthy",
                "port": self.config.monitoring.metrics_port
            }
            
            # Update overall status
            all_healthy = all(
                comp["status"] == "healthy" 
                for comp in health["components"].values()
            )
            
            health["status"] = "healthy" if all_healthy else "unhealthy"
            self.health_status.set(1 if all_healthy else 0)
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
            self.health_status.set(0)
        
        return health
    
    async def stop(self):
        """Stop monitoring system"""
        self.logger.info("Monitoring system stopped")
