"""
Risk management for trading bot
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.config import TradingBotConfig
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics"""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0


class RiskManager:
    """Manages trading risk"""
    
    def __init__(self, config: TradingBotConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        self.metrics = RiskMetrics()
        self.daily_start = datetime.now().replace(hour=0, minute=0, second=0)
        self.cooldown_until = None
        
        self.logger.info("Risk manager initialized")
    
    async def evaluate_trade(self, recommendation: Dict[str, Any]) -> bool:
        """Evaluate if trade should be executed based on risk"""
        
        # Check cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            self.logger.warning("In cooldown period, skipping trade")
            return False
        
        # Check daily loss limit
        if self.metrics.daily_pnl <= -self.config.risk.max_daily_loss_percent:
            self.logger.warning(f"Daily loss limit reached: {self.metrics.daily_pnl:.2f}%")
            return False
        
        # Check consecutive losses
        if self.metrics.consecutive_losses >= self.config.risk.max_consecutive_losses:
            self.logger.warning(f"Max consecutive losses reached: {self.metrics.consecutive_losses}")
            self._start_cooldown()
            return False
        
        # Check position concentration
        if not self._check_position_concentration(recommendation):
            return False
        
        # Check market conditions
        if not await self._check_market_conditions(recommendation):
            return False
        
        return True
    
    def _check_position_concentration(self, recommendation: Dict[str, Any]) -> bool:
        """Check position concentration limits"""
        # For now, simple check - can be expanded
        pair = recommendation.get("pair", "")
        
        # Check if pair is in blacklist (none for now)
        if pair in []:  # Could load from config
            self.logger.warning(f"Pair in blacklist: {pair}")
            return False
        
        return True
    
    async def _check_market_conditions(self, recommendation: Dict[str, Any]) -> bool:
        """Check market conditions"""
        # Check first Friday of month
        if self.config.trading.pause_first_friday:
            now = datetime.now()
            if now.weekday() == 4 and now.day <= 7:  # First Friday
                self.logger.warning("Trading paused (first Friday of month)")
                return False
        
        # Check volatility
        if self.metrics.volatility > self.config.trading.max_volatility_percent:
            self.logger.warning(f"High volatility: {self.metrics.volatility:.2f}%")
            return False
        
        return True
    
    def update_metrics(self, pnl: float, is_win: bool):
        """Update risk metrics after trade"""
        # Update daily PnL
        self.metrics.daily_pnl += pnl
        self.metrics.daily_trades += 1
        
        # Update consecutive losses
        if is_win:
            self.metrics.consecutive_losses = 0
        else:
            self.metrics.consecutive_losses += 1
        
        # Update drawdown
        if pnl < 0:
            self.metrics.current_drawdown += abs(pnl)
            self.metrics.max_drawdown = max(
                self.metrics.max_drawdown,
                self.metrics.current_drawdown
            )
        else:
            self.metrics.current_drawdown = max(0, self.metrics.current_drawdown - pnl)
        
        # Reset daily metrics if new day
        now = datetime.now()
        if now.date() > self.daily_start.date():
            self._reset_daily_metrics()
    
    def _reset_daily_metrics(self):
        """Reset daily metrics"""
        self.metrics.daily_pnl = 0.0
        self.metrics.daily_trades = 0
        self.daily_start = datetime.now().replace(hour=0, minute=0, second=0)
        self.logger.info("Daily metrics reset")
    
    def _start_cooldown(self):
        """Start cooldown period"""
        self.cooldown_until = datetime.now() + timedelta(
            minutes=self.config.risk.cooldown_minutes
        )
        self.logger.warning(
            f"Cooldown started until {self.cooldown_until}"
        )
    
    def update_volatility(self, volatility: float):
        """Update volatility metric"""
        self.metrics.volatility = volatility
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get risk metrics"""
        return {
            "daily_pnl": self.metrics.daily_pnl,
            "daily_trades": self.metrics.daily_trades,
            "consecutive_losses": self.metrics.consecutive_losses,
            "max_drawdown": self.metrics.max_drawdown,
            "current_drawdown": self.metrics.current_drawdown,
            "volatility": self.metrics.volatility,
            "in_cooldown": self.cooldown_until is not None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None
        }

