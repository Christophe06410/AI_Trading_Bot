"""
Tests for trading engine
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.trading_engine import TradingEngine, TradingMetrics
from src.config import TradingBotConfig
from src.position_manager import Position, PositionStatus


class TestTradingEngine:
    """Test trading engine"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        config = TradingBotConfig()
        config.trading.pairs = ["SOL/USDC"]
        config.trading.max_positions = 3
        config.trading.position_size_usd = 100.0
        return config
    
    @pytest.fixture
    def trading_engine(self, config):
        """Test trading engine"""
        return TradingEngine(config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, trading_engine):
        """Test initialization"""
        # Mock components
        trading_engine.dex_executor.initialize = AsyncMock()
        trading_engine.ai_client.initialize = AsyncMock()
        trading_engine.market_data.initialize = AsyncMock()
        
        # Mock wallet manager
        mock_wallet_manager = Mock()
        mock_wallet_manager.get_keypair.return_value = None
        
        await trading_engine.initialize(mock_wallet_manager)
        
        # Verify components initialized
        trading_engine.dex_executor.initialize.assert_called_once()
        trading_engine.ai_client.initialize.assert_called_once()
        trading_engine.market_data.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_market_data_update(self, trading_engine):
        """Test market data update"""
        # Mock market data
        trading_engine.market_data.get_price = AsyncMock(return_value=100.0)
        
        await trading_engine._update_market_data()
        
        assert "SOL/USDC" in trading_engine.last_prices
        assert trading_engine.last_prices["SOL/USDC"] == 100.0
    
    def test_position_size_calculation(self, trading_engine):
        """Test position size calculation"""
        # Test with current price
        position_size = trading_engine._calculate_position_size(100.0)
        
        # Should be position_size_usd / max_positions
        expected = trading_engine.config.trading.position_size_usd / trading_engine.config.trading.max_positions
        assert position_size == expected
    
    def test_stop_loss_calculation(self, trading_engine):
        """Test stop loss calculation"""
        # Test LONG position
        entry_price = 100.0
        stop_loss = trading_engine._calculate_stop_loss(entry_price, "LONG")
        
        expected = entry_price * (1 - trading_engine.config.trading.stop_loss.initial_percent)
        assert stop_loss == expected
        
        # Test SHORT position
        stop_loss = trading_engine._calculate_stop_loss(entry_price, "SHORT")
        
        expected = entry_price * (1 + trading_engine.config.trading.stop_loss.initial_percent)
        assert stop_loss == expected
    
    @pytest.mark.asyncio
    async def test_trading_cycle(self, trading_engine):
        """Test trading cycle execution"""
        # Mock all components
        trading_engine._update_market_data = AsyncMock()
        trading_engine._check_stop_losses = AsyncMock()
        trading_engine._get_recommendations = AsyncMock(return_value=[])
        trading_engine._update_metrics = AsyncMock()
        trading_engine._log_status = Mock()
        
        await trading_engine.trading_cycle()
        
        # Verify all methods called
        trading_engine._update_market_data.assert_called_once()
        trading_engine._check_stop_losses.assert_called_once()
        trading_engine._get_recommendations.assert_called_once()
        trading_engine._update_metrics.assert_called_once()
        trading_engine._log_status.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
