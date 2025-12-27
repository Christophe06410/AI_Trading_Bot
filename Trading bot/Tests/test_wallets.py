"""
Tests for wallet management
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from solders.keypair import Keypair

from src.wallet_manager import WalletManager
from src.config import TradingBotConfig


class TestWalletManager:
    """Test wallet manager"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        config = TradingBotConfig()
        config.wallet.private_key_storage = "env"
        return config
    
    @pytest.fixture
    def wallet_manager(self, config):
        """Test wallet manager"""
        return WalletManager(config)
    
    @pytest.mark.asyncio
    async def test_wallet_creation(self, wallet_manager):
        """Test wallet creation"""
        # Mock environment variable
        with patch('os.getenv', return_value=None):
            result = await wallet_manager.initialize()
            
            # Should create test wallet when no env var
            assert not result  # Should fail without private key
    
    @pytest.mark.asyncio
    async def test_wallet_balance(self, wallet_manager):
        """Test balance checking"""
        # Mock RPC client
        mock_client = AsyncMock()
        mock_client.get_balance.return_value = Mock(value=1_000_000_000)  # 1 SOL
        
        wallet_manager.client = mock_client
        wallet_manager.wallet = Keypair()
        
        balance = await wallet_manager.update_balance()
        
        assert "SOL" in balance
        assert balance["SOL"] == 1.0  # 1 SOL
    
    def test_public_key(self, wallet_manager):
        """Test public key retrieval"""
        test_wallet = Keypair()
        wallet_manager.wallet = test_wallet
        
        pubkey = wallet_manager.get_public_key()
        
        assert pubkey == test_wallet.pubkey()
    
    @pytest.mark.asyncio
    async def test_connection_check(self, wallet_manager):
        """Test connection checking"""
        # Mock successful connection
        mock_client = AsyncMock()
        mock_client.get_version.return_value = Mock(value={"solana-core": "1.14.0"})
        
        wallet_manager.client = mock_client
        wallet_manager.wallet = Keypair()
        
        # Mock update_balance
        wallet_manager.update_balance = AsyncMock(return_value={"SOL": 1.0})
        
        connected = await wallet_manager.check_connection()
        
        assert connected is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
