"""
Secure wallet management for Solana
"""

import os
import base58
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient

from src.config import TradingBotConfig
from src.utils import get_logger, encrypt_data, decrypt_data

logger = get_logger(__name__)


@dataclass
class WalletInfo:
    """Wallet information"""
    public_key: str
    balance_sol: float = 0.0
    balance_usdc: float = 0.0
    is_loaded: bool = False


class WalletManager:
    """Manages Solana wallet securely"""
    
    def __init__(self, config: TradingBotConfig):
        self.config = config
        self.wallet: Optional[Keypair] = None
        self.client: Optional[AsyncClient] = None
        self.info = WalletInfo(public_key="")
        
        logger.info("Wallet manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize wallet connection"""
        try:
            # Load wallet
            if not await self._load_wallet():
                logger.error("Failed to load wallet")
                return False
            
            # Initialize RPC client
            self.client = AsyncClient(self.config.solana.rpc_endpoint)
            
            # Check balance
            await self.update_balance()
            
            self.info.is_loaded = True
            logger.info("Wallet initialized successfully", public_key=self.info.public_key)
            return True
            
        except Exception as e:
            logger.error("Wallet initialization failed", error=str(e))
            return False
    
    async def _load_wallet(self) -> bool:
        """Load wallet from secure storage"""
        try:
            private_key = None
            
            if self.config.wallet.private_key_storage == "env":
                # Load from environment variable
                private_key_b58 = os.getenv("SOLANA_PRIVATE_KEY")
                if not private_key_b58:
                    logger.error("SOLANA_PRIVATE_KEY not found in environment")
                    return False
                
                # Decode from base58
                try:
                    private_key = base58.b58decode(private_key_b58)
                except Exception as e:
                    logger.error("Failed to decode private key", error=str(e))
                    return False
                    
            elif self.config.wallet.private_key_storage == "encrypted_file":
                # Load from encrypted file
                file_path = self.config.wallet.encrypted_file_path
                if not os.path.exists(file_path):
                    logger.error(f"Encrypted key file not found: {file_path}")
                    return False
                
                # Read and decrypt
                with open(file_path, 'rb') as f:
                    encrypted_data = f.read()
                
                encryption_key = os.getenv("WALLET_ENCRYPTION_KEY")
                if not encryption_key:
                    logger.error("WALLET_ENCRYPTION_KEY not found in environment")
                    return False
                
                try:
                    private_key = decrypt_data(encrypted_data, encryption_key)
                except Exception as e:
                    logger.error("Failed to decrypt private key", error=str(e))
                    return False
            
            else:
                logger.error(f"Unknown storage type: {self.config.wallet.private_key_storage}")
                return False
            
            # Create keypair
            if private_key:
                self.wallet = Keypair.from_bytes(private_key)
                self.info.public_key = str(self.wallet.pubkey())
                
                # Verify wallet address matches config if provided
                if self.config.wallet.address:
                    if self.info.public_key != self.config.wallet.address:
                        logger.warning(
                            "Wallet address mismatch",
                            config_address=self.config.wallet.address,
                            loaded_address=self.info.public_key
                        )
                
                logger.info("Wallet loaded successfully", public_key=self.info.public_key)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to load wallet", error=str(e))
            return False
    
    async def update_balance(self) -> Dict[str, float]:
        """Update wallet balances"""
        if not self.wallet or not self.client:
            logger.warning("Wallet or client not initialized")
            return {}
        
        try:
            # Get SOL balance
            balance = await self.client.get_balance(self.wallet.pubkey())
            
            if balance.value:
                sol_balance = balance.value / 1_000_000_000  # lamports to SOL
                self.info.balance_sol = sol_balance
                
                logger.info(
                    "Balance updated",
                    sol_balance=sol_balance,
                    public_key=self.info.public_key[:10] + "..."
                )
                
                return {"SOL": sol_balance}
            else:
                logger.warning("Zero balance")
                return {"SOL": 0.0}
                
        except Exception as e:
            logger.error("Failed to update balance", error=str(e))
            return {}
    
    async def check_connection(self) -> bool:
        """Check wallet and RPC connection"""
        if not self.wallet or not self.client:
            return False
        
        try:
            # Check RPC connection
            version = await self.client.get_version()
            if version.value:
                # Check wallet is accessible
                balance = await self.update_balance()
                return bool(balance)
            return False
            
        except Exception as e:
            logger.error("Connection check failed", error=str(e))
            return False
    
    def get_public_key(self) -> Optional[Pubkey]:
        """Get wallet public key"""
        return self.wallet.pubkey() if self.wallet else None
    
    def get_keypair(self) -> Optional[Keypair]:
        """Get wallet keypair"""
        return self.wallet
    
    async def close(self):
        """Close connections"""
        if self.client:
            await self.client.close()
            logger.info("Wallet client closed")
    
    def save_to_encrypted_file(self, encryption_key: str) -> bool:
        """Save wallet to encrypted file (for backup)"""
        if not self.wallet:
            logger.error("No wallet to save")
            return False
        
        try:
            # Get private key bytes
            private_key = bytes(self.wallet)
            
            # Encrypt
            encrypted_data = encrypt_data(private_key, encryption_key)
            
            # Save to file
            file_path = self.config.wallet.encrypted_file_path
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info("Wallet saved to encrypted file", file_path=file_path)
            return True
            
        except Exception as e:
            logger.error("Failed to save wallet to encrypted file", error=str(e))
            return False
