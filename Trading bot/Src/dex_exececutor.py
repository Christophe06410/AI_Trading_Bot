"""
DEX execution for Solana (Jupiter/Raydium) - Updated with better error handling
"""

import aiohttp
import json
import base64
import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts

from src.config import TradingBotConfig
from src.utils import get_logger, retry_with_backoff

logger = get_logger(__name__)


@dataclass
class SwapQuote:
    """Swap quote from DEX"""
    input_mint: str
    output_mint: str
    in_amount: int
    out_amount: int
    slippage_bps: int
    swap_mode: str
    price_impact_pct: float
    other_amount_threshold: int
    route_plan: List[Dict[str, Any]]
    context_slot: Optional[int] = None
    
    @property
    def price(self) -> float:
        """Calculate effective price"""
        if self.in_amount > 0 and self.out_amount > 0:
            return self.out_amount / self.in_amount
        return 0.0


@dataclass
class SwapResult:
    """Swap execution result"""
    success: bool
    transaction_hash: Optional[str] = None
    error_message: Optional[str] = None
    input_amount: float = 0.0
    output_amount: float = 0.0
    price_impact: float = 0.0
    fee: float = 0.0
    slot: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "transaction_hash": self.transaction_hash,
            "error_message": self.error_message,
            "input_amount": self.input_amount,
            "output_amount": self.output_amount,
            "price_impact": self.price_impact,
            "fee": self.fee,
            "slot": self.slot
        }


class DEXExecutor:
    """Executes trades on Solana DEXes with proper error handling"""
    
    def __init__(self, config: TradingBotConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.client: Optional[AsyncClient] = None
        self.logger = get_logger(__name__)
        
        # Token mint addresses
        self.token_mints = {
            "SOL": "So11111111111111111111111111111111111111112",
            "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
            "SRM": "SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt",
            "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
        }
    
    async def initialize(self):
        """Initialize DEX executor"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self.client = AsyncClient(self.config.solana.rpc_endpoint)
        self.logger.info("DEX executor initialized")
    
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: float,
        slippage_bps: int = 100
    ) -> Optional[SwapQuote]:
        """Get swap quote from Jupiter"""
        try:
            # Convert to lamports if SOL
            if input_mint == "SOL":
                amount_lamports = int(amount * 1_000_000_000)  # SOL to lamports
            else:
                # For tokens, need to know decimals (simplified)
                amount_lamports = int(amount * 1_000_000)  # Assume 6 decimals
            
            # Get mint addresses
            input_mint_addr = self.token_mints.get(input_mint, input_mint)
            output_mint_addr = self.token_mints.get(output_mint, output_mint)
            
            url = f"{self.config.solana.dex.jupiter_api_url}/quote"
            
            params = {
                "inputMint": input_mint_addr,
                "outputMint": output_mint_addr,
                "amount": str(amount_lamports),
                "slippageBps": str(slippage_bps),
                "swapMode": "ExactIn",
                "onlyDirectRoutes": "false",
                "asLegacyTransaction": "false"
            }
            
            self.logger.debug(f"Getting quote: {params}")
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return SwapQuote(
                        input_mint=data.get("inputMint"),
                        output_mint=data.get("outputMint"),
                        in_amount=int(data.get("inAmount", 0)),
                        out_amount=int(data.get("outAmount", 0)),
                        slippage_bps=data.get("slippageBps", slippage_bps),
                        swap_mode=data.get("swapMode", "ExactIn"),
                        price_impact_pct=data.get("priceImpactPct", 0.0),
                        other_amount_threshold=data.get("otherAmountThreshold", 0),
                        route_plan=data.get("routePlan", []),
                        context_slot=data.get("contextSlot")
                    )
                else:
                    error_text = await response.text()
                    self.logger.error(f"Quote request failed {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to get quote: {e}")
            return None
    
    @retry_with_backoff(max_retries=3)
    async def execute_swap(
        self,
        wallet: Keypair,
        quote: SwapQuote,
        priority_fee: Optional[int] = None
    ) -> SwapResult:
        """Execute swap on Jupiter"""
        try:
            # 1. Get swap transaction
            swap_url = f"{self.config.solana.dex.jupiter_api_url}/swap"
            
            swap_data = {
                "quoteResponse": {
                    "inputMint": quote.input_mint,
                    "outputMint": quote.output_mint,
                    "inAmount": str(quote.in_amount),
                    "outAmount": str(quote.out_amount),
                    "slippageBps": quote.slippage_bps,
                    "swapMode": quote.swap_mode,
                    "priceImpactPct": quote.price_impact_pct,
                    "otherAmountThreshold": str(quote.other_amount_threshold),
                    "contextSlot": quote.context_slot,
                    "routePlan": quote.route_plan
                },
                "userPublicKey": str(wallet.pubkey()),
                "wrapAndUnwrapSol": True,
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": priority_fee or 0,
                "useSharedAccounts": True,
                "asLegacyTransaction": False
            }
            
            self.logger.info(f"Executing swap: {swap_data['quoteResponse']['inputMint']} -> {swap_data['quoteResponse']['outputMint']}")
            
            async with self.session.post(
                swap_url,
                json=swap_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"Swap request failed: {error_text}")
                    return SwapResult(
                        success=False,
                        error_message=f"Swap API error: {response.status} - {error_text}"
                    )
                
                swap_result = await response.json()
                
                # Check for errors in response
                if "error" in swap_result:
                    return SwapResult(
                        success=False,
                        error_message=swap_result["error"]
                    )
                
                # 2. Send transaction
                tx_hash = await self._send_transaction(
                    wallet,
                    swap_result.get("swapTransaction")
                )
                
                if tx_hash:
                    # Calculate amounts in readable units
                    input_amount = quote.in_amount / 1_000_000_000 if quote.input_mint == self.token_mints["SOL"] else quote.in_amount / 1_000_000
                    output_amount = quote.out_amount / 1_000_000_000 if quote.output_mint == self.token_mints["SOL"] else quote.out_amount / 1_000_000
                    
                    return SwapResult(
                        success=True,
                        transaction_hash=tx_hash,
                        input_amount=input_amount,
                        output_amount=output_amount,
                        price_impact=quote.price_impact_pct,
                        fee=0.0005,  # Estimated fee
                        slot=quote.context_slot
                    )
                else:
                    return SwapResult(
                        success=False,
                        error_message="Transaction failed"
                    )
                    
        except Exception as e:
            self.logger.error(f"Swap execution failed: {e}")
            return SwapResult(
                success=False,
                error_message=str(e)
            )
    
    async def _send_transaction(
        self,
        wallet: Keypair,
        transaction_data: str
    ) -> Optional[str]:
        """Send transaction to Solana network"""
        try:
            # Deserialize transaction
            tx_bytes = base64.b64decode(transaction_data)
            transaction = VersionedTransaction.from_bytes(tx_bytes)
            
            # Sign transaction with wallet
            transaction.sign([wallet])
            
            # Send transaction
            opts = TxOpts(
                skip_preflight=False,
                preflight_commitment=Confirmed,
                max_retries=3
            )
            
            result = await self.client.send_transaction(
                transaction,
                opts=opts
            )
            
            if result.value:
                tx_hash = str(result.value)
                self.logger.info(f"Transaction sent: {tx_hash}")
                
                # Wait for confirmation
                confirmed = await self._confirm_transaction(tx_hash)
                
                if confirmed:
                    return tx_hash
                else:
                    self.logger.warning(f"Transaction not confirmed: {tx_hash}")
                    return None
            else:
                self.logger.error("Failed to send transaction")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to send transaction: {e}")
            return None
    
    async def _confirm_transaction(self, tx_hash: str, timeout: int = 30) -> bool:
        """Wait for transaction confirmation"""
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Check signature status
                status = await self.client.get_signature_statuses([tx_hash], search_transaction_history=True)
                
                if status.value and status.value[0]:
                    sig_status = status.value[0]
                    
                    if sig_status.confirmation_status:
                        if sig_status.confirmation_status in ["confirmed", "finalized"]:
                            self.logger.info(f"Transaction confirmed: {tx_hash}")
                            return True
                        elif sig_status.confirmation_status == "processed":
                            # Still processing
                            pass
                    elif sig_status.err:
                        self.logger.error(f"Transaction error: {sig_status.err}")
                        return False
                
                await asyncio.sleep(1)
            
            self.logger.warning(f"Transaction confirmation timeout: {tx_hash}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error confirming transaction: {e}")
            return False
    
    async def get_price(self, mint: str) -> Optional[float]:
        """Get token price from Jupiter"""
        try:
            mint_addr = self.token_mints.get(mint, mint)
            
            url = f"{self.config.solana.dex.jupiter_api_url}/price"
            params = {"ids": mint_addr}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "data" in data and mint_addr in data["data"]:
                        price = float(data["data"][mint_addr]["price"])
                        return price
                    else:
                        self.logger.warning(f"No price data for {mint}")
                        return None
                else:
                    self.logger.error(f"Price request failed: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to get price: {e}")
            return None
    
    async def get_balance(self, wallet_pubkey: Pubkey, mint: str = "SOL") -> float:
        """Get token balance for wallet"""
        try:
            mint_addr = self.token_mints.get(mint, mint)
            
            if mint == "SOL":
                # SOL balance
                balance = await self.client.get_balance(wallet_pubkey)
                if balance.value:
                    return balance.value / 1_000_000_000  # lamports to SOL
            else:
                # Token balance
                from solana.rpc.types import TokenAccountOpts
                accounts = await self.client.get_token_accounts_by_owner(
                    wallet_pubkey,
                    TokenAccountOpts(mint=mint_addr)
                )
                
                if accounts.value:
                    # Parse token account data
                    import base64
                    for account in accounts.value:
                        data = base64.b64decode(account.account.data)
                        # Token account layout: mint(32) + owner(32) + amount(8) + ...
                        if len(data) >= 72:  # Minimum size
                            amount = int.from_bytes(data[64:72], 'little')
                            decimals = 6  # Default for most tokens
                            return amount / (10 ** decimals)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    async def check_connection(self) -> bool:
        """Check DEX connection"""
        try:
            price = await self.get_price("SOL")
            return price is not None and price > 0
        except Exception as e:
            self.logger.warning(f"DEX connection check failed: {e}")
            return False
    
    async def close(self):
        """Close connections"""
        if self.session:
            await self.session.close()
        if self.client:
            await self.client.close()
        self.logger.info("DEX executor closed")