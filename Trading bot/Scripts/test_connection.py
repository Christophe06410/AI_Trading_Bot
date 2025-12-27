#!/usr/bin/env python3
"""
Test connections for Trading Bot
Tests wallet, RPC, AI service, and DEX connections
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.config import TradingBotConfig
from src.wallet_manager import WalletManager
from src.dex_executor import DEXExecutor
from src.ai_client import AIClient
from src.market_data import MarketData


class ConnectionTester:
    """Test all trading bot connections"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = TradingBotConfig.load(config_path)
        self.results = {}
        
    async def test_all(self):
        """Test all connections"""
        print("=" * 60)
        print("TRADING BOT CONNECTION TEST")
        print("=" * 60)
        
        tests = [
            ("Configuration", self.test_config),
            ("Wallet", self.test_wallet),
            ("Solana RPC", self.test_solana_rpc),
            ("Jupiter DEX", self.test_jupiter_dex),
            ("AI Service", self.test_ai_service),
            ("Market Data", self.test_market_data),
        ]
        
        for name, test_func in tests:
            print(f"\n🔍 Testing {name}...")
            try:
                result = await test_func()
                self.results[name] = result
                if result:
                    print(f"   ✅ {name}: PASSED")
                else:
                    print(f"   ❌ {name}: FAILED")
            except Exception as e:
                self.results[name] = False
                print(f"   ❌ {name}: ERROR - {e}")
        
        self.print_summary()
    
    async def test_config(self) -> bool:
        """Test configuration loading"""
        try:
            # Check config exists
            if not os.path.exists("config/config.yaml"):
                print("   ⚠️  config.yaml not found, using defaults")
            
            # Check .env exists
            if not os.path.exists(".env"):
                print("   ⚠️  .env file not found")
            
            # Validate config
            print(f"   • Trading Mode: {self.config.trading_mode}")
            print(f"   • Pairs: {', '.join(self.config.trading.pairs)}")
            print(f"   • Max Positions: {self.config.trading.max_positions}")
            print(f"   • Position Size: ${self.config.trading.position_size_usd}")
            
            return True
        except Exception as e:
            print(f"   • Error: {e}")
            return False
    
    async def test_wallet(self) -> bool:
        """Test wallet connection"""
        try:
            wallet_manager = WalletManager(self.config)
            
            # Initialize wallet
            success = await wallet_manager.initialize()
            if not success:
                print("   • Wallet initialization failed")
                return False
            
            # Get wallet info
            pubkey = wallet_manager.get_public_key()
            if pubkey:
                print(f"   • Wallet Address: {pubkey}")
                
                # Check balance
                balance = await wallet_manager.update_balance()
                if "SOL" in balance:
                    print(f"   • Balance: {balance['SOL']:.4f} SOL")
                    
                    # Warning if zero balance
                    if balance["SOL"] == 0:
                        print("   ⚠️  Zero balance - get testnet SOL from https://solfaucet.com/")
                
                await wallet_manager.close()
                return True
            else:
                print("   • No wallet loaded")
                return False
                
        except Exception as e:
            print(f"   • Error: {e}")
            return False
    
    async def test_solana_rpc(self) -> bool:
        """Test Solana RPC connection"""
        try:
            from solana.rpc.async_api import AsyncClient
            
            async with AsyncClient(self.config.solana.rpc_endpoint) as client:
                # Get version
                version = await client.get_version()
                if version.value:
                    print(f"   • RPC Endpoint: {self.config.solana.rpc_endpoint}")
                    print(f"   • Solana Version: {version.value['solana-core']}")
                    return True
                else:
                    print("   • No response from RPC")
                    return False
                    
        except Exception as e:
            print(f"   • Error: {e}")
            return False
    
    async def test_jupiter_dex(self) -> bool:
        """Test Jupiter DEX connection"""
        try:
            dex = DEXExecutor(self.config)
            await dex.initialize()
            
            # Get SOL price
            price = await dex.get_price("SOL")
            if price:
                print(f"   • SOL Price: ${price:.4f}")
                await dex.close()
                return True
            else:
                print("   • Failed to get price")
                await dex.close()
                return False
                
        except Exception as e:
            print(f"   • Error: {e}")
            return False
    
    async def test_ai_service(self) -> bool:
        """Test AI service connection"""
        try:
            ai_client = AIClient(self.config)
            await ai_client.initialize()
            
            # Check connection
            connected = await ai_client.check_connection()
            if connected:
                print(f"   • AI Service: {self.config.get_full_ai_endpoint()}")
                print(f"   • Status: Connected")
                await ai_client.close()
                return True
            else:
                print(f"   • AI Service: {self.config.get_full_ai_endpoint()}")
                print(f"   • Status: Not connected")
                await ai_client.close()
                return False
                
        except Exception as e:
            print(f"   • Error: {e}")
            print(f"   • Tip: Make sure AI service is running on port 8000")
            return False
    
    async def test_market_data(self) -> bool:
        """Test market data connections"""
        try:
            market_data = MarketData(self.config)
            await market_data.initialize()
            
            # Test price fetching
            for pair in self.config.trading.pairs:
                price = await market_data.get_price(pair)
                if price:
                    print(f"   • {pair}: ${price:.4f}")
                else:
                    print(f"   • {pair}: Failed to get price")
            
            await market_data.close()
            return any([
                await market_data.get_price(pair) 
                for pair in self.config.trading.pairs
            ])
            
        except Exception as e:
            print(f"   • Error: {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.results.values() if result)
        total = len(self.results)
        
        print(f"\n📊 Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ All tests passed! Trading bot is ready.")
            print("\n🚀 Next steps:")
            print("   1. Run test mode: python src/main.py --test")
            print("   2. Start trading: python src/main.py start")
        else:
            print("⚠️  Some tests failed:")
            for name, result in self.results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"   {status} {name}")
            
            print("\n🔧 Troubleshooting:")
            if not self.results.get("Wallet"):
                print("   • Check SOLANA_PRIVATE_KEY in .env file")
                print("   • Get testnet SOL: https://solfaucet.com/")
            
            if not self.results.get("Solana RPC"):
                print("   • Check RPC endpoint in config.yaml")
                print("   • Try different RPC: https://publicnodes.solana.com/")
            
            if not self.results.get("AI Service"):
                print("   • Start AI service: uvicorn ai_service.main:app --reload")
                print("   • Check AI_SERVICE_URL in config.yaml")


async def main():
    """Main function"""
    tester = ConnectionTester()
    await tester.test_all()


if __name__ == "__main__":
    asyncio.run(main())
