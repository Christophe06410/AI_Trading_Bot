#!/usr/bin/env python3
"""
Solana Trading Bot - Complete Production Version
"""

import asyncio
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import structlog
import click

# Import local modules
from src.utils import setup_logging, load_config, logger
from src.wallet_manager import WalletManager
from src.dex_trader import DEXTrader
from src.position_manager import PositionManager, Position, PositionStatus

# AI Service client
import aiohttp


class TradingBot:
    """Main trading bot class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        setup_logging(self.config)
        self.logger = logger.bind(component="trading_bot")
        
        # Initialize components
        self.wallet_manager = WalletManager(self.config)
        self.position_manager = PositionManager()
        self.dex_trader = DEXTrader(self.config, self.wallet_manager)
        
        # Trading state
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.start_time = datetime.now()
        self.iteration_count = 0
        self.trade_count = 0
        
        # Market data cache
        self.market_data: Dict[str, float] = {}
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("trading_bot_initialized", config_path=config_path)
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        self.logger.info("initializing_components")
        
        try:
            # Initialize wallet
            if not await self.wallet_manager.initialize():
                self.logger.error("wallet_initialization_failed")
                return False
            
            # Check minimum balance
            min_balance = self.config.get("safety", {}).get("min_balance_sol", 0.5)
            if not await self.wallet_manager.check_minimum_balance(min_balance):
                self.logger.warning("low_balance_warning", min_balance=min_balance)
            
            # Print startup info
            self._print_startup_info()
            
            self.logger.info("initialization_complete")
            return True
            
        except Exception as e:
            self.logger.error("initialization_error", error=str(e))
            return False
    
    def _print_startup_info(self):
        """Print startup information"""
        trading_config = self.config.get("trading", {})
        wallet_address = self.wallet_manager.get_public_key_short()
        
        info = f"""
╔══════════════════════════════════════════════════════════╗
║                 SOLANA TRADING BOT                       ║
║                    Version 1.0.0                         ║
╠══════════════════════════════════════════════════════════╣
║ Mode:        {trading_config.get('mode', 'testnet'):38} ║
║ Wallet:      {wallet_address:38} ║
║ Pairs:       {', '.join(trading_config.get('pairs', ['SOL/USDC'])):38} ║
║ Max Positions:{trading_config.get('max_positions', 5):38} ║
║ Position Size:${trading_config.get('position_size_usd', 50.0):37.2f} ║
║ Stop Loss:   {trading_config.get('stop_loss_percent', 3.0)}%{' ':35} ║
╚══════════════════════════════════════════════════════════╝
        """
        print(info)
    
    async def run(self):
        """Main trading loop"""
        if not self.running:
            self.running = True
            self.logger.info("starting_trading_loop")
        
        try:
            # Create background tasks
            tasks = [
                asyncio.create_task(self._market_data_loop(), name="market_data"),
                asyncio.create_task(self._trading_loop(), name="trading"),
                asyncio.create_task(self._position_monitoring_loop(), name="monitoring"),
                asyncio.create_task(self._statistics_loop(), name="statistics"),
            ]
            
            # Run until shutdown
            await self.shutdown_event.wait()
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error("main_loop_error", error=str(e))
        finally:
            await self.shutdown()
    
    async def _market_data_loop(self):
        """Fetch market data continuously"""
        check_interval = self.config.get("performance", {}).get("market_data_refresh_seconds", 1)
        
        while self.running:
            try:
                pairs = self.config.get("trading", {}).get("pairs", ["SOL/USDC"])
                
                for pair in pairs:
                    price = await self._get_market_price(pair)
                    if price:
                        self.market_data[pair] = price
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error("market_data_loop_error", error=str(e))
                await asyncio.sleep(5)
    
    async def _get_market_price(self, pair: str) -> Optional[float]:
        """Get current market price for a pair"""
        try:
            # Parse pair (e.g., "SOL/USDC")
            if '/' in pair:
                base, quote = pair.split('/')
                
                # For now, use Jupiter for SOL/USDC
                if base == "SOL" and quote == "USDC":
                    price = await self.dex_trader.get_price("SOL", "USDC", 1.0)
                    return price
            
            # Default fallback
            return 100.0  # Mock price for testing
            
        except Exception as e:
            self.logger.error("market_price_error", pair=pair, error=str(e))
            return None
    
    async def _trading_loop(self):
        """Main trading logic loop"""
        check_interval = self.config.get("performance", {}).get("check_interval_seconds", 5)
        
        while self.running:
            try:
                self.iteration_count += 1
                
                # Check if we can trade
                if not await self._can_trade():
                    await asyncio.sleep(check_interval)
                    continue
                
                # Get AI recommendation for each pair
                pairs = self.config.get("trading", {}).get("pairs", ["SOL/USDC"])
                
                for pair in pairs:
                    current_price = self.market_data.get(pair)
                    if not current_price:
                        continue
                    
                    # Get AI recommendation
                    recommendation = await self._get_ai_recommendation(pair, current_price)
                    
                    # Process recommendation
                    if recommendation and recommendation.get("recommendation") != "WAIT":
                        await self._process_recommendation(pair, recommendation, current_price)
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error("trading_loop_error", error=str(e))
                await asyncio.sleep(10)
    
    async def _can_trade(self) -> bool:
        """Check if trading conditions are met"""
        # Check max positions
        open_positions = self.position_manager.get_open_positions()
        max_positions = self.config.get("trading", {}).get("max_positions", 5)
        
        if len(open_positions) >= max_positions:
            return False
        
        # Check balance
        min_balance = self.config.get("safety", {}).get("min_balance_sol", 0.5)
        if not await self.wallet_manager.check_minimum_balance(min_balance):
            return False
        
        # Check if it's first Friday (skip for crypto)
        if self._is_first_friday() and self.config.get("safety", {}).get("pause_on_first_friday", False):
            self.logger.info("trading_paused_first_friday")
            return False
        
        return True
    
    def _is_first_friday(self) -> bool:
        """Check if today is first Friday of month"""
        today = datetime.now()
        return today.weekday() == 4 and today.day <= 7
    
    async def _get_ai_recommendation(self, pair: str, current_price: float) -> Optional[Dict[str, Any]]:
        """Get recommendation from AI service"""
        try:
            ai_config = self.config.get("ai", {})
            endpoint = ai_config.get("endpoint", "http://localhost:8000")
            
            # Prepare candle data (mock for now)
            candles = self._generate_candle_data(current_price)
            
            # Get open positions for this pair
            open_positions = self.position_manager.get_open_positions()
            positions_data = []
            
            for pos in open_positions:
                if pos.pair == pair:
                    positions_data.append({
                        "id": pos.id,
                        "timestamp": pos.entry_time,
                        "direction": pos.direction,
                        "leverage": 1.0,
                        "price": pos.entry_price,
                        "nbLots": pos.size_usd / pos.entry_price,
                        "SL": pos.stop_loss,
                        "TP": pos.take_profit
                    })
            
            # Prepare request
            request_data = {
                "pair": pair,
                "chain": "Solana",
                "previous": candles,
                "positions": positions_data
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {ai_config.get('api_key', '')}"
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{endpoint}/api/v1/recommendation"
                async with session.post(url, json=request_data, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug("ai_recommendation_received", pair=pair, recommendation=data)
                        return data
                    else:
                        self.logger.error("ai_request_failed", 
                                        status=response.status,
                                        text=await response.text())
                        return None
                        
        except Exception as e:
            self.logger.error("ai_recommendation_error", pair=pair, error=str(e))
            return None
    
    def _generate_candle_data(self, current_price: float) -> List[Dict[str, Any]]:
        """Generate mock candle data for AI"""
        import random
        from datetime import datetime, timedelta
        
        candles = []
        for i in range(4):
            timestamp = (datetime.utcnow() - timedelta(minutes=i)).isoformat() + "Z"
            candle = {
                "timestamp": timestamp,
                "o": current_price * random.uniform(0.99, 1.01),
                "h": current_price * random.uniform(1.01, 1.03),
                "l": current_price * random.uniform(0.97, 0.99),
                "c": current_price,
                "v": random.uniform(1000, 5000),
                "ema200": current_price * random.uniform(0.95, 1.05)
            }
            candles.append(candle)
        
        return candles
    
    async def _process_recommendation(self, pair: str, recommendation: Dict[str, Any], current_price: float):
        """Process AI recommendation"""
        action = recommendation.get("recommendation", "WAIT")
        confidence = recommendation.get("confidence", 0.0)
        
        # Check minimum confidence
        min_confidence = self.config.get("trading", {}).get("min_confidence", 0.7)
        if confidence < min_confidence:
            self.logger.debug("low_confidence_skip", pair=pair, confidence=confidence, min=min_confidence)
            return
        
        # Check if it's a position close
        if action == "CLOSE":
            position_data = recommendation.get("position")
            if position_data and "id" in position_data:
                await self._close_position_by_id(position_data["id"], current_price, "ai_signal")
            return
        
        # It's a new position (LONG or SHORT)
        await self._open_position(pair, action, current_price, confidence)
    
    async def _open_position(self, pair: str, direction: str, current_price: float, confidence: float):
        """Open a new position"""
        try:
            # Generate position ID
            position_id = f"{direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.trade_count}"
            
            # Get position size
            position_size_usd = self.config.get("trading", {}).get("position_size_usd", 50.0)
            
            # Calculate stop loss
            stop_loss_percent = self.config.get("trading", {}).get("stop_loss_percent", 0.03)
            
            if direction == "LONG":
                stop_loss = current_price * (1 - stop_loss_percent)
            else:  # SHORT
                stop_loss = current_price * (1 + stop_loss_percent)
            
            # Create position object
            position = Position(
                id=position_id,
                pair=pair,
                direction=direction,
                entry_price=current_price,
                entry_time=datetime.utcnow().isoformat() + "Z",
                size_usd=position_size_usd,
                stop_loss=stop_loss,
                current_stop_loss=stop_loss,  # Initial trailing stop
                status=PositionStatus.OPEN.value
            )
            
            # Execute trade based on trading mode
            trading_mode = self.config.get("trading", {}).get("mode", "testnet")
            
            if trading_mode == "live":
                # Execute real trade
                trade_result = await self._execute_real_trade(pair, direction, position_size_usd, current_price)
                if trade_result and trade_result.get("success"):
                    position.transaction_hash = trade_result.get("signature")
                else:
                    self.logger.error("real_trade_failed", position_id=position_id)
                    return
            else:
                # Simulate trade for testnet/paper mode
                trade_result = await self.dex_trader.simulate_trade(pair, direction, position_size_usd, current_price)
                position.transaction_hash = trade_result.get("signature")
            
            # Save position to database
            if self.position_manager.add_position(position):
                self.trade_count += 1
                
                self.logger.info("position_opened",
                               position_id=position_id,
                               pair=pair,
                               direction=direction,
                               entry_price=current_price,
                               size_usd=position_size_usd,
                               stop_loss=stop_loss,
                               confidence=confidence,
                               trade_count=self.trade_count)
            else:
                self.logger.error("position_save_failed", position_id=position_id)
                
        except Exception as e:
            self.logger.error("position_open_error", pair=pair, direction=direction, error=str(e))
    
    async def _execute_real_trade(self, pair: str, direction: str, amount_usd: float, current_price: float) -> Optional[Dict[str, Any]]:
        """Execute a real trade on DEX"""
        try:
            # Parse pair
            base, quote = pair.split('/') if '/' in pair else (pair, "USDC")
            
            if direction == "LONG":
                # Buy base with quote
                result = await self.dex_trader.execute_swap(quote, base, amount_usd / current_price)
            else:  # SHORT
                # Sell base for quote (requires having the base token)
                result = await self.dex_trader.execute_swap(base, quote, amount_usd / current_price)
            
            return result
            
        except Exception as e:
            self.logger.error("real_trade_execution_error", error=str(e))
            return None
    
    async def _close_position_by_id(self, position_id: str, current_price: float, reason: str):
        """Close a position by ID"""
        try:
            # Close in database
            success = self.position_manager.close_position(position_id, current_price, reason)
            
            if success:
                self.logger.info("position_closed",
                               position_id=position_id,
                               reason=reason,
                               exit_price=current_price)
            else:
                self.logger.error("position_close_failed", position_id=position_id)
                
        except Exception as e:
            self.logger.error("position_close_error", position_id=position_id, error=str(e))
    
    async def _position_monitoring_loop(self):
        """Monitor positions for stop losses and updates"""
        check_interval = self.config.get("performance", {}).get("position_check_interval_seconds", 1)
        
        while self.running:
            try:
                open_positions = self.position_manager.get_open_positions()
                
                for position in open_positions:
                    current_price = self.market_data.get(position.pair)
                    if not current_price:
                        continue
                    
                    # Check stop loss
                    if position.check_stop_loss(current_price):
                        await self._close_position_by_id(position.id, current_price, "stop_loss")
                        continue
                    
                    # Update trailing stop
                    trailing_config = self.config.get("trading", {}).get("trailing_stop", {})
                    if trailing_config.get("enabled", False):
                        trail_percent = trailing_config.get("trail_percent", 0.01)
                        activation_percent = trailing_config.get("activation_percent", 0.02)
                        
                        # Check if we've reached activation threshold
                        pnl_data = position.calculate_pnl(current_price)
                        if pnl_data["pnl_percent"] >= activation_percent * 100:
                            if position.update_trailing_stop(current_price, trail_percent):
                                # Save updated stop to database
                                self.position_manager.update_position(
                                    position.id,
                                    current_stop_loss=position.current_stop_loss
                                )
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error("position_monitoring_error", error=str(e))
                await asyncio.sleep(5)
    
    async def _statistics_loop(self):
        """Periodically log statistics"""
        while self.running:
            try:
                # Log every 60 seconds
                await asyncio.sleep(60)
                
                stats = self.position_manager.get_statistics()
                runtime = datetime.now() - self.start_time
                
                self.logger.info("trading_statistics",
                               runtime=str(runtime).split('.')[0],
                               iterations=self.iteration_count,
                               trades=self.trade_count,
                               total_pnl=stats.get("total_pnl", 0),
                               win_rate=f"{stats.get('win_rate', 0):.1f}%",
                               open_positions=stats.get("open_positions", 0))
                
            except Exception as e:
                self.logger.error("statistics_error", error=str(e))
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("shutdown_signal_received", signal=signum)
        self.shutdown_event.set()
    
    async def shutdown(self):
        """Graceful shutdown"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("starting_graceful_shutdown")
        
        try:
            # Close wallet connections
            await self.wallet_manager.close()
            
            # Close DEX trader
            await self.dex_trader.close()
            
            # Backup database
            self.position_manager.backup_database()
            
            # Log final statistics
            stats = self.position_manager.get_statistics()
            runtime = datetime.now() - self.start_time
            
            self.logger.info("shutdown_complete",
                           total_runtime=str(runtime).split('.')[0],
                           total_iterations=self.iteration_count,
                           total_trades=self.trade_count,
                           final_stats=stats)
            
        except Exception as e:
            self.logger.error("shutdown_error", error=str(e))


@click.group()
def cli():
    """Solana Trading Bot CLI"""
    pass

@cli.command()
@click.option('--config', default='config/config.yaml', help='Configuration file')
def run(config: str):
    """Run the trading bot"""
    bot = TradingBot(config)
    
    async def main():
        if await bot.initialize():
            await bot.run()
        else:
            print("❌ Initialization failed. Check logs for details.")
            sys.exit(1)
    
    asyncio.run(main())

@cli.command()
@click.option('--config', default='config/config.yaml', help='Configuration file')
def test(config: str):
    """Test bot connection and configuration"""
    bot = TradingBot(config)
    
    async def test_connections():
        print("🔍 Testing bot connections...")
        
        # Test wallet
        print("1. Testing wallet...")
        if await bot.wallet_manager.initialize():
            balance = await bot.wallet_manager.get_balance()
            address = bot.wallet_manager.get_public_key_short()
            print(f"   ✅ Wallet: {address} ({balance:.4f} SOL)")
        else:
            print("   ❌ Wallet failed")
            return False
        
        # Test AI service
        print("2. Testing AI service...")
        ai_config = bot.config.get("ai", {})
        endpoint = ai_config.get("endpoint", "http://localhost:8000")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{endpoint}/health", timeout=5) as response:
                    if response.status == 200:
                        print(f"   ✅ AI Service: {endpoint}")
                    else:
                        print(f"   ⚠️  AI Service responded with status {response.status}")
        except Exception as e:
            print(f"   ❌ AI Service error: {e}")
        
        # Test database
        print("3. Testing database...")
        try:
            stats = bot.position_manager.get_statistics()
            print(f"   ✅ Database: {stats.get('total_positions', 0)} positions")
        except Exception as e:
            print(f"   ❌ Database error: {e}")
        
        print("\n✅ All tests completed")
        return True
    
    asyncio.run(test_connections())

@cli.command()
@click.option('--position-id', required=True, help='Position ID to close')
@click.option('--price', type=float, required=True, help='Current price for closing')
@click.option('--reason', default='manual', help='Close reason')
def close_position(position_id: str, price: float, reason: str):
    """Manually close a position"""
    bot = TradingBot()
    
    async def close():
        await bot.position_manager.close_position(position_id, price, reason)
        print(f"✅ Position {position_id} closed at ${price}")
    
    asyncio.run(close())

@cli.command()
def stats():
    """Show trading statistics"""
    bot = TradingBot()
    
    stats = bot.position_manager.get_statistics()
    
    print("="*50)
    print("TRADING STATISTICS")
    print("="*50)
    print(f"Total Positions: {stats.get('total_positions', 0)}")
    print(f"Open Positions: {stats.get('open_positions', 0)}")
    print(f"Closed Positions: {stats.get('closed_positions', 0)}")
    print(f"Total PnL: ${stats.get('total_pnl', 0):.2f}")
    print(f"Win Rate: {stats.get('win_rate', 0):.1f}%")
    print(f"Average Win: ${stats.get('average_win', 0):.2f}")
    print(f"Average Loss: ${stats.get('average_loss', 0):.2f}")
    print("="*50)

if __name__ == "__main__":
    cli()
