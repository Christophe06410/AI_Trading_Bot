#!/usr/bin/env python3
"""
Solana Trading Bot - Production Version
Main entry point
"""

import asyncio
import signal
import sys
import logging
from datetime import datetime
import click
from pathlib import Path

from src.config import TradingBotConfig
from src.trading_engine import TradingEngine
from src.wallet_manager import WalletManager
from src.monitoring import MonitoringSystem
from src.utils import setup_logging, get_logger

logger = get_logger(__name__)


class TradingBot:
    """Main Trading Bot class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize trading bot"""
        self.config = TradingBotConfig.load(config_path)
        
        # Setup logging
        setup_logging(self.config.logging.dict())
        self.logger = get_logger("trading_bot")
        
        # Initialize components
        self.monitoring = MonitoringSystem(self.config)
        self.wallet_manager = WalletManager(self.config)
        self.trading_engine = TradingEngine(self.config)
        
        # State
        self.running = False
        self.start_time = None
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Trading Bot initialized")
    
    async def initialize(self):
        """Initialize all components"""
        self.logger.info("Starting initialization...")
        
        try:
            # Start monitoring
            await self.monitoring.start()
            
            # Initialize wallet
            if not await self.wallet_manager.initialize():
                raise RuntimeError("Wallet initialization failed")
            
            # Initialize trading engine
            await self.trading_engine.initialize(self.wallet_manager)
            
            # Health check
            await self._health_check()
            
            self.start_time = datetime.now()
            self.logger.info("Initialization complete")
            
        except Exception as e:
            self.logger.error("Initialization failed", error=str(e))
            await self.shutdown()
            raise
    
    async def _health_check(self):
        """Perform health checks"""
        self.logger.info("Performing health checks...")
        
        health_checks = [
            ("wallet", self.wallet_manager.check_connection),
            ("solana_rpc", self.trading_engine.check_rpc_connection),
            ("ai_service", self.trading_engine.check_ai_service),
            ("dex", self.trading_engine.check_dex_connection),
        ]
        
        all_healthy = True
        
        for name, check_func in health_checks:
            try:
                result = await check_func()
                if result:
                    self.logger.debug(f"Health check passed: {name}")
                else:
                    self.logger.error(f"Health check failed: {name}")
                    all_healthy = False
            except Exception as e:
                self.logger.error(f"Health check error for {name}", error=str(e))
                all_healthy = False
        
        if not all_healthy:
            raise RuntimeError("Health checks failed")
        
        self.logger.info("All health checks passed")
    
    async def run(self):
        """Main trading loop"""
        if not self.running:
            await self.initialize()
        
        self.running = True
        self.logger.info("Starting trading engine...")
        
        try:
            # Main trading loop
            while self.running:
                try:
                    # Execute trading cycle
                    await self.trading_engine.trading_cycle()
                    
                    # Update metrics
                    await self.monitoring.update_metrics(
                        self.trading_engine.get_metrics()
                    )
                    
                    # Sleep between cycles
                    await asyncio.sleep(self.config.trading.check_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error("Error in trading cycle", error=str(e))
                    await asyncio.sleep(5)
            
        except KeyboardInterrupt:
            self.logger.info("Trading stopped by user")
        except Exception as e:
            self.logger.error("Fatal error in trading loop", error=str(e))
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Initiating shutdown...")
        
        try:
            # Close positions if configured
            if self.config.trading.close_on_shutdown:
                self.logger.info("Closing all positions...")
                await self.trading_engine.close_all_positions()
            
            # Stop trading engine
            await self.trading_engine.shutdown()
            
            # Stop monitoring
            await self.monitoring.stop()
            
            # Log runtime
            if self.start_time:
                runtime = datetime.now() - self.start_time
                self.logger.info(
                    "Shutdown complete",
                    runtime=str(runtime),
                    trades=self.trading_engine.get_metrics()['total_trades']
                )
            
        except Exception as e:
            self.logger.error("Error during shutdown", error=str(e))
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}")
        asyncio.create_task(self.shutdown())
    
    async def test_mode(self):
        """Run in test mode"""
        self.logger.info("Running in test mode...")
        
        await self.initialize()
        
        # Run a few test cycles
        for i in range(3):
            self.logger.info(f"Test cycle {i + 1}/3")
            await self.trading_engine.test_cycle()
            await asyncio.sleep(2)
        
        await self.shutdown()


@click.group()
def cli():
    """Solana Trading Bot CLI"""
    pass


@cli.command()
@click.option('--config', default='config/config.yaml', help='Configuration file')
@click.option('--test', is_flag=True, help='Run in test mode')
def start(config: str, test: bool):
    """Start the trading bot"""
    
    # Check config file exists
    if not Path(config).exists():
        click.echo(f"❌ Config file not found: {config}")
        click.echo("Create config/config.yaml first")
        return
    
    bot = TradingBot(config)
    
    if test:
        asyncio.run(bot.test_mode())
    else:
        asyncio.run(bot.run())


@cli.command()
@click.option('--config', default='config/config.yaml', help='Configuration file')
def status(config: str):
    """Check bot status"""
    bot = TradingBot(config)
    
    try:
        # Quick status check
        asyncio.run(bot.initialize())
        click.echo("✅ Bot components initialized successfully")
        click.echo(f"   Trading Mode: {bot.config.trading_mode}")
        click.echo(f"   Wallet: {bot.config.wallet.address[:10]}...")
        click.echo(f"   Pairs: {', '.join(bot.config.trading.pairs)}")
        click.echo(f"   AI Service: {bot.config.get_full_ai_endpoint()}")
        
    except Exception as e:
        click.echo(f"❌ Status check failed: {e}")
    finally:
        asyncio.run(bot.shutdown())


@cli.command()
@click.option('--config', default='config/config.yaml', help='Configuration file')
@click.option('--force', is_flag=True, help='Force close all positions')
def stop(config: str, force: bool):
    """Stop the trading bot"""
    bot = TradingBot(config)
    
    if force:
        click.confirm('Are you sure you want to force close all positions?', abort=True)
        asyncio.run(bot.trading_engine.close_all_positions())
    
    asyncio.run(bot.shutdown())
    click.echo("✅ Bot stopped")


if __name__ == "__main__":
    cli()