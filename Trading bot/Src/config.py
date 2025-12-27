"""
Configuration management for Trading Bot
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TradingPair:
    """Trading pair configuration"""
    symbol: str
    base: str
    quote: str
    min_size: float
    max_size: float


class StopLossConfig(BaseModel):
    """Stop loss configuration"""
    type: str = "trailing"
    initial_percent: float = 0.01
    activation_percent: float = 0.02
    min_distance: float = 0.005
    
    @validator('initial_percent', 'activation_percent')
    def validate_percent(cls, v):
        if not 0.001 <= v <= 0.1:
            raise ValueError('Stop loss percent must be between 0.1% and 10%')
        return v


class TakeProfitConfig(BaseModel):
    """Take profit configuration"""
    enabled: bool = False
    percent: float = 0.05
    
    @validator('percent')
    def validate_percent(cls, v):
        if not 0.01 <= v <= 0.2:
            raise ValueError('Take profit percent must be between 1% and 20%')
        return v


class TradingConfig(BaseModel):
    """Trading configuration"""
    pairs: List[str] = Field(default_factory=lambda: ["SOL/USDC"])
    max_positions: int = 5
    position_size_usd: float = 50.0
    min_position_size_usd: float = 10.0
    check_interval: int = 10
    market_data_interval: int = 5
    position_check_interval: int = 2
    order_type: str = "market"
    slippage_tolerance: float = 0.01
    time_in_force: str = "IOC"
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)
    pause_first_friday: bool = True
    max_volatility_percent: float = 5.0
    close_on_shutdown: bool = False
    
    @validator('slippage_tolerance')
    def validate_slippage(cls, v):
        if not 0.001 <= v <= 0.1:
            raise ValueError('Slippage tolerance must be between 0.1% and 10%')
        return v


class AIServiceConfig(BaseModel):
    """AI Service configuration"""
    url: str = "http://localhost:8000"
    endpoint: str = "/api/v1/recommendation"
    timeout: int = 10
    retries: int = 3
    min_confidence: float = 0.70
    
    @validator('min_confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0 and 1')
        return v


class DEXConfig(BaseModel):
    """DEX configuration"""
    primary: str = "jupiter"
    jupiter_api_url: str = "https://quote-api.jup.ag/v6"
    slippage_bps: int = 100
    fee_bps: int = 10


class SolanaConfig(BaseModel):
    """Solana network configuration"""
    rpc_endpoint: str = "https://api.devnet.solana.com"
    websocket_endpoint: str = "wss://api.devnet.solana.com"
    commitment: str = "confirmed"
    max_retries: int = 3
    priority_fee: str = "auto"
    dex: DEXConfig = Field(default_factory=DEXConfig)


class WalletConfig(BaseModel):
    """Wallet configuration"""
    address: str = ""
    private_key_storage: str = "env"
    encrypted_file_path: str = "wallet/encrypted_key.bin"
    max_hot_wallet_balance_sol: float = 10.0
    
    @validator('private_key_storage')
    def validate_storage(cls, v):
        if v not in ['env', 'encrypted_file']:
            raise ValueError('Storage must be "env" or "encrypted_file"')
        return v


class RiskConfig(BaseModel):
    """Risk management configuration"""
    max_daily_loss_percent: float = 20.0
    max_position_percent: float = 5.0
    max_consecutive_losses: int = 3
    cooldown_minutes: int = 30


class DatabaseConfig(BaseModel):
    """Database configuration"""
    type: str = "sqlite"
    connection_string: str = "sqlite:///data/trading_bot.db"
    keep_trades_days: int = 90
    keep_logs_days: int = 30


class ConsoleLoggingConfig(BaseModel):
    """Console logging configuration"""
    enabled: bool = True
    level: str = "INFO"
    format: str = "colored"


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/trading_bot.log"
    position_log_file: str = "logs/positions.log"
    error_log_file: str = "logs/errors.log"
    console: ConsoleLoggingConfig = Field(default_factory=ConsoleLoggingConfig)


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    enabled: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30


class PerformanceConfig(BaseModel):
    """Performance configuration"""
    max_concurrent_requests: int = 100
    connection_timeout: int = 10
    cache_ttl: int = 30


class TradingBotConfig(BaseModel):
    """Main trading bot configuration"""
    environment: str = "development"
    trading_mode: str = "testnet"
    trading: TradingConfig = Field(default_factory=TradingConfig)
    ai_service: AIServiceConfig = Field(default_factory=AIServiceConfig)
    solana: SolanaConfig = Field(default_factory=SolanaConfig)
    wallet: WalletConfig = Field(default_factory=WalletConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> 'TradingBotConfig':
        """Load configuration from YAML file"""
        import yaml
        
        # Load default config
        config = cls()
        
        # Load from file if exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config = cls(**file_config)
        
        # Override with environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        env_wallet = os.getenv("WALLET_ADDRESS")
        if env_wallet:
            config.wallet.address = env_wallet
        
        env_rpc = os.getenv("SOLANA_RPC_ENDPOINT")
        if env_rpc:
            config.solana.rpc_endpoint = env_rpc
        
        env_mode = os.getenv("TRADING_MODE")
        if env_mode:
            config.trading_mode = env_mode
        
        return config
    
    def save(self, config_path: str = "config/config.yaml"):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        config_dict = self.dict()
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def get_full_ai_endpoint(self) -> str:
        """Get full AI endpoint URL"""
        return f"{self.ai_service.url}{self.ai_service.endpoint}"
    
    def is_live_trading(self) -> bool:
        """Check if trading mode is live"""
        return self.trading_mode.lower() == "live"
    
    def is_testnet(self) -> bool:
        """Check if using testnet"""
        return self.trading_mode.lower() == "testnet"
    
    def is_paper_trading(self) -> bool:
        """Check if paper trading"""
        return self.trading_mode.lower() == "paper"
