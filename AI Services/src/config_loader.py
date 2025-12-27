"""
Configuration loader for AI Service
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False

@dataclass
class MLConfig:
    model_path: str = "../ML-Pipeline/models/production/"
    min_confidence: float = 0.70
    models: list = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = [
                {"name": "random_forest", "file": "random_forest_model.joblib"},
                {"name": "xgboost", "file": "xgboost_model.json"}
            ]

@dataclass
class CacheConfig:
    redis_url: str = "redis://localhost:6379/0"
    ttl: int = 3600
    enabled: bool = True

@dataclass
class SecurityConfig:
    api_key_header: str = "X-API-Key"
    require_auth: bool = True
    valid_api_keys: list = None
    
    def __post_init__(self):
        if self.valid_api_keys is None:
            self.valid_api_keys = []

@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/ai_service.log"

@dataclass
class Config:
    server: ServerConfig = None
    ml: MLConfig = None
    cache: CacheConfig = None
    security: SecurityConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        if self.server is None:
            self.server = ServerConfig()
        if self.ml is None:
            self.ml = MLConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
    
    def load(self, config_path: str = "config/config.yaml") -> 'Config':
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update configuration from file
            if 'server' in config_data:
                for key, value in config_data['server'].items():
                    setattr(self.server, key, value)
            
            if 'ml' in config_data:
                for key, value in config_data['ml'].items():
                    setattr(self.ml, key, value)
            
            if 'cache' in config_data:
                for key, value in config_data['cache'].items():
                    setattr(self.cache, key, value)
            
            if 'security' in config_data:
                for key, value in config_data['security'].items():
                    setattr(self.security, key, value)
            
            if 'logging' in config_data:
                for key, value in config_data['logging'].items():
                    setattr(self.logging, key, value)
        
        # Load API keys from environment
        api_keys = os.getenv("API_KEYS", "")
        if api_keys:
            self.security.valid_api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]
        
        return self
