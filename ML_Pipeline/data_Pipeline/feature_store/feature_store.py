"""
Enterprise Feature Store for ML Pipeline
Stores, version, and serves features for training and inference
"""

import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import redis.asyncio as redis
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features"""
    REALTIME = "realtime"
    HISTORICAL = "historical"
    ALTERNATIVE = "alternative"
    DERIVED = "derived"


class StorageBackend(Enum):
    """Storage backends for features"""
    REDIS = "redis"
    PARQUET = "parquet"
    POSTGRES = "postgres"
    MEMORY = "memory"


@dataclass
class FeatureMetadata:
    """Metadata for a feature"""
    name: str
    feature_type: FeatureType
    data_type: str
    description: str
    created_at: str
    version: str = "1.0.0"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    source: str = "unknown"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class FeatureStore:
    """Enterprise feature store for ML pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_backend = StorageBackend(config.get("storage_backend", "redis"))
        self.feature_registry: Dict[str, FeatureMetadata] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.data_directory = Path(config.get("data_directory", "data/feature_store"))
        
        # Statistics
        self.stats = {
            "features_registered": 0,
            "features_served": 0,
            "storage_usage_bytes": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Ensure data directory exists
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Feature store initialized with {self.storage_backend.value} backend")
    
    async def initialize(self):
        """Initialize feature store connections"""
        if self.storage_backend == StorageBackend.REDIS:
            await self._initialize_redis()
        
        # Load existing feature registry
        await self._load_feature_registry()
        
        logger.info(f"Feature store ready. {len(self.feature_registry)} features registered")
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        redis_url = self.config.get("redis_url", "redis://localhost:6379/0")
        self.redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=False)
        
        try:
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Fall back to memory storage
            self.storage_backend = StorageBackend.MEMORY
    
    async def _load_feature_registry(self):
        """Load feature registry from storage"""
        try:
            if self.storage_backend == StorageBackend.REDIS and self.redis_client:
                registry_data = await self.redis_client.get("feature_registry")
                if registry_data:
                    registry_dict = json.loads(registry_data)
                    for name, meta_dict in registry_dict.items():
                        self.feature_registry[name] = FeatureMetadata(**meta_dict)
            
            # Also try to load from file
            registry_file = self.data_directory / "registry.json"
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    registry_dict = json.load(f)
                    for name, meta_dict in registry_dict.items():
                        if name not in self.feature_registry:
                            self.feature_registry[name] = FeatureMetadata(**meta_dict)
            
            logger.info(f"Loaded {len(self.feature_registry)} features from registry")
            
        except Exception as e:
            logger.error(f"Failed to load feature registry: {e}")
    
    async def _save_feature_registry(self):
        """Save feature registry to storage"""
        try:
            registry_dict = {
                name: asdict(metadata)
                for name, metadata in self.feature_registry.items()
            }
            
            # Save to Redis
            if self.storage_backend == StorageBackend.REDIS and self.redis_client:
                await self.redis_client.set(
                    "feature_registry",
                    json.dumps(registry_dict)
                )
            
            # Save to file
            registry_file = self.data_directory / "registry.json"
            with open(registry_file, 'w') as f:
                json.dump(registry_dict, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save feature registry: {e}")
    
    async def register_feature(self, name: str, feature_type: FeatureType,
                              data_type: str, description: str, 
                              source: str = "unknown", tags: List[str] = None) -> bool:
        """Register a new feature in the store"""
        try:
            # Check if feature already exists
            if name in self.feature_registry:
                logger.warning(f"Feature '{name}' already registered")
                return False
            
            # Create metadata
            metadata = FeatureMetadata(
                name=name,
                feature_type=feature_type,
                data_type=data_type,
                description=description,
                created_at=datetime.now().isoformat(),
                source=source,
                tags=tags or []
            )
            
            # Add to registry
            self.feature_registry[name] = metadata
            self.stats["features_registered"] += 1
            
            # Save registry
            await self._save_feature_registry()
            
            logger.info(f"Registered feature '{name}' ({feature_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register feature '{name}': {e}")
            return False
    
    async def store_features(self, features: Dict[str, Any], 
                           symbol: str, timestamp: str = None) -> bool:
        """Store features for a specific symbol and timestamp"""
        if not timestamp:
            timestamp = datetime.now().isoformat()
        
        try:
            # Create feature set ID
            feature_set_id = self._create_feature_set_id(symbol, timestamp)
            
            # Prepare feature data
            feature_data = {
                "symbol": symbol,
                "timestamp": timestamp,
                "features": features,
                "metadata": {
                    "feature_count": len(features),
                    "storage_time": datetime.now().isoformat()
                }
            }
            
            # Store based on backend
            if self.storage_backend == StorageBackend.REDIS:
                await self._store_in_redis(feature_set_id, feature_data)
            elif self.storage_backend == StorageBackend.PARQUET:
                await self._store_in_parquet(feature_set_id, feature_data)
            else:
                await self._store_in_memory(feature_set_id, feature_data)
            
            # Update statistics for features
            for feature_name in features.keys():
                if feature_name in self.feature_registry:
                    await self._update_feature_stats(feature_name, features[feature_name])
            
            logger.debug(f"Stored {len(features)} features for {symbol} at {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store features for {symbol}: {e}")
            return False
    
    async def _store_in_redis(self, feature_set_id: str, feature_data: Dict[str, Any]):
        """Store features in Redis"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        
        # Store with TTL (default 7 days)
        ttl = self.config.get("redis_ttl_seconds", 604800)
        
        await self.redis_client.setex(
            f"features:{feature_set_id}",
            ttl,
            pickle.dumps(feature_data)
        )
    
    async def _store_in_parquet(self, feature_set_id: str, feature_data: Dict[str, Any]):
        """Store features in Parquet files"""
        # Convert to DataFrame
        features_flat = {
            "symbol": feature_data["symbol"],
            "timestamp": feature_data["timestamp"],
            **feature_data["features"]
        }
        
        df = pd.DataFrame([features_flat])
        
        # Create directory structure: data/feature_store/{symbol}/{date}/
        date_str = feature_data["timestamp"][:10]  # YYYY-MM-DD
        symbol_dir = self.data_directory / feature_data["symbol"]
        date_dir = symbol_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # File path
        file_path = date_dir / f"{feature_set_id}.parquet"
        
        # Write to Parquet
        df.to_parquet(file_path, index=False)
        
        # Update storage usage
        self.stats["storage_usage_bytes"] += file_path.stat().st_size
    
    async def _store_in_memory(self, feature_set_id: str, feature_data: Dict[str, Any]):
        """Store features in memory (fallback)"""
        # Simple in-memory storage for development
        if not hasattr(self, '_memory_store'):
            self._memory_store = {}
        
        self._memory_store[feature_set_id] = feature_data
    
    async def get_features(self, symbol: str, timestamp: str = None, 
                          lookback_minutes: int = 0) -> Dict[str, Any]:
        """Get features for a symbol"""
        try:
            if timestamp is None:
                # Get most recent features
                feature_set_id = await self._find_latest_features(symbol)
            else:
                feature_set_id = self._create_feature_set_id(symbol, timestamp)
            
            # Try to get from cache/storage
            feature_data = await self._retrieve_features(feature_set_id)
            
            if feature_data:
                self.stats["cache_hits"] += 1
                
                # If lookback requested, get historical features
                if lookback_minutes > 0:
                    historical_features = await self._get_historical_features(
                        symbol, timestamp, lookback_minutes
                    )
                    feature_data["historical"] = historical_features
                
                return feature_data
            else:
                self.stats["cache_misses"] += 1
                logger.warning(f"Features not found for {symbol} at {timestamp}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get features for {symbol}: {e}")
            return {}
    
    async def _retrieve_features(self, feature_set_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve features from storage"""
        if self.storage_backend == StorageBackend.REDIS and self.redis_client:
            data = await self.redis_client.get(f"features:{feature_set_id}")
            if data:
                return pickle.loads(data)
        
        elif self.storage_backend == StorageBackend.PARQUET:
            # Parse symbol and date from ID
            parts = feature_set_id.split(":")
            if len(parts) >= 3:
                symbol = parts[0]
                date_str = parts[1][:10]
                
                file_path = self.data_directory / symbol / date_str / f"{feature_set_id}.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    if not df.empty:
                        features = df.iloc[0].to_dict()
                        return {
                            "symbol": symbol,
                            "timestamp": features.pop("timestamp"),
                            "features": features
                        }
        
        elif hasattr(self, '_memory_store'):
            return self._memory_store.get(feature_set_id)
        
        return None
    
    async def _find_latest_features(self, symbol: str) -> Optional[str]:
        """Find the latest feature set for a symbol"""
        # This is a simplified implementation
        # In production, you'd query a timestamp index
        
        if self.storage_backend == StorageBackend.REDIS and self.redis_client:
            # Use Redis keys command (not efficient for production)
            pattern = f"features:{symbol}:*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                # Sort by timestamp in key
                keys.sort(reverse=True)
                return keys[0].decode('utf-8').replace("features:", "")
        
        return None
    
    async def _get_historical_features(self, symbol: str, timestamp: str, 
                                      lookback_minutes: int) -> List[Dict[str, Any]]:
        """Get historical features for lookback period"""
        historical = []
        
        try:
            # Convert timestamp to datetime
            current_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Calculate time points for lookback
            time_points = []
            for i in range(lookback_minutes):
                time_point = current_time - timedelta(minutes=i)
                time_points.append(time_point.isoformat())
            
            # Get features for each time point
            for time_point in time_points:
                feature_set_id = self._create_feature_set_id(symbol, time_point)
                data = await self._retrieve_features(feature_set_id)
                
                if data:
                    historical.append({
                        "timestamp": time_point,
                        "features": data.get("features", {})
                    })
            
        except Exception as e:
            logger.error(f"Failed to get historical features: {e}")
        
        return historical
    
    async def _update_feature_stats(self, feature_name: str, value: Any):
        """Update statistics for a feature"""
        if feature_name not in self.feature_registry:
            return
        
        metadata = self.feature_registry[feature_name]
        
        try:
            # Convert to float if possible
            if isinstance(value, (int, float, np.number)):
                num_value = float(value)
                
                # Update min/max
                if metadata.min_value is None or num_value < metadata.min_value:
                    metadata.min_value = num_value
                
                if metadata.max_value is None or num_value > metadata.max_value:
                    metadata.max_value = num_value
                
                # Update mean and std (simplified moving average)
                if metadata.mean_value is None:
                    metadata.mean_value = num_value
                    metadata.std_value = 0
                else:
                    # Simple online update (Welford's algorithm simplified)
                    old_mean = metadata.mean_value
                    metadata.mean_value = old_mean + (num_value - old_mean) / 100
                    
                    if metadata.std_value is not None:
                        old_std = metadata.std_value
                        new_std = np.sqrt(
                            old_std**2 + (num_value - old_mean) * (num_value - metadata.mean_value) / 99
                        )
                        metadata.std_value = new_std
            
            # Save updated metadata
            self.feature_registry[feature_name] = metadata
            
        except Exception as e:
            logger.debug(f"Failed to update stats for {feature_name}: {e}")
    
    def _create_feature_set_id(self, symbol: str, timestamp: str) -> str:
        """Create unique ID for feature set"""
        # Normalize timestamp
        normalized_timestamp = timestamp.replace(':', '_').replace('.', '_')
        return f"{symbol}:{normalized_timestamp}"
    
    async def get_feature_metadata(self, feature_name: str = None) -> Dict[str, Any]:
        """Get metadata for features"""
        if feature_name:
            if feature_name in self.feature_registry:
                return asdict(self.feature_registry[feature_name])
            else:
                return {}
        else:
            return {
                name: asdict(metadata)
                for name, metadata in self.feature_registry.items()
            }
    
    async def search_features(self, query: str, 
                             feature_type: FeatureType = None) -> List[str]:
        """Search for features by name, description, or tags"""
        results = []
        query_lower = query.lower()
        
        for name, metadata in self.feature_registry.items():
            # Check name
            if query_lower in name.lower():
                results.append(name)
                continue
            
            # Check description
            if query_lower in metadata.description.lower():
                results.append(name)
                continue
            
            # Check tags
            for tag in metadata.tags:
                if query_lower in tag.lower():
                    results.append(name)
                    break
            
            # Check feature type
            if feature_type and metadata.feature_type == feature_type:
                results.append(name)
        
        return list(set(results))  # Remove duplicates
    
    async def export_features(self, symbol: str, start_time: str, 
                             end_time: str, output_format: str = "parquet") -> Optional[Path]:
        """Export features for a time range"""
        try:
            # Convert times to datetime
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
            # Collect features
            all_features = []
            
            current_dt = start_dt
            while current_dt <= end_dt:
                timestamp = current_dt.isoformat()
                feature_set_id = self._create_feature_set_id(symbol, timestamp)
                
                data = await self._retrieve_features(feature_set_id)
                if data:
                    all_features.append({
                        "timestamp": timestamp,
                        **data.get("features", {})
                    })
                
                # Move to next minute
                current_dt += timedelta(minutes=1)
            
            if not all_features:
                logger.warning(f"No features found for {symbol} in time range")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(all_features)
            
            # Export based on format
            export_dir = self.data_directory / "exports"
            export_dir.mkdir(exist_ok=True)
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timestamp_str}"
            
            if output_format == "parquet":
                file_path = export_dir / f"{filename}.parquet"
                df.to_parquet(file_path, index=False)
            elif output_format == "csv":
                file_path = export_dir / f"{filename}.csv"
                df.to_csv(file_path, index=False)
            elif output_format == "json":
                file_path = export_dir / f"{filename}.json"
                df.to_json(file_path, orient="records", indent=2)
            else:
                logger.error(f"Unsupported output format: {output_format}")
                return None
            
            logger.info(f"Exported {len(df)} features to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to export features: {e}")
            return None
    
    async def cleanup_old_features(self, days_to_keep: int = 30):
        """Clean up old features"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            
            if self.storage_backend == StorageBackend.REDIS and self.redis_client:
                # Redis handles TTL automatically
                logger.info("Redis TTL handles old feature cleanup")
            
            elif self.storage_backend == StorageBackend.PARQUET:
                # Clean up old Parquet files
                deleted_count = 0
                
                for symbol_dir in self.data_directory.iterdir():
                    if symbol_dir.is_dir():
                        for date_dir in symbol_dir.iterdir():
                            if date_dir.is_dir():
                                # Parse date from directory name
                                try:
                                    dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                                    if dir_date < cutoff_time:
                                        # Delete directory and contents
                                        import shutil
                                        shutil.rmtree(date_dir)
                                        deleted_count += 1
                                except ValueError:
                                    pass
                
                logger.info(f"Cleaned up {deleted_count} old feature directories")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old features: {e}")
    
    async def close(self):
        """Close feature store connections"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Feature store connections closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feature store statistics"""
        return {
            **self.stats,
            "features_registered": len(self.feature_registry),
            "storage_backend": self.storage_backend.value
        }
