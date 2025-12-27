"""
ENTERPRISE MODEL REGISTRY v1.0
Maintains 100% backward compatibility with AI-Service
"""
# ML-Pipeline/enterprise/model_registry.py

import json
import yaml
import pickle
import joblib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model status in registry"""
    EXPERIMENTAL = "experimental"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

class DeploymentStrategy(Enum):
    """Deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ALL_AT_ONCE = "all_at_once"

@dataclass
class ModelMetadata:
    """Complete model metadata"""
    model_id: str
    version: str
    symbol: str
    model_type: str  # "ensemble", "rl", "xgboost", etc.
    status: ModelStatus
    created_at: datetime
    trained_at: datetime
    performance_metrics: Dict[str, float]
    training_data: Dict[str, Any]
    features_used: List[str]
    hyperparameters: Dict[str, Any]
    dependencies: Dict[str, str]
    author: str = "ml_pipeline"
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.trained_at, str):
            self.trained_at = datetime.fromisoformat(self.trained_at)
        if isinstance(self.status, str):
            self.status = ModelStatus(self.status)

@dataclass
class ExperimentConfig:
    """A/B experiment configuration"""
    experiment_id: str
    model_a_id: str
    model_b_id: str
    traffic_split: float  # 0.0 to 1.0 for model_a
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if isinstance(self.start_time, str):
            self.start_time = datetime.fromisoformat(self.start_time)
        if self.end_time and isinstance(self.end_time, str):
            self.end_time = datetime.fromisoformat(self.end_time)

class ModelRegistry:
    """
    Enterprise Model Registry with backward compatibility
    
    Key Features:
    1. Version control for models
    2. A/B testing support
    3. Blue-green deployment
    4. Full backward compatibility with AI-Service
    5. Automated symlinking for legacy paths
    """
    
    def __init__(self, base_path: str = "registry", config_path: str = "config/enterprise_config.yaml"):
        self.base_path = Path(base_path)
        self.config = self._load_config(config_path)
        
        # Ensure directory structure exists
        self._ensure_directories()
        
        # Load existing registry
        self.registry_file = self.base_path / "registry.json"
        self.registry = self._load_registry()
        
        # Backward compatibility setup
        self.legacy_model_path = Path(self.config['backward_compatibility']['model_path'])
        self._setup_backward_compatibility()
        
        logger.info(f"Model Registry initialized at {self.base_path}")
        logger.info(f"Backward compatibility: {self.config['backward_compatibility']['enabled']}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load enterprise configuration"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'backward_compatibility': {
                    'enabled': True,
                    'model_path': 'models/',
                    'symlink_models': True
                }
            }
    
    def _ensure_directories(self):
        """Ensure all registry directories exist"""
        directories = [
            self.base_path / "production",
            self.base_path / "experiments",
            self.base_path / "archived",
            self.base_path / "metadata"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from JSON file"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'models': {},
                'experiments': {},
                'deployments': {},
                'version': '1.0.0',
                'created_at': datetime.now().isoformat()
            }
    
    def _save_registry(self):
        """Save registry to JSON file"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def _setup_backward_compatibility(self):
        """Setup backward compatibility with AI-Service"""
        if not self.config['backward_compatibility']['enabled']:
            return
        
        # Ensure legacy model directory exists
        self.legacy_model_path.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks for existing production models
        self._create_backward_compatibility_links()
    
    def _create_backward_compatibility_links(self):
        """Create symlinks from legacy location to current production models"""
        production_path = self.base_path / "production"
        
        if not production_path.exists():
            return
        
        for symbol_dir in production_path.iterdir():
            if symbol_dir.is_dir():
                current_link = symbol_dir / "current"
                if current_link.exists() and current_link.is_symlink():
                    # Get actual model file
                    actual_model = current_link.resolve()
                    
                    if actual_model.exists():
                        # Create legacy symlink
                        symbol_name = symbol_dir.name.replace('_', '/')
                        legacy_filename = f"ensemble_model_{symbol_name.replace('/', '_')}.joblib"
                        legacy_path = self.legacy_model_path / legacy_filename
                        
                        try:
                            if legacy_path.exists():
                                legacy_path.unlink()
                            
                            legacy_path.symlink_to(actual_model / "model.joblib")
                            logger.info(f"Created backward compatibility link: {legacy_path} -> {actual_model}")
                        except Exception as e:
                            logger.warning(f"Failed to create symlink for {symbol_name}: {e}")
    
    def register_model(self, model, metadata: ModelMetadata, model_path: Optional[Path] = None):
        """
        Register a new model in the registry
        
        Args:
            model: Trained model object
            metadata: Complete model metadata
            model_path: Optional custom path to save model
        
        Returns:
            str: Model ID
        """
        try:
            # Generate unique model ID if not provided
            if not metadata.model_id:
                metadata.model_id = self._generate_model_id(metadata)
            
            # Create version directory
            version_path = self._get_version_path(metadata)
            version_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_filename = self._save_model_file(model, version_path)
            
            # Save metadata
            metadata_path = version_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            # Update registry
            if metadata.symbol not in self.registry['models']:
                self.registry['models'][metadata.symbol] = {}
            
            self.registry['models'][metadata.symbol][metadata.version] = {
                'model_id': metadata.model_id,
                'path': str(version_path.relative_to(self.base_path)),
                'status': metadata.status.value,
                'performance': metadata.performance_metrics,
                'created_at': metadata.created_at.isoformat()
            }
            
            self._save_registry()
            
            # Setup backward compatibility if this is production model
            if metadata.status == ModelStatus.PRODUCTION:
                self._promote_to_production(metadata, version_path)
            
            logger.info(f"Registered model {metadata.model_id} v{metadata.version} for {metadata.symbol}")
            return metadata.model_id
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def _generate_model_id(self, metadata: ModelMetadata) -> str:
        """Generate unique model ID"""
        content = f"{metadata.symbol}_{metadata.model_type}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _get_version_path(self, metadata: ModelMetadata) -> Path:
        """Get path for model version"""
        symbol_safe = metadata.symbol.replace('/', '_').lower()
        
        if metadata.status == ModelStatus.PRODUCTION:
            base_dir = self.base_path / "production" / symbol_safe / f"v{metadata.version}"
        elif metadata.status == ModelStatus.EXPERIMENTAL:
            base_dir = self.base_path / "experiments" / symbol_safe / metadata.model_id
        else:
            base_dir = self.base_path / "archived" / symbol_safe / metadata.model_id
        
        return base_dir
    
    def _save_model_file(self, model, version_path: Path) -> str:
        """Save model to file based on type"""
        import xgboost as xgb
        
        # Determine model type and save appropriately
        if isinstance(model, xgb.Booster):
            filename = "model.json"
            model.save_model(str(version_path / filename))
        elif hasattr(model, 'save'):
            # Keras/TensorFlow model
            filename = "model.h5"
            model.save(str(version_path / filename))
        else:
            # Scikit-learn style model
            filename = "model.joblib"
            joblib.dump(model, version_path / filename)
        
        return filename
    
    def _promote_to_production(self, metadata: ModelMetadata, version_path: Path):
        """Promote model to production and setup backward compatibility"""
        symbol_safe = metadata.symbol.replace('/', '_').lower()
        production_dir = self.base_path / "production" / symbol_safe
        
        # Update current symlink
        current_link = production_dir / "current"
        if current_link.exists():
            current_link.unlink()
        
        current_link.symlink_to(version_path, target_is_directory=True)
        
        # Update backward compatibility symlinks
        if self.config['backward_compatibility']['symlink_models']:
            legacy_filename = f"ensemble_model_{symbol_safe}.joblib"
            legacy_path = self.legacy_model_path / legacy_filename
            
            try:
                if legacy_path.exists():
                    legacy_path.unlink()
                
                # Link to the actual model file inside version directory
                model_file = version_path / "model.joblib"
                if model_file.exists():
                    legacy_path.symlink_to(model_file)
                    logger.info(f"Updated backward compatibility link: {legacy_path}")
            except Exception as e:
                logger.warning(f"Failed to update backward compatibility link: {e}")
    
    def get_production_model(self, symbol: str) -> Tuple[Any, ModelMetadata]:
        """
        Get current production model for a symbol
        
        Args:
            symbol: Trading symbol (e.g., "SOL/USDT")
            
        Returns:
            Tuple of (model, metadata)
        """
        symbol_safe = symbol.replace('/', '_').lower()
        production_dir = self.base_path / "production" / symbol_safe
        
        # Check current symlink
        current_link = production_dir / "current"
        if not current_link.exists():
            # Try legacy path for backward compatibility
            legacy_path = self.legacy_model_path / f"ensemble_model_{symbol_safe}.joblib"
            if legacy_path.exists():
                model = joblib.load(legacy_path)
                # Create minimal metadata
                metadata = ModelMetadata(
                    model_id="legacy",
                    version="1.0.0",
                    symbol=symbol,
                    model_type="ensemble",
                    status=ModelStatus.PRODUCTION,
                    created_at=datetime.now(),
                    trained_at=datetime.now(),
                    performance_metrics={},
                    training_data={},
                    features_used=[],
                    hyperparameters={},
                    dependencies={}
                )
                return model, metadata
            
            raise FileNotFoundError(f"No production model found for {symbol}")
        
        # Load from current version
        version_path = current_link.resolve()
        model = joblib.load(version_path / "model.joblib")
        
        # Load metadata
        metadata_path = version_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        metadata = ModelMetadata(**metadata_dict)
        
        return model, metadata
    
    def list_models(self, symbol: Optional[str] = None, status: Optional[ModelStatus] = None) -> List[Dict[str, Any]]:
        """List all models in registry with optional filters"""
        models_list = []
        
        for sym, versions in self.registry['models'].items():
            if symbol and sym != symbol:
                continue
            
            for version, model_info in versions.items():
                if status and ModelStatus(model_info['status']) != status:
                    continue
                
                models_list.append({
                    'symbol': sym,
                    'version': version,
                    'model_id': model_info['model_id'],
                    'status': model_info['status'],
                    'created_at': model_info['created_at'],
                    'performance': model_info.get('performance', {})
                })
        
        return sorted(models_list, key=lambda x: x['created_at'], reverse=True)
    
    def create_experiment(self, experiment_config: ExperimentConfig) -> str:
        """Create A/B experiment"""
        experiment_id = experiment_config.experiment_id
        
        # Verify both models exist
        model_a = self._find_model_by_id(experiment_config.model_a_id)
        model_b = self._find_model_by_id(experiment_config.model_b_id)
        
        if not model_a or not model_b:
            raise ValueError("One or both models not found")
        
        # Store experiment configuration
        self.registry['experiments'][experiment_id] = asdict(experiment_config)
        self._save_registry()
        
        logger.info(f"Created experiment {experiment_id}: {model_a['symbol']} v{model_a['version']} vs {model_b['symbol']} v{model_b['version']}")
        
        return experiment_id
    
    def _find_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Find model by ID in registry"""
        for symbol, versions in self.registry['models'].items():
            for version, model_info in versions.items():
                if model_info['model_id'] == model_id:
                    return {
                        'symbol': symbol,
                        'version': version,
                        **model_info
                    }
        return None
    
    def deploy_model(self, model_id: str, strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN) -> str:
        """Deploy model using specified strategy"""
        model_info = self._find_model_by_id(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")
        
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        deployment_info = {
            'deployment_id': deployment_id,
            'model_id': model_id,
            'strategy': strategy.value,
            'start_time': datetime.now().isoformat(),
            'status': 'in_progress',
            'symbol': model_info['symbol']
        }
        
        self.registry['deployments'][deployment_id] = deployment_info
        self._save_registry()
        
        # Update model status to staging
        symbol = model_info['symbol']
        version = model_info['version']
        
        if symbol in self.registry['models'] and version in self.registry['models'][symbol]:
            self.registry['models'][symbol][version]['status'] = ModelStatus.STAGING.value
        
        logger.info(f"Started deployment {deployment_id} for model {model_id} using {strategy.value}")
        
        return deployment_id
    
    def archive_model(self, model_id: str, reason: str = "") -> bool:
        """Archive a model"""
        model_info = self._find_model_by_id(model_id)
        if not model_info:
            return False
        
        symbol = model_info['symbol']
        version = model_info['version']
        
        # Update status
        if symbol in self.registry['models'] and version in self.registry['models'][symbol]:
            self.registry['models'][symbol][version]['status'] = ModelStatus.ARCHIVED.value
            
            # Move to archived directory
            old_path = self.base_path / model_info['path']
            symbol_safe = symbol.replace('/', '_').lower()
            new_path = self.base_path / "archived" / symbol_safe / model_id
            
            if old_path.exists():
                shutil.move(old_path, new_path)
                
                # Update path in registry
                self.registry['models'][symbol][version]['path'] = str(new_path.relative_to(self.base_path))
            
            self._save_registry()
            logger.info(f"Archived model {model_id}: {reason}")
            return True
        
        return False
    
    def get_model_performance_history(self, symbol: str) -> pd.DataFrame:
        """Get performance history for a symbol"""
        if symbol not in self.registry['models']:
            return pd.DataFrame()
        
        performance_data = []
        for version, model_info in self.registry['models'][symbol].items():
            if 'performance' in model_info:
                row = {
                    'version': version,
                    'created_at': pd.to_datetime(model_info['created_at']),
                    'status': model_info['status'],
                    **model_info['performance']
                }
                performance_data.append(row)
        
        return pd.DataFrame(performance_data).sort_values('created_at')
    
    def cleanup_old_versions(self, keep_last_n: int = 5, older_than_days: int = 30):
        """Cleanup old model versions"""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        for symbol, versions in self.registry['models'].items():
            # Sort versions by date
            sorted_versions = sorted(
                versions.items(),
                key=lambda x: x[1]['created_at'],
                reverse=True
            )
            
            for i, (version, model_info) in enumerate(sorted_versions):
                created_at = datetime.fromisoformat(model_info['created_at'])
                status = ModelStatus(model_info['status'])
                
                # Skip if production or recent
                if status == ModelStatus.PRODUCTION:
                    continue
                
                if i >= keep_last_n and created_at < cutoff_date:
                    # Archive old version
                    self.archive_model(
                        model_info['model_id'],
                        reason=f"Auto-cleanup: older than {older_than_days} days"
                    )
