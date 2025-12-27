"""
REGISTRY MANAGER - Core registry operations
Manages model lifecycle from training to deployment
"""

import os
import sys
import json
import yaml
import shutil
import hashlib
import pickle
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Enterprise imports
from enterprise.model_registry import ModelMetadata, ModelVersion, get_model_registry

class RegistryManager:
    """
    Manages model registry operations with audit trail
    """
    
    def __init__(self, registry_root: Path = None):
        self.registry = get_model_registry()
        self.registry_root = self.registry.registry_root
        
        # Create necessary subdirectories
        self._create_registry_structure()
        
        # Audit trail
        self.audit_log = self.registry_root / 'audit_log.jsonl'
        self._setup_audit_trail()
        
        # Performance cache
        self.performance_cache = {}
        
        print(f"📁 Registry Manager initialized at {self.registry_root}")
    
    def _create_registry_structure(self):
        """Create complete registry directory structure"""
        directories = [
            'production',
            'staging', 
            'experiments',
            'archived',
            'tmp',
            'backups',
            'metadata',
            'performance',
            'audit',
            'exports'
        ]
        
        for directory in directories:
            (self.registry_root / directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_audit_trail(self):
        """Setup audit trail logging"""
        if not self.audit_log.exists():
            self.audit_log.touch()
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log event to audit trail"""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'user': os.getenv('USER', 'unknown'),
            'host': os.getenv('HOSTNAME', 'unknown')
        }
        
        with open(self.audit_log, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
    
    def register_trained_model(
        self,
        model: Any,
        symbol: str,
        model_type: str = "ensemble",
        performance_metrics: Dict[str, float] = None,
        feature_names: List[str] = None,
        training_config: Dict[str, Any] = None,
        experiment_id: str = None,
        auto_promote: bool = False
    ) -> ModelVersion:
        """
        Register a newly trained model in the registry
        Includes comprehensive metadata and performance tracking
        """
        print(f"📝 Registering trained {model_type} model for {symbol}...")
        
        # Create metadata
        metadata = ModelMetadata(model_type, symbol)
        
        # Set performance metrics
        if performance_metrics:
            for metric, value in performance_metrics.items():
                if hasattr(metadata, metric):
                    setattr(metadata, metric, value)
        
        # Set feature information
        if feature_names:
            metadata.feature_names = feature_names
            metadata.feature_count = len(feature_names)
        
        # Set training configuration
        if training_config:
            metadata.training_config = training_config
        
        # Set experiment ID if provided
        if experiment_id:
            metadata.experiment_id = experiment_id
        
        # Determine initial status
        if auto_promote:
            metadata.status = "production"
        else:
            metadata.status = "staging"
        
        # Calculate model hash for integrity checking
        model_hash = self._calculate_model_hash(model, metadata)
        metadata.model_hash = model_hash
        
        # Register in registry
        model_version = self.registry.register_model(
            model=model,
            model_type=model_type,
            symbol=symbol,
            metadata=metadata,
            status=metadata.status
        )
        
        # Log audit event
        self._log_audit_event('model_registered', {
            'symbol': symbol,
            'model_type': model_type,
            'version': metadata.version,
            'status': metadata.status,
            'accuracy': metadata.accuracy,
            'auto_promote': auto_promote
        })
        
        # Store performance metrics
        if performance_metrics:
            self._store_performance_metrics(
                symbol, model_type, metadata.version, performance_metrics
            )
        
        # Store training configuration
        if training_config:
            self._store_training_config(
                symbol, model_type, metadata.version, training_config
            )
        
        print(f"✅ Model registered: {symbol}/{model_type} v{metadata.version}")
        print(f"   Status: {metadata.status}")
        print(f"   Accuracy: {metadata.accuracy:.3f}")
        print(f"   Model hash: {model_hash[:16]}...")
        
        return model_version
    
    def _calculate_model_hash(self, model: Any, metadata: ModelMetadata) -> str:
        """Calculate hash of model for integrity checking"""
        import io
        
        try:
            # Serialize model to bytes
            buffer = io.BytesIO()
            
            if metadata.model_format == "joblib":
                joblib.dump(model, buffer)
            elif metadata.model_format == "pickle":
                pickle.dump(model, buffer)
            else:
                # For other formats, use pickle as fallback
                pickle.dump(model, buffer)
            
            buffer.seek(0)
            model_bytes = buffer.read()
            
            # Calculate hash
            model_hash = hashlib.sha256(model_bytes).hexdigest()
            
            # Also hash metadata
            metadata_str = json.dumps(metadata.to_dict(), sort_keys=True)
            metadata_hash = hashlib.sha256(metadata_str.encode()).hexdigest()
            
            # Combined hash
            combined = model_hash + metadata_hash
            final_hash = hashlib.sha256(combined.encode()).hexdigest()
            
            return final_hash
            
        except Exception as e:
            print(f"⚠️  Could not calculate model hash: {e}")
            return "unknown"
    
    def _store_performance_metrics(
        self,
        symbol: str,
        model_type: str,
        version: str,
        metrics: Dict[str, float]
    ):
        """Store performance metrics separately"""
        metrics_file = (
            self.registry_root / 'performance' / 
            f"{symbol}_{model_type}_{version}_metrics.json"
        )
        
        metrics_file.parent.mkdir(exist_ok=True)
        
        metrics_data = {
            'symbol': symbol,
            'model_type': model_type,
            'version': version,
            'metrics': metrics,
            'stored_at': datetime.now().isoformat()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def _store_training_config(
        self,
        symbol: str,
        model_type: str,
        version: str,
        config: Dict[str, Any]
    ):
        """Store training configuration"""
        config_file = (
            self.registry_root / 'metadata' / 
            f"{symbol}_{model_type}_{version}_config.json"
        )
        
        config_file.parent.mkdir(exist_ok=True)
        
        config_data = {
            'symbol': symbol,
            'model_type': model_type,
            'version': version,
            'config': config,
            'stored_at': datetime.now().isoformat()
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def promote_to_production(
        self,
        symbol: str,
        model_type: str = "ensemble",
        version: str = None,
        reason: str = "manual_promotion",
        validation_metrics: Dict[str, float] = None
    ) -> ModelVersion:
        """
        Promote model to production with validation checks
        Includes zero-downtime deployment
        """
        print(f"🚀 Promoting {symbol}/{model_type} to production...")
        
        # Get model to promote
        if version:
            model_version = self._get_model_version(symbol, model_type, version)
        else:
            # Get latest staging model
            staging_models = self._get_models_by_status(symbol, model_type, "staging")
            if not staging_models:
                raise ValueError(f"No staging models found for {symbol}/{model_type}")
            
            # Get most recent
            staging_models.sort(
                key=lambda x: x.metadata.trained_at,
                reverse=True
            )
            model_version = staging_models[0]
            version = model_version.metadata.version
        
        # Validate model before promotion
        if not self._validate_model_for_production(model_version):
            raise ValueError(f"Model validation failed for {symbol}/{model_type} v{version}")
        
        # Get current production model (for rollback reference)
        current_prod = self.registry.get_production_model(symbol, model_type)
        
        # Promote to production
        promoted_version = self.registry._promote_to_production(model_version)
        
        # Update metadata
        promoted_version.metadata.deployment_reason = reason
        
        if validation_metrics:
            promoted_version.metadata.validation_metrics = validation_metrics
        
        # Save updated metadata
        metadata_file = promoted_version.model_path.parent / 'metadata.json'
        promoted_version.metadata.save(metadata_file)
        
        # Log promotion in audit trail
        self._log_audit_event('model_promoted', {
            'symbol': symbol,
            'model_type': model_type,
            'version': version,
            'previous_version': current_prod.metadata.version if current_prod else None,
            'reason': reason,
            'validation_metrics': validation_metrics
        })
        
        # Create deployment record
        self._create_deployment_record(
            symbol, model_type, version, current_prod.metadata.version if current_prod else None
        )
        
        # Send notification
        self._send_deployment_notification(
            symbol, model_type, version, "promoted"
        )
        
        print(f"✅ {symbol}/{model_type} v{version} promoted to production")
        print(f"   Previous version: {current_prod.metadata.version if current_prod else 'None'}")
        print(f"   Reason: {reason}")
        
        return promoted_version
    
    def _validate_model_for_production(self, model_version: ModelVersion) -> bool:
        """Validate model before promotion to production"""
        metadata = model_version.metadata
        
        # Basic validation checks
        checks = []
        
        # 1. Minimum accuracy requirement
        if hasattr(metadata, 'accuracy') and metadata.accuracy < 0.5:
            print(f"❌ Validation failed: Accuracy {metadata.accuracy:.3f} < 0.5")
            checks.append(False)
        
        # 2. Model file exists and is readable
        if not model_version.model_path.exists():
            print(f"❌ Validation failed: Model file not found")
            checks.append(False)
        
        # 3. Model can be loaded
        try:
            model_version.load_model()
            if model_version.load_error:
                print(f"❌ Validation failed: Model loading error: {model_version.load_error}")
                checks.append(False)
        except Exception as e:
            print(f"❌ Validation failed: Model loading exception: {e}")
            checks.append(False)
        
        # 4. Metadata is complete
        required_fields = ['model_type', 'symbol', 'version', 'trained_at']
        for field in required_fields:
            if not getattr(metadata, field, None):
                print(f"❌ Validation failed: Missing metadata field: {field}")
                checks.append(False)
        
        return all(checks) if checks else True
    
    def _get_model_version(self, symbol: str, model_type: str, version: str) -> ModelVersion:
        """Get specific model version"""
        symbol_key = symbol.upper().replace('/', '_')
        
        if (symbol_key in self.registry.models and 
            model_type in self.registry.models[symbol_key]):
            
            for model_version in self.registry.models[symbol_key][model_type]:
                if model_version.metadata.version == version:
                    return model_version
        
        raise ValueError(f"Model version {version} not found for {symbol}/{model_type}")
    
    def _get_models_by_status(self, symbol: str, model_type: str, status: str) -> List[ModelVersion]:
        """Get models by status"""
        symbol_key = symbol.upper().replace('/', '_')
        
        if (symbol_key in self.registry.models and 
            model_type in self.registry.models[symbol_key]):
            
            return [
                mv for mv in self.registry.models[symbol_key][model_type]
                if mv.metadata.status == status
            ]
        
        return []
    
    def _create_deployment_record(
        self,
        symbol: str,
        model_type: str,
        version: str,
        previous_version: Optional[str] = None
    ):
        """Create deployment record"""
        deployments_file = self.registry_root / 'deployments.jsonl'
        
        deployment_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'model_type': model_type,
            'version': version,
            'previous_version': previous_version,
            'deployment_type': 'promotion',
            'status': 'success'
        }
        
        with open(deployments_file, 'a') as f:
            f.write(json.dumps(deployment_record) + '\n')
    
    def _send_deployment_notification(
        self,
        symbol: str,
        model_type: str,
        version: str,
        action: str
    ):
        """Send deployment notification (webhook, email, etc.)"""
        # In production, this would send to Slack, email, etc.
        notification = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'model_type': model_type,
            'version': version,
            'action': action,
            'message': f"Model {action} completed"
        }
        
        # Log to file for now
        notifications_file = self.registry_root / 'notifications.jsonl'
        with open(notifications_file, 'a') as f:
            f.write(json.dumps(notification) + '\n')
        
        print(f"📢 Notification: {symbol}/{model_type} v{version} {action}")
    
    def rollback_model(
        self,
        symbol: str,
        model_type: str = "ensemble",
        target_version: str = None,
        reason: str = "performance_issue"
    ) -> ModelVersion:
        """
        Rollback to previous version with comprehensive tracking
        """
        print(f"🔄 Rolling back {symbol}/{model_type}...")
        
        # Get current production model
        current_prod = self.registry.get_production_model(symbol, model_type)
        if not current_prod:
            raise ValueError(f"No production model found for {symbol}/{model_type}")
        
        # Get target version
        if target_version:
            target_model = self._get_model_version(symbol, model_type, target_version)
        else:
            # Get previous version (second most recent production)
            all_versions = self.registry.get_model_history(symbol, model_type)
            production_versions = [
                v for v in all_versions if v['status'] == 'production'
            ]
            
            if len(production_versions) < 2:
                raise ValueError(f"No previous production version to rollback to")
            
            # Sort by deployment time
            production_versions.sort(
                key=lambda x: x.get('deployed_at', x['trained_at']),
                reverse=True
            )
            
            target_version = production_versions[1]['version']
            target_model = self._get_model_version(symbol, model_type, target_version)
        
        # Update metadata
        current_prod.metadata.rollback_reason = reason
        current_prod.metadata.rollback_at = datetime.now().isoformat()
        current_prod.metadata.rollback_to = target_version
        
        # Save current model metadata
        current_metadata_file = current_prod.model_path.parent / 'metadata.json'
        current_prod.metadata.save(current_metadata_file)
        
        # Promote target version to production
        target_model.metadata.status = "production"
        target_model.metadata.deployed_at = datetime.now().isoformat()
        target_model.metadata.promotion_reason = f"rollback from {current_prod.metadata.version}"
        
        promoted_version = self.registry._promote_to_production(target_model)
        
        # Log rollback in audit trail
        self._log_audit_event('model_rollback', {
            'symbol': symbol,
            'model_type': model_type,
            'from_version': current_prod.metadata.version,
            'to_version': target_version,
            'reason': reason,
            'current_accuracy': current_prod.metadata.accuracy,
            'target_accuracy': target_model.metadata.accuracy
        })
        
        # Send notification
        self._send_deployment_notification(
            symbol, model_type, target_version, "rollback"
        )
        
        print(f"✅ Rollback complete: {current_prod.metadata.version} → {target_version}")
        print(f"   Reason: {reason}")
        
        return promoted_version
    
    def export_model(
        self,
        symbol: str,
        model_type: str = "ensemble",
        version: str = None,
        format: str = "joblib",
        include_metadata: bool = True
    ) -> Path:
        """
        Export model for sharing or deployment elsewhere
        """
        print(f"📤 Exporting {symbol}/{model_type} model...")
        
        # Get model version
        if version:
            model_version = self._get_model_version(symbol, model_type, version)
        else:
            model_version = self.registry.get_production_model(symbol, model_type)
        
        if not model_version:
            raise ValueError(f"No model found for {symbol}/{model_type}")
        
        # Create export directory
        export_dir = self.registry_root / 'exports' / symbol / model_type
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"{symbol}_{model_type}_{model_version.metadata.version}_{timestamp}"
        
        # Export model
        if format == "joblib":
            export_path = export_dir / f"{export_filename}.joblib"
            joblib.dump(model_version.model, export_path)
        elif format == "pickle":
            export_path = export_dir / f"{export_filename}.pkl"
            with open(export_path, 'wb') as f:
                pickle.dump(model_version.model, f)
        elif format == "onnx":
            # Convert to ONNX if possible
            export_path = self._export_to_onnx(model_version, export_dir, export_filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Export metadata if requested
        if include_metadata:
            metadata_path = export_dir / f"{export_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_version.metadata.to_dict(), f, indent=2)
        
        # Log export
        self._log_audit_event('model_exported', {
            'symbol': symbol,
            'model_type': model_type,
            'version': model_version.metadata.version,
            'export_format': format,
            'export_path': str(export_path),
            'include_metadata': include_metadata
        })
        
        print(f"✅ Model exported to {export_path}")
        
        return export_path
    
    def _export_to_onnx(self, model_version: ModelVersion, export_dir: Path, filename: str) -> Path:
        """Export model to ONNX format"""
        try:
            import onnx
            import onnxmltools
            from skl2onnx import convert_sklearn
            
            # Convert sklearn model to ONNX
            model = model_version.load_model()
            
            # Get feature names from metadata
            feature_names = model_version.metadata.feature_names
            
            # Create initial types
            from skl2onnx.common.data_types import FloatTensorType
            
            initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
            
            # Convert to ONNX
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            # Save ONNX model
            export_path = export_dir / f"{filename}.onnx"
            with open(export_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            return export_path
            
        except ImportError:
            print("⚠️  ONNX export requires skl2onnx and onnx packages")
            print("   Install with: pip install skl2onnx onnx")
            raise
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            raise
    
    def get_model_performance_history(
        self,
        symbol: str,
        model_type: str = "ensemble",
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get performance history for a model
        """
        # Get model history
        history = self.registry.get_model_history(symbol, model_type)
        
        if not history:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_history = pd.DataFrame(history)
        
        # Parse dates
        df_history['trained_at'] = pd.to_datetime(df_history['trained_at'])
        df_history['deployed_at'] = pd.to_datetime(df_history['deployed_at'], errors='coerce')
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        df_filtered = df_history[df_history['trained_at'] >= cutoff_date]
        
        # Sort by date
        df_sorted = df_filtered.sort_values('trained_at', ascending=True)
        
        return df_sorted
    
    def cleanup_old_exports(self, older_than_days: int = 7):
        """Cleanup old export files"""
        exports_dir = self.registry_root / 'exports'
        
        if not exports_dir.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        
        for export_file in exports_dir.rglob('*'):
            if export_file.is_file():
                file_time = datetime.fromtimestamp(export_file.stat().st_mtime)
                if file_time < cutoff_time:
                    try:
                        export_file.unlink()
                        print(f"🗑️  Deleted old export: {export_file}")
                    except Exception as e:
                        print(f"⚠️  Could not delete {export_file}: {e}")
    
    def backup_registry(self, backup_dir: Path = None):
        """Create backup of entire registry"""
        if backup_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.registry_root / 'backups' / f"registry_backup_{timestamp}"
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Creating registry backup to {backup_dir}...")
        
        # Copy registry files
        for item in self.registry_root.iterdir():
            if item.name in ['production', 'staging', 'experiments', 'archived', 'metadata']:
                dest = backup_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
        
        # Copy registry.json
        registry_file = self.registry_root / 'registry.json'
        if registry_file.exists():
            shutil.copy2(registry_file, backup_dir / 'registry.json')
        
        # Create backup manifest
        manifest = {
            'backup_time': datetime.now().isoformat(),
            'registry_root': str(self.registry_root),
            'backup_dir': str(backup_dir),
            'total_models': sum(len(models) for models in self.registry.models.values())
        }
        
        manifest_file = backup_dir / 'backup_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✅ Registry backup created: {backup_dir}")
        
        return backup_dir
    
    def restore_registry(self, backup_dir: Path, dry_run: bool = True):
        """Restore registry from backup"""
        if not backup_dir.exists():
            raise ValueError(f"Backup directory not found: {backup_dir}")
        
        # Check backup manifest
        manifest_file = backup_dir / 'backup_manifest.json'
        if not manifest_file.exists():
            raise ValueError("Invalid backup: missing manifest")
        
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        print(f"🔄 Restoring registry from backup: {backup_dir}")
        print(f"   Backup time: {manifest.get('backup_time')}")
        print(f"   Total models: {manifest.get('total_models', 'unknown')}")
        
        if dry_run:
            print("🔍 DRY RUN - No changes will be made")
            return
        
        # Backup current registry first
        current_backup = self.backup_registry()
        print(f"   Current registry backed up to: {current_backup}")
        
        # Restore directories
        directories_to_restore = ['production', 'staging', 'experiments', 'archived', 'metadata']
        
        for dir_name in directories_to_restore:
            src_dir = backup_dir / dir_name
            dest_dir = self.registry_root / dir_name
            
            if src_dir.exists():
                # Remove existing directory
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                
                # Copy backup directory
                shutil.copytree(src_dir, dest_dir)
                print(f"   Restored: {dir_name}")
        
        # Restore registry.json
        backup_registry_file = backup_dir / 'registry.json'
        if backup_registry_file.exists():
            shutil.copy2(backup_registry_file, self.registry_root / 'registry.json')
            print(f"   Restored: registry.json")
        
        # Reload registry
        self.registry._load_registry()
        
        print("✅ Registry restore completed")
        print("⚠️  IMPORTANT: Restart AI-Service to load restored models")


# Singleton instance
_registry_manager_instance = None

def get_registry_manager() -> RegistryManager:
    """Get singleton registry manager instance"""
    global _registry_manager_instance
    
    if _registry_manager_instance is None:
        _registry_manager_instance = RegistryManager()
    
    return _registry_manager_instance


# Command Line Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enterprise Registry Manager CLI")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a trained model')
    register_parser.add_argument('--symbol', required=True, help='Trading symbol')
    register_parser.add_argument('--model-type', default='ensemble', help='Type of model')
    register_parser.add_argument('--model-path', required=True, help='Path to model file')
    register_parser.add_argument('--accuracy', type=float, help='Model accuracy')
    register_parser.add_argument('--promote', action='store_true', help='Auto-promote to production')
    
    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote model to production')
    promote_parser.add_argument('--symbol', required=True, help='Trading symbol')
    promote_parser.add_argument('--model-type', default='ensemble', help='Type of model')
    promote_parser.add_argument('--version', help='Specific version to promote')
    promote_parser.add_argument('--reason', default='manual', help='Promotion reason')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List models')
    list_parser.add_argument('--symbol', help='Filter by symbol')
    list_parser.add_argument('--type', help='Filter by model type')
    list_parser.add_argument('--status', help='Filter by status')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback model')
    rollback_parser.add_argument('--symbol', required=True, help='Trading symbol')
    rollback_parser.add_argument('--model-type', default='ensemble', help='Type of model')
    rollback_parser.add_argument('--version', help='Version to rollback to')
    rollback_parser.add_argument('--reason', default='performance', help='Rollback reason')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup registry')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup old exports')
    cleanup_parser.add_argument('--days', type=int, default=7, help='Delete exports older than X days')
    
    args = parser.parse_args()
    
    manager = get_registry_manager()
    
    if args.command == 'register':
        # Load model
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            sys.exit(1)
        
        model = joblib.load(model_path) if model_path.suffix == '.joblib' else None
        
        # Prepare performance metrics
        performance = {}
        if args.accuracy:
            performance['accuracy'] = args.accuracy
        
        # Register model
        manager.register_trained_model(
            model=model,
            symbol=args.symbol,
            model_type=args.model_type,
            performance_metrics=performance,
            auto_promote=args.promote
        )
    
    elif args.command == 'promote':
        manager.promote_to_production(
            symbol=args.symbol,
            model_type=args.model_type,
            version=args.version,
            reason=args.reason
        )
    
    elif args.command == 'list':
        registry = get_model_registry()
        models = registry.list_models(args.symbol, args.type)
        
        if not models:
            print("No models found")
        else:
            print(f"\n📋 Models in Registry ({len(models)} total):")
            print("=" * 100)
            
            for model in models:
                if args.status and model['status'] != args.status:
                    continue
                
                status_icon = {
                    'production': '🚀',
                    'staging': '🧪',
                    'experiment': '🔬',
                    'archived': '📦'
                }.get(model['status'], '❓')
                
                print(f"{status_icon} {model['symbol']}/{model['model_type']}")
                print(f"   Version: {model['version']}")
                print(f"   Status: {model['status']}")
                print(f"   Accuracy: {model['accuracy']:.3f}")
                print(f"   Trained: {model['trained_at'][:10]}")
                print(f"   Size: {model['model_size_mb']} MB")
                print("-" * 80)
    
    elif args.command == 'rollback':
        manager.rollback_model(
            symbol=args.symbol,
            model_type=args.model_type,
            target_version=args.version,
            reason=args.reason
        )
    
    elif args.command == 'backup':
        backup_dir = manager.backup_registry()
        print(f"✅ Backup created at: {backup_dir}")
    
    elif args.command == 'cleanup':
        manager.cleanup_old_exports(args.days)
    
    else:
        parser.print_help()
