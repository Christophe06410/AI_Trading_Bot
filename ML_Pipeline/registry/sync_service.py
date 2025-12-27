"""
REGISTRY SYNC SERVICE
Synchronizes registry with AI-Service and Trading Bot
Ensures backward compatibility
"""

import os
import sys
import json
import yaml
import time
import shutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
import logging
import warnings
warnings.filterwarnings('ignore')

from enterprise.model_registry import get_model_registry

class RegistrySyncService:
    """
    Synchronizes model registry with AI-Service
    Ensures AI-Service always has latest models
    """
    
    def __init__(self, config_path: Path = None):
        self.registry = get_model_registry()
        self.config = self._load_config(config_path)
        
        # AI-Service configuration
        self.ai_service_url = self.config.get('ai_service', {}).get('url', 'http://localhost:8000')
        self.ai_service_api_key = self.config.get('ai_service', {}).get('api_key', '')
        
        # Sync intervals
        self.sync_interval = self.config.get('sync', {}).get('interval_seconds', 300)
        self.last_sync = {}
        
        # Logging
        self.log_file = self.registry.registry_root / 'sync_log.jsonl'
        self.log_file.touch(exist_ok=True)
        
        # Monitoring
        self.monitoring = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'last_error': None
        }
        
        print(f"🔄 Registry Sync Service initialized")
        print(f"   AI-Service URL: {self.ai_service_url}")
        print(f"   Sync interval: {self.sync_interval}s")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load sync configuration"""
        default_config = {
            'ai_service': {
                'url': 'http://localhost:8000',
                'api_key': '',
                'health_endpoint': '/health',
                'model_info_endpoint': '/api/v1/model-info',
                'reload_endpoint': '/api/v1/reload-models'
            },
            'sync': {
                'interval_seconds': 300,
                'auto_sync': True,
                'notify_on_change': True,
                'verify_checksums': True
            },
            'backward_compatibility': {
                'ensure_legacy_models': True,
                'max_retries': 3,
                'retry_delay': 10
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                default_config.update(config_data)
        
        return default_config
    
    async def start_sync_service(self):
        """Start continuous sync service"""
        print("🚀 Starting registry sync service...")
        
        while True:
            try:
                await self.sync_all_symbols()
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                print(f"❌ Sync service error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def sync_all_symbols(self):
        """Sync all symbols with AI-Service"""
        print(f"🔄 Syncing all symbols with AI-Service...")
        
        # Get all production models
        for symbol in self.registry.production_models:
            for model_type in self.registry.production_models[symbol]:
                await self.sync_model(symbol, model_type)
    
    async def sync_model(
        self,
        symbol: str,
        model_type: str = "ensemble",
        force: bool = False
    ):
        """
        Sync specific model with AI-Service
        """
        model_key = f"{symbol}/{model_type}"
        
        # Check if sync is needed
        if not force and not self._needs_sync(symbol, model_type):
            return
        
        print(f"📡 Syncing {symbol}/{model_type}...")
        
        try:
            # 1. Verify AI-Service is reachable
            if not await self._check_ai_service_health():
                print(f"❌ AI-Service not reachable")
                self._log_sync_error(symbol, model_type, "AI-Service unreachable")
                return
            
            # 2. Get current model from registry
            model_version = self.registry.get_production_model(symbol, model_type)
            if not model_version:
                print(f"❌ No production model found for {symbol}/{model_type}")
                return
            
            # 3. Check if AI-Service already has this version
            ai_service_version = await self._get_ai_service_model_version(symbol, model_type)
            
            if ai_service_version == model_version.metadata.version and not force:
                print(f"✅ AI-Service already has latest version {model_version.metadata.version}")
                self._update_last_sync(symbol, model_type, 'skipped')
                return
            
            # 4. Ensure backward compatibility symlink exists
            self._ensure_backward_compatibility(model_version)
            
            # 5. Verify model checksum
            if not self._verify_model_integrity(model_version):
                print(f"❌ Model integrity check failed for {symbol}/{model_type}")
                self._log_sync_error(symbol, model_type, "Model integrity check failed")
                return
            
            # 6. Notify AI-Service about new model (if it supports hot reload)
            if await self._notify_ai_service(model_version):
                print(f"✅ AI-Service notified about new model version")
                self.monitoring['successful_syncs'] += 1
            else:
                print(f"⚠️  AI-Service notification failed, but model is available")
                # Still count as success since model file is in place
                self.monitoring['successful_syncs'] += 1
            
            # 7. Update sync tracking
            self._update_last_sync(symbol, model_type, 'success')
            
            # 8. Log successful sync
            self._log_sync_success(symbol, model_type, model_version.metadata.version)
            
            print(f"✅ Successfully synced {symbol}/{model_type} v{model_version.metadata.version}")
            
        except Exception as e:
            print(f"❌ Sync failed for {symbol}/{model_type}: {e}")
            self.monitoring['failed_syncs'] += 1
            self.monitoring['last_error'] = str(e)
            self._log_sync_error(symbol, model_type, str(e))
    
    def _needs_sync(self, symbol: str, model_type: str) -> bool:
        """Check if model needs syncing"""
        model_key = f"{symbol}/{model_type}"
        
        if model_key not in self.last_sync:
            return True
        
        last_sync_time = self.last_sync[model_key].get('timestamp')
        if not last_sync_time:
            return True
        
        # Convert string time to datetime
        try:
            last_sync_dt = datetime.fromisoformat(last_sync_time)
            time_since_sync = (datetime.now() - last_sync_dt).total_seconds()
            return time_since_sync > self.sync_interval
        except:
            return True
    
    async def _check_ai_service_health(self) -> bool:
        """Check if AI-Service is healthy"""
        try:
            health_url = f"{self.ai_service_url}{self.config['ai_service']['health_endpoint']}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=10) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        return health_data.get('status') == 'healthy'
                    else:
                        return False
        except Exception as e:
            print(f"⚠️  AI-Service health check failed: {e}")
            return False
    
    async def _get_ai_service_model_version(
        self, 
        symbol: str, 
        model_type: str
    ) -> Optional[str]:
        """Get current model version from AI-Service"""
        try:
            # This assumes AI-Service has an endpoint to report loaded model versions
            # If not, we'll implement fallback
            model_info_url = f"{self.ai_service_url}{self.config['ai_service']['model_info_endpoint']}"
            
            headers = {}
            if self.ai_service_api_key:
                headers['X-API-Key'] = self.ai_service_api_key
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    model_info_url,
                    headers=headers,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        model_info = await response.json()
                        
                        # Parse response to find version for this symbol/model_type
                        # This depends on AI-Service API structure
                        # For now, return None to force sync
                        return None
                    else:
                        return None
                        
        except Exception as e:
            print(f"⚠️  Could not get AI-Service model version: {e}")
            return None
    
    def _ensure_backward_compatibility(self, model_version):
        """Ensure backward compatibility symlink exists"""
        # This is already handled by the registry
        # Double-check that symlink exists
        legacy_path = self.registry.legacy_models_path / self.registry._get_legacy_filename(
            model_version.metadata.symbol,
            model_version.metadata.model_type
        )
        
        if not legacy_path.exists():
            print(f"⚠️  Backward compatibility symlink missing, recreating...")
            self.registry._ensure_backward_compatibility()
    
    def _verify_model_integrity(self, model_version) -> bool:
        """Verify model file integrity"""
        model_path = model_version.model_path
        
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return False
        
        # Check file size
        file_size = model_path.stat().st_size
        if file_size < 1000:  # Less than 1KB is suspicious
            print(f"⚠️  Model file suspiciously small: {file_size} bytes")
            return False
        
        # Calculate checksum
        try:
            import hashlib
            with open(model_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            # Store checksum in metadata
            checksum_file = model_path.parent / 'checksum.md5'
            if not checksum_file.exists():
                checksum_file.write_text(file_hash)
            else:
                stored_hash = checksum_file.read_text().strip()
                if stored_hash != file_hash:
                    print(f"❌ Model checksum mismatch!")
                    print(f"   Stored: {stored_hash}")
                    print(f"   Actual: {file_hash}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"⚠️  Could not verify model integrity: {e}")
            return False
    
    async def _notify_ai_service(self, model_version) -> bool:
        """Notify AI-Service about new model (if supported)"""
        # Check if AI-Service supports hot reload
        reload_endpoint = self.config['ai_service'].get('reload_endpoint')
        
        if not reload_endpoint:
            # AI-Service doesn't support hot reload
            # It will load new model on next restart or via file watch
            return True
        
        try:
            reload_url = f"{self.ai_service_url}{reload_endpoint}"
            
            headers = {'Content-Type': 'application/json'}
            if self.ai_service_api_key:
                headers['X-API-Key'] = self.ai_service_api_key
            
            payload = {
                'symbol': model_version.metadata.symbol,
                'model_type': model_version.metadata.model_type,
                'version': model_version.metadata.version,
                'model_path': str(model_version.model_path)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    reload_url,
                    json=payload,
                    headers=headers,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        print(f"✅ AI-Service acknowledged model reload")
                        return True
                    else:
                        print(f"❌ AI-Service reload failed: {response.status}")
                        return False
                        
        except Exception as e:
            print(f"⚠️  AI-Service notification failed: {e}")
            return False
    
    def _update_last_sync(self, symbol: str, model_type: str, status: str):
        """Update last sync timestamp"""
        model_key = f"{symbol}/{model_type}"
        
        self.last_sync[model_key] = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'symbol': symbol,
            'model_type': model_type
        }
        
        self.monitoring['total_syncs'] += 1
    
    def _log_sync_success(self, symbol: str, model_type: str, version: str):
        """Log successful sync"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'sync_success',
            'symbol': symbol,
            'model_type': model_type,
            'version': version,
            'status': 'success'
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _log_sync_error(self, symbol: str, model_type: str, error: str):
        """Log sync error"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'sync_error',
            'symbol': symbol,
            'model_type': model_type,
            'error': error,
            'status': 'failed'
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get sync service status"""
        status = {
            'monitoring': self.monitoring,
            'last_syncs': self.last_sync,
            'config': {
                'ai_service_url': self.ai_service_url,
                'sync_interval': self.sync_interval,
                'auto_sync': self.config['sync'].get('auto_sync', True)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return status
    
    async def verify_sync_state(self) -> Dict[str, Any]:
        """
        Verify sync state between registry and AI-Service
        Returns detailed verification report
        """
        print("🔍 Verifying sync state...")
        
        verification_report = {
            'timestamp': datetime.now().isoformat(),
            'ai_service_reachable': False,
            'models_verified': [],
            'issues_found': [],
            'summary': {}
        }
        
        # Check AI-Service health
        ai_healthy = await self._check_ai_service_health()
        verification_report['ai_service_reachable'] = ai_healthy
        
        if not ai_healthy:
            verification_report['issues_found'].append(
                "AI-Service is not reachable. Check if it's running."
            )
        
        # Verify each production model
        for symbol in self.registry.production_models:
            for model_type, model_version in self.registry.production_models[symbol].items():
                model_verification = await self._verify_model_sync(
                    symbol, model_type, model_version
                )
                verification_report['models_verified'].append(model_verification)
                
                if not model_verification['sync_status'] == 'in_sync':
                    verification_report['issues_found'].append(
                        f"{symbol}/{model_type}: {model_verification['issue']}"
                    )
        
        # Generate summary
        total_models = len(verification_report['models_verified'])
        in_sync_models = len([m for m in verification_report['models_verified'] 
                             if m['sync_status'] == 'in_sync'])
        
        verification_report['summary'] = {
            'total_models': total_models,
            'in_sync': in_sync_models,
            'out_of_sync': total_models - in_sync_models,
            'ai_service_healthy': ai_healthy,
            'issues_count': len(verification_report['issues_found'])
        }
        
        # Save verification report
        reports_dir = self.registry.registry_root / 'sync_reports'
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(verification_report, f, indent=2)
        
        print(f"✅ Verification complete. Report saved to {report_file}")
        
        return verification_report
    
    async def _verify_model_sync(self, symbol: str, model_type: str, model_version) -> Dict[str, Any]:
        """Verify individual model sync status"""
        verification = {
            'symbol': symbol,
            'model_type': model_type,
            'registry_version': model_version.metadata.version,
            'ai_service_version': None,
            'sync_status': 'unknown',
            'issue': None,
            'checks_passed': []
        }
        
        # Check 1: Model file exists
        if model_version.model_path.exists():
            verification['checks_passed'].append('model_file_exists')
        else:
            verification['sync_status'] = 'error'
            verification['issue'] = 'Model file not found in registry'
            return verification
        
        # Check 2: Backward compatibility symlink exists
        legacy_path = self.registry.legacy_models_path / self.registry._get_legacy_filename(
            symbol, model_type
        )
        
        if legacy_path.exists():
            verification['checks_passed'].append('backward_compatibility_ok')
        else:
            verification['sync_status'] = 'warning'
            verification['issue'] = 'Backward compatibility symlink missing'
        
        # Check 3: AI-Service has correct version (if we can check)
        ai_version = await self._get_ai_service_model_version(symbol, model_type)
        if ai_version:
            verification['ai_service_version'] = ai_version
            
            if ai_version == model_version.metadata.version:
                verification['sync_status'] = 'in_sync'
                verification['checks_passed'].append('versions_match')
            else:
                verification['sync_status'] = 'out_of_sync'
                verification['issue'] = f'Version mismatch: registry={model_version.metadata.version}, ai_service={ai_version}'
        else:
            verification['sync_status'] = 'unknown'
            verification['issue'] = 'Could not determine AI-Service version'
        
        return verification
