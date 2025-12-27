"""
REGISTRY CLEANUP SERVICE
Automated cleanup of old models and temporary files
"""

import os
import sys
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from enterprise.model_registry import get_model_registry

class RegistryCleanupService:
    """
    Automated cleanup service for registry
    Removes old models, temporary files, and optimizes storage
    """
    
    def __init__(self, registry_root: Path = None):
        self.registry = get_model_registry()
        self.registry_root = self.registry.registry_root
        
        # Cleanup policies
        self.policies = {
            'staging_models': {
                'keep_last': 5,
                'max_age_days': 30,
                'enabled': True
            },
            'experiment_models': {
                'keep_last': 3,
                'max_age_days': 90,
                'enabled': True
            },
            'archived_models': {
                'keep_last': 10,
                'max_age_days': 365,
                'enabled': True
            },
            'temporary_files': {
                'max_age_hours': 24,
                'enabled': True
            },
            'performance_data': {
                'max_age_days': 180,
                'enabled': True
            },
            'log_files': {
                'max_age_days': 30,
                'max_size_mb': 100,
                'enabled': True
            }
        }
        
        # Statistics
        self.cleanup_stats = {
            'last_run': None,
            'total_cleaned': 0,
            'space_saved_mb': 0,
            'errors': 0
        }
        
        print(f"🧹 Registry Cleanup Service initialized")
    
    def run_cleanup(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Run complete cleanup based on policies
        Returns cleanup report
        """
        print(f"{'🔍 DRY RUN' if dry_run else '🧹 RUNNING'} cleanup...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': dry_run,
            'policies_applied': [],
            'items_cleaned': [],
            'errors': [],
            'summary': {
                'total_items': 0,
                'space_saved_mb': 0
            }
        }
        
        # Apply each cleanup policy
        if self.policies['staging_models']['enabled']:
            staging_report = self._cleanup_staging_models(dry_run)
            report['policies_applied'].append('staging_models')
            report['items_cleaned'].extend(staging_report['cleaned'])
            report['errors'].extend(staging_report['errors'])
        
        if self.policies['experiment_models']['enabled']:
            experiment_report = self._cleanup_experiment_models(dry_run)
            report['policies_applied'].append('experiment_models')
            report['items_cleaned'].extend(experiment_report['cleaned'])
            report['errors'].extend(experiment_report['errors'])
        
        if self.policies['archived_models']['enabled']:
            archived_report = self._cleanup_archived_models(dry_run)
            report['policies_applied'].append('archived_models')
            report['items_cleaned'].extend(archived_report['cleaned'])
            report['errors'].extend(archived_report['errors'])
        
        if self.policies['temporary_files']['enabled']:
            temp_report = self._cleanup_temporary_files(dry_run)
            report['policies_applied'].append('temporary_files')
            report['items_cleaned'].extend(temp_report['cleaned'])
            report['errors'].extend(temp_report['errors'])
        
        if self.policies['performance_data']['enabled']:
            perf_report = self._cleanup_performance_data(dry_run)
            report['policies_applied'].append('performance_data')
            report['items_cleaned'].extend(perf_report['cleaned'])
            report['errors'].extend(perf_report['errors'])
        
        if self.policies['log_files']['enabled']:
            log_report = self._cleanup_log_files(dry_run)
            report['policies_applied'].append('log_files')
            report['items_cleaned'].extend(log_report['cleaned'])
            report['errors'].extend(log_report['errors'])
        
        # Calculate summary
        report['summary']['total_items'] = len(report['items_cleaned'])
        report['summary']['space_saved_mb'] = sum(
            item.get('size_mb', 0) for item in report['items_cleaned']
        )
        
        # Update statistics if not dry run
        if not dry_run:
            self.cleanup_stats['last_run'] = datetime.now().isoformat()
            self.cleanup_stats['total_cleaned'] += report['summary']['total_items']
            self.cleanup_stats['space_saved_mb'] += report['summary']['space_saved_mb']
            self.cleanup_stats['errors'] += len(report['errors'])
        
        # Save report
        reports_dir = self.registry_root / 'cleanup_reports'
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Cleanup complete: {report['summary']['total_items']} items, "
              f"{report['summary']['space_saved_mb']:.1f} MB saved")
        
        if dry_run:
            print("📋 Dry run report saved, no changes made")
        
        return report
    
    def _cleanup_staging_models(self, dry_run: bool) -> Dict[str, Any]:
        """Cleanup old staging models"""
        report = {
            'policy': 'staging_models',
            'cleaned': [],
            'errors': []
        }
        
        staging_dir = self.registry_root / 'staging'
        if not staging_dir.exists():
            return report
        
        keep_last = self.policies['staging_models']['keep_last']
        max_age_days = self.policies['staging_models']['max_age_days']
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Process each symbol/model_type
        for symbol_dir in staging_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
            
            for model_type_dir in symbol_dir.iterdir():
                if not model_type_dir.is_dir():
                    continue
                
                # Get all versions
                versions = []
                for version_dir in model_type_dir.iterdir():
                    if version_dir.is_dir():
                        metadata_file = version_dir / 'metadata.json'
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                
                                trained_at = datetime.fromisoformat(metadata.get('trained_at', '2000-01-01'))
                                
                                versions.append({
                                    'path': version_dir,
                                    'trained_at': trained_at,
                                    'version': metadata.get('version', 'unknown')
                                })
                            except Exception as e:
                                report['errors'].append(f"Error reading {metadata_file}: {e}")
                
                # Sort by date (oldest first)
                versions.sort(key=lambda x: x['trained_at'])
                
                # Determine which to keep
                keep_versions = []
                remove_versions = []
                
                for i, version in enumerate(versions):
                    # Always keep recent versions
                    if i >= len(versions) - keep_last:
                        keep_versions.append(version)
                    # Keep versions younger than max age
                    elif version['trained_at'] > cutoff_date:
                        keep_versions.append(version)
                    else:
                        remove_versions.append(version)
                
                # Remove old versions
                for version in remove_versions:
                    try:
                        # Calculate size before deletion
                        size_mb = self._get_directory_size(version['path']) / (1024 * 1024)
                        
                        if not dry_run:
                            shutil.rmtree(version['path'])
                        
                        report['cleaned'].append({
                            'type': 'staging_model',
                            'symbol': symbol_dir.name,
                            'model_type': model_type_dir.name,
                            'version': version['version'],
                            'trained_at': version['trained_at'].isoformat(),
                            'size_mb': size_mb,
                            'reason': f'Old version (beyond keep_last={keep_last} or max_age={max_age_days} days)'
                        })
                        
                        print(f"   🗑️  Staging: {symbol_dir.name}/{model_type_dir.name} v{version['version']}")
                        
                    except Exception as e:
                        report['errors'].append(f"Error removing {version['path']}: {e}")
        
        return report
    
    def _cleanup_experiment_models(self, dry_run: bool) -> Dict[str, Any]:
        """Cleanup old experiment models"""
        report = {
            'policy': 'experiment_models',
            'cleaned': [],
            'errors': []
        }
        
        experiments_dir = self.registry_root / 'experiments'
        if not experiments_dir.exists():
            return report
        
        keep_last = self.policies['experiment_models']['keep_last']
        max_age_days = self.policies['experiment_models']['max_age_days']
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Process experiment directories
        for exp_dir in experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Check if experiment is old
            exp_time = datetime.fromtimestamp(exp_dir.stat().st_mtime)
            
            if exp_time < cutoff_date:
                try:
                    # Calculate size
                    size_mb = self._get_directory_size(exp_dir) / (1024 * 1024)
                    
                    if not dry_run:
                        shutil.rmtree(exp_dir)
                    
                    report['cleaned'].append({
                        'type': 'experiment',
                        'path': str(exp_dir),
                        'last_modified': exp_time.isoformat(),
                        'size_mb': size_mb,
                        'reason': f'Older than {max_age_days} days'
                    })
                    
                    print(f"   🗑️  Experiment: {exp_dir.name}")
                    
                except Exception as e:
                    report['errors'].append(f"Error removing {exp_dir}: {e}")
        
        return report
    
    def _cleanup_archived_models(self, dry_run: bool) -> Dict[str, Any]:
        """Cleanup old archived models"""
        report = {
            'policy': 'archived_models',
            'cleaned': [],
            'errors': []
        }
        
        archived_dir = self.registry_root / 'archived'
        if not archived_dir.exists():
            return report
        
        keep_last = self.policies['archived_models']['keep_last']
        max_age_days = self.policies['archived_models']['max_age_days']
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Process each symbol/model_type in archived
        for symbol_dir in archived_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
            
            for model_type_dir in symbol_dir.iterdir():
                if not model_type_dir.is_dir():
                    continue
                
                # Get all archived versions
                versions = []
                for version_dir in model_type_dir.iterdir():
                    if version_dir.is_dir():
                        metadata_file = version_dir / 'metadata.json'
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                
                                archived_at = datetime.fromisoformat(
                                    metadata.get('archived_at', metadata.get('trained_at', '2000-01-01'))
                                )
                                
                                versions.append({
                                    'path': version_dir,
                                    'archived_at': archived_at,
                                    'version': metadata.get('version', 'unknown')
                                })
                            except Exception as e:
                                report['errors'].append(f"Error reading {metadata_file}: {e}")
                
                # Sort by date (oldest first)
                versions.sort(key=lambda x: x['archived_at'])
                
                # Keep only recent versions
                if len(versions) > keep_last:
                    remove_versions = versions[:len(versions) - keep_last]
                    
                    for version in remove_versions:
                        # Also check age
                        if version['archived_at'] < cutoff_date:
                            try:
                                size_mb = self._get_directory_size(version['path']) / (1024 * 1024)
                                
                                if not dry_run:
                                    shutil.rmtree(version['path'])
                                
                                report['cleaned'].append({
                                    'type': 'archived_model',
                                    'symbol': symbol_dir.name,
                                    'model_type': model_type_dir.name,
                                    'version': version['version'],
                                    'archived_at': version['archived_at'].isoformat(),
                                    'size_mb': size_mb,
                                    'reason': f'Beyond keep_last={keep_last} and older than {max_age_days} days'
                                })
                                
                                print(f"   🗑️  Archived: {symbol_dir.name}/{model_type_dir.name} v{version['version']}")
                                
                            except Exception as e:
                                report['errors'].append(f"Error removing {version['path']}: {e}")
        
        return report
    
    def _cleanup_temporary_files(self, dry_run: bool) -> Dict[str, Any]:
        """Cleanup temporary files"""
        report = {
            'policy': 'temporary_files',
            'cleaned': [],
            'errors': []
        }
        
        tmp_dir = self.registry_root / 'tmp'
        if not tmp_dir.exists():
            return report
        
        max_age_hours = self.policies['temporary_files']['max_age_hours']
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for tmp_file in tmp_dir.rglob('*'):
            if tmp_file.is_file():
                file_time = datetime.fromtimestamp(tmp_file.stat().st_mtime)
                
                if file_time < cutoff_time:
                    try:
                        size_mb = tmp_file.stat().st_size / (1024 * 1024)
                        
                        if not dry_run:
                            tmp_file.unlink()
                        
                        report['cleaned'].append({
                            'type': 'temporary_file',
                            'path': str(tmp_file),
                            'last_modified': file_time.isoformat(),
                            'size_mb': size_mb,
                            'reason': f'Older than {max_age_hours} hours'
                        })
                        
                    except Exception as e:
                        report['errors'].append(f"Error removing {tmp_file}: {e}")
        
        # Cleanup empty directories
        for tmp_subdir in tmp_dir.rglob('*'):
            if tmp_subdir.is_dir() and not any(tmp_subdir.iterdir()):
                try:
                    if not dry_run:
                        tmp_subdir.rmdir()
                    
                    report['cleaned'].append({
                        'type': 'empty_directory',
                        'path': str(tmp_subdir),
                        'size_mb': 0,
                        'reason': 'Empty directory'
                    })
                    
                except Exception as e:
                    report['errors'].append(f"Error removing {tmp_subdir}: {e}")
        
        return report
    
    def _cleanup_performance_data(self, dry_run: bool) -> Dict[str, Any]:
        """Cleanup old performance data"""
        report = {
            'policy': 'performance_data',
            'cleaned': [],
            'errors': []
        }
        
        perf_dir = self.registry_root / 'performance'
        if not perf_dir.exists():
            return report
        
        max_age_days = self.policies['performance_data']['max_age_days']
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Cleanup old report files
        reports_dir = perf_dir / 'reports'
        if reports_dir.exists():
            for report_file in reports_dir.glob('*.json'):
                file_time = datetime.fromtimestamp(report_file.stat().st_mtime)
                
                if file_time < cutoff_date:
                    try:
                        size_mb = report_file.stat().st_size / (1024 * 1024)
                        
                        if not dry_run:
                            report_file.unlink()
                        
                        report['cleaned'].append({
                            'type': 'performance_report',
                            'path': str(report_file),
                            'last_modified': file_time.isoformat(),
                            'size_mb': size_mb,
                            'reason': f'Older than {max_age_days} days'
                        })
                        
                    except Exception as e:
                        report['errors'].append(f"Error removing {report_file}: {e}")
        
        # Cleanup old database backups (if any)
        for db_file in perf_dir.glob('*.db.*'):  # .db-wal, .db-shm, etc.
            file_time = datetime.fromtimestamp(db_file.stat().st_mtime)
            
            if file_time < cutoff_date:
                try:
                    size_mb = db_file.stat().st_size / (1024 * 1024)
                    
                    if not dry_run:
                        db_file.unlink()
                    
                    report['cleaned'].append({
                        'type': 'database_backup',
                        'path': str(db_file),
                        'last_modified': file_time.isoformat(),
                        'size_mb': size_mb,
                        'reason': f'Older than {max_age_days} days'
                    })
                    
                except Exception as e:
                    report['errors'].append(f"Error removing {db_file}: {e}")
        
        return report
    
    def _cleanup_log_files(self, dry_run: bool) -> Dict[str, Any]:
        """Cleanup old log files"""
        report = {
            'policy': 'log_files',
            'cleaned': [],
            'errors': []
        }
        
        max_age_days = self.policies['log_files']['max_age_days']
        max_size_mb = self.policies['log_files']['max_size_mb']
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Find log files in registry
        log_files = []
        
        for log_file in self.registry_root.rglob('*.log'):
            if log_file.is_file():
                log_files.append(log_file)
        
        for log_file in self.registry_root.rglob('*.jsonl'):
            if log_file.is_file():
                log_files.append(log_file)
        
        for log_file in log_files:
            file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            file_size_mb = log_file.stat().st_size / (1024 * 1024)
            
            # Check age
            if file_time < cutoff_date:
                try:
                    if not dry_run:
                        log_file.unlink()
                    
                    report['cleaned'].append({
                        'type': 'log_file',
                        'path': str(log_file),
                        'last_modified': file_time.isoformat(),
                        'size_mb': file_size_mb,
                        'reason': f'Older than {max_age_days} days'
                    })
                    
                    print(f"   🗑️  Log file: {log_file.name}")
                    
                except Exception as e:
                    report['errors'].append(f"Error removing {log_file}: {e}")
            
            # Check size (for current files)
            elif file_size_mb > max_size_mb:
                # Rotate log file instead of deleting
                try:
                    if not dry_run:
                        # Create backup with timestamp
                        backup_name = f"{log_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{log_file.suffix}"
                        backup_path = log_file.parent / backup_name
                        shutil.copy2(log_file, backup_path)
                        
                        # Truncate original file
                        with open(log_file, 'w') as f:
                            f.write(f"# Log rotated at {datetime.now().isoformat()}\n")
                    
                    report['cleaned'].append({
                        'type': 'log_rotation',
                        'path': str(log_file),
                        'size_mb': file_size_mb,
                        'reason': f'Size exceeded {max_size_mb} MB'
                    })
                    
                    print(f"   🔄 Rotated log file: {log_file.name}")
                    
                except Exception as e:
                    report['errors'].append(f"Error rotating {log_file}: {e}")
        
        return report
    
    def _get_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes"""
        total_size = 0
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics"""
        usage = {
            'timestamp': datetime.now().isoformat(),
            'directories': {},
            'total_size_mb': 0,
            'recommendations': []
        }
        
        # Calculate size for each directory
        directories = [
            'production',
            'staging',
            'experiments',
            'archived',
            'tmp',
            'performance',
            'exports',
            'backups'
        ]
        
        for dir_name in directories:
            dir_path = self.registry_root / dir_name
            
            if dir_path.exists():
                size_bytes = self._get_directory_size(dir_path)
                size_mb = size_bytes / (1024 * 1024)
                
                usage['directories'][dir_name] = {
                    'size_mb': round(size_mb, 2),
                    'files_count': sum(1 for _ in dir_path.rglob('*') if _.is_file())
                }
                
                usage['total_size_mb'] += size_mb
        
        # Generate recommendations
        total_size_gb = usage['total_size_mb'] / 1024
        
        if total_size_gb > 10:
            usage['recommendations'].append(
                f"Registry size is {total_size_gb:.1f} GB. Consider increasing cleanup frequency."
            )
        
        if usage['directories'].get('tmp', {}).get('size_mb', 0) > 100:
            usage['recommendations'].append(
                "Temporary files are using significant space. Reduce tmp cleanup interval."
            )
        
        if usage['directories'].get('archived', {}).get('size_mb', 0) > 500:
            usage['recommendations'].append(
                "Archived models using >500 MB. Consider reducing keep_last for archived models."
            )
        
        return usage
