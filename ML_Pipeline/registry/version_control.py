"""
MODEL VERSION CONTROL SYSTEM
Git-like versioning for ML models with branching and merging
"""

import os
import sys
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from enterprise.model_registry import ModelMetadata, ModelVersion

class ModelVersionControl:
    """
    Git-like version control for ML models
    Supports branching, tagging, and model lineage
    """
    
    def __init__(self, registry_root: Path):
        self.registry_root = registry_root
        self.vcs_dir = registry_root / '.vcs'
        self.vcs_dir.mkdir(exist_ok=True)
        
        # Initialize if needed
        self._initialize_vcs()
    
    def _initialize_vcs(self):
        """Initialize version control system"""
        vcs_files = ['HEAD', 'config', 'index', 'refs']
        
        for file in vcs_files:
            file_path = self.vcs_dir / file
            if not file_path.exists():
                if file == 'HEAD':
                    file_path.write_text('ref: refs/heads/main\n')
                elif file == 'config':
                    config = {
                        'core': {
                            'repositoryformatversion': 0,
                            'filemode': False,
                            'bare': False
                        }
                    }
                    file_path.write_text(json.dumps(config, indent=2))
                else:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.touch()
    
    def create_branch(self, branch_name: str, from_version: str = None):
        """
        Create new branch for model experimentation
        """
        branches_dir = self.vcs_dir / 'refs' / 'heads'
        branches_dir.mkdir(parents=True, exist_ok=True)
        
        if from_version:
            # Create branch from specific version
            branch_content = from_version
        else:
            # Create from current HEAD
            head_content = self._read_head()
            branch_content = head_content
        
        branch_file = branches_dir / branch_name
        branch_file.write_text(branch_content)
        
        print(f"🌿 Created branch: {branch_name}")
        return branch_name
    
    def checkout_branch(self, branch_name: str):
        """Switch to branch"""
        branch_file = self.vcs_dir / 'refs' / 'heads' / branch_name
        
        if not branch_file.exists():
            raise ValueError(f"Branch {branch_name} does not exist")
        
        # Update HEAD
        head_file = self.vcs_dir / 'HEAD'
        head_file.write_text(f'ref: refs/heads/{branch_name}\n')
        
        print(f"✓ Checked out branch: {branch_name}")
        return branch_name
    
    def commit_model(
        self,
        model_version: ModelVersion,
        message: str,
        author: str = None,
        branch: str = None
    ) -> str:
        """
        Commit model version to VCS with message
        Returns commit hash
        """
        # Generate commit hash
        commit_data = {
            'model_version': model_version.metadata.version,
            'model_type': model_version.metadata.model_type,
            'symbol': model_version.metadata.symbol,
            'message': message,
            'author': author or os.getenv('USER', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'parent': self._get_current_commit(),
            'tree': self._create_tree_hash(model_version)
        }
        
        commit_hash = self._hash_object(commit_data)
        
        # Store commit
        commit_dir = self.vcs_dir / 'objects' / commit_hash[:2]
        commit_dir.mkdir(parents=True, exist_ok=True)
        
        commit_file = commit_dir / commit_hash[2:]
        commit_file.write_text(json.dumps(commit_data, indent=2))
        
        # Update branch reference
        if branch is None:
            branch = self._get_current_branch()
        
        branch_file = self.vcs_dir / 'refs' / 'heads' / branch
        branch_file.write_text(commit_hash)
        
        # Update HEAD if on this branch
        current_branch = self._get_current_branch()
        if current_branch == branch:
            head_file = self.vcs_dir / 'HEAD'
            head_file.write_text(f'ref: refs/heads/{branch}\n')
        
        print(f"💾 Committed model v{model_version.metadata.version}")
        print(f"   Commit: {commit_hash[:8]}")
        print(f"   Message: {message}")
        print(f"   Branch: {branch}")
        
        return commit_hash
    
    def tag_version(self, model_version: ModelVersion, tag_name: str, message: str = ""):
        """
        Create tag for model version
        """
        tags_dir = self.vcs_dir / 'refs' / 'tags'
        tags_dir.mkdir(parents=True, exist_ok=True)
        
        tag_data = {
            'model_version': model_version.metadata.version,
            'tag': tag_name,
            'message': message,
            'created_at': datetime.now().isoformat(),
            'tagger': os.getenv('USER', 'unknown')
        }
        
        tag_file = tags_dir / tag_name
        tag_file.write_text(json.dumps(tag_data, indent=2))
        
        print(f"🏷️  Created tag: {tag_name} for v{model_version.metadata.version}")
    
    def get_model_lineage(self, symbol: str, model_type: str) -> List[Dict[str, Any]]:
        """
        Get lineage/graph of model versions
        """
        # Find all commits for this model
        objects_dir = self.vcs_dir / 'objects'
        
        lineage = []
        
        if objects_dir.exists():
            for commit_dir in objects_dir.iterdir():
                if commit_dir.is_dir():
                    for commit_file in commit_dir.iterdir():
                        try:
                            with open(commit_file, 'r') as f:
                                commit_data = json.load(f)
                            
                            if (commit_data.get('symbol') == symbol and 
                                commit_data.get('model_type') == model_type):
                                
                                lineage.append({
                                    'commit': f"{commit_dir.name}{commit_file.name}",
                                    'model_version': commit_data['model_version'],
                                    'message': commit_data['message'],
                                    'timestamp': commit_data['timestamp'],
                                    'author': commit_data['author'],
                                    'parent': commit_data.get('parent')
                                })
                        except:
                            continue
        
        # Sort by timestamp
        lineage.sort(key=lambda x: x['timestamp'])
        
        return lineage
    
    def diff_models(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare differences between two model versions
        """
        # Get model metadata
        model1 = self._get_model_by_version(version1)
        model2 = self._get_model_by_version(version2)
        
        if not model1 or not model2:
            return {'error': 'One or both versions not found'}
        
        diff = {
            'versions': {
                'from': version1,
                'to': version2
            },
            'metadata_changes': {},
            'performance_changes': {},
            'feature_changes': {}
        }
        
        # Compare metadata
        md1 = model1.metadata
        md2 = model2.metadata
        
        # Compare accuracy
        if hasattr(md1, 'accuracy') and hasattr(md2, 'accuracy'):
            diff['performance_changes']['accuracy'] = {
                'from': md1.accuracy,
                'to': md2.accuracy,
                'delta': md2.accuracy - md1.accuracy
            }
        
        # Compare feature counts
        if hasattr(md1, 'feature_count') and hasattr(md2, 'feature_count'):
            diff['feature_changes']['count'] = {
                'from': md1.feature_count,
                'to': md2.feature_count,
                'delta': md2.feature_count - md1.feature_count
            }
        
        # Compare feature lists
        if hasattr(md1, 'feature_names') and hasattr(md2, 'feature_names'):
            features1 = set(md1.feature_names or [])
            features2 = set(md2.feature_names or [])
            
            diff['feature_changes']['added'] = list(features2 - features1)
            diff['feature_changes']['removed'] = list(features1 - features2)
            diff['feature_changes']['common'] = list(features1 & features2)
        
        return diff
    
    def _read_head(self) -> str:
        """Read current HEAD"""
        head_file = self.vcs_dir / 'HEAD'
        
        if not head_file.exists():
            return ''
        
        content = head_file.read_text().strip()
        if content.startswith('ref: '):
            # Follow reference
            ref_path = content[5:]
            ref_file = self.vcs_dir / ref_path
            
            if ref_file.exists():
                return ref_file.read_text().strip()
            else:
                return ''
        else:
            # Direct commit hash
            return content
    
    def _get_current_commit(self) -> Optional[str]:
        """Get current commit hash"""
        return self._read_head()
    
    def _get_current_branch(self) -> Optional[str]:
        """Get current branch name"""
        head_file = self.vcs_dir / 'HEAD'
        
        if not head_file.exists():
            return None
        
        content = head_file.read_text().strip()
        if content.startswith('ref: refs/heads/'):
            return content[16:]
        
        return None
    
    def _hash_object(self, data: Dict[str, Any]) -> str:
        """Generate hash for object"""
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha1(content.encode()).hexdigest()
    
    def _create_tree_hash(self, model_version: ModelVersion) -> str:
        """Create tree hash for model files"""
        model_dir = model_version.model_path.parent
        
        tree_items = []
        
        for file_path in model_dir.iterdir():
            if file_path.is_file():
                file_hash = hashlib.sha1(file_path.read_bytes()).hexdigest()
                tree_items.append({
                    'file': file_path.name,
                    'hash': file_hash,
                    'size': file_path.stat().st_size
                })
        
        tree_data = {
            'model_version': model_version.metadata.version,
            'files': tree_items,
            'timestamp': datetime.now().isoformat()
        }
        
        return self._hash_object(tree_data)
    
    def _get_model_by_version(self, version: str) -> Optional[ModelVersion]:
        """Find model by version number"""
        # This would need integration with the main registry
        # For now, return None - this is a stub
        return None


# File 6: `registry/ab_testing.py`
class ABTestingFramework:
    """
    A/B Testing Framework for ML Models
    Test multiple model versions in production
    """
    
    def __init__(self, registry_root: Path):
        self.registry_root = registry_root
        self.experiments_dir = registry_root / 'experiments'
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Load existing experiments
        self.experiments = self._load_experiments()
    
    def _load_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Load existing experiments"""
        experiments = {}
        
        for exp_file in self.experiments_dir.glob('*.json'):
            try:
                with open(exp_file, 'r') as f:
                    exp_data = json.load(f)
                experiments[exp_data['experiment_id']] = exp_data
            except:
                continue
        
        return experiments
    
    def create_experiment(
        self,
        symbol: str,
        model_type: str,
        treatment_version: str,
        experiment_name: str,
        traffic_percentage: float = 0.1,
        duration_hours: int = 24,
        metrics: List[str] = None
    ) -> str:
        """
        Create A/B test experiment
        Returns experiment ID
        """
        from datetime import datetime, timedelta
        
        experiment_id = hashlib.md5(
            f"{symbol}_{model_type}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        experiment = {
            'experiment_id': experiment_id,
            'name': experiment_name,
            'symbol': symbol,
            'model_type': model_type,
            'treatment_version': treatment_version,
            'traffic_percentage': traffic_percentage,
            'start_time': datetime.now().isoformat(),
            'planned_end_time': (datetime.now() + timedelta(hours=duration_hours)).isoformat(),
            'actual_end_time': None,
            'status': 'running',
            'metrics': metrics or ['accuracy', 'profit', 'sharpe_ratio'],
            'results': {},
            'participants': 0,
            'decisions': {'treatment': 0, 'control': 0}
        }
        
        # Save experiment
        exp_file = self.experiments_dir / f"{experiment_id}.json"
        with open(exp_file, 'w') as f:
            json.dump(experiment, f, indent=2)
        
        # Add to in-memory cache
        self.experiments[experiment_id] = experiment
        
        print(f"🔬 Created experiment: {experiment_name}")
        print(f"   ID: {experiment_id}")
        print(f"   Treatment: {treatment_version}")
        print(f"   Traffic: {traffic_percentage:.1%}")
        print(f"   Duration: {duration_hours} hours")
        
        return experiment_id
    
    def get_assignment(
        self,
        experiment_id: str,
        user_id: str = None
    ) -> Tuple[str, str]:
        """
        Get model assignment for a user/request
        Returns (assignment, version)
        """
        if experiment_id not in self.experiments:
            return 'control', None
        
        experiment = self.experiments[experiment_id]
        
        if experiment['status'] != 'running':
            return 'control', None
        
        # Simple hash-based assignment
        if user_id:
            user_hash = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
            assignment_prob = user_hash % 100 / 100.0
        else:
            # Random assignment
            import random
            assignment_prob = random.random()
        
        # Check if user gets treatment
        if assignment_prob < experiment['traffic_percentage']:
            assignment = 'treatment'
            version = experiment['treatment_version']
            experiment['decisions']['treatment'] += 1
        else:
            assignment = 'control'
            version = None  # Use current production
            experiment['decisions']['control'] += 1
        
        experiment['participants'] += 1
        
        # Update experiment file
        self._save_experiment(experiment)
        
        return assignment, version
    
    def record_result(
        self,
        experiment_id: str,
        assignment: str,
        metrics: Dict[str, float]
    ):
        """
        Record experiment result
        """
        if experiment_id not in self.experiments:
            return
        
        experiment = self.experiments[experiment_id]
        
        # Initialize results if needed
        if 'results' not in experiment:
            experiment['results'] = {}
        
        # Record metrics
        for metric_name, metric_value in metrics.items():
            if metric_name not in experiment['results']:
                experiment['results'][metric_name] = {
                    'treatment': {'values': [], 'mean': 0.0},
                    'control': {'values': [], 'mean': 0.0}
                }
            
            experiment['results'][metric_name][assignment]['values'].append(metric_value)
            
            # Update mean
            values = experiment['results'][metric_name][assignment]['values']
            experiment['results'][metric_name][assignment]['mean'] = sum(values) / len(values)
        
        # Save updated experiment
        self._save_experiment(experiment)
    
    def _save_experiment(self, experiment: Dict[str, Any]):
        """Save experiment to file"""
        exp_file = self.experiments_dir / f"{experiment['experiment_id']}.json"
        with open(exp_file, 'w') as f:
            json.dump(experiment, f, indent=2)
    
    def end_experiment(self, experiment_id: str, promote_winner: bool = True):
        """
        End experiment and optionally promote winning version
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment['status'] != 'running':
            print(f"⚠️  Experiment {experiment_id} is not running")
            return
        
        # Update status
        experiment['status'] = 'ended'
        experiment['actual_end_time'] = datetime.now().isoformat()
        
        # Calculate statistical significance
        results = self._analyze_results(experiment)
        
        # Determine winner
        winner = self._determine_winner(results)
        
        experiment['analysis'] = results
        experiment['winner'] = winner
        
        # Save final state
        self._save_experiment(experiment)
        
        print(f"🏁 Experiment ended: {experiment['name']}")
        print(f"   Participants: {experiment['participants']}")
        print(f"   Treatment decisions: {experiment['decisions']['treatment']}")
        print(f"   Control decisions: {experiment['decisions']['control']}")
        
        if winner and winner != 'none':
            print(f"   Winner: {winner}")
            
            if promote_winner and winner == 'treatment':
                print(f"   Promoting treatment version to production...")
                # This would call the registry promotion function
                # For now, just print
                print(f"   Would promote {experiment['treatment_version']} to production")
        else:
            print(f"   No clear winner found")
        
        return experiment
    
    def _analyze_results(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results"""
        import numpy as np
        from scipy import stats
        
        analysis = {
            'metrics': {},
            'confidence_intervals': {},
            'p_values': {},
            'effect_sizes': {}
        }
        
        for metric_name, metric_data in experiment.get('results', {}).items():
            treatment_values = metric_data['treatment']['values']
            control_values = metric_data['control']['values']
            
            if len(treatment_values) > 5 and len(control_values) > 5:
                # Calculate statistics
                treatment_mean = np.mean(treatment_values)
                control_mean = np.mean(control_values)
                
                # T-test for statistical significance
                t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
                
                # Effect size
                pooled_std = np.sqrt(
                    (np.var(treatment_values) + np.var(control_values)) / 2
                )
                effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
                
                analysis['metrics'][metric_name] = {
                    'treatment_mean': float(treatment_mean),
                    'control_mean': float(control_mean),
                    'difference': float(treatment_mean - control_mean),
                    'p_value': float(p_value),
                    'effect_size': float(effect_size),
                    'significant': p_value < 0.05
                }
        
        return analysis
    
    def _determine_winner(self, analysis: Dict[str, Any]) -> str:
        """Determine experiment winner"""
        if not analysis.get('metrics'):
            return 'none'
        
        # Check primary metrics (accuracy, profit, etc.)
        primary_metrics = ['accuracy', 'profit', 'sharpe_ratio']
        
        for metric in primary_metrics:
            if metric in analysis['metrics']:
                metric_data = analysis['metrics'][metric]
                
                if metric_data['significant'] and metric_data['difference'] > 0:
                    return 'treatment'
                elif metric_data['significant'] and metric_data['difference'] < 0:
                    return 'control'
        
        # No significant difference
        return 'none'
