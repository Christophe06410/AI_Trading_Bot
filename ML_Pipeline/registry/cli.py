"""
REGISTRY COMMAND LINE INTERFACE
Complete CLI for managing the model registry
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add enterprise to path
sys.path.append(str(Path(__file__).parent.parent))

from enterprise.model_registry import get_model_registry
from registry.registry_manager import get_registry_manager
from registry.performance_tracker import PerformanceTracker
from registry.sync_service import RegistrySyncService
from registry.cleanup_service import RegistryCleanupService

class RegistryCLI:
    """
    Command Line Interface for Registry Management
    """
    
    def __init__(self):
        self.registry = get_model_registry()
        self.manager = get_registry_manager()
        self.performance_tracker = PerformanceTracker(self.registry.registry_root)
        self.sync_service = RegistrySyncService()
        self.cleanup_service = RegistryCleanupService()
    
    def run(self):
        """Main CLI entry point"""
        parser = argparse.ArgumentParser(
            description='Enterprise Model Registry CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  registry-cli list --status production
  registry-cli info --symbol SOL_USDT
  registry-cli promote --symbol SOL_USDT --version 1.2.3
  registry-cli sync --verify
  registry-cli cleanup --dry-run
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Command to execute')
        
        # List command
        list_parser = subparsers.add_parser('list', help='List models in registry')
        list_parser.add_argument('--symbol', help='Filter by symbol')
        list_parser.add_argument('--type', help='Filter by model type')
        list_parser.add_argument('--status', help='Filter by status')
        list_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
        
        # Info command
        info_parser = subparsers.add_parser('info', help='Get detailed model information')
        info_parser.add_argument('--symbol', required=True, help='Trading symbol')
        info_parser.add_argument('--type', default='ensemble', help='Model type')
        info_parser.add_argument('--version', help='Specific version (default: latest)')
        
        # History command
        history_parser = subparsers.add_parser('history', help='Show model history')
        history_parser.add_argument('--symbol', required=True, help='Trading symbol')
        history_parser.add_argument('--type', default='ensemble', help='Model type')
        history_parser.add_argument('--days', type=int, default=30, help='Number of days')
        
        # Promote command
        promote_parser = subparsers.add_parser('promote', help='Promote model to production')
        promote_parser.add_argument('--symbol', required=True, help='Trading symbol')
        promote_parser.add_argument('--type', default='ensemble', help='Model type')
        promote_parser.add_argument('--version', help='Specific version to promote')
        promote_parser.add_argument('--reason', default='cli_promotion', help='Promotion reason')
        
        # Rollback command
        rollback_parser = subparsers.add_parser('rollback', help='Rollback to previous version')
        rollback_parser.add_argument('--symbol', required=True, help='Trading symbol')
        rollback_parser.add_argument('--type', default='ensemble', help='Model type')
        rollback_parser.add_argument('--version', help='Version to rollback to')
        rollback_parser.add_argument('--reason', default='cli_rollback', help='Rollback reason')
        
        # Sync command
        sync_parser = subparsers.add_parser('sync', help='Sync registry with AI-Service')
        sync_parser.add_argument('--symbol', help='Specific symbol to sync')
        sync_parser.add_argument('--type', help='Specific model type to sync')
        sync_parser.add_argument('--verify', action='store_true', help='Verify sync state')
        sync_parser.add_argument('--force', action='store_true', help='Force sync even if not needed')
        
        # Performance command
        perf_parser = subparsers.add_parser('performance', help='Performance analysis')
        perf_parser.add_argument('--symbol', required=True, help='Trading symbol')
        perf_parser.add_argument('--type', default='ensemble', help='Model type')
        perf_parser.add_argument('--days', type=int, default=30, help='Days to analyze')
        perf_parser.add_argument('--report', action='store_true', help='Generate detailed report')
        
        # Cleanup command
        cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup registry')
        cleanup_parser.add_argument('--dry-run', action='store_true', help='Show what would be cleaned')
        cleanup_parser.add_argument('--storage', action='store_true', help='Show storage usage')
        
        # Backup command
        backup_parser = subparsers.add_parser('backup', help='Backup registry')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Registry status')
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        # Execute command
        if args.command == 'list':
            self.command_list(args)
        elif args.command == 'info':
            self.command_info(args)
        elif args.command == 'history':
            self.command_history(args)
        elif args.command == 'promote':
            self.command_promote(args)
        elif args.command == 'rollback':
            self.command_rollback(args)
        elif args.command == 'sync':
            self.command_sync(args)
        elif args.command == 'performance':
            self.command_performance(args)
        elif args.command == 'cleanup':
            self.command_cleanup(args)
        elif args.command == 'backup':
            self.command_backup(args)
        elif args.command == 'status':
            self.command_status(args)
    
    def command_list(self, args):
        """List models command"""
        models = self.registry.list_models(args.symbol, args.type)
        
        if not models:
            print("No models found in registry")
            return
        
        # Filter by status if specified
        if args.status:
            models = [m for m in models if m['status'] == args.status]
        
        print(f"\n📋 Models in Registry ({len(models)} total):")
        print("=" * 120)
        
        for model in models:
            if args.detailed:
                self._print_model_detailed(model)
            else:
                self._print_model_summary(model)
        
        print("\nLegend: 🚀 Production | 🧪 Staging | 🔬 Experiment | 📦 Archived")
    
    def _print_model_summary(self, model: Dict[str, Any]):
        """Print model summary"""
        status_icon = {
            'production': '🚀',
            'staging': '🧪',
            'experiment': '🔬',
            'archived': '📦'
        }.get(model['status'], '❓')
        
        print(f"{status_icon} {model['symbol']}/{model['model_type']}")
        print(f"   Version: {model['version']}")
        print(f"   Accuracy: {model['accuracy']:.3f}")
        print(f"   Status: {model['status']}")
        print(f"   Trained: {model['trained_at'][:16]}")
        print(f"   Size: {model['model_size_mb']} MB")
        print("-" * 80)
    
    def _print_model_detailed(self, model: Dict[str, Any]):
        """Print detailed model information"""
        status_icon = {
            'production': '🚀',
            'staging': '🧪',
            'experiment': '🔬',
            'archived': '📦'
        }.get(model['status'], '❓')
        
        print(f"{status_icon} {model['symbol']}/{model['model_type']} v{model['version']}")
        print(f"   Status: {model['status']}")
        print(f"   Accuracy: {model['accuracy']:.3f}")
        print(f"   Trained: {model['trained_at']}")
        
        if 'deployed_at' in model and model['deployed_at']:
            print(f"   Deployed: {model['deployed_at']}")
        
        print(f"   Model format: {model.get('model_type', 'unknown')}")
        print(f"   Size: {model['model_size_mb']} MB")
        print(f"   Path: {model['model_path']}")
        print(f"   Features: {model.get('feature_count', 'unknown')}")
        
        if 'feature_names' in model and model['feature_names']:
            print(f"   Top features: {', '.join(model['feature_names'][:5])}")
            if len(model['feature_names']) > 5:
                print(f"   ... and {len(model['feature_names']) - 5} more")
        
        print("=" * 80)
    
    def command_info(self, args):
        """Get model information command"""
        # Get specific version or latest
        if args.version:
            # Get specific version
            history = self.registry.get_model_history(args.symbol, args.type)
            model_info = next(
                (m for m in history if m['version'] == args.version),
                None
            )
            
            if not model_info:
                print(f"❌ Version {args.version} not found for {args.symbol}/{args.type}")
                return
        else:
            # Get latest production model
            model_version = self.registry.get_production_model(args.symbol, args.type)
            if not model_version:
                print(f"❌ No production model found for {args.symbol}/{args.type}")
                return
            
            model_info = model_version.get_info()
        
        print(f"\n📊 Model Information:")
        print("=" * 80)
        
        for key, value in model_info.items():
            if isinstance(value, list):
                print(f"{key:20}: {', '.join(map(str, value[:5]))}")
                if len(value) > 5:
                    print(f"{'':20}  ... and {len(value) - 5} more")
            elif isinstance(value, dict):
                print(f"{key:20}:")
                for k, v in value.items():
                    print(f"{'':22}{k}: {v}")
            else:
                print(f"{key:20}: {value}")
    
    def command_history(self, args):
        """Show model history command"""
        history = self.registry.get_model_history(args.symbol, args.type)
        
        if not history:
            print(f"No history found for {args.symbol}/{args.type}")
            return
        
        # Filter by days if specified
        if args.days:
            cutoff_date = datetime.now() - timedelta(days=args.days)
            history = [
                h for h in history 
                if datetime.fromisoformat(h['trained_at']) >= cutoff_date
            ]
        
        print(f"\n📜 Model History for {args.symbol}/{args.type} ({len(history)} versions):")
        print("=" * 120)
        
        for i, model in enumerate(history):
            status_icon = {
                'production': '🚀',
                'staging': '🧪',
                'experiment': '🔬',
                'archived': '📦'
            }.get(model['status'], '❓')
            
            print(f"{i+1:3d}. {status_icon} v{model['version']}")
            print(f"     Accuracy: {model['accuracy']:.3f}")
            print(f"     Status: {model['status']}")
            print(f"     Trained: {model['trained_at'][:16]}")
            print(f"     Size: {model['model_size_mb']} MB")
            
            if model['status'] == 'production' and 'deployed_at' in model:
                print(f"     Deployed: {model['deployed_at'][:16]}")
            
            print()
    
    def command_promote(self, args):
        """Promote model command"""
        try:
            model_version = self.manager.promote_to_production(
                symbol=args.symbol,
                model_type=args.type,
                version=args.version,
                reason=args.reason
            )
            
            print(f"\n✅ Successfully promoted {args.symbol}/{args.type}")
            print(f"   Version: {model_version.metadata.version}")
            print(f"   Accuracy: {model_version.metadata.accuracy:.3f}")
            print(f"   Reason: {args.reason}")
            
        except Exception as e:
            print(f"❌ Promotion failed: {e}")
    
    def command_rollback(self, args):
        """Rollback model command"""
        try:
            model_version = self.manager.rollback_model(
                symbol=args.symbol,
                model_type=args.type,
                target_version=args.version,
                reason=args.reason
            )
            
            print(f"\n✅ Successfully rolled back {args.symbol}/{args.type}")
            print(f"   To version: {model_version.metadata.version}")
            print(f"   Accuracy: {model_version.metadata.accuracy:.3f}")
            print(f"   Reason: {args.reason}")
            
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
    
    def command_sync(self, args):
        """Sync registry command"""
        import asyncio
        
        if args.verify:
            # Verify sync state
            print("🔍 Verifying sync state...")
            
            async def verify():
                report = await self.sync_service.verify_sync_state()
                print(f"\n📋 Sync Verification Report:")
                print(f"   AI-Service reachable: {report['ai_service_reachable']}")
                print(f"   Total models checked: {report['summary']['total_models']}")
                print(f"   Models in sync: {report['summary']['in_sync']}")
                print(f"   Models out of sync: {report['summary']['out_of_sync']}")
                print(f"   Issues found: {report['summary']['issues_count']}")
                
                if report['issues_found']:
                    print(f"\n⚠️  Issues:")
                    for issue in report['issues_found'][:5]:  # Show first 5
                        print(f"   • {issue}")
                    if len(report['issues_found']) > 5:
                        print(f"   ... and {len(report['issues_found']) - 5} more")
            
            asyncio.run(verify())
        
        else:
            # Perform sync
            print("🔄 Syncing registry with AI-Service...")
            
            async def sync():
                if args.symbol:
                    # Sync specific symbol
                    if args.type:
                        await self.sync_service.sync_model(args.symbol, args.type, args.force)
                    else:
                        # Sync all model types for symbol
                        for model_type in self.registry.production_models.get(args.symbol, {}):
                            await self.sync_service.sync_model(args.symbol, model_type, args.force)
                else:
                    # Sync all symbols
                    await self.sync_service.sync_all_symbols()
                
                # Show sync status
                status = self.sync_service.get_sync_status()
                print(f"\n📊 Sync Status:")
                print(f"   Total syncs: {status['monitoring']['total_syncs']}")
                print(f"   Successful: {status['monitoring']['successful_syncs']}")
                print(f"   Failed: {status['monitoring']['failed_syncs']}")
                
                if status['monitoring']['last_error']:
                    print(f"   Last error: {status['monitoring']['last_error']}")
            
            asyncio.run(sync())
    
    def command_performance(self, args):
        """Performance analysis command"""
        if args.report:
            # Generate detailed report
            print(f"📊 Generating performance report for {args.symbol}/{args.type}...")
            
            report = self.performance_tracker.generate_performance_report(
                symbol=args.symbol,
                model_type=args.type,
                days=args.days
            )
            
            print(f"\n📈 Performance Report:")
            print("=" * 80)
            print(f"Symbol: {report['symbol']}")
            print(f"Model Type: {report['model_type']}")
            print(f"Period: {report['report_period_days']} days")
            print(f"Generated: {report['generated_at'][:16]}")
            
            summary = report['summary']
            print(f"\n📋 Summary:")
            print(f"   Accuracy: {summary['accuracy']:.3f}")
            print(f"   Total Trades: {summary['total_trades']}")
            print(f"   Win Rate: {summary['win_rate']:.1%}")
            print(f"   Total Profit: ${summary['total_profit']:.2f}")
            print(f"   Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
            
            if report['recommendations']:
                print(f"\n💡 Recommendations:")
                for rec in report['recommendations']:
                    print(f"   • {rec}")
            
            print(f"\n📁 Report saved to registry/performance/reports/")
        
        else:
            # Show performance metrics
            print(f"📊 Performance metrics for {args.symbol}/{args.type}...")
            
            metrics = self.performance_tracker.calculate_performance_metrics(
                symbol=args.symbol,
                model_type=args.type,
                start_date=datetime.now() - timedelta(days=args.days)
            )
            
            print(f"\n📈 Performance ({args.days} days):")
            print("=" * 80)
            
            if 'total_predictions' in metrics:
                print(f"Total Predictions: {metrics['total_predictions']}")
            
            if 'accuracy' in metrics:
                print(f"Accuracy: {metrics['accuracy']:.3f}")
            
            if 'trading' in metrics:
                trading = metrics['trading']
                print(f"\n📊 Trading Performance:")
                print(f"   Total Trades: {trading['total_trades']}")
                print(f"   Winning Trades: {trading['winning_trades']}")
                print(f"   Win Rate: {trading['win_rate']:.1%}")
                print(f"   Total Profit: ${trading['total_profit']:.2f}")
                print(f"   Avg Profit/Trade: ${trading['avg_profit_per_trade']:.2f}")
                print(f"   Sharpe Ratio: {trading['sharpe_ratio']:.2f}")
                print(f"   Max Drawdown: ${trading['max_drawdown']:.2f}")
    
    def command_cleanup(self, args):
        """Cleanup registry command"""
        if args.storage:
            # Show storage usage
            usage = self.cleanup_service.get_storage_usage()
            
            print(f"\n💾 Registry Storage Usage:")
            print("=" * 80)
            
            for dir_name, info in usage['directories'].items():
                print(f"{dir_name:15}: {info['size_mb']:8.1f} MB ({info['files_count']} files)")
            
            print(f"\nTotal: {usage['total_size_mb']:.1f} MB ({usage['total_size_mb']/1024:.1f} GB)")
            
            if usage['recommendations']:
                print(f"\n💡 Recommendations:")
                for rec in usage['recommendations']:
                    print(f"   • {rec}")
        
        else:
            # Run cleanup
            report = self.cleanup_service.run_cleanup(dry_run=args.dry_run)
            
            print(f"\n{'🔍 Dry Run Results:' if args.dry_run else '🧹 Cleanup Results:'}")
            print("=" * 80)
            print(f"Total items: {report['summary']['total_items']}")
            print(f"Space saved: {report['summary']['space_saved_mb']:.1f} MB")
            print(f"Policies applied: {', '.join(report['policies_applied'])}")
            
            if report['errors']:
                print(f"\n⚠️  Errors ({len(report['errors'])}):")
                for error in report['errors'][:3]:
                    print(f"   • {error}")
                if len(report['errors']) > 3:
                    print(f"   ... and {len(report['errors']) - 3} more")
    
    def command_backup(self, args):
        """Backup registry command"""
        print("💾 Creating registry backup...")
        
        backup_dir = self.manager.backup_registry()
        
        print(f"\n✅ Backup created:")
        print(f"   Location: {backup_dir}")
        
        # Show backup contents
        import shutil
        total_size = sum(
            f.stat().st_size for f in backup_dir.rglob('*') if f.is_file()
        )
        print(f"   Size: {total_size / (1024*1024):.1f} MB")
        
        # Count files
        file_count = sum(1 for _ in backup_dir.rglob('*') if _.is_file())
        print(f"   Files: {file_count}")
    
    def command_status(self, args):
        """Registry status command"""
        print(f"\n🏢 Enterprise Model Registry Status")
        print("=" * 80)
        
        # Basic info
        print(f"📁 Registry Root: {self.registry.registry_root}")
        print(f"📅 Initialized: {self.registry.registry_data.get('created_at', 'unknown')}")
        print(f"🔄 Last Updated: {self.registry.registry_data.get('updated_at', 'unknown')}")
        
        # Model counts
        total_models = sum(
            len(model_types) for model_types in self.registry.models.values()
        )
        print(f"\n📊 Model Statistics:")
        print(f"   Total Symbols: {len(self.registry.models)}")
        print(f"   Total Models: {total_models}")
        
        # Status breakdown
        status_counts = {}
        for symbol in self.registry.models:
            for model_type, versions in self.registry.models[symbol].items():
                for version in versions:
                    status = version.metadata.status
                    status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\n📈 Model Status:")
        for status, count in status_counts.items():
            status_icon = {
                'production': '🚀',
                'staging': '🧪',
                'experiment': '🔬',
                'archived': '📦'
            }.get(status, '❓')
            print(f"   {status_icon} {status}: {count}")
        
        # Production models
        print(f"\n🎯 Production Models:")
        for symbol in self.registry.production_models:
            for model_type, model_version in self.registry.production_models[symbol].items():
                print(f"   • {symbol}/{model_type}: v{model_version.metadata.version}")
        
        # Storage info
        usage = self.cleanup_service.get_storage_usage()
        print(f"\n💾 Storage:")
        print(f"   Total: {usage['total_size_mb']:.1f} MB")
        
        # Sync status
        sync_status = self.sync_service.get_sync_status()
        print(f"\n🔄 Sync Status:")
        print(f"   AI-Service: {self.sync_service.ai_service_url}")
        print(f"   Total Syncs: {sync_status['monitoring']['total_syncs']}")
        print(f"   Last Error: {sync_status['monitoring']['last_error'] or 'None'}")


def main():
    """Main entry point"""
    cli = RegistryCLI()
    cli.run()


if __name__ == "__main__":
    main()
