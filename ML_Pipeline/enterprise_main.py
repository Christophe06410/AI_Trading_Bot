#!/usr/bin/env python3
"""
Enterprise Main Entry Point - ROOT LEVEL

Unified interface for:
1. Original MLPipeline (backward compatible)
2. Enterprise features (1000x upgrade)
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enterprise.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_original_pipeline(args):
    """Run original MLPipeline (backward compatibility)"""
    print("\n🚀 Running ORIGINAL MLPipeline (Backward Compatible)")
    print("=" * 50)
    
    import subprocess
    
    # Build command for original main.py
    cmd = [sys.executable, "src/main.py"]
    
    if args.train:
        cmd.append("--train")
    if args.predict:
        cmd.append("--predict")
    if args.symbol:
        cmd.extend(["--symbol", args.symbol])
    if args.timeframe:
        cmd.extend(["--timeframe", args.timeframe])
    
    # Run original pipeline
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode

def run_enterprise_orchestrator(args):
    """Run enterprise orchestrator"""
    print("\n🚀 Running ENTERPRISE Orchestrator (1000x Upgrade)")
    print("=" * 50)
    
    try:
        # Import enterprise orchestrator
        from enterprise.orchestrator import main as orchestrator_main
        import asyncio
        
        # Prepare arguments for orchestrator
        orchestrator_args = []
        
        if args.train:
            orchestrator_args.append("--train")
        if args.deploy:
            orchestrator_args.append("--deploy")
        if args.monitor:
            orchestrator_args.append("--monitor")
        if args.continuous:
            orchestrator_args.append("--continuous")
        if args.list_models:
            orchestrator_args.append("--list-models")
        if args.cleanup:
            orchestrator_args.append("--cleanup")
        if args.symbol:
            orchestrator_args.extend(["--symbol", args.symbol])
        
        # Save original argv and replace
        original_argv = sys.argv
        sys.argv = ["enterprise_orchestrator"] + orchestrator_args
        
        try:
            # Run orchestrator
            print(f"Enterprise command: {' '.join(sys.argv)}")
            asyncio.run(orchestrator_main())
        finally:
            # Restore original argv
            sys.argv = original_argv
            
    except ImportError as e:
        print(f"❌ Cannot import enterprise modules: {e}")
        print("Did you run setup_enterprise.py?")
        return 1
    except Exception as e:
        logger.error(f"Enterprise orchestrator failed: {e}", exc_info=True)
        return 1
    
    return 0

def run_setup():
    """Run enterprise setup"""
    print("\n⚙️  Running Enterprise Setup")
    print("=" * 50)
    
    setup_script = current_dir / "setup_enterprise.py"
    if not setup_script.exists():
        print(f"❌ Setup script not found: {setup_script}")
        return 1
    
    import subprocess
    cmd = [sys.executable, str(setup_script)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode

def run_tests():
    """Run compatibility tests"""
    print("\n🧪 Running Compatibility Tests")
    print("=" * 50)
    
    test_script = current_dir / "test_backward_compatibility.py"
    if not test_script.exists():
        print(f"❌ Test script not found: {test_script}")
        return 1
    
    import subprocess
    cmd = [sys.executable, str(test_script)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode

def show_status():
    """Show system status"""
    print("\n📊 Enterprise System Status")
    print("=" * 50)
    
    from pathlib import Path
    
    # Check critical directories
    directories = [
        ("src/", "Original source code", True),
        ("enterprise/", "Enterprise modules", True),
        ("registry/", "Model registry", False),  # Will be created
        ("models/", "Backward compatibility", False),  # Will be created
        ("config/", "Configuration", True)
    ]
    
    for dir_path, description, required in directories:
        exists = Path(dir_path).exists()
        if required and not exists:
            status = "❌ MISSING"
        elif exists:
            status = "✅ EXISTS"
        else:
            status = "⚠️  WILL CREATE"
        print(f"{status:12} {dir_path:20} {description}")
    
    # Check enterprise modules
    print("\nEnterprise Modules:")
    enterprise_modules = [
        "orchestrator.py",
        "model_registry.py", 
        "online_learner.py"
    ]
    
    for module in enterprise_modules:
        module_path = Path("enterprise") / module
        if module_path.exists():
            print(f"  ✅ {module}")
        else:
            print(f"  ❌ {module} (missing)")
    
    # Check backward compatibility
    models_path = Path("models")
    if models_path.exists():
        # Count symlinks
        symlinks = [f for f in models_path.iterdir() if f.is_symlink()]
        regular_files = [f for f in models_path.iterdir() if f.is_file() and not f.is_symlink()]
        
        print(f"\n🔗 Backward Compatibility:")
        print(f"  Symlinks to registry: {len(symlinks)}")
        print(f"  Regular model files: {len(regular_files)}")
        
        if symlinks:
            print(f"  Example: {symlinks[0].name} -> registry/...")

def show_help():
    """Show help message"""
    help_text = """
    ENTERPRISE ML-PIPELINE v4.0
    ===========================
    
    1000x upgrade with 100% backward compatibility.
    
    USAGE:
    ------
    
    [BACKWARD COMPATIBLE] - Original pipeline:
    python enterprise_main.py original --train --symbol "SOL/USDT"
    
    [ENTERPRISE UPGRADE] - New features:
    python enterprise_main.py enterprise --train          # Train all symbols
    python enterprise_main.py enterprise --list-models    # List registered models
    python enterprise_main.py enterprise --deploy         # Deploy to production
    
    [SETUP & TESTING]:
    python enterprise_main.py setup                       # First-time setup
    python enterprise_main.py test                        # Verify compatibility
    python enterprise_main.py status                      # System status
    
    [ORIGINAL COMMANDS STILL WORK]:
    python src/main.py --train --symbol "SOL/USDT"       # Unchanged!
    
    AI-SERVICE & TRADING BOT:
    -------------------------
    ✅ NO CHANGES REQUIRED
    ✅ Automatically get better models
    ✅ Same API endpoints and formats
    """
    
    print(help_text)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enterprise ML-Pipeline v4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Original pipeline
    original_parser = subparsers.add_parser('original', 
        help='Run original MLPipeline (backward compatible)')
    original_parser.add_argument('--train', action='store_true', 
        help='Train models')
    original_parser.add_argument('--predict', action='store_true', 
        help='Make prediction')
    original_parser.add_argument('--symbol', default='SOL/USDT', 
        help='Trading symbol (default: SOL/USDT)')
    original_parser.add_argument('--timeframe', default='5m', 
        help='Candle timeframe (default: 5m)')
    
    # Enterprise orchestrator
    enterprise_parser = subparsers.add_parser('enterprise', 
        help='Run enterprise orchestrator (1000x upgrade)')
    enterprise_parser.add_argument('--train', action='store_true', 
        help='Train all symbols')
    enterprise_parser.add_argument('--deploy', action='store_true', 
        help='Deploy models to production')
    enterprise_parser.add_argument('--monitor', action='store_true', 
        help='Start performance monitoring')
    enterprise_parser.add_argument('--continuous', action='store_true', 
        help='Continuous improvement loop')
    enterprise_parser.add_argument('--list-models', action='store_true', 
        help='List all registered models')
    enterprise_parser.add_argument('--cleanup', action='store_true', 
        help='Cleanup old model versions')
    enterprise_parser.add_argument('--symbol', 
        help='Specific symbol (for single training)')
    
    # Other commands
    subparsers.add_parser('setup', help='Setup enterprise features')
    subparsers.add_parser('test', help='Test backward compatibility')
    subparsers.add_parser('status', help='Show system status')
    subparsers.add_parser('help', help='Show this help message')
    
    args = parser.parse_args()
    
    if not args.command or args.command == 'help':
        show_help()
        return 0
    
    try:
        if args.command == 'original':
            return run_original_pipeline(args)
        elif args.command == 'enterprise':
            return run_enterprise_orchestrator(args)
        elif args.command == 'setup':
            return run_setup()
        elif args.command == 'test':
            return run_tests()
        elif args.command == 'status':
            show_status()
            return 0
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled")
        return 1
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
