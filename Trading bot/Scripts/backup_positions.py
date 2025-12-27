#!/usr/bin/env python3
"""
Backup trading positions to JSON file
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.config import TradingBotConfig


class PositionBackup:
    """Backup positions from database"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = TradingBotConfig.load(config_path)
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
    
    def backup(self):
        """Create backup of all positions"""
        try:
            # Connect to database
            db_path = self.config.database.connection_string.replace("sqlite:///", "")
            
            if not Path(db_path).exists():
                print(f"❌ Database not found: {db_path}")
                return False
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all positions
            cursor.execute('SELECT * FROM positions ORDER BY entry_time DESC')
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            # Convert to dictionary
            positions = []
            for row in rows:
                position = dict(zip(columns, row))
                
                # Convert string timestamps
                for time_field in ['entry_time', 'exit_time', 'created_at', 'updated_at']:
                    if position.get(time_field):
                        position[time_field] = position[time_field]
                
                # Parse metadata JSON
                if position.get('metadata'):
                    try:
                        position['metadata'] = json.loads(position['metadata'])
                    except:
                        position['metadata'] = {}
                
                positions.append(position)
            
            conn.close()
            
            # Create backup file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"positions_backup_{timestamp}.json"
            
            # Save to JSON
            with open(backup_file, 'w') as f:
                json.dump({
                    "backup_date": datetime.now().isoformat(),
                    "positions_count": len(positions),
                    "positions": positions
                }, f, indent=2)
            
            print(f"✅ Backup created: {backup_file}")
            print(f"   Positions: {len(positions)}")
            
            # Create latest backup symlink
            latest_file = self.backup_dir / "positions_backup_latest.json"
            if latest_file.exists():
                latest_file.unlink()
            latest_file.symlink_to(backup_file)
            
            return True
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def restore(self, backup_file: str):
        """Restore positions from backup"""
        try:
            # Load backup
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            positions = backup_data.get("positions", [])
            if not positions:
                print("❌ No positions in backup file")
                return False
            
            # Connect to database
            db_path = self.config.database.connection_string.replace("sqlite:///", "")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Clear existing positions (optional)
            clear = input("Clear existing positions before restore? (y/n): ").lower() == 'y'
            if clear:
                cursor.execute('DELETE FROM positions')
                print("   Cleared existing positions")
            
            # Restore positions
            restored_count = 0
            for position in positions:
                try:
                    # Prepare data for insertion
                    columns = []
                    values = []
                    
                    for key, value in position.items():
                        if key == 'metadata' and isinstance(value, dict):
                            value = json.dumps(value)
                        columns.append(key)
                        values.append(value)
                    
                    # Create SQL
                    placeholders = ', '.join(['?' for _ in columns])
                    column_names = ', '.join(columns)
                    
                    cursor.execute(
                        f'INSERT OR REPLACE INTO positions ({column_names}) VALUES ({placeholders})',
                        values
                    )
                    
                    restored_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️  Failed to restore position {position.get('id')}: {e}")
            
            conn.commit()
            conn.close()
            
            print(f"✅ Restored {restored_count}/{len(positions)} positions")
            return True
            
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return False
    
    def list_backups(self):
        """List available backups"""
        backups = list(self.backup_dir.glob("positions_backup_*.json"))
        
        if not backups:
            print("No backups found")
            return
        
        print(f"Available backups ({len(backups)}):")
        for i, backup in enumerate(sorted(backups, reverse=True), 1):
            size_mb = backup.stat().st_size / 1024 / 1024
            print(f"  {i}. {backup.name} ({size_mb:.2f} MB)")
    
    def cleanup_old_backups(self, keep_days: int = 30):
        """Clean up old backups"""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        backups = list(self.backup_dir.glob("positions_backup_*.json"))
        
        deleted = 0
        for backup in backups:
            try:
                # Parse date from filename
                date_str = backup.stem.replace("positions_backup_", "")
                backup_date = datetime.strptime(date_str[:15], "%Y%m%d_%H%M%S")
                
                if backup_date < cutoff_date:
                    backup.unlink()
                    deleted += 1
                    print(f"   Deleted: {backup.name}")
            except:
                continue
        
        print(f"✅ Cleaned up {deleted} old backups")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backup and restore trading positions")
    parser.add_argument('action', choices=['backup', 'restore', 'list', 'cleanup'],
                       help='Action to perform')
    parser.add_argument('--file', help='Backup file for restore')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Configuration file')
    parser.add_argument('--keep-days', type=int, default=30,
                       help='Days to keep backups (cleanup only)')
    
    args = parser.parse_args()
    
    backup = PositionBackup(args.config)
    
    if args.action == 'backup':
        backup.backup()
    elif args.action == 'restore':
        if not args.file:
            print("❌ Please specify backup file with --file")
            return
        backup.restore(args.file)
    elif args.action == 'list':
        backup.list_backups()
    elif args.action == 'cleanup':
        backup.cleanup_old_backups(args.keep_days)


if __name__ == "__main__":
    main()
    main()
