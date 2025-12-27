"""
Position management with SQLite database
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

from src.config import TradingBotConfig
from src.utils import get_logger

logger = get_logger(__name__)


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class Position:
    """Trading position"""
    id: str
    pair: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    entry_time: datetime
    size: float  # in quote currency (USD)
    stop_loss: float
    take_profit: Optional[float] = None
    current_stop_loss: Optional[float] = None  # For trailing stop
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    transaction_hash: Optional[str] = None
    close_transaction_hash: Optional[str] = None
    close_reason: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.status, str):
            self.status = PositionStatus(self.status)
        if isinstance(self.entry_time, str):
            self.entry_time = datetime.fromisoformat(self.entry_time)
        if isinstance(self.exit_time, str):
            self.exit_time = datetime.fromisoformat(self.exit_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        data['entry_time'] = self.entry_time.isoformat()
        if self.exit_time:
            data['exit_time'] = self.exit_time.isoformat()
        return data
    
    def update_pnl(self, current_price: float):
        """Update PnL based on current price"""
        if self.status != PositionStatus.OPEN:
            return
        
        if self.direction == "LONG":
            self.pnl = (current_price - self.entry_price) * self.size
            self.pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            self.pnl = (self.entry_price - current_price) * self.size
            self.pnl_percent = ((self.entry_price - current_price) / self.entry_price) * 100


class PositionManager:
    """Manages trading positions with database storage"""
    
    def __init__(self, config: TradingBotConfig):
        self.config = config
        self.db_path = config.database.connection_string.replace("sqlite:///", "")
        self.conn: Optional[sqlite3.Connection] = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    pair TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    size REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL,
                    current_stop_loss REAL,
                    exit_price REAL,
                    exit_time TEXT,
                    pnl REAL,
                    pnl_percent REAL,
                    status TEXT NOT NULL,
                    transaction_hash TEXT,
                    close_transaction_hash TEXT,
                    close_reason TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON positions(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pair ON positions(pair)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entry_time ON positions(entry_time)')
            
            self.conn.commit()
            logger.info("Database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_position(self, position: Position) -> bool:
        """Create a new position"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO positions (
                    id, pair, direction, entry_price, entry_time, size,
                    stop_loss, take_profit, current_stop_loss, status,
                    transaction_hash, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.id,
                position.pair,
                position.direction,
                position.entry_price,
                position.entry_time.isoformat(),
                position.size,
                position.stop_loss,
                position.take_profit,
                position.current_stop_loss,
                position.status.value,
                position.transaction_hash,
                json.dumps(position.metadata)
            ))
            
            self.conn.commit()
            logger.info(f"Position created: {position.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create position: {e}")
            return False
    
    def update_position(self, position: Position) -> bool:
        """Update an existing position"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                UPDATE positions SET
                    exit_price = ?,
                    exit_time = ?,
                    pnl = ?,
                    pnl_percent = ?,
                    status = ?,
                    close_transaction_hash = ?,
                    close_reason = ?,
                    current_stop_loss = ?,
                    metadata = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (
                position.exit_price,
                position.exit_time.isoformat() if position.exit_time else None,
                position.pnl,
                position.pnl_percent,
                position.status.value,
                position.close_transaction_hash,
                position.close_reason,
                position.current_stop_loss,
                json.dumps(position.metadata),
                position.id
            ))
            
            self.conn.commit()
            logger.debug(f"Position updated: {position.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update position: {e}")
            return False
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('SELECT * FROM positions WHERE id = ?', (position_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_position(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return None
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT * FROM positions 
                WHERE status = 'open' 
                ORDER BY entry_time DESC
            ''')
            
            rows = cursor.fetchall()
            return [self._row_to_position(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []
    
    def get_closed_positions(self, limit: int = 100) -> List[Position]:
        """Get recent closed positions"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT * FROM positions 
                WHERE status = 'closed' 
                ORDER BY exit_time DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            return [self._row_to_position(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get closed positions: {e}")
            return []
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        close_reason: str,
        transaction_hash: Optional[str] = None
    ) -> bool:
        """Close a position"""
        try:
            position = self.get_position(position_id)
            if not position:
                logger.error(f"Position not found: {position_id}")
                return False
            
            position.exit_price = exit_price
            position.exit_time = datetime.now()
            position.status = PositionStatus.CLOSED
            position.close_reason = close_reason
            position.close_transaction_hash = transaction_hash
            
            # Calculate PnL
            position.update_pnl(exit_price)
            
            return self.update_position(position)
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
    
    def update_trailing_stop(self, position_id: str, new_stop_price: float) -> bool:
        """Update trailing stop loss"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                UPDATE positions SET
                    current_stop_loss = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'open'
            ''', (new_stop_price, position_id))
            
            self.conn.commit()
            
            if cursor.rowcount > 0:
                logger.debug(f"Trailing stop updated: {position_id} -> {new_stop_price}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update trailing stop: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics"""
        try:
            cursor = self.conn.cursor()
            
            stats = {}
            
            # Total positions
            cursor.execute('SELECT COUNT(*) FROM positions')
            stats['total_positions'] = cursor.fetchone()[0]
            
            # Open positions
            cursor.execute('SELECT COUNT(*) FROM positions WHERE status = "open"')
            stats['open_positions'] = cursor.fetchone()[0]
            
            # Closed positions
            cursor.execute('SELECT COUNT(*) FROM positions WHERE status = "closed"')
            stats['closed_positions'] = cursor.fetchone()[0]
            
            # Total PnL
            cursor.execute('SELECT COALESCE(SUM(pnl), 0) FROM positions WHERE status = "closed"')
            stats['total_pnl'] = cursor.fetchone()[0] or 0
            
            # Win rate
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins
                FROM positions 
                WHERE status = "closed"
            ''')
            row = cursor.fetchone()
            if row and row[0] > 0:
                stats['win_rate'] = (row[1] / row[0]) * 100
            else:
                stats['win_rate'] = 0
            
            # Average PnL
            cursor.execute('''
                SELECT 
                    AVG(pnl) as avg_pnl,
                    AVG(pnl_percent) as avg_pnl_percent
                FROM positions 
                WHERE status = "closed"
            ''')
            row = cursor.fetchone()
            stats['avg_pnl'] = row[0] or 0
            stats['avg_pnl_percent'] = row[1] or 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def _row_to_position(self, row) -> Position:
        """Convert database row to Position object"""
        return Position(
            id=row[0],
            pair=row[1],
            direction=row[2],
            entry_price=row[3],
            entry_time=datetime.fromisoformat(row[4]),
            size=row[5],
            stop_loss=row[6],
            take_profit=row[7],
            current_stop_loss=row[8],
            exit_price=row[9],
            exit_time=datetime.fromisoformat(row[10]) if row[10] else None,
            pnl=row[11],
            pnl_percent=row[12],
            status=PositionStatus(row[13]),
            transaction_hash=row[14],
            close_transaction_hash=row[15],
            close_reason=row[16],
            metadata=json.loads(row[17]) if row[17] else {}
        )
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
