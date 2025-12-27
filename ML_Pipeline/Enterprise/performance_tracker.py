"""
ENTERPRISE PERFORMANCE TRACKER v1.0
Tracks model performance, trading outcomes, and business metrics
Provides comprehensive analytics for decision making
"""
# ML-Pipeline/enterprise/performance_tracker.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TradeOutcome(Enum):
    """Outcome of a trade"""
    WIN = "win"
    LOSS = "loss"
    BREAK_EVEN = "break_even"
    CANCELLED = "cancelled"

class PerformanceMetric(Enum):
    """Performance metrics to track"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    AVERAGE_WIN = "average_win"
    AVERAGE_LOSS = "average_loss"
    RISK_REWARD_RATIO = "risk_reward_ratio"

@dataclass
class TradeRecord:
    """Record of a completed trade"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_type: str  # "LONG" or "SHORT"
    size: float
    pnl: float
    pnl_percentage: float
    duration_minutes: float
    prediction_id: Optional[str] = None
    model_used: Optional[str] = None
    confidence: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    fees: float = 0.0
    notes: Optional[str] = None
    
    @property
    def outcome(self) -> TradeOutcome:
        """Determine trade outcome"""
        if abs(self.pnl) < 0.01:  # Less than 1 cent
            return TradeOutcome.BREAK_EVEN
        elif self.pnl > 0:
            return TradeOutcome.WIN
        else:
            return TradeOutcome.LOSS
    
    @property
    def is_win(self) -> bool:
        """Check if trade was winning"""
        return self.outcome == TradeOutcome.WIN
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk-reward ratio"""
        if self.stop_loss is None or self.take_profit is None:
            return 0.0
        
        if self.position_type == "LONG":
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:  # SHORT
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        
        if risk == 0:
            return 0.0
        
        return reward / risk

@dataclass
class PredictionRecord:
    """Record of a model prediction"""
    prediction_id: str
    symbol: str
    timestamp: datetime
    model_type: str
    recommendation: str  # "LONG", "SHORT", "WAIT", "CLOSE"
    confidence: float
    features_used: List[str]
    trade_id: Optional[str] = None  # If prediction led to trade
    actual_outcome: Optional[str] = None  # What actually happened
    correct: Optional[bool] = None  # Whether prediction was correct

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    period_start: datetime
    period_end: datetime
    symbol: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    average_win: float
    average_loss: float
    best_trade: Optional[TradeRecord]
    worst_trade: Optional[TradeRecord]
    model_performance: Dict[str, Dict[str, float]]  # model_type -> metrics
    daily_performance: List[Dict[str, Any]]

class EnterprisePerformanceTracker:
    """
    Enterprise Performance Tracker
    
    Tracks:
    1. Model prediction performance
    2. Trading outcomes and PnL
    3. Risk metrics (Sharpe, drawdown, etc.)
    4. Model comparison and benchmarking
    5. Real-time performance dashboards
    
    Maintains backward compatibility with existing trading data
    """
    
    def __init__(self, config: Dict[str, Any], model_registry):
        self.config = config
        self.registry = model_registry
        
        # Storage
        self.trades: Dict[str, TradeRecord] = {}  # trade_id -> TradeRecord
        self.predictions: Dict[str, PredictionRecord] = {}  # prediction_id -> PredictionRecord
        
        # Performance cache
        self.performance_cache = {}
        self.reports_dir = Path("monitoring/performance/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self._initialize_metrics()
        
        logger.info("Enterprise Performance Tracker initialized")
    
    def _initialize_metrics(self):
        """Initialize performance metrics tracking"""
        self.metrics = {
            'daily': {
                'trades': [],
                'pnl': [],
                'accuracy': []
            },
            'weekly': {
                'trades': [],
                'pnl': [],
                'accuracy': []
            },
            'monthly': {
                'trades': [],
                'pnl': [],
                'accuracy': []
            }
        }
    
    async def record_prediction(self, prediction: PredictionRecord):
        """Record a model prediction"""
        self.predictions[prediction.prediction_id] = prediction
        
        # Keep predictions bounded (optional - in production would use database)
        if len(self.predictions) > 10000:
            # Remove oldest predictions
            oldest_ids = sorted(
                self.predictions.keys(),
                key=lambda k: self.predictions[k].timestamp
            )[:1000]
            for pred_id in oldest_ids:
                del self.predictions[pred_id]
        
        logger.debug(f"Recorded prediction {prediction.prediction_id} for {prediction.symbol}")
    
    async def record_trade(self, trade: TradeRecord):
        """Record a completed trade"""
        self.trades[trade.trade_id] = trade
        
        # Link prediction if available
        if trade.prediction_id and trade.prediction_id in self.predictions:
            self.predictions[trade.prediction_id].trade_id = trade.trade_id
            self.predictions[trade.prediction_id].actual_outcome = trade.outcome.value
            self.predictions[trade.prediction_id].correct = (
                (trade.position_type == "LONG" and trade.pnl > 0) or
                (trade.position_type == "SHORT" and trade.pnl > 0)
            )
        
        # Update performance metrics
        await self._update_performance_metrics(trade)
        
        # Generate trade analysis
        analysis = await self._analyze_trade(trade)
        
        logger.info(
            f"Recorded trade {trade.trade_id}: {trade.symbol} {trade.position_type} "
            f"PNL: ${trade.pnl:.2f} ({trade.outcome.value})"
        )
        
        return analysis
    
    async def _update_performance_metrics(self, trade: TradeRecord):
        """Update performance metrics with new trade"""
        today = datetime.now().date()
        
        # Update daily metrics
        if not self.metrics['daily']['trades'] or self.metrics['daily']['trades'][-1]['date'] != today:
            self.metrics['daily']['trades'].append({
                'date': today,
                'count': 1,
                'wins': 1 if trade.is_win else 0,
                'losses': 0 if trade.is_win else 1,
                'pnl': trade.pnl
            })
        else:
            last_day = self.metrics['daily']['trades'][-1]
            last_day['count'] += 1
            last_day['wins'] += 1 if trade.is_win else 0
            last_day['losses'] += 0 if trade.is_win else 1
            last_day['pnl'] += trade.pnl
        
        # Keep only last 90 days
        if len(self.metrics['daily']['trades']) > 90:
            self.metrics['daily']['trades'] = self.metrics['daily']['trades'][-90:]
    
    async def _analyze_trade(self, trade: TradeRecord) -> Dict[str, Any]:
        """Analyze a single trade for insights"""
        analysis = {
            'trade_id': trade.trade_id,
            'symbol': trade.symbol,
            'outcome': trade.outcome.value,
            'pnl': trade.pnl,
            'pnl_percentage': trade.pnl_percentage,
            'duration_minutes': trade.duration_minutes,
            'risk_reward_ratio': trade.risk_reward_ratio,
            'insights': []
        }
        
        # Generate insights
        if trade.pnl_percentage > 0.05:  # > 5% gain
            analysis['insights'].append(
                f"Excellent trade! {trade.pnl_percentage:.1%} gain in {trade.duration_minutes:.0f} minutes"
            )
        elif trade.pnl_percentage < -0.03:  # > 3% loss
            analysis['insights'].append(
                f"Significant loss: {trade.pnl_percentage:.1%}. Review stop-loss settings."
            )
        
        if trade.duration_minutes < 5:
            analysis['insights'].append("Very short-term trade - consider if aligns with strategy")
        elif trade.duration_minutes > 240:  # 4 hours
            analysis['insights'].append("Long holding period - ensure adequate position sizing")
        
        # Compare to prediction if available
        if trade.prediction_id and trade.prediction_id in self.predictions:
            pred = self.predictions[trade.prediction_id]
            if pred.correct is not None:
                if pred.correct:
                    analysis['insights'].append("Model prediction was correct")
                else:
                    analysis['insights'].append("Model prediction was incorrect")
        
        return analysis
    
    async def calculate_model_performance(self, symbol: Optional[str] = None,
                                        start_date: Optional[datetime] = None,
                                        end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Calculate performance metrics for models"""
        
        # Filter predictions
        filtered_predictions = self._filter_predictions(symbol, start_date, end_date)
        
        if not filtered_predictions:
            return {
                'total_predictions': 0,
                'message': 'No predictions in specified period'
            }
        
        # Group by model type
        model_groups = {}
        for pred in filtered_predictions:
            model_type = pred.model_type
            if model_type not in model_groups:
                model_groups[model_type] = []
            model_groups[model_type].append(pred)
        
        # Calculate metrics per model
        model_performance = {}
        for model_type, predictions in model_groups.items():
            # Filter predictions with known outcomes
            evaluated = [p for p in predictions if p.correct is not None]
            
            if not evaluated:
                continue
            
            # Calculate metrics
            correct = sum(1 for p in evaluated if p.correct)
            total = len(evaluated)
            
            # Get confidence scores
            confidences = [p.confidence for p in evaluated]
            correct_confidences = [p.confidence for p in evaluated if p.correct]
            
            model_performance[model_type] = {
                'total_predictions': len(predictions),
                'evaluated_predictions': total,
                'accuracy': correct / total if total > 0 else 0.0,
                'average_confidence': np.mean(confidences) if confidences else 0.0,
                'correct_confidence': np.mean(correct_confidences) if correct_confidences else 0.0,
                'prediction_distribution': self._count_predictions_by_type(predictions)
            }
        
        # Overall performance
        all_evaluated = [p for p in filtered_predictions if p.correct is not None]
        if all_evaluated:
            overall_accuracy = sum(1 for p in all_evaluated if p.correct) / len(all_evaluated)
        else:
            overall_accuracy = 0.0
        
        return {
            'period': {
                'start': start_date.isoformat() if start_date else 'beginning',
                'end': end_date.isoformat() if end_date else 'now'
            },
            'symbol': symbol or 'all',
            'total_predictions': len(filtered_predictions),
            'evaluated_predictions': len(all_evaluated),
            'overall_accuracy': overall_accuracy,
            'model_performance': model_performance,
            'prediction_volume_by_day': self._calculate_prediction_volume(filtered_predictions)
        }
    
    async def calculate_trading_performance(self, symbol: Optional[str] = None,
                                          start_date: Optional[datetime] = None,
                                          end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Calculate trading performance metrics"""
        
        # Filter trades
        filtered_trades = self._filter_trades(symbol, start_date, end_date)
        
        if not filtered_trades:
            return {
                'total_trades': 0,
                'message': 'No trades in specified period'
            }
        
        # Calculate basic metrics
        winning_trades = [t for t in filtered_trades if t.is_win]
        losing_trades = [t for t in filtered_trades if not t.is_win]
        
        total_trades = len(filtered_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        # PnL metrics
        total_pnl = sum(t.pnl for t in filtered_trades)
        average_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        
        average_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        average_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(filtered_trades)
        max_drawdown = self._calculate_max_drawdown(filtered_trades)
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Find best and worst trades
        best_trade = max(filtered_trades, key=lambda t: t.pnl) if filtered_trades else None
        worst_trade = min(filtered_trades, key=lambda t: t.pnl) if filtered_trades else None
        
        # Daily performance
        daily_performance = self._calculate_daily_performance(filtered_trades)
        
        return {
            'period': {
                'start': start_date.isoformat() if start_date else 'beginning',
                'end': end_date.isoformat() if end_date else 'now'
            },
            'symbol': symbol or 'all',
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'average_pnl': average_pnl,
            'average_win': average_win,
            'average_loss': average_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'best_trade': {
                'id': best_trade.trade_id,
                'pnl': best_trade.pnl,
                'date': best_trade.exit_time.date().isoformat()
            } if best_trade else None,
            'worst_trade': {
                'id': worst_trade.trade_id,
                'pnl': worst_trade.pnl,
                'date': worst_trade.exit_time.date().isoformat()
            } if worst_trade else None,
            'daily_performance': daily_performance
        }
    
    async def generate_performance_report(self, symbol: str,
                                        period: str = 'monthly') -> PerformanceReport:
        """Generate comprehensive performance report"""
        
        # Determine date range
        end_date = datetime.now()
        if period == 'daily':
            start_date = end_date - timedelta(days=1)
        elif period == 'weekly':
            start_date = end_date - timedelta(weeks=1)
        elif period == 'monthly':
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=7)  # Default to weekly
        
        # Get trading performance
        trading_perf = await self.calculate_trading_performance(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get model performance
        model_perf = await self.calculate_model_performance(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get best and worst trades
        filtered_trades = self._filter_trades(symbol, start_date, end_date)
        best_trade = max(filtered_trades, key=lambda t: t.pnl) if filtered_trades else None
        worst_trade = min(filtered_trades, key=lambda t: t.pnl) if filtered_trades else None
        
        # Create report
        report = PerformanceReport(
            period_start=start_date,
            period_end=end_date,
            symbol=symbol,
            total_trades=trading_perf['total_trades'],
            winning_trades=trading_perf['winning_trades'],
            losing_trades=trading_perf['losing_trades'],
            win_rate=trading_perf['win_rate'],
            total_pnl=trading_perf['total_pnl'],
            average_pnl=trading_perf['average_pnl'],
            sharpe_ratio=trading_perf['sharpe_ratio'],
            max_drawdown=trading_perf['max_drawdown'],
            profit_factor=trading_perf['profit_factor'],
            average_win=trading_perf['average_win'],
            average_loss=trading_perf['average_loss'],
            best_trade=best_trade,
            worst_trade=worst_trade,
            model_performance=model_perf.get('model_performance', {}),
            daily_performance=trading_perf.get('daily_performance', [])
        )
        
        # Save report
        await self._save_performance_report(report, period)
        
        return report
    
    def _filter_predictions(self, symbol: Optional[str], 
                          start_date: Optional[datetime],
                          end_date: Optional[datetime]) -> List[PredictionRecord]:
        """Filter predictions based on criteria"""
        filtered = []
        
        for pred in self.predictions.values():
            # Filter by symbol
            if symbol and pred.symbol != symbol:
                continue
            
            # Filter by date
            if start_date and pred.timestamp < start_date:
                continue
            if end_date and pred.timestamp > end_date:
                continue
            
            filtered.append(pred)
        
        return filtered
    
    def _filter_trades(self, symbol: Optional[str],
                     start_date: Optional[datetime],
                     end_date: Optional[datetime]) -> List[TradeRecord]:
        """Filter trades based on criteria"""
        filtered = []
        
        for trade in self.trades.values():
            # Filter by symbol
            if symbol and trade.symbol != symbol:
                continue
            
            # Filter by date
            if start_date and trade.exit_time < start_date:
                continue
            if end_date and trade.exit_time > end_date:
                continue
            
            filtered.append(trade)
        
        return filtered
    
    def _count_predictions_by_type(self, predictions: List[PredictionRecord]) -> Dict[str, int]:
        """Count predictions by recommendation type"""
        counts = {'LONG': 0, 'SHORT': 0, 'WAIT': 0, 'CLOSE': 0}
        
        for pred in predictions:
            if pred.recommendation in counts:
                counts[pred.recommendation] += 1
        
        return counts
    
    def _calculate_prediction_volume(self, predictions: List[PredictionRecord]) -> List[Dict[str, Any]]:
        """Calculate prediction volume by day"""
        if not predictions:
            return []
        
        # Group by date
        by_date = {}
        for pred in predictions:
            date_str = pred.timestamp.date().isoformat()
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append(pred)
        
        # Create volume data
        volume_data = []
        for date_str, date_predictions in sorted(by_date.items()):
            counts = self._count_predictions_by_type(date_predictions)
            volume_data.append({
                'date': date_str,
                'total': len(date_predictions),
                **counts
            })
        
        return volume_data
    
    def _calculate_sharpe_ratio(self, trades: List[TradeRecord], 
                              risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from trades"""
        if not trades or len(trades) < 2:
            return 0.0
        
        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        
        # Calculate returns
        returns = [t.pnl_percentage for t in sorted_trades]
        
        # Calculate Sharpe ratio
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming daily trades)
        sharpe = (avg_return - risk_free_rate/252) / std_return * np.sqrt(252)
        
        return float(sharpe)
    
    def _calculate_max_drawdown(self, trades: List[TradeRecord]) -> float:
        """Calculate maximum drawdown from trades"""
        if not trades:
            return 0.0
        
        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        
        # Calculate cumulative PnL
        cumulative_pnl = []
        current_pnl = 0.0
        
        for trade in sorted_trades:
            current_pnl += trade.pnl
            cumulative_pnl.append(current_pnl)
        
        if not cumulative_pnl:
            return 0.0
        
        # Calculate drawdown
        peak = cumulative_pnl[0]
        max_drawdown = 0.0
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = (peak - pnl) / (abs(peak) + 1e-10)  # Avoid division by zero
            max_drawdown = max(max_drawdown, drawdown)
        
        return float(max_drawdown)
    
    def _calculate_daily_performance(self, trades: List[TradeRecord]) -> List[Dict[str, Any]]:
        """Calculate daily performance metrics"""
        if not trades:
            return []
        
        # Group trades by date
        by_date = {}
        for trade in trades:
            date_str = trade.exit_time.date().isoformat()
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append(trade)
        
        # Calculate daily metrics
        daily_performance = []
        
        for date_str, date_trades in sorted(by_date.items()):
            winning = [t for t in date_trades if t.is_win]
            losing = [t for t in date_trades if not t.is_win]
            
            total_pnl = sum(t.pnl for t in date_trades)
            
            daily_performance.append({
                'date': date_str,
                'total_trades': len(date_trades),
                'winning_trades': len(winning),
                'losing_trades': len(losing),
                'win_rate': len(winning) / len(date_trades) if date_trades else 0.0,
                'total_pnl': total_pnl,
                'average_pnl': total_pnl / len(date_trades) if date_trades else 0.0,
                'best_trade_pnl': max(t.pnl for t in date_trades) if date_trades else 0.0,
                'worst_trade_pnl': min(t.pnl for t in date_trades) if date_trades else 0.0
            })
        
        return daily_performance
    
    async def _save_performance_report(self, report: PerformanceReport, period: str):
        """Save performance report to file"""
        # Convert report to dictionary
        report_dict = {
            'period_start': report.period_start.isoformat(),
            'period_end': report.period_end.isoformat(),
            'symbol': report.symbol,
            'total_trades': report.total_trades,
            'winning_trades': report.winning_trades,
            'losing_trades': report.losing_trades,
            'win_rate': report.win_rate,
            'total_pnl': report.total_pnl,
            'average_pnl': report.average_pnl,
            'sharpe_ratio': report.sharpe_ratio,
            'max_drawdown': report.max_drawdown,
            'profit_factor': report.profit_factor,
            'average_win': report.average_win,
            'average_loss': report.average_loss,
            'best_trade': {
                'id': report.best_trade.trade_id if report.best_trade else None,
                'pnl': report.best_trade.pnl if report.best_trade else 0.0
            },
            'worst_trade': {
                'id': report.worst_trade.trade_id if report.worst_trade else None,
                'pnl': report.worst_trade.pnl if report.worst_trade else 0.0
            },
            'model_performance': report.model_performance,
            'daily_performance': report.daily_performance,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save to file
        filename = f"performance_{report.symbol}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.reports_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Saved performance report: {filepath}")
    
    async def get_real_time_metrics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get real-time performance metrics"""
        # Last 24 hours
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        # Get recent trades
        recent_trades = self._filter_trades(symbol, start_time, end_time)
        
        # Get recent predictions
        recent_predictions = self._filter_predictions(symbol, start_time, end_time)
        
        # Calculate metrics
        if recent_trades:
            winning = sum(1 for t in recent_trades if t.is_win)
            total_pnl = sum(t.pnl for t in recent_trades)
            win_rate = winning / len(recent_trades)
        else:
            winning = 0
            total_pnl = 0.0
            win_rate = 0.0
        
        if recent_predictions:
            evaluated = [p for p in recent_predictions if p.correct is not None]
            if evaluated:
                accuracy = sum(1 for p in evaluated if p.correct) / len(evaluated)
            else:
                accuracy = 0.0
        else:
            accuracy = 0.0
        
        return {
            'timestamp': end_time.isoformat(),
            'symbol': symbol or 'all',
            'last_24_hours': {
                'trades': len(recent_trades),
                'winning_trades': winning,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'predictions': len(recent_predictions),
                'model_accuracy': accuracy
            },
            'active_alerts': len(self._get_active_alerts(symbol))
        }
    
    def _get_active_alerts(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active performance alerts (simplified)"""
        # In production, this would check various conditions
        alerts = []
        
        # Example alert: Low win rate in last 24 hours
        recent_trades = self._filter_trades(
            symbol, 
            datetime.now() - timedelta(hours=24),
            datetime.now()
        )
        
        if recent_trades and len(recent_trades) >= 5:
            win_rate = sum(1 for t in recent_trades if t.is_win) / len(recent_trades)
            if win_rate < 0.3:
                alerts.append({
                    'type': 'low_win_rate',
                    'severity': 'warning',
                    'message': f'Win rate below 30% in last 24 hours ({win_rate:.1%})',
                    'symbol': symbol or 'all'
                })
        
        return alerts
    
    async def export_performance_data(self, format: str = 'json', 
                                    symbol: Optional[str] = None,
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> Any:
        """Export performance data in specified format"""
        
        # Filter data
        trades = self._filter_trades(symbol, start_date, end_date)
        predictions = self._filter_predictions(symbol, start_date, end_date)
        
        # Prepare export data
        export_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'symbol': symbol or 'all',
                'period': {
                    'start': start_date.isoformat() if start_date else 'beginning',
                    'end': end_date.isoformat() if end_date else 'now'
                }
            },
            'trades': [
                {
                    'id': t.trade_id,
                    'symbol': t.symbol,
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat(),
                    'position_type': t.position_type,
                    'pnl': t.pnl,
                    'pnl_percentage': t.pnl_percentage,
                    'duration_minutes': t.duration_minutes,
                    'outcome': t.outcome.value
                }
                for t in trades
            ],
            'predictions': [
                {
                    'id': p.prediction_id,
                    'symbol': p.symbol,
                    'timestamp': p.timestamp.isoformat(),
                    'model_type': p.model_type,
                    'recommendation': p.recommendation,
                    'confidence': p.confidence,
                    'correct': p.correct
                }
                for p in predictions
            ]
        }
        
        # Export in requested format
        if format == 'json':
            return json.dumps(export_data, indent=2)
        elif format == 'csv':
            # Convert to CSV
            import io
            import csv
            
            output = io.StringIO()
            
            # Write trades CSV
            if trades:
                trade_writer = csv.DictWriter(output, fieldnames=[
                    'id', 'symbol', 'entry_time', 'exit_time', 'position_type',
                    'pnl', 'pnl_percentage', 'duration_minutes', 'outcome'
                ])
                trade_writer.writeheader()
                for t in trades:
                    trade_writer.writerow({
                        'id': t.trade_id,
                        'symbol': t.symbol,
                        'entry_time': t.entry_time.isoformat(),
                        'exit_time': t.exit_time.isoformat(),
                        'position_type': t.position_type,
                        'pnl': t.pnl,
                        'pnl_percentage': t.pnl_percentage,
                        'duration_minutes': t.duration_minutes,
                        'outcome': t.outcome.value
                    })
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")
