"""
MODEL PERFORMANCE TRACKER
Tracks model performance over time with real outcomes
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

class PerformanceTracker:
    """
    Tracks model predictions and actual outcomes
    Calculates real-world performance metrics
    """
    
    def __init__(self, registry_root: Path):
        self.registry_root = registry_root
        self.performance_db = registry_root / 'performance' / 'performance.db'
        self.performance_db.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Cache for quick access
        self.performance_cache = {}
    
    def _init_database(self):
        """Initialize performance tracking database"""
        conn = sqlite3.connect(self.performance_db)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            model_type TEXT,
            model_version TEXT,
            prediction TEXT,
            confidence REAL,
            features_hash TEXT,
            metadata TEXT
        )
        ''')
        
        # Outcomes table (linked to predictions)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            timestamp DATETIME,
            symbol TEXT,
            actual_direction TEXT,
            actual_return REAL,
            profit_loss REAL,
            trade_duration_seconds INTEGER,
            metadata TEXT,
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
        ''')
        
        # Performance metrics table (aggregated)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            model_type TEXT,
            model_version TEXT,
            period TEXT,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            total_trades INTEGER,
            winning_trades INTEGER,
            total_profit REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            metadata TEXT
        )
        ''')
        
        # Model comparison table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            model_type TEXT,
            version_a TEXT,
            version_b TEXT,
            metric TEXT,
            value_a REAL,
            value_b REAL,
            difference REAL,
            significance REAL,
            metadata TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_prediction(
        self,
        symbol: str,
        model_type: str,
        model_version: str,
        prediction: str,
        confidence: float,
        features_hash: str = None,
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Record a model prediction
        Returns prediction ID for later outcome recording
        """
        conn = sqlite3.connect(self.performance_db)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO predictions 
        (timestamp, symbol, model_type, model_version, prediction, confidence, features_hash, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            symbol,
            model_type,
            model_version,
            prediction,
            confidence,
            features_hash or '',
            json.dumps(metadata or {})
        ))
        
        prediction_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def record_outcome(
        self,
        prediction_id: int,
        symbol: str,
        actual_direction: str,
        actual_return: float,
        profit_loss: float = None,
        trade_duration_seconds: int = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Record actual outcome for a prediction
        """
        conn = sqlite3.connect(self.performance_db)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO outcomes 
        (prediction_id, timestamp, symbol, actual_direction, actual_return, 
         profit_loss, trade_duration_seconds, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_id,
            datetime.now().isoformat(),
            symbol,
            actual_direction,
            actual_return,
            profit_loss,
            trade_duration_seconds,
            json.dumps(metadata or {})
        ))
        
        conn.commit()
        conn.close()
        
        # Update cached metrics
        self._update_cached_metrics(symbol)
    
    def calculate_performance_metrics(
        self,
        symbol: str = None,
        model_type: str = None,
        model_version: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics
        """
        conn = sqlite3.connect(self.performance_db)
        
        # Build WHERE clause
        where_clauses = []
        params = []
        
        if symbol:
            where_clauses.append("p.symbol = ?")
            params.append(symbol)
        
        if model_type:
            where_clauses.append("p.model_type = ?")
            params.append(model_type)
        
        if model_version:
            where_clauses.append("p.model_version = ?")
            params.append(model_version)
        
        if start_date:
            where_clauses.append("p.timestamp >= ?")
            params.append(start_date.isoformat())
        
        if end_date:
            where_clauses.append("p.timestamp <= ?")
            params.append(end_date.isoformat())
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Query to get predictions with outcomes
        query = f'''
        SELECT 
            p.prediction,
            p.confidence,
            o.actual_direction,
            o.actual_return,
            o.profit_loss,
            o.trade_duration_seconds
        FROM predictions p
        LEFT JOIN outcomes o ON p.id = o.prediction_id
        WHERE {where_clause}
        AND o.actual_direction IS NOT NULL
        ORDER BY p.timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return {
                'total_predictions': 0,
                'predictions_with_outcomes': 0,
                'message': 'No predictions with outcomes found'
            }
        
        # Calculate metrics
        metrics = {
            'total_predictions': len(df),
            'period': {
                'start': df.iloc[0]['timestamp'] if 'timestamp' in df.columns else None,
                'end': df.iloc[-1]['timestamp'] if 'timestamp' in df.columns else None
            }
        }
        
        # Classification metrics
        if 'prediction' in df.columns and 'actual_direction' in df.columns:
            # Convert to binary for classification metrics
            df['correct'] = df['prediction'] == df['actual_direction']
            
            metrics['accuracy'] = float(df['correct'].mean())
            
            # For LONG/SHORT classification
            true_positives = len(df[(df['prediction'] == 'LONG') & (df['actual_direction'] == 'LONG')])
            false_positives = len(df[(df['prediction'] == 'LONG') & (df['actual_direction'] != 'LONG')])
            false_negatives = len(df[(df['prediction'] != 'LONG') & (df['actual_direction'] == 'LONG')])
            
            if true_positives + false_positives > 0:
                metrics['precision'] = true_positives / (true_positives + false_positives)
            else:
                metrics['precision'] = 0.0
            
            if true_positives + false_negatives > 0:
                metrics['recall'] = true_positives / (true_positives + false_negatives)
            else:
                metrics['recall'] = 0.0
            
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1_score'] = 0.0
        
        # Trading metrics
        if 'profit_loss' in df.columns and df['profit_loss'].notna().any():
            profit_series = df['profit_loss'].dropna()
            
            metrics['trading'] = {
                'total_trades': len(profit_series),
                'winning_trades': len(profit_series[profit_series > 0]),
                'losing_trades': len(profit_series[profit_series < 0]),
                'total_profit': float(profit_series.sum()),
                'avg_profit_per_trade': float(profit_series.mean()),
                'profit_std': float(profit_series.std()),
                'max_profit': float(profit_series.max()),
                'max_loss': float(profit_series.min()),
                'win_rate': len(profit_series[profit_series > 0]) / len(profit_series) if len(profit_series) > 0 else 0.0
            }
            
            # Sharpe ratio (simplified)
            if len(profit_series) > 1 and profit_series.std() > 0:
                metrics['trading']['sharpe_ratio'] = float(
                    profit_series.mean() / profit_series.std() * np.sqrt(252)  # Annualized
                )
            else:
                metrics['trading']['sharpe_ratio'] = 0.0
            
            # Max drawdown
            cumulative = profit_series.cumsum()
            running_max = cumulative.expanding().max()
            drawdown = cumulative - running_max
            if len(drawdown) > 0:
                metrics['trading']['max_drawdown'] = float(drawdown.min())
            else:
                metrics['trading']['max_drawdown'] = 0.0
        
        # Confidence analysis
        if 'confidence' in df.columns and 'correct' in df.columns:
            confidence_series = df['confidence']
            correct_series = df['correct']
            
            metrics['confidence_analysis'] = {
                'avg_confidence': float(confidence_series.mean()),
                'confidence_when_correct': float(confidence_series[correct_series].mean()),
                'confidence_when_incorrect': float(confidence_series[~correct_series].mean()),
                'calibration_error': float(abs(confidence_series.mean() - metrics.get('accuracy', 0)))
            }
        
        # Store in database
        self._store_aggregated_metrics(symbol, model_type, model_version, metrics)
        
        return metrics
    
    def _store_aggregated_metrics(
        self,
        symbol: str,
        model_type: str,
        model_version: str,
        metrics: Dict[str, Any]
    ):
        """Store aggregated metrics in database"""
        conn = sqlite3.connect(self.performance_db)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO performance_metrics 
        (timestamp, symbol, model_type, model_version, period, 
         accuracy, precision, recall, f1_score, total_trades, 
         winning_trades, total_profit, sharpe_ratio, max_drawdown, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            symbol,
            model_type,
            model_version,
            'custom',
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0),
            metrics.get('trading', {}).get('total_trades', 0),
            metrics.get('trading', {}).get('winning_trades', 0),
            metrics.get('trading', {}).get('total_profit', 0),
            metrics.get('trading', {}).get('sharpe_ratio', 0),
            metrics.get('trading', {}).get('max_drawdown', 0),
            json.dumps(metrics)
        ))
        
        conn.commit()
        conn.close()
    
    def _update_cached_metrics(self, symbol: str):
        """Update cached performance metrics"""
        cache_key = f"{symbol}_latest"
        
        # Calculate latest metrics
        try:
            metrics = self.calculate_performance_metrics(
                symbol=symbol,
                start_date=datetime.now() - timedelta(days=7)
            )
            self.performance_cache[cache_key] = {
                'metrics': metrics,
                'calculated_at': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"⚠️  Could not update cache for {symbol}: {e}")
    
    def get_model_comparison(
        self,
        symbol: str,
        model_type: str,
        version_a: str,
        version_b: str,
        metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Compare performance of two model versions
        """
        # Get metrics for both versions
        metrics_a = self.calculate_performance_metrics(
            symbol=symbol,
            model_type=model_type,
            model_version=version_a,
            start_date=datetime.now() - timedelta(days=30)
        )
        
        metrics_b = self.calculate_performance_metrics(
            symbol=symbol,
            model_type=model_type,
            model_version=version_b,
            start_date=datetime.now() - timedelta(days=30)
        )
        
        # Extract metric values
        if metric == 'accuracy':
            value_a = metrics_a.get('accuracy', 0)
            value_b = metrics_b.get('accuracy', 0)
        elif metric == 'sharpe_ratio':
            value_a = metrics_a.get('trading', {}).get('sharpe_ratio', 0)
            value_b = metrics_b.get('trading', {}).get('sharpe_ratio', 0)
        elif metric == 'win_rate':
            value_a = metrics_a.get('trading', {}).get('win_rate', 0)
            value_b = metrics_b.get('trading', {}).get('win_rate', 0)
        else:
            value_a = 0
            value_b = 0
        
        comparison = {
            'symbol': symbol,
            'model_type': model_type,
            'version_a': version_a,
            'version_b': version_b,
            'metric': metric,
            'value_a': value_a,
            'value_b': value_b,
            'difference': value_b - value_a,
            'percentage_change': ((value_b - value_a) / value_a * 100) if value_a != 0 else 0,
            'comparison_date': datetime.now().isoformat()
        }
        
        # Store in database
        conn = sqlite3.connect(self.performance_db)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO model_comparisons 
        (timestamp, symbol, model_type, version_a, version_b, 
         metric, value_a, value_b, difference, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            symbol,
            model_type,
            version_a,
            version_b,
            metric,
            value_a,
            value_b,
            value_b - value_a,
            json.dumps(comparison)
        ))
        
        conn.commit()
        conn.close()
        
        return comparison
    
    def get_performance_trend(
        self,
        symbol: str,
        model_type: str,
        days: int = 30,
        metric: str = 'accuracy'
    ) -> pd.DataFrame:
        """
        Get performance trend over time
        """
        conn = sqlite3.connect(self.performance_db)
        
        query = '''
        SELECT 
            DATE(timestamp) as date,
            AVG(accuracy) as avg_accuracy,
            AVG(sharpe_ratio) as avg_sharpe,
            AVG(win_rate) as avg_win_rate,
            COUNT(*) as predictions_count
        FROM performance_metrics
        WHERE symbol = ? AND model_type = ?
        AND timestamp >= DATE('now', ?)
        GROUP BY DATE(timestamp)
        ORDER BY date
        '''
        
        df = pd.read_sql_query(
            query, 
            conn, 
            params=[symbol, model_type, f'-{days} days']
        )
        
        conn.close()
        
        return df
    
    def generate_performance_report(
        self,
        symbol: str,
        model_type: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        """
        # Get latest metrics
        latest_metrics = self.calculate_performance_metrics(
            symbol=symbol,
            model_type=model_type,
            start_date=datetime.now() - timedelta(days=days)
        )
        
        # Get trend
        trend = self.get_performance_trend(symbol, model_type, days)
        
        # Get model versions in period
        conn = sqlite3.connect(self.performance_db)
        
        versions_query = '''
        SELECT DISTINCT model_version
        FROM predictions
        WHERE symbol = ? AND model_type = ?
        AND timestamp >= DATE('now', ?)
        ORDER BY timestamp DESC
        '''
        
        versions_df = pd.read_sql_query(
            versions_query,
            conn,
            params=[symbol, model_type, f'-{days} days']
        )
        
        conn.close()
        
        # Compile report
        report = {
            'symbol': symbol,
            'model_type': model_type,
            'report_period_days': days,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'accuracy': latest_metrics.get('accuracy', 0),
                'total_trades': latest_metrics.get('trading', {}).get('total_trades', 0),
                'win_rate': latest_metrics.get('trading', {}).get('win_rate', 0),
                'total_profit': latest_metrics.get('trading', {}).get('total_profit', 0),
                'sharpe_ratio': latest_metrics.get('trading', {}).get('sharpe_ratio', 0)
            },
            'detailed_metrics': latest_metrics,
            'performance_trend': trend.to_dict('records') if not trend.empty else [],
            'model_versions_used': versions_df['model_version'].tolist() if not versions_df.empty else [],
            'recommendations': self._generate_recommendations(latest_metrics)
        }
        
        # Save report
        reports_dir = self.registry_root / 'performance' / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance"""
        recommendations = []
        
        # Accuracy recommendations
        accuracy = metrics.get('accuracy', 0)
        if accuracy < 0.5:
            recommendations.append("Model accuracy below 50%. Consider retraining with more data or different features.")
        elif accuracy < 0.6:
            recommendations.append("Model accuracy moderate. Could benefit from hyperparameter tuning.")
        elif accuracy > 0.75:
            recommendations.append("Model accuracy excellent. Consider A/B testing new features.")
        
        # Trading recommendations
        trading_metrics = metrics.get('trading', {})
        win_rate = trading_metrics.get('win_rate', 0)
        
        if win_rate < 0.4:
            recommendations.append("Win rate below 40%. Review stop-loss and take-profit strategies.")
        elif win_rate > 0.6:
            recommendations.append("Win rate excellent. Consider increasing position size gradually.")
        
        # Sharpe ratio recommendations
        sharpe_ratio = trading_metrics.get('sharpe_ratio', 0)
        if sharpe_ratio < 1.0:
            recommendations.append(f"Sharpe ratio {sharpe_ratio:.2f} indicates moderate risk-adjusted returns.")
        elif sharpe_ratio > 2.0:
            recommendations.append(f"Sharpe ratio {sharpe_ratio:.2f} indicates excellent risk-adjusted returns!")
        
        # Confidence calibration
        confidence_analysis = metrics.get('confidence_analysis', {})
        calibration_error = confidence_analysis.get('calibration_error', 0)
        if calibration_error > 0.1:
            recommendations.append(f"High calibration error ({calibration_error:.3f}). Model is overconfident. Consider recalibration.")
        
        return recommendations
