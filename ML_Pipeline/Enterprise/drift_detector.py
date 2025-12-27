"""
ENTERPRISE DRIFT DETECTOR v1.0
Detects concept drift and data drift in real-time
Maintains backward compatibility with existing predictions
"""
# ML-Pipeline/enterprise/drift_detector.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of drift that can occur"""
    CONCEPT_DRIFT = "concept_drift"  # Relationship between X and y changes
    DATA_DRIFT = "data_drift"        # Distribution of X changes
    LABEL_DRIFT = "label_drift"      # Distribution of y changes
    FEATURE_DRIFT = "feature_drift"  # Individual feature distribution changes
    COVARIATE_SHIFT = "covariate_shift"  # P(X) changes
    PRIOR_SHIFT = "prior_shift"      # P(y) changes

class DriftSeverity(Enum):
    """Severity of detected drift"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DriftAlert:
    """Drift detection alert"""
    alert_id: str
    symbol: str
    drift_type: DriftType
    severity: DriftSeverity
    detection_time: datetime
    confidence: float
    features_affected: List[str]
    metrics: Dict[str, float]
    recommendation: str
    
    def __post_init__(self):
        if isinstance(self.drift_type, str):
            self.drift_type = DriftType(self.drift_type)
        if isinstance(self.severity, str):
            self.severity = DriftSeverity(self.severity)

@dataclass  
class DriftMetrics:
    """Comprehensive drift metrics"""
    ks_statistic: float  # Kolmogorov-Smirnov statistic
    psi: float           # Population Stability Index
    wasserstein_distance: float
    hellinger_distance: float
    jensen_shannon_divergence: float
    feature_correlation_change: float
    prediction_distribution_change: float
    
class DriftDetector:
    """
    Enterprise Drift Detector
    
    Monitors for:
    1. Concept Drift - Model performance degradation
    2. Data Drift - Feature distribution changes
    3. Covariate Shift - Input distribution changes
    4. Prior Shift - Label distribution changes
    
    Maintains backward compatibility by working alongside existing predictions
    """
    
    def __init__(self, config: Dict[str, Any], model_registry):
        self.config = config
        self.registry = model_registry
        
        # Drift detection thresholds
        self.thresholds = {
            'psi': 0.25,      # PSI > 0.25 indicates drift
            'ks_statistic': 0.3,  # KS > 0.3 indicates drift
            'accuracy_drop': 0.15,  # 15% accuracy drop
            'confidence_drop': 0.2,  # 20% confidence drop
        }
        
        # Monitoring windows
        self.window_sizes = {
            'short': 100,     # Last 100 predictions
            'medium': 1000,   # Last 1000 predictions
            'long': 10000,    # Last 10000 predictions
        }
        
        # Storage for monitoring data
        self.prediction_history = {}  # symbol -> list of predictions
        self.feature_history = {}     # symbol -> list of feature distributions
        self.performance_history = {} # symbol -> list of performance metrics
        self.drift_alerts = []        # All drift alerts
        
        # Drift detection methods
        self.detection_methods = [
            self._detect_concept_drift,
            self._detect_data_drift,
            self._detect_feature_drift,
            self._detect_covariate_shift
        ]
        
        logger.info(f"Drift Detector initialized with thresholds: {self.thresholds}")
    
    async def monitor_prediction(self, symbol: str, prediction: Dict[str, Any], 
                               features: pd.DataFrame, actual_outcome: Optional[Dict] = None):
        """
        Monitor a single prediction for drift detection
        
        Args:
            symbol: Trading symbol
            prediction: Prediction from model
            features: Features used for prediction
            actual_outcome: Actual trade outcome (if available)
        """
        try:
            # Initialize storage for symbol if needed
            if symbol not in self.prediction_history:
                self._initialize_symbol_monitoring(symbol)
            
            # Store prediction
            prediction_record = {
                'timestamp': datetime.now(),
                'prediction': prediction,
                'features': features.iloc[-1].to_dict() if len(features) > 0 else {},
                'actual_outcome': actual_outcome,
                'confidence': prediction.get('confidence', 0.5)
            }
            
            self.prediction_history[symbol].append(prediction_record)
            
            # Keep history bounded
            if len(self.prediction_history[symbol]) > self.window_sizes['long']:
                self.prediction_history[symbol] = self.prediction_history[symbol][-self.window_sizes['long']:]
            
            # Store feature distribution
            if not features.empty:
                feature_stats = self._calculate_feature_statistics(features)
                self.feature_history[symbol].append({
                    'timestamp': datetime.now(),
                    'stats': feature_stats
                })
                
                # Keep feature history bounded
                if len(self.feature_history[symbol]) > 1000:
                    self.feature_history[symbol] = self.feature_history[symbol][-1000:]
            
            # Store performance if outcome available
            if actual_outcome:
                performance = {
                    'timestamp': datetime.now(),
                    'correct': actual_outcome.get('correct', False),
                    'pnl': actual_outcome.get('pnl', 0),
                    'confidence': prediction.get('confidence', 0.5)
                }
                self.performance_history[symbol].append(performance)
                
                # Keep performance history bounded
                if len(self.performance_history[symbol]) > self.window_sizes['long']:
                    self.performance_history[symbol] = self.performance_history[symbol][-self.window_sizes['long']:]
            
            # Run drift detection periodically
            if len(self.prediction_history[symbol]) % 50 == 0:  # Every 50 predictions
                await self._run_drift_detection(symbol)
            
        except Exception as e:
            logger.error(f"Error monitoring prediction for {symbol}: {e}")
    
    def _initialize_symbol_monitoring(self, symbol: str):
        """Initialize monitoring storage for a symbol"""
        self.prediction_history[symbol] = []
        self.feature_history[symbol] = []
        self.performance_history[symbol] = []
    
    def _calculate_feature_statistics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for feature monitoring"""
        stats = {}
        
        for column in features.columns:
            if pd.api.types.is_numeric_dtype(features[column]):
                stats[column] = {
                    'mean': float(features[column].mean()),
                    'std': float(features[column].std()),
                    'min': float(features[column].min()),
                    'max': float(features[column].max()),
                    'percentiles': {
                        '25': float(features[column].quantile(0.25)),
                        '50': float(features[column].quantile(0.50)),
                        '75': float(features[column].quantile(0.75))
                    }
                }
        
        return stats
    
    async def _run_drift_detection(self, symbol: str):
        """Run all drift detection methods for a symbol"""
        if len(self.prediction_history[symbol]) < self.window_sizes['medium']:
            return  # Not enough data
        
        alerts = []
        
        for detection_method in self.detection_methods:
            try:
                alert = await detection_method(symbol)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.error(f"Drift detection method failed for {symbol}: {e}")
        
        # Process alerts
        for alert in alerts:
            await self._handle_drift_alert(alert)
    
    async def _detect_concept_drift(self, symbol: str) -> Optional[DriftAlert]:
        """
        Detect concept drift - degradation in model performance
        
        Concept drift occurs when P(y|X) changes over time
        """
        if len(self.performance_history[symbol]) < self.window_sizes['medium']:
            return None
        
        # Split data into reference and current windows
        ref_window = self.performance_history[symbol][-self.window_sizes['long']:-self.window_sizes['medium']]
        current_window = self.performance_history[symbol][-self.window_sizes['medium']:]
        
        if not ref_window or not current_window:
            return None
        
        # Calculate accuracy in reference window
        ref_correct = [p['correct'] for p in ref_window if p['correct'] is not None]
        ref_accuracy = sum(ref_correct) / len(ref_correct) if ref_correct else 0.5
        
        # Calculate accuracy in current window
        curr_correct = [p['correct'] for p in current_window if p['correct'] is not None]
        curr_accuracy = sum(curr_correct) / len(curr_correct) if curr_correct else 0.5
        
        # Check for significant accuracy drop
        accuracy_drop = ref_accuracy - curr_accuracy
        
        if accuracy_drop > self.thresholds['accuracy_drop']:
            # Calculate confidence scores
            ref_confidences = [p['confidence'] for p in ref_window]
            curr_confidences = [p['confidence'] for p in current_window]
            
            avg_ref_confidence = np.mean(ref_confidences) if ref_confidences else 0.5
            avg_curr_confidence = np.mean(curr_confidences) if curr_confidences else 0.5
            
            confidence_drop = avg_ref_confidence - avg_curr_confidence
            
            # Determine severity
            severity = self._determine_severity(accuracy_drop, confidence_drop)
            
            alert = DriftAlert(
                alert_id=f"concept_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=severity,
                detection_time=datetime.now(),
                confidence=min(accuracy_drop, 1.0),
                features_affected=["all"],  # Concept drift affects all features
                metrics={
                    'reference_accuracy': ref_accuracy,
                    'current_accuracy': curr_accuracy,
                    'accuracy_drop': accuracy_drop,
                    'reference_confidence': avg_ref_confidence,
                    'current_confidence': avg_curr_confidence,
                    'confidence_drop': confidence_drop
                },
                recommendation="Retrain model with recent data"
            )
            
            return alert
        
        return None
    
    async def _detect_data_drift(self, symbol: str) -> Optional[DriftAlert]:
        """
        Detect data drift - changes in feature distributions
        
        Data drift occurs when P(X) changes over time
        """
        if len(self.feature_history[symbol]) < self.window_sizes['medium']:
            return None
        
        # Get feature statistics from reference and current windows
        ref_stats = self.feature_history[symbol][-self.window_sizes['long']:-self.window_sizes['medium']]
        curr_stats = self.feature_history[symbol][-self.window_sizes['medium']:]
        
        if not ref_stats or not curr_stats:
            return None
        
        # Calculate PSI (Population Stability Index) for each feature
        psi_scores = {}
        affected_features = []
        
        # Get all feature names
        if ref_stats and ref_stats[0]['stats']:
            feature_names = list(ref_stats[0]['stats'].keys())
        else:
            return None
        
        for feature in feature_names[:20]:  # Limit to first 20 features for performance
            try:
                # Extract values from history
                ref_values = []
                for stat_record in ref_stats:
                    if feature in stat_record['stats']:
                        ref_values.append(stat_record['stats'][feature]['mean'])
                
                curr_values = []
                for stat_record in curr_stats:
                    if feature in stat_record['stats']:
                        curr_values.append(stat_record['stats'][feature]['mean'])
                
                if len(ref_values) > 10 and len(curr_values) > 10:
                    psi = self._calculate_psi(ref_values, curr_values)
                    psi_scores[feature] = psi
                    
                    if psi > self.thresholds['psi']:
                        affected_features.append(feature)
            except Exception as e:
                logger.debug(f"Could not calculate PSI for {feature}: {e}")
        
        # Check if significant drift detected
        if affected_features:
            max_psi = max(psi_scores.values()) if psi_scores else 0
            severity = self._determine_psi_severity(max_psi)
            
            alert = DriftAlert(
                alert_id=f"data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                detection_time=datetime.now(),
                confidence=min(max_psi, 1.0),
                features_affected=affected_features[:10],  # Limit to 10 features
                metrics={
                    'max_psi': max_psi,
                    'affected_features_count': len(affected_features),
                    'psi_scores': {k: v for k, v in list(psi_scores.items())[:5]}
                },
                recommendation=f"Check data pipeline for {', '.join(affected_features[:3])}"
            )
            
            return alert
        
        return None
    
    async def _detect_feature_drift(self, symbol: str) -> Optional[DriftAlert]:
        """
        Detect feature-level drift using Kolmogorov-Smirnov test
        """
        if len(self.prediction_history[symbol]) < self.window_sizes['medium']:
            return None
        
        # Get recent predictions with features
        recent_predictions = self.prediction_history[symbol][-self.window_sizes['medium']:]
        
        if not recent_predictions or len(recent_predictions) < 100:
            return None
        
        # Split into two halves for comparison
        mid_point = len(recent_predictions) // 2
        first_half = recent_predictions[:mid_point]
        second_half = recent_predictions[mid_point:]
        
        # Find common features
        if not first_half or not second_half:
            return None
        
        sample_features_first = first_half[0]['features']
        sample_features_second = second_half[0]['features']
        
        if not sample_features_first or not sample_features_second:
            return None
        
        common_features = set(sample_features_first.keys()) & set(sample_features_second.keys())
        
        ks_results = {}
        affected_features = []
        
        for feature in list(common_features)[:15]:  # Limit to 15 features
            try:
                # Extract feature values
                values_first = [p['features'].get(feature, 0) for p in first_half 
                              if feature in p['features']]
                values_second = [p['features'].get(feature, 0) for p in second_half 
                               if feature in p['features']]
                
                if len(values_first) > 30 and len(values_second) > 30:
                    ks_stat = self._calculate_ks_statistic(values_first, values_second)
                    ks_results[feature] = ks_stat
                    
                    if ks_stat > self.thresholds['ks_statistic']:
                        affected_features.append(feature)
            except Exception as e:
                logger.debug(f"Could not calculate KS for {feature}: {e}")
        
        if affected_features:
            max_ks = max(ks_results.values()) if ks_results else 0
            severity = self._determine_ks_severity(max_ks)
            
            alert = DriftAlert(
                alert_id=f"feature_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                drift_type=DriftType.FEATURE_DRIFT,
                severity=severity,
                detection_time=datetime.now(),
                confidence=min(max_ks, 1.0),
                features_affected=affected_features[:10],
                metrics={
                    'max_ks': max_ks,
                    'affected_features': len(affected_features),
                    'ks_scores': {k: v for k, v in list(ks_results.items())[:5]}
                },
                recommendation="Review feature engineering pipeline"
            )
            
            return alert
        
        return None
    
    async def _detect_covariate_shift(self, symbol: str) -> Optional[DriftAlert]:
        """
        Detect covariate shift using domain classifier approach
        Simplified implementation
        """
        if len(self.prediction_history[symbol]) < self.window_sizes['long']:
            return None
        
        # Split data into source (old) and target (new) domains
        source_data = self.prediction_history[symbol][-self.window_sizes['long']:-self.window_sizes['medium']]
        target_data = self.prediction_history[symbol][-self.window_sizes['medium']:]
        
        if len(source_data) < 100 or len(target_data) < 100:
            return None
        
        # Simple approach: Compare prediction confidence distributions
        source_confidences = [p['confidence'] for p in source_data]
        target_confidences = [p['confidence'] for p in target_data]
        
        if not source_confidences or not target_confidences:
            return None
        
        # Calculate Wasserstein distance (Earth Mover's Distance)
        try:
            from scipy.stats import wasserstein_distance
            
            wasserstein_dist = wasserstein_distance(source_confidences, target_confidences)
            
            if wasserstein_dist > 0.1:  # Threshold for covariate shift
                # Also check if means are significantly different
                source_mean = np.mean(source_confidences)
                target_mean = np.mean(target_confidences)
                mean_diff = abs(source_mean - target_mean)
                
                severity = DriftSeverity.MEDIUM if wasserstein_dist > 0.15 else DriftSeverity.LOW
                
                alert = DriftAlert(
                    alert_id=f"covariate_shift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    drift_type=DriftType.COVARIATE_SHIFT,
                    severity=severity,
                    detection_time=datetime.now(),
                    confidence=min(wasserstein_dist * 5, 1.0),  # Scale to [0, 1]
                    features_affected=["prediction_confidence"],
                    metrics={
                        'wasserstein_distance': wasserstein_dist,
                        'source_mean_confidence': source_mean,
                        'target_mean_confidence': target_mean,
                        'mean_difference': mean_diff
                    },
                    recommendation="Model may need adaptation to new data distribution"
                )
                
                return alert
        except ImportError:
            logger.warning("scipy not available for Wasserstein distance calculation")
        except Exception as e:
            logger.error(f"Error calculating covariate shift: {e}")
        
        return None
    
    def _calculate_psi(self, reference: List[float], current: List[float], 
                      bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        PSI < 0.1: No significant population change
        PSI 0.1-0.25: Minor change
        PSI > 0.25: Significant change
        """
        if not reference or not current:
            return 0.0
        
        # Create bins based on reference distribution
        ref_min, ref_max = min(reference), max(reference)
        if ref_max - ref_min < 1e-10:
            return 0.0
        
        bin_edges = np.linspace(ref_min, ref_max, bins + 1)
        
        # Calculate bin counts
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Add small epsilon to avoid division by zero
        ref_counts = ref_counts + 1e-10
        curr_counts = curr_counts + 1e-10
        
        # Normalize to get probabilities
        ref_probs = ref_counts / len(reference)
        curr_probs = curr_counts / len(current)
        
        # Calculate PSI
        psi = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
        
        return float(psi)
    
    def _calculate_ks_statistic(self, sample1: List[float], sample2: List[float]) -> float:
        """
        Calculate Kolmogorov-Smirnov statistic
        
        KS statistic measures the maximum distance between two empirical CDFs
        """
        if not sample1 or not sample2:
            return 0.0
        
        try:
            from scipy.stats import ks_2samp
            statistic, _ = ks_2samp(sample1, sample2)
            return float(statistic)
        except ImportError:
            # Fallback implementation
            from collections import Counter
            
            # Combine and sort all values
            all_values = sorted(sample1 + sample2)
            
            # Calculate empirical CDFs
            cdf1 = []
            cdf2 = []
            
            for value in all_values:
                cdf1.append(sum(1 for x in sample1 if x <= value) / len(sample1))
                cdf2.append(sum(1 for x in sample2 if x <= value) / len(sample2))
            
            # Calculate maximum difference
            ks_stat = max(abs(c1 - c2) for c1, c2 in zip(cdf1, cdf2))
            return ks_stat
    
    def _determine_severity(self, accuracy_drop: float, confidence_drop: float) -> DriftSeverity:
        """Determine severity based on accuracy and confidence drops"""
        if accuracy_drop > 0.25 or confidence_drop > 0.3:
            return DriftSeverity.CRITICAL
        elif accuracy_drop > 0.15 or confidence_drop > 0.2:
            return DriftSeverity.HIGH
        elif accuracy_drop > 0.08 or confidence_drop > 0.1:
            return DriftSeverity.MEDIUM
        elif accuracy_drop > 0.04 or confidence_drop > 0.05:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE
    
    def _determine_psi_severity(self, psi: float) -> DriftSeverity:
        """Determine severity based on PSI value"""
        if psi > 0.5:
            return DriftSeverity.CRITICAL
        elif psi > 0.25:
            return DriftSeverity.HIGH
        elif psi > 0.1:
            return DriftSeverity.MEDIUM
        elif psi > 0.05:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE
    
    def _determine_ks_severity(self, ks: float) -> DriftSeverity:
        """Determine severity based on KS statistic"""
        if ks > 0.4:
            return DriftSeverity.CRITICAL
        elif ks > 0.3:
            return DriftSeverity.HIGH
        elif ks > 0.2:
            return DriftSeverity.MEDIUM
        elif ks > 0.1:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE
    
    async def _handle_drift_alert(self, alert: DriftAlert):
        """Handle a drift alert (log, notify, trigger actions)"""
        self.drift_alerts.append(alert)
        
        # Keep alerts bounded
        if len(self.drift_alerts) > 1000:
            self.drift_alerts = self.drift_alerts[-1000:]
        
        # Log based on severity
        if alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            logger.warning(
                f"🚨 DRIFT ALERT [{alert.severity.value}] {alert.drift_type.value} "
                f"detected for {alert.symbol}: {alert.recommendation}"
            )
            
            # For high/critical drift, suggest immediate action
            if alert.drift_type == DriftType.CONCEPT_DRIFT:
                await self._trigger_retraining(alert.symbol, "concept_drift_detected")
        else:
            logger.info(
                f"⚠️  Drift detected [{alert.severity.value}] for {alert.symbol}: "
                f"{alert.drift_type.value}"
            )
        
        # Store alert for analysis
        await self._store_alert(alert)
    
    async def _trigger_retraining(self, symbol: str, reason: str):
        """Trigger model retraining due to drift"""
        logger.info(f"Triggering retraining for {symbol} due to {reason}")
        
        # TODO: Integrate with orchestrator to trigger retraining
        # This would call the orchestrator's retraining method
        
        # For now, log the action
        print(f"⏰ [ACTION REQUIRED] Retrain model for {symbol}: {reason}")
    
    async def _store_alert(self, alert: DriftAlert):
        """Store drift alert for analysis"""
        # TODO: Store in database or file
        # For now, just keep in memory
        
        # Log to file
        import json
        from pathlib import Path
        
        alerts_dir = Path("monitoring/drift/alerts")
        alerts_dir.mkdir(parents=True, exist_ok=True)
        
        alert_file = alerts_dir / f"alert_{alert.alert_id}.json"
        
        alert_dict = {
            'alert_id': alert.alert_id,
            'symbol': alert.symbol,
            'drift_type': alert.drift_type.value,
            'severity': alert.severity.value,
            'detection_time': alert.detection_time.isoformat(),
            'confidence': alert.confidence,
            'features_affected': alert.features_affected,
            'metrics': alert.metrics,
            'recommendation': alert.recommendation
        }
        
        with open(alert_file, 'w') as f:
            json.dump(alert_dict, f, indent=2)
    
    def get_drift_summary(self, symbol: Optional[str] = None, 
                         days: int = 7) -> Dict[str, Any]:
        """Get drift detection summary"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        filtered_alerts = [
            alert for alert in self.drift_alerts
            if alert.detection_time >= cutoff_date
            and (symbol is None or alert.symbol == symbol)
        ]
        
        if not filtered_alerts:
            return {
                'total_alerts': 0,
                'symbol': symbol or 'all',
                'period_days': days
            }
        
        # Count by drift type
        drift_type_counts = {}
        severity_counts = {}
        
        for alert in filtered_alerts:
            drift_type = alert.drift_type.value
            severity = alert.severity.value
            
            drift_type_counts[drift_type] = drift_type_counts.get(drift_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Get most affected symbols
        symbol_counts = {}
        for alert in filtered_alerts:
            symbol_counts[alert.symbol] = symbol_counts.get(alert.symbol, 0) + 1
        
        most_affected_symbols = sorted(
            symbol_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            'total_alerts': len(filtered_alerts),
            'period_days': days,
            'drift_type_distribution': drift_type_counts,
            'severity_distribution': severity_counts,
            'most_affected_symbols': dict(most_affected_symbols),
            'recent_alerts': [
                {
                    'symbol': alert.symbol,
                    'type': alert.drift_type.value,
                    'severity': alert.severity.value,
                    'time': alert.detection_time.isoformat(),
                    'confidence': alert.confidence
                }
                for alert in filtered_alerts[-5:]  # Last 5 alerts
            ]
        }
    
    async def run_continuous_monitoring(self):
        """Run continuous monitoring loop"""
        logger.info("Starting continuous drift monitoring...")
        
        while True:
            try:
                # Check all symbols with prediction history
                for symbol in list(self.prediction_history.keys()):
                    if len(self.prediction_history[symbol]) >= self.window_sizes['medium']:
                        await self._run_drift_detection(symbol)
                
                # Sleep before next monitoring cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Continuous monitoring failed: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
