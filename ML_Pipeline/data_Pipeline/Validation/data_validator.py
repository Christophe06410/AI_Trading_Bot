"""
Data quality validation for streaming and batch data
Ensures data integrity for ML pipeline
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result types"""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationRule:
    """Data validation rule"""
    name: str
    field: str
    rule_type: str  # "range", "pattern", "not_null", "unique", "custom"
    condition: Any
    severity: ValidationResult = ValidationResult.ERROR
    description: str = ""


@dataclass
class ValidationReport:
    """Validation report"""
    timestamp: str
    data_source: str
    total_checks: int
    passed_checks: int
    warnings: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    critical_errors: List[Dict[str, Any]]
    is_valid: bool


class DataValidator:
    """Validates data quality for ML pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.validation_history: List[ValidationReport] = []
        
        # Statistics
        self.stats = {
            "validations_performed": 0,
            "data_points_validated": 0,
            "validation_errors": 0,
            "validation_warnings": 0
        }
        
        # Load default validation rules
        self._load_default_rules()
        
        logger.info("Data validator initialized")
    
    def _load_default_rules(self):
        """Load default validation rules for common data types"""
        
        # OHLCV data rules
        ohlcv_rules = [
            ValidationRule(
                name="price_positive",
                field="close",
                rule_type="range",
                condition={"min": 0.000001, "max": 1000000},
                severity=ValidationResult.ERROR,
                description="Price must be positive and reasonable"
            ),
            ValidationRule(
                name="high_gte_low",
                field="high",
                rule_type="custom",
                condition=lambda data: data.get("high", 0) >= data.get("low", float('inf')),
                severity=ValidationResult.ERROR,
                description="High must be >= low"
            ),
            ValidationRule(
                name="close_in_range",
                field="close",
                rule_type="custom",
                condition=lambda data: data.get("low", 0) <= data.get("close", 0) <= data.get("high", float('inf')),
                severity=ValidationResult.ERROR,
                description="Close must be between low and high"
            ),
            ValidationRule(
                name="volume_non_negative",
                field="volume",
                rule_type="range",
                condition={"min": 0, "max": 1e12},
                severity=ValidationResult.WARNING,
                description="Volume must be non-negative"
            ),
            ValidationRule(
                name="timestamp_format",
                field="timestamp",
                rule_type="pattern",
                condition=r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
                severity=ValidationResult.ERROR,
                description="Timestamp must be ISO format"
            )
        ]
        
        self.validation_rules["ohlcv"] = ohlcv_rules
        
        # Trade data rules
        trade_rules = [
            ValidationRule(
                name="trade_price_positive",
                field="price",
                rule_type="range",
                condition={"min": 0.000001, "max": 1000000},
                severity=ValidationResult.ERROR
            ),
            ValidationRule(
                name="trade_quantity_positive",
                field="quantity",
                rule_type="range",
                condition={"min": 0, "max": 1e9},
                severity=ValidationResult.ERROR
            ),
            ValidationRule(
                name="trade_timestamp_recent",
                field="timestamp",
                rule_type="custom",
                condition=lambda data: self._is_timestamp_recent(data.get("timestamp"), max_age_minutes=10),
                severity=ValidationResult.WARNING
            )
        ]
        
        self.validation_rules["trade"] = trade_rules
        
        # Order book rules
        orderbook_rules = [
            ValidationRule(
                name="bid_ask_spread_positive",
                field="spread",
                rule_type="range",
                condition={"min": 0, "max": 0.1},  # Max 10% spread
                severity=ValidationResult.WARNING
            ),
            ValidationRule(
                name="bids_descending",
                field="bids",
                rule_type="custom",
                condition=lambda data: self._are_prices_descending(data.get("bids", [])),
                severity=ValidationResult.ERROR
            ),
            ValidationRule(
                name="asks_ascending",
                field="asks",
                rule_type="custom",
                condition=lambda data: self._are_prices_ascending(data.get("asks", [])),
                severity=ValidationResult.ERROR
            )
        ]
        
        self.validation_rules["orderbook"] = orderbook_rules
        
        logger.info(f"Loaded validation rules for {len(self.validation_rules)} data types")
    
    def add_validation_rule(self, data_type: str, rule: ValidationRule):
        """Add custom validation rule"""
        if data_type not in self.validation_rules:
            self.validation_rules[data_type] = []
        
        self.validation_rules[data_type].append(rule)
        logger.info(f"Added validation rule '{rule.name}' for {data_type}")
    
    async def validate(self, data: Dict[str, Any], data_type: str = "ohlcv") -> ValidationReport:
        """Validate data against rules"""
        self.stats["validations_performed"] += 1
        
        warnings = []
        errors = []
        critical_errors = []
        
        # Get rules for this data type
        rules = self.validation_rules.get(data_type, [])
        
        for rule in rules:
            try:
                # Get field value
                field_value = data.get(rule.field)
                
                # Apply validation rule
                is_valid, message = self._apply_rule(rule, field_value, data)
                
                if not is_valid:
                    error_info = {
                        "rule": rule.name,
                        "field": rule.field,
                        "value": field_value,
                        "message": message,
                        "severity": rule.severity.value
                    }
                    
                    if rule.severity == ValidationResult.WARNING:
                        warnings.append(error_info)
                        self.stats["validation_warnings"] += 1
                    elif rule.severity == ValidationResult.ERROR:
                        errors.append(error_info)
                        self.stats["validation_errors"] += 1
                    elif rule.severity == ValidationResult.CRITICAL:
                        critical_errors.append(error_info)
                        self.stats["validation_errors"] += 1
                    
                    logger.debug(f"Validation failed: {rule.name} - {message}")
                
            except Exception as e:
                logger.error(f"Error applying rule '{rule.name}': {e}")
                continue
        
        # Create validation report
        total_checks = len(rules)
        passed_checks = total_checks - len(errors) - len(critical_errors)
        
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            data_source=data_type,
            total_checks=total_checks,
            passed_checks=passed_checks,
            warnings=warnings,
            errors=errors,
            critical_errors=critical_errors,
            is_valid=len(errors) == 0 and len(critical_errors) == 0
        )
        
        # Add to history
        self.validation_history.append(report)
        
        # Keep only last 1000 reports
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        # Log summary
        if not report.is_valid:
            logger.warning(
                f"Data validation failed: {len(errors)} errors, "
                f"{len(critical_errors)} critical, {len(warnings)} warnings"
            )
        else:
            logger.debug(f"Data validation passed: {passed_checks}/{total_checks} checks")
        
        return report
    
    def _apply_rule(self, rule: ValidationRule, value: Any, full_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Apply a single validation rule"""
        
        if rule.rule_type == "range":
            # Range validation
            if value is None:
                return False, f"Field {rule.field} is null"
            
            try:
                num_value = float(value)
                min_val = rule.condition.get("min")
                max_val = rule.condition.get("max")
                
                if min_val is not None and num_value < min_val:
                    return False, f"Value {num_value} below minimum {min_val}"
                
                if max_val is not None and num_value > max_val:
                    return False, f"Value {num_value} above maximum {max_val}"
                
                return True, "Range validation passed"
                
            except (ValueError, TypeError):
                return False, f"Value {value} is not numeric"
        
        elif rule.rule_type == "pattern":
            # Pattern/regex validation
            if value is None:
                return False, f"Field {rule.field} is null"
            
            import re
            if not re.match(rule.condition, str(value)):
                return False, f"Value {value} does not match pattern {rule.condition}"
            
            return True, "Pattern validation passed"
        
        elif rule.rule_type == "not_null":
            # Not null validation
            if value is None:
                return False, f"Field {rule.field} is null"
            
            return True, "Not null validation passed"
        
        elif rule.rule_type == "unique":
            # Unique validation (would need context)
            # For now, always pass
            return True, "Unique validation passed"
        
        elif rule.rule_type == "custom":
            # Custom validation function
            try:
                result = rule.condition(full_data)
                if result:
                    return True, "Custom validation passed"
                else:
                    return False, "Custom validation failed"
            except Exception as e:
                return False, f"Custom validation error: {e}"
        
        else:
            return False, f"Unknown rule type: {rule.rule_type}"
    
    def _is_timestamp_recent(self, timestamp_str: str, max_age_minutes: int = 10) -> bool:
        """Check if timestamp is recent"""
        try:
            if not timestamp_str:
                return False
            
            # Parse timestamp
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            now = datetime.now(timestamp.tzinfo) if timestamp.tzinfo else datetime.now()
            
            age_minutes = (now - timestamp).total_seconds() / 60
            return age_minutes <= max_age_minutes
            
        except Exception:
            return False
    
    def _are_prices_descending(self, bids: List[List[Any]]) -> bool:
        """Check if bid prices are in descending order"""
        if not bids or len(bids) < 2:
            return True
        
        try:
            prices = [float(bid[0]) for bid in bids]
            return all(prices[i] >= prices[i+1] for i in range(len(prices)-1))
        except Exception:
            return False
    
    def _are_prices_ascending(self, asks: List[List[Any]]) -> bool:
        """Check if ask prices are in ascending order"""
        if not asks or len(asks) < 2:
            return True
        
        try:
            prices = [float(ask[0]) for ask in asks]
            return all(prices[i] <= prices[i+1] for i in range(len(prices)-1))
        except Exception:
            return False
    
    async def validate_batch(self, data_list: List[Dict[str, Any]], 
                           data_type: str = "ohlcv") -> List[ValidationReport]:
        """Validate a batch of data"""
        self.stats["data_points_validated"] += len(data_list)
        
        reports = []
        for data in data_list:
            report = await self.validate(data, data_type)
            reports.append(report)
        
        return reports
    
    async def validate_dataframe(self, df: pd.DataFrame, data_type: str = "ohlcv") -> ValidationReport:
        """Validate DataFrame data"""
        if df.empty:
            return ValidationReport(
                timestamp=datetime.now().isoformat(),
                data_source=data_type,
                total_checks=0,
                passed_checks=0,
                warnings=[{"message": "Empty DataFrame"}],
                errors=[],
                critical_errors=[],
                is_valid=False
            )
        
        # Convert DataFrame to list of dicts
        data_list = df.to_dict('records')
        reports = await self.validate_batch(data_list, data_type)
        
        # Aggregate reports
        if reports:
            # Use the first report as base
            aggregated = reports[0]
            
            # Aggregate counts
            for report in reports[1:]:
                aggregated.total_checks += report.total_checks
                aggregated.passed_checks += report.passed_checks
                aggregated.warnings.extend(report.warnings)
                aggregated.errors.extend(report.errors)
                aggregated.critical_errors.extend(report.critical_errors)
                aggregated.is_valid = aggregated.is_valid and report.is_valid
            
            return aggregated
        else:
            return ValidationReport(
                timestamp=datetime.now().isoformat(),
                data_source=data_type,
                total_checks=0,
                passed_checks=0,
                warnings=[],
                errors=[],
                critical_errors=[],
                is_valid=True
            )
    
    async def get_data_quality_score(self, data_type: str = None, 
                                   time_window_hours: int = 24) -> Dict[str, Any]:
        """Calculate data quality score"""
        # Filter reports by time window
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        relevant_reports = [
            report for report in self.validation_history
            if datetime.fromisoformat(report.timestamp) >= cutoff_time
        ]
        
        if data_type:
            relevant_reports = [
                report for report in relevant_reports
                if report.data_source == data_type
            ]
        
        if not relevant_reports:
            return {"score": 0, "message": "No validation data available"}
        
        # Calculate scores
        total_checks = sum(r.total_checks for r in relevant_reports)
        passed_checks = sum(r.passed_checks for r in relevant_reports)
        total_errors = sum(len(r.errors) + len(r.critical_errors) for r in relevant_reports)
        
        if total_checks > 0:
            success_rate = passed_checks / total_checks
            error_rate = total_errors / total_checks
            
            # Quality score (0-100)
            quality_score = success_rate * 100
            
            # Determine quality level
            if quality_score >= 95:
                quality_level = "excellent"
            elif quality_score >= 85:
                quality_level = "good"
            elif quality_score >= 70:
                quality_level = "fair"
            else:
                quality_level = "poor"
            
            return {
                "score": round(quality_score, 2),
                "level": quality_level,
                "success_rate": round(success_rate, 4),
                "error_rate": round(error_rate, 4),
                "total_validations": len(relevant_reports),
                "time_window_hours": time_window_hours,
                "data_type": data_type or "all"
            }
        else:
            return {"score": 0, "message": "No validation checks performed"}
    
    async def generate_validation_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate validation summary report"""
        quality_score = await self.get_data_quality_score(time_window_hours=time_window_hours)
        
        # Get recent validation issues
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_issues = []
        
        for report in self.validation_history:
            if datetime.fromisoformat(report.timestamp) >= cutoff_time:
                if not report.is_valid:
                    recent_issues.append({
                        "timestamp": report.timestamp,
                        "data_source": report.data_source,
                        "errors": len(report.errors) + len(report.critical_errors),
                        "warnings": len(report.warnings)
                    })
        
        return {
            "summary": {
                "total_validations": self.stats["validations_performed"],
                "data_points_validated": self.stats["data_points_validated"],
                "total_errors": self.stats["validation_errors"],
                "total_warnings": self.stats["validation_warnings"],
                "quality_score": quality_score,
                "timestamp": datetime.now().isoformat()
            },
            "recent_issues": recent_issues[-20:],  # Last 20 issues
            "validation_rules": {
                data_type: len(rules)
                for data_type, rules in self.validation_rules.items()
            }
        }
    
    async def detect_anomalies(self, data: pd.DataFrame, 
                              field: str, 
                              window: int = 100) -> List[Dict[str, Any]]:
        """Detect anomalies in time series data"""
        anomalies = []
        
        try:
            if field not in data.columns:
                logger.warning(f"Field {field} not found in data")
                return anomalies
            
            values = data[field].dropna()
            
            if len(values) < window:
                return anomalies
            
            # Calculate rolling statistics
            rolling_mean = values.rolling(window=window, min_periods=1).mean()
            rolling_std = values.rolling(window=window, min_periods=1).std()
            
            # Detect anomalies (values outside 3 standard deviations)
            z_scores = (values - rolling_mean) / rolling_std.replace(0, 1)
            
            anomaly_indices = np.where(np.abs(z_scores) > 3)[0]
            
            for idx in anomaly_indices:
                if idx < len(data):
                    row = data.iloc[idx]
                    anomalies.append({
                        "timestamp": row.get("timestamp", str(idx)),
                        "field": field,
                        "value": float(values.iloc[idx]),
                        "z_score": float(z_scores.iloc[idx]),
                        "rolling_mean": float(rolling_mean.iloc[idx]),
                        "rolling_std": float(rolling_std.iloc[idx]),
                        "description": f"Anomaly detected: z-score = {z_scores.iloc[idx]:.2f}"
                    })
            
            logger.info(f"Detected {len(anomalies)} anomalies in field {field}")
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    async def validate_and_clean(self, data: pd.DataFrame, 
                                data_type: str = "ohlcv") -> Tuple[pd.DataFrame, ValidationReport]:
        """Validate data and return cleaned version"""
        # Validate
        report = await self.validate_dataframe(data, data_type)
        
        if report.is_valid:
            return data, report
        
        # Try to clean data
        cleaned_data = data.copy()
        
        for error in report.errors + report.critical_errors:
            field = error.get("field")
            
            if field in cleaned_data.columns:
                # Simple cleaning: replace outliers with median
                if error.get("rule") == "range":
                    median_val = cleaned_data[field].median()
                    cleaned_data[field] = cleaned_data[field].apply(
                        lambda x: median_val if pd.isna(x) or not (0.000001 <= float(x) <= 1000000) else x
                    )
        
        # Validate cleaned data
        cleaned_report = await self.validate_dataframe(cleaned_data, data_type)
        
        return cleaned_data, cleaned_report
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics"""
        return {
            **self.stats,
            "validation_rules_total": sum(len(rules) for rules in self.validation_rules.values()),
            "validation_history_size": len(self.validation_history),
            "data_types_supported": list(self.validation_rules.keys())
        }
