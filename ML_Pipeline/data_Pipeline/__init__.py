"""
Enterprise Data Pipeline for ML Trading System
Real-time streaming, alternative data, feature store, and validation
"""

__version__ = "1.0.0"
__author__ = "ML Pipeline Enterprise"
__description__ = "Enterprise-grade data pipeline for trading ML"

from data_pipeline.orchestrator import DataPipelineOrchestrator, PipelineConfig
from data_pipeline.feature_store.feature_store import FeatureStore, FeatureType, StorageBackend
from data_pipeline.validation.data_validator import DataValidator, ValidationResult
from data_pipeline.streaming.realtime_processor import RealTimeProcessor

__all__ = [
    "DataPipelineOrchestrator",
    "PipelineConfig",
    "FeatureStore",
    "FeatureType",
    "StorageBackend",
    "DataValidator",
    "ValidationResult",
    "RealTimeProcessor"
]
