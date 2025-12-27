"""
ML Predictor - Loads models from ML-Pipeline project
"""

import joblib
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)

class MLModel:
    """Wrapper for a single ML model"""
    
    def __init__(self, name: str, model_path: Path):
        self.name = name
        self.model_path = model_path
        self.model = None
        self.model_type = "unknown"
        self.accuracy = 0.0
        self.feature_names = []
        self.loaded_at = None
        self.is_loaded = False
    
    async def load(self):
        """Load model from disk"""
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Try different serialization formats
            if self.model_path.suffix == '.joblib':
                self.model = joblib.load(self.model_path)
                self.model_type = "joblib"
            elif self.model_path.suffix == '.pkl':
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.feature_names = model_data.get('feature_names', [])
                    self.accuracy = model_data.get('accuracy', 0.0)
                else:
                    self.model = model_data
                self.model_type = "pickle"
            elif self.model_path.suffix == '.json':
                # XGBoost JSON format
                import xgboost as xgb
                self.model = xgb.Booster()
                self.model.load_model(str(self.model_path))
                self.model_type = "xgboost"
            else:
                logger.error(f"Unsupported model format: {self.model_path.suffix}")
                return False
            
            self.loaded_at = datetime.now()
            self.is_loaded = True
            
            logger.info(f"Model '{self.name}' loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model '{self.name}': {e}")
            return False
    
    async def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with loaded model"""
        if not self.is_loaded:
            raise ValueError(f"Model '{self.name}' not loaded")
        
        try:
            # Ensure features match expected format
            if self.feature_names and len(self.feature_names) > 0:
                # Reorder features to match model expectations
                missing_features = set(self.feature_names) - set(features.columns)
                if missing_features:
                    # Add missing features with default values
                    for feat in missing_features:
                        features[feat] = 0.0
                features = features[self.feature_names]
            
            # Make prediction based on model type
            if self.model_type == "xgboost":
                import xgboost as xgb
                dmatrix = xgb.DMatrix(features)
                prediction = self.model.predict(dmatrix)
            elif hasattr(self.model, 'predict_proba'):
                # Scikit-learn style model
                prediction = self.model.predict_proba(features)
            elif hasattr(self.model, 'predict'):
                prediction = self.model.predict(features)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            return {
                "prediction": prediction,
                "model_name": self.name,
                "model_type": self.model_type,
                "confidence": self._calculate_confidence(prediction)
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for model '{self.name}': {e}")
            raise
    
    def _calculate_confidence(self, prediction) -> float:
        """Calculate confidence from prediction output"""
        if isinstance(prediction, np.ndarray):
            if len(prediction.shape) == 2:  # Probability array
                return float(np.max(prediction))
            else:  # Single prediction
                return 0.7  # Default confidence
        elif isinstance(prediction, (float, np.float32, np.float64)):
            return float(abs(prediction))
        else:
            return 0.5  # Default


class MLPredictor:
    """Main ML predictor that loads models from ML-Pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.models: Dict[str, MLModel] = {}
        self.model_path = Path(config.ml.model_path).resolve()
        self.is_initialized = False
        
        logger.info(f"ML Predictor initialized. Model path: {self.model_path}")
    
    async def initialize(self):
        """Load all models from ML-Pipeline directory"""
        if not self.model_path.exists():
            logger.error(f"ML-Pipeline model directory not found: {self.model_path}")
            logger.info("Please run ML-Pipeline training first")
            return
        
        # Load each model configuration
        for model_config in self.config.ml.models:
            model_name = model_config["name"]
            model_file = model_config["file"]
            model_full_path = self.model_path / model_file
            
            if not model_full_path.exists():
                logger.warning(f"Model file not found: {model_full_path}")
                continue
            
            model = MLModel(model_name, model_full_path)
            if await model.load():
                self.models[model_name] = model
        
        if len(self.models) == 0:
            logger.warning("No models loaded. Creating fallback model...")
            await self._create_fallback_model()
        
        self.is_initialized = True
        logger.info(f"Loaded {len(self.models)} models from ML-Pipeline")
    
    async def _create_fallback_model(self):
        """Create a fallback model if no ML-Pipeline models are available"""
        logger.info("Creating fallback model for testing")
        
        # Create a simple model for testing
        class FallbackModel:
            def predict(self, features):
                # Random predictions for testing
                import random
                return [[random.random(), random.random()]]
        
        fallback_model = MLModel("fallback", Path("dummy"))
        fallback_model.model = FallbackModel()
        fallback_model.model_type = "fallback"
        fallback_model.accuracy = 0.5
        fallback_model.feature_names = ["price", "volume", "rsi", "macd"]
        fallback_model.is_loaded = True
        fallback_model.loaded_at = datetime.now()
        
        self.models["fallback"] = fallback_model
    
    async def predict_ensemble(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction using ensemble of all loaded models"""
        if not self.is_initialized:
            raise ValueError("ML Predictor not initialized")
        
        predictions = []
        confidences = []
        model_results = []
        
        # Get predictions from all models
        for model_name, model in self.models.items():
            try:
                result = await model.predict(features)
                predictions.append(result["prediction"])
                confidences.append(result["confidence"])
                model_results.append({
                    "model": model_name,
                    "confidence": result["confidence"],
                    "type": model.model_type
                })
            except Exception as e:
                logger.error(f"Model '{model_name}' prediction error: {e}")
        
        if not predictions:
            raise ValueError("No models produced predictions")
        
        # Ensemble voting (simple majority)
        # For binary classification: 0 = WAIT, 1 = LONG, 2 = SHORT
        vote_counts = {"LONG": 0, "SHORT": 0, "WAIT": 0}
        
        for pred, confidence in zip(predictions, confidences):
            if isinstance(pred, np.ndarray) and len(pred.shape) == 2:
                # Probability array - take class with highest probability
                class_idx = np.argmax(pred[0])
                if class_idx == 0:
                    vote = "WAIT"
                elif class_idx == 1:
                    vote = "LONG"
                else:
                    vote = "SHORT"
            else:
                # Single value prediction
                if pred > 0.5:
                    vote = "LONG"
                elif pred < -0.5:
                    vote = "SHORT"
                else:
                    vote = "WAIT"
            
            # Weight vote by model confidence
            vote_counts[vote] += confidence
        
        # Determine winner
        winner = max(vote_counts, key=vote_counts.get)
        total_votes = sum(vote_counts.values())
        confidence = vote_counts[winner] / total_votes if total_votes > 0 else 0.5
        
        return {
            "recommendation": winner,
            "confidence": confidence,
            "model_results": model_results,
            "ensemble_votes": vote_counts
        }