"""
Ensemble ML Model for Trading Predictions
Combines XGBoost, Random Forest, LightGBM, and Neural Network
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb

class TradingEnsembleModel:
    """
    Ensemble model combining multiple ML algorithms for trading predictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.feature_names = []
        self.is_trained = False
        self.performance_metrics = {}
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all ML models"""
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=self.config['models']['xgboost']['n_estimators'],
            max_depth=self.config['models']['xgboost']['max_depth'],
            learning_rate=self.config['models']['xgboost']['learning_rate'],
            subsample=self.config['models']['xgboost']['subsample'],
            colsample_bytree=self.config['models']['xgboost']['colsample_bytree'],
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=self.config['models']['random_forest']['n_estimators'],
            max_depth=self.config['models']['random_forest']['max_depth'],
            min_samples_split=self.config['models']['random_forest']['min_samples_split'],
            min_samples_leaf=self.config['models']['random_forest']['min_samples_leaf'],
            random_state=42,
            n_jobs=-1
        )
        
        # LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=self.config['models']['lightgbm']['n_estimators'],
            num_leaves=self.config['models']['lightgbm']['num_leaves'],
            learning_rate=self.config['models']['lightgbm']['learning_rate'],
            feature_fraction=self.config['models']['lightgbm']['feature_fraction'],
            bagging_fraction=self.config['models']['lightgbm']['bagging_fraction'],
            random_state=42,
            n_jobs=-1
        )
        
        # Neural Network
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=tuple(self.config['models']['neural_network']['layers']),
            dropout=self.config['models']['neural_network']['dropout'],
            activation=self.config['models']['neural_network']['activation'],
            learning_rate='adaptive',
            max_iter=self.config['models']['neural_network']['epochs'],
            batch_size=self.config['models']['neural_network']['batch_size'],
            random_state=42
        )
        
        # Create ensemble with soft voting
        self.ensemble = VotingClassifier(
            estimators=[
                ('xgboost', self.models['xgboost']),
                ('random_forest', self.models['random_forest']),
                ('lightgbm', self.models['lightgbm']),
                ('neural_network', self.models['neural_network'])
            ],
            voting='soft',  # Use probability voting
            weights=[1.2, 1.0, 1.1, 0.8]  # Weight by expected performance
        )
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training
        """
        
        # Create target: 1 if price increases, 0 if decreases
        lookahead = self.config['training']['target']['lookahead_period']
        threshold = self.config['training']['target']['classification_threshold']
        
        # Future returns
        df['future_return'] = df['close'].pct_change(lookahead).shift(-lookahead)
        
        # Binary classification target
        df['target'] = (df['future_return'] > threshold).astype(int)
        
        # Drop rows with NaN
        df_clean = df.dropna().copy()
        
        # Separate features and target
        feature_cols = [col for col in df_clean.columns if col not in ['target', 'future_return', 'timestamp']]
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train the ensemble model with time-series cross-validation
        """
        
        print("🚀 Training Ensemble Model...")
        print(f"   Samples: {len(X)}, Features: {X.shape[1]}")
        print(f"   Class distribution: {y.value_counts().to_dict()}")
        
        # Time-series split (no data leakage)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Handle class imbalance
        if self.config['training']['handle_imbalance']:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"   After SMOTE: {len(X_resampled)} samples")
        else:
            X_resampled, y_resampled = X, y
        
        # Scale features
        self.scaler = RobustScaler()  # Robust to outliers
        X_scaled = self.scaler.fit_transform(X_resampled)
        
        # Train-test split (temporal)
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_resampled[:split_idx], y_resampled[split_idx:]
        
        # Train individual models
        model_performance = {}
        
        for name, model in self.models.items():
            print(f"   Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            performance = self._calculate_metrics(y_test, y_pred, y_proba)
            model_performance[name] = performance
            
            print(f"     Accuracy: {performance['accuracy']:.3f}, "
                  f"Precision: {performance['precision']:.3f}, "
                  f"Recall: {performance['recall']:.3f}")
        
        # Train ensemble
        print("   Training ensemble...")
        self.ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = self.ensemble.predict(X_test)
        y_proba_ensemble = self.ensemble.predict_proba(X_test)[:, 1]
        
        ensemble_performance = self._calculate_metrics(y_test, y_pred_ensemble, y_proba_ensemble)
        model_performance['ensemble'] = ensemble_performance
        
        print(f"   Ensemble - Accuracy: {ensemble_performance['accuracy']:.3f}, "
              f"Precision: {ensemble_performance['precision']:.3f}, "
              f"Recall: {ensemble_performance['recall']:.3f}")
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.ensemble, X_scaled, y_resampled, 
                                   cv=tscv, scoring='accuracy', n_jobs=-1)
        
        print(f"   Cross-validation accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        self.is_trained = True
        self.performance_metrics = model_performance
        
        return model_performance
    
    def _calculate_metrics(self, y_true, y_pred, y_proba=None) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Make trading predictions with confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)
                predictions[name] = {
                    'prediction': model.predict(X_scaled)[-1],  # Latest prediction
                    'probability': float(proba[-1, 1]),  # Probability of class 1
                    'confidence': float(np.max(proba[-1]))  # Max probability
                }
        
        # Ensemble prediction
        ensemble_proba = self.ensemble.predict_proba(X_scaled)
        ensemble_pred = self.ensemble.predict(X_scaled)
        
        predictions['ensemble'] = {
            'prediction': int(ensemble_pred[-1]),
            'probability': float(ensemble_proba[-1, 1]),
            'confidence': float(np.max(ensemble_proba[-1]))
        }
        
        # Generate final recommendation
        final_pred = self._generate_final_recommendation(predictions)
        
        return {
            'predictions': predictions,
            'final_recommendation': final_pred,
            'model_agreement': self._calculate_model_agreement(predictions)
        }
    
    def _generate_final_recommendation(self, predictions: Dict) -> Dict[str, Any]:
        """Generate final trading recommendation from model predictions"""
        
        # Weighted voting based on model confidence
        weights = {
            'xgboost': 1.2,
            'random_forest': 1.0,
            'lightgbm': 1.1,
            'neural_network': 0.8,
            'ensemble': 1.5  # Highest weight for ensemble
        }
        
        buy_votes = 0
        sell_votes = 0
        total_confidence = 0
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 1.0)
            confidence = pred['confidence']
            
            if pred['prediction'] == 1:  # Buy signal
                buy_votes += weight * confidence
            else:  # Sell or hold
                sell_votes += weight * confidence
            
            total_confidence += confidence
        
        # Normalize votes
        buy_score = buy_votes / total_confidence if total_confidence > 0 else 0
        sell_score = sell_votes / total_confidence if total_confidence > 0 else 0
        
        # Determine recommendation
        if buy_score > 0.6:  # Strong buy signal
            recommendation = "LONG"
            confidence = buy_score
            reasoning = f"Strong buy consensus ({buy_score:.1%})"
            
        elif sell_score > 0.6:  # Strong sell signal
            recommendation = "SHORT"
            confidence = sell_score
            reasoning = f"Strong sell consensus ({sell_score:.1%})"
            
        else:  # Wait for clearer signal
            recommendation = "WAIT"
            confidence = max(buy_score, sell_score)
            reasoning = f"Inconclusive signals (Buy: {buy_score:.1%}, Sell: {sell_score:.1%})"
        
        return {
            'recommendation': recommendation,
            'confidence': float(confidence),
            'reasoning': reasoning,
            'buy_score': float(buy_score),
            'sell_score': float(sell_score)
        }
    
    def _calculate_model_agreement(self, predictions: Dict) -> float:
        """Calculate agreement percentage among models"""
        
        predictions_list = [pred['prediction'] for pred in predictions.values()]
        agreement = sum(predictions_list) / len(predictions_list)
        
        # Convert to agreement percentage
        if agreement > 0.5:
            agreement_pct = agreement  # Majority predict 1
        else:
            agreement_pct = 1 - agreement  # Majority predict 0
        
        return float(agreement_pct)
    
    def save(self, path: str):
        """Save trained model"""
        
        model_data = {
            'ensemble': self.ensemble,
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'config': self.config,
            'trained_at': datetime.now().isoformat()
        }
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save using joblib (better for large models)
        joblib.dump(model_data, path)
        print(f"✅ Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        
        model_data = joblib.load(path)
        
        self.ensemble = model_data['ensemble']
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.performance_metrics = model_data['performance_metrics']
        self.is_trained = True
        
        print(f"✅ Model loaded from {path}")
        print(f"   Trained at: {model_data.get('trained_at', 'unknown')}")
        print(f"   Performance: {self.performance_metrics.get('ensemble', {})}")
