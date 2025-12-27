"""
ENTERPRISE EXPLAINER v1.0
Explainable AI (XAI) for model predictions
Provides insights into why models make specific predictions
"""
# ML-Pipeline/enterprise/explainer.py

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class FeatureContribution:
    """Contribution of a single feature to prediction"""
    feature_name: str
    contribution_score: float  # How much this feature contributed
    feature_value: float       # Actual feature value
    feature_importance: float  # Global importance of this feature
    direction: str            # "positive", "negative", "neutral"
    percentile: float         # Where this value falls in distribution
    
    def __post_init__(self):
        # Ensure contribution_score is between -1 and 1
        self.contribution_score = max(-1.0, min(1.0, self.contribution_score))

@dataclass
class ModelExplanation:
    """Complete explanation for a model prediction"""
    prediction_id: str
    symbol: str
    model_type: str
    prediction: str  # "LONG", "SHORT", "WAIT", "CLOSE"
    confidence: float
    explanation_time: datetime
    top_features: List[FeatureContribution]
    decision_boundary_distance: float  # How close to decision boundary
    model_confidence_breakdown: Dict[str, float]  # Per-class confidence
    counterfactual_explanation: Optional[str] = None  # What would change prediction
    
class EnterpriseExplainer:
    """
    Enterprise Model Explainer
    
    Provides explainable AI (XAI) for:
    1. Feature importance for individual predictions
    2. Model decision explanations
    3. Counterfactual explanations
    4. Confidence breakdowns
    
    Works with existing models without requiring retraining
    """
    
    def __init__(self, config: Dict[str, Any], model_registry):
        self.config = config
        self.registry = model_registry
        self.explanation_methods = {}
        
        # Initialize explanation methods
        self._initialize_explanation_methods()
        
        # Cache for explanations
        self.explanation_cache = {}
        self.feature_importance_cache = {}
        
        logger.info("Enterprise Explainer initialized")
    
    def _initialize_explanation_methods(self):
        """Initialize different explanation methods"""
        self.explanation_methods = {
            'shap': self._explain_with_shap,
            'lime': self._explain_with_lime,
            'feature_importance': self._explain_with_feature_importance,
            'gradient_based': self._explain_with_gradients,
            'rule_based': self._explain_with_rules
        }
    
    async def explain_prediction(self, symbol: str, model_type: str, 
                               features: pd.DataFrame, 
                               prediction: Dict[str, Any],
                               method: str = 'auto') -> ModelExplanation:
        """
        Explain a model prediction
        
        Args:
            symbol: Trading symbol
            model_type: Type of model (ensemble, xgboost, etc.)
            features: Features used for prediction
            prediction: Prediction result
            method: Explanation method ('auto', 'shap', 'lime', 'feature_importance')
            
        Returns:
            ModelExplanation with detailed breakdown
        """
        try:
            prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"
            
            # Get model from registry
            model, metadata = self.registry.get_production_model(symbol)
            
            # Select explanation method
            if method == 'auto':
                explanation_method = self._select_best_explanation_method(model_type)
            else:
                explanation_method = self.explanation_methods.get(method)
                
            if not explanation_method:
                explanation_method = self._explain_with_feature_importance
            
            # Generate explanation
            explanation = await explanation_method(
                model, features, prediction, symbol, model_type
            )
            
            # Add metadata
            explanation.prediction_id = prediction_id
            explanation.symbol = symbol
            explanation.model_type = model_type
            explanation.explanation_time = datetime.now()
            
            # Cache explanation
            self.explanation_cache[prediction_id] = explanation
            
            # Keep cache bounded
            if len(self.explanation_cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(self.explanation_cache.keys())[:100]
                for key in oldest_keys:
                    del self.explanation_cache[key]
            
            logger.info(f"Generated explanation for {symbol} prediction using {method}")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain prediction for {symbol}: {e}")
            
            # Return basic explanation as fallback
            return self._create_basic_explanation(symbol, model_type, features, prediction)
    
    def _select_best_explanation_method(self, model_type: str) -> callable:
        """Select the best explanation method for the model type"""
        method_preferences = {
            'ensemble': self._explain_with_shap,
            'xgboost': self._explain_with_shap,
            'random_forest': self._explain_with_feature_importance,
            'lightgbm': self._explain_with_shap,
            'neural_network': self._explain_with_gradients,
            'rl': self._explain_with_rules
        }
        
        return method_preferences.get(model_type, self._explain_with_feature_importance)
    
    async def _explain_with_shap(self, model, features: pd.DataFrame,
                                prediction: Dict[str, Any], symbol: str,
                                model_type: str) -> ModelExplanation:
        """
        Explain using SHAP (SHapley Additive exPlanations)
        
        SHAP provides theoretically optimal feature attributions
        """
        try:
            import shap
            
            # Prepare features for SHAP
            X = features.values
            
            # Create SHAP explainer based on model type
            if hasattr(model, 'predict_proba'):
                # Tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # For binary classification, get values for predicted class
                predicted_class = int(prediction.get('prediction', 0))
                if len(shap_values.shape) == 3:  # Multi-class
                    shap_values_class = shap_values[predicted_class][0]
                else:
                    shap_values_class = shap_values[0]
                    
            elif hasattr(model, 'coef_'):
                # Linear models
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)
                shap_values_class = shap_values[0]
                
            else:
                # Kernel SHAP as fallback
                explainer = shap.KernelExplainer(model.predict_proba, X[:100])
                shap_values = explainer.shap_values(X[0:1])
                shap_values_class = shap_values[0]
            
            # Convert SHAP values to feature contributions
            feature_contributions = []
            for i, (feature_name, shap_value) in enumerate(zip(features.columns, shap_values_class)):
                # Normalize SHAP value
                normalized_contribution = shap_value / (np.abs(shap_values_class).sum() + 1e-10)
                
                # Get feature value
                feature_value = float(features.iloc[0, i])
                
                # Get global feature importance from cache
                global_importance = self._get_global_feature_importance(symbol, feature_name)
                
                # Determine direction
                if shap_value > 0:
                    direction = "positive"
                elif shap_value < 0:
                    direction = "negative"
                else:
                    direction = "neutral"
                
                # Calculate percentile (simplified)
                percentile = 0.5  # Placeholder
                
                contribution = FeatureContribution(
                    feature_name=feature_name,
                    contribution_score=float(normalized_contribution),
                    feature_value=feature_value,
                    feature_importance=global_importance,
                    direction=direction,
                    percentile=percentile
                )
                feature_contributions.append(contribution)
            
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: abs(x.contribution_score), reverse=True)
            
            # Calculate decision boundary distance
            decision_boundary_distance = self._calculate_decision_boundary_distance(
                shap_values_class, prediction.get('confidence', 0.5)
            )
            
            # Create model explanation
            explanation = ModelExplanation(
                prediction_id="",  # Will be set by caller
                symbol=symbol,
                model_type=model_type,
                prediction=prediction.get('recommendation', 'WAIT'),
                confidence=prediction.get('confidence', 0.5),
                explanation_time=datetime.now(),
                top_features=feature_contributions[:10],  # Top 10 features
                decision_boundary_distance=decision_boundary_distance,
                model_confidence_breakdown=self._get_confidence_breakdown(model, features),
                counterfactual_explanation=self._generate_counterfactual(
                    feature_contributions, prediction
                )
            )
            
            return explanation
            
        except ImportError:
            logger.warning("SHAP not available, falling back to feature importance")
            return await self._explain_with_feature_importance(
                model, features, prediction, symbol, model_type
            )
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return await self._explain_with_feature_importance(
                model, features, prediction, symbol, model_type
            )
    
    async def _explain_with_lime(self, model, features: pd.DataFrame,
                               prediction: Dict[str, Any], symbol: str,
                               model_type: str) -> ModelExplanation:
        """
        Explain using LIME (Local Interpretable Model-agnostic Explanations)
        
        LIME creates local surrogate models to explain individual predictions
        """
        try:
            import lime
            import lime.lime_tabular
            
            # Prepare data for LIME
            X = features.values
            feature_names = features.columns.tolist()
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X,
                feature_names=feature_names,
                class_names=['WAIT', 'LONG', 'SHORT'],
                mode='classification'
            )
            
            # Generate explanation
            exp = explainer.explain_instance(
                X[0], 
                model.predict_proba,
                num_features=10
            )
            
            # Convert LIME explanation to feature contributions
            feature_contributions = []
            for feature, weight in exp.as_list():
                # Parse feature from LIME output
                if ' <= ' in feature or ' > ' in feature:
                    # Feature with condition
                    feature_name = feature.split(' ')[0]
                else:
                    feature_name = feature
                
                # Find feature index
                try:
                    feature_idx = feature_names.index(feature_name)
                    feature_value = float(features.iloc[0, feature_idx])
                except (ValueError, IndexError):
                    feature_value = 0.0
                
                # Get global importance
                global_importance = self._get_global_feature_importance(symbol, feature_name)
                
                # Determine direction
                direction = "positive" if weight > 0 else "negative"
                
                contribution = FeatureContribution(
                    feature_name=feature_name,
                    contribution_score=float(weight),
                    feature_value=feature_value,
                    feature_importance=global_importance,
                    direction=direction,
                    percentile=0.5
                )
                feature_contributions.append(contribution)
            
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: abs(x.contribution_score), reverse=True)
            
            # Create explanation
            explanation = ModelExplanation(
                prediction_id="",
                symbol=symbol,
                model_type=model_type,
                prediction=prediction.get('recommendation', 'WAIT'),
                confidence=prediction.get('confidence', 0.5),
                explanation_time=datetime.now(),
                top_features=feature_contributions[:10],
                decision_boundary_distance=0.1,  # Simplified
                model_confidence_breakdown=self._get_confidence_breakdown(model, features),
                counterfactual_explanation=self._generate_counterfactual(
                    feature_contributions, prediction
                )
            )
            
            return explanation
            
        except ImportError:
            logger.warning("LIME not available, falling back to feature importance")
            return await self._explain_with_feature_importance(
                model, features, prediction, symbol, model_type
            )
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return await self._explain_with_feature_importance(
                model, features, prediction, symbol, model_type
            )
    
    async def _explain_with_feature_importance(self, model, features: pd.DataFrame,
                                             prediction: Dict[str, Any], symbol: str,
                                             model_type: str) -> ModelExplanation:
        """
        Explain using feature importance (model-agnostic)
        
        Uses permutation importance or built-in feature importance
        """
        try:
            # Try to get feature importance from model
            if hasattr(model, 'feature_importances_'):
                # Model has built-in feature importance
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear model coefficients
                importances = np.abs(model.coef_[0])
            else:
                # Calculate permutation importance
                importances = await self._calculate_permutation_importance(
                    model, features
                )
            
            # Normalize importances
            if importances is not None and len(importances) > 0:
                importances = importances / (importances.sum() + 1e-10)
            
            # Create feature contributions
            feature_contributions = []
            for i, feature_name in enumerate(features.columns):
                if importances is not None and i < len(importances):
                    importance_score = float(importances[i])
                else:
                    importance_score = 1.0 / len(features.columns)
                
                feature_value = float(features.iloc[0, i])
                
                # Determine contribution direction based on feature value
                # This is simplified - in reality would need more sophisticated logic
                if feature_name in ['rsi', 'stoch_k']:
                    # For momentum indicators, high values might suggest LONG
                    if feature_value > 70:
                        direction = "positive"
                    elif feature_value < 30:
                        direction = "negative"
                    else:
                        direction = "neutral"
                elif feature_name in ['bb_position']:
                    # Bollinger band position
                    if feature_value > 0.7:
                        direction = "positive"  # Near top of band
                    elif feature_value < 0.3:
                        direction = "negative"  # Near bottom of band
                    else:
                        direction = "neutral"
                else:
                    direction = "neutral"
                
                # Adjust contribution based on prediction
                if prediction.get('recommendation') == 'LONG' and direction == "positive":
                    contribution_score = importance_score
                elif prediction.get('recommendation') == 'SHORT' and direction == "negative":
                    contribution_score = importance_score
                else:
                    contribution_score = importance_score * 0.5
                
                contribution = FeatureContribution(
                    feature_name=feature_name,
                    contribution_score=contribution_score,
                    feature_value=feature_value,
                    feature_importance=importance_score,
                    direction=direction,
                    percentile=0.5
                )
                feature_contributions.append(contribution)
            
            # Sort by contribution
            feature_contributions.sort(key=lambda x: x.contribution_score, reverse=True)
            
            # Create explanation
            explanation = ModelExplanation(
                prediction_id="",
                symbol=symbol,
                model_type=model_type,
                prediction=prediction.get('recommendation', 'WAIT'),
                confidence=prediction.get('confidence', 0.5),
                explanation_time=datetime.now(),
                top_features=feature_contributions[:10],
                decision_boundary_distance=0.15,
                model_confidence_breakdown=self._get_confidence_breakdown(model, features),
                counterfactual_explanation=self._generate_counterfactual(
                    feature_contributions, prediction
                )
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Feature importance explanation failed: {e}")
            return self._create_basic_explanation(symbol, model_type, features, prediction)
    
    async def _explain_with_gradients(self, model, features: pd.DataFrame,
                                    prediction: Dict[str, Any], symbol: str,
                                    model_type: str) -> ModelExplanation:
        """
        Explain using gradient-based methods (for neural networks)
        """
        # Placeholder for gradient-based explanations
        # In production, would use Integrated Gradients, DeepLIFT, etc.
        
        # Fall back to feature importance
        return await self._explain_with_feature_importance(
            model, features, prediction, symbol, model_type
        )
    
    async def _explain_with_rules(self, model, features: pd.DataFrame,
                                prediction: Dict[str, Any], symbol: str,
                                model_type: str) -> ModelExplanation:
        """
        Explain using rule extraction (for RL agents and rule-based systems)
        """
        # For RL agents, we can explain based on state features
        feature_contributions = []
        
        for feature_name in features.columns:
            feature_value = float(features.iloc[0][feature_name])
            
            # Simple rule-based explanations for common features
            if feature_name == 'rsi':
                if feature_value > 70:
                    direction = "negative"  # Overbought -> potential SHORT
                    score = 0.8
                elif feature_value < 30:
                    direction = "positive"  # Oversold -> potential LONG
                    score = 0.8
                else:
                    direction = "neutral"
                    score = 0.3
            elif feature_name == 'price_vs_ema200':
                if feature_value > 1.05:
                    direction = "positive"  # Price above EMA -> bullish
                    score = 0.7
                elif feature_value < 0.95:
                    direction = "negative"  # Price below EMA -> bearish
                    score = 0.7
                else:
                    direction = "neutral"
                    score = 0.3
            elif 'macd' in feature_name:
                if feature_value > 0:
                    direction = "positive"
                    score = 0.6
                else:
                    direction = "negative"
                    score = 0.6
            else:
                direction = "neutral"
                score = 0.2
            
            contribution = FeatureContribution(
                feature_name=feature_name,
                contribution_score=score,
                feature_value=feature_value,
                feature_importance=score,
                direction=direction,
                percentile=0.5
            )
            feature_contributions.append(contribution)
        
        # Sort by contribution
        feature_contributions.sort(key=lambda x: x.contribution_score, reverse=True)
        
        # Create explanation
        explanation = ModelExplanation(
            prediction_id="",
            symbol=symbol,
            model_type=model_type,
            prediction=prediction.get('recommendation', 'WAIT'),
            confidence=prediction.get('confidence', 0.5),
            explanation_time=datetime.now(),
            top_features=feature_contributions[:10],
            decision_boundary_distance=0.2,
            model_confidence_breakdown={'WAIT': 0.33, 'LONG': 0.33, 'SHORT': 0.33},
            counterfactual_explanation="Rule-based model decision based on technical indicators"
        )
        
        return explanation
    
    async def _calculate_permutation_importance(self, model, features: pd.DataFrame, 
                                              n_repeats: int = 10) -> np.ndarray:
        """Calculate permutation importance"""
        try:
            from sklearn.inspection import permutation_importance
            
            # Use a sample of data for efficiency
            if len(features) > 100:
                sample = features.sample(100, random_state=42)
            else:
                sample = features
            
            X = sample.values
            
            # Get predictions for baseline
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X)[:, 1]
            else:
                y_pred = model.predict(X)
            
            # Calculate permutation importance
            result = permutation_importance(
                model, X, y_pred,
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=-1
            )
            
            return result.importances_mean
            
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
            return None
    
    def _get_global_feature_importance(self, symbol: str, feature_name: str) -> float:
        """Get global feature importance from cache or calculate"""
        cache_key = f"{symbol}_{feature_name}"
        
        if cache_key in self.feature_importance_cache:
            return self.feature_importance_cache[cache_key]
        
        # Default importance
        importance = 0.5
        
        # Cache it
        self.feature_importance_cache[cache_key] = importance
        
        return importance
    
    def _calculate_decision_boundary_distance(self, shap_values: np.ndarray, 
                                           confidence: float) -> float:
        """Calculate distance to decision boundary"""
        # Simplified calculation
        # In reality, would depend on model and feature space
        
        # More confident predictions are farther from boundary
        if confidence > 0.8:
            return 0.9
        elif confidence > 0.7:
            return 0.7
        elif confidence > 0.6:
            return 0.5
        elif confidence > 0.55:
            return 0.3
        else:
            return 0.1
    
    def _get_confidence_breakdown(self, model, features: pd.DataFrame) -> Dict[str, float]:
        """Get confidence breakdown per class"""
        try:
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(features)
                if len(probas.shape) == 2 and probas.shape[1] >= 3:
                    return {
                        'WAIT': float(probas[0, 0]),
                        'LONG': float(probas[0, 1]),
                        'SHORT': float(probas[0, 2])
                    }
        except:
            pass
        
        # Default uniform distribution
        return {'WAIT': 0.33, 'LONG': 0.33, 'SHORT': 0.34}
    
    def _generate_counterfactual(self, top_features: List[FeatureContribution],
                               prediction: Dict[str, Any]) -> str:
        """Generate counterfactual explanation"""
        pred = prediction.get('recommendation', 'WAIT')
        
        if pred == 'WAIT':
            return "No strong signal from any feature"
        
        # Find the strongest feature
        strongest_feature = top_features[0] if top_features else None
        
        if not strongest_feature:
            return "Multiple features contributed to this decision"
        
        # Generate counterfactual based on strongest feature
        feature_name = strongest_feature.feature_name
        direction = strongest_feature.direction
        
        if pred == 'LONG':
            if direction == 'positive':
                return f"If {feature_name} was lower, the prediction might be WAIT or SHORT"
            else:
                return f"Despite negative {feature_name}, other features suggest LONG"
        else:  # SHORT
            if direction == 'negative':
                return f"If {feature_name} was higher, the prediction might be WAIT or LONG"
            else:
                return f"Despite positive {feature_name}, other features suggest SHORT"
    
    def _create_basic_explanation(self, symbol: str, model_type: str,
                                features: pd.DataFrame, prediction: Dict[str, Any]) -> ModelExplanation:
        """Create a basic explanation when advanced methods fail"""
        # Simple feature ranking by value magnitude
        feature_contributions = []
        
        for feature_name in features.columns:
            feature_value = float(features.iloc[0][feature_name])
            
            # Simple scoring based on absolute value (normalized)
            score = min(abs(feature_value) / 10.0, 1.0)
            
            contribution = FeatureContribution(
                feature_name=feature_name,
                contribution_score=score,
                feature_value=feature_value,
                feature_importance=score,
                direction="neutral",
                percentile=0.5
            )
            feature_contributions.append(contribution)
        
        # Sort by contribution
        feature_contributions.sort(key=lambda x: x.contribution_score, reverse=True)
        
        return ModelExplanation(
            prediction_id=f"basic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol,
            model_type=model_type,
            prediction=prediction.get('recommendation', 'WAIT'),
            confidence=prediction.get('confidence', 0.5),
            explanation_time=datetime.now(),
            top_features=feature_contributions[:5],
            decision_boundary_distance=0.3,
            model_confidence_breakdown={'WAIT': 0.33, 'LONG': 0.33, 'SHORT': 0.34},
            counterfactual_explanation="Basic explanation based on feature magnitudes"
        )
    
    def get_explanation_summary(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get summary of recent explanations for a symbol"""
        # Filter explanations for symbol
        symbol_explanations = [
            exp for exp in self.explanation_cache.values()
            if exp.symbol == symbol
        ]
        
        if not symbol_explanations:
            return {'total_explanations': 0, 'symbol': symbol}
        
        # Get most common influential features
        feature_counts = {}
        for exp in symbol_explanations:
            for feature in exp.top_features[:3]:  # Top 3 features per explanation
                feature_counts[feature.feature_name] = feature_counts.get(feature.feature_name, 0) + 1
        
        # Sort features by frequency
        common_features = sorted(
            feature_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Calculate average confidence by prediction type
        confidence_by_type = {'LONG': [], 'SHORT': [], 'WAIT': []}
        for exp in symbol_explanations:
            pred_type = exp.prediction
            if pred_type in confidence_by_type:
                confidence_by_type[pred_type].append(exp.confidence)
        
        avg_confidence = {
            pred_type: np.mean(confs) if confs else 0.0
            for pred_type, confs in confidence_by_type.items()
        }
        
        # Get recent explanations
        recent_explanations = sorted(
            symbol_explanations,
            key=lambda x: x.explanation_time,
            reverse=True
        )[:limit]
        
        return {
            'total_explanations': len(symbol_explanations),
            'symbol': symbol,
            'most_influential_features': dict(common_features),
            'average_confidence_by_type': avg_confidence,
            'recent_explanations': [
                {
                    'prediction': exp.prediction,
                    'confidence': exp.confidence,
                    'time': exp.explanation_time.isoformat(),
                    'top_features': [f.feature_name for f in exp.top_features[:3]]
                }
                for exp in recent_explanations
            ]
        }
    
    async def generate_explanation_report(self, symbol: str, 
                                        start_date: datetime,
                                        end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive explanation report for a period"""
        # Filter explanations by date
        period_explanations = [
            exp for exp in self.explanation_cache.values()
            if exp.symbol == symbol
            and start_date <= exp.explanation_time <= end_date
        ]
        
        if not period_explanations:
            return {
                'period': f"{start_date.date()} to {end_date.date()}",
                'symbol': symbol,
                'total_explanations': 0,
                'message': 'No explanations in this period'
            }
        
        # Calculate statistics
        predictions = [exp.prediction for exp in period_explanations]
        confidences = [exp.confidence for exp in period_explanations]
        
        # Feature importance analysis
        feature_scores = {}
        for exp in period_explanations:
            for feature in exp.top_features:
                feature_name = feature.feature_name
                current_score = feature_scores.get(feature_name, 0)
                feature_scores[feature_name] = current_score + feature.contribution_score
        
        # Normalize feature scores
        if feature_scores:
            max_score = max(feature_scores.values())
            for feature_name in feature_scores:
                feature_scores[feature_name] /= max_score
        
        # Sort features by importance
        top_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]
        
        # Generate insights
        insights = []
        
        # Insight 1: Most reliable features
        if top_features:
            insights.append(
                f"Top influential feature: {top_features[0][0]} "
                f"(score: {top_features[0][1]:.2f})"
            )
        
        # Insight 2: Prediction distribution
        pred_counts = {p: predictions.count(p) for p in set(predictions)}
        most_common_pred = max(pred_counts.items(), key=lambda x: x[1])
        insights.append(
            f"Most common prediction: {most_common_pred[0]} "
            f"({most_common_pred[1]} times, {most_common_pred[1]/len(predictions):.1%})"
        )
        
        # Insight 3: Average confidence
        avg_confidence = np.mean(confidences) if confidences else 0
        insights.append(f"Average confidence: {avg_confidence:.1%}")
        
        return {
            'period': f"{start_date.date()} to {end_date.date()}",
            'symbol': symbol,
            'total_explanations': len(period_explanations),
            'prediction_distribution': pred_counts,
            'average_confidence': avg_confidence,
            'top_features': dict(top_features),
            'insights': insights,
            'sample_explanations': [
                {
                    'time': exp.explanation_time.isoformat(),
                    'prediction': exp.prediction,
                    'confidence': exp.confidence,
                    'decision_boundary': exp.decision_boundary_distance
                }
                for exp in period_explanations[:5]  # First 5
            ]
        }
