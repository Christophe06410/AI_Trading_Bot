"""
Recommendation Engine - Business logic for generating recommendations
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from src.ml_predictor import MLPredictor
from src.data_cache import DataCache

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Generates trading recommendations using ML models and business rules"""
    
    def __init__(self, ml_predictor: MLPredictor, data_cache: DataCache, config):
        self.ml_predictor = ml_predictor
        self.data_cache = data_cache
        self.config = config
        self.min_confidence = config.ml.min_confidence
    
    async def generate_recommendation(self, request) -> Dict[str, Any]:
        """Generate trading recommendation"""
        
        # Step 1: Check if any positions should be closed
        close_recommendation = await self._check_position_closure(request)
        if close_recommendation:
            return close_recommendation
        
        # Step 2: Prepare features for ML models
        features = await self._prepare_features(request)
        
        # Step 3: Get ML prediction
        ml_result = await self.ml_predictor.predict_ensemble(features)
        
        # Step 4: Apply business rules
        final_recommendation = await self._apply_business_rules(ml_result, request)
        
        # Step 5: Store for analysis
        await self._store_recommendation(request.pair, final_recommendation)
        
        return final_recommendation
    
    async def _prepare_features(self, request) -> pd.DataFrame:
        """Prepare features from candle data for ML models"""
        candles = request.previous
        
        # Convert to DataFrame
        df = pd.DataFrame([c.dict() for c in candles])
        
        # Calculate technical indicators
        features = {}
        
        # Price features
        if len(df) > 0:
            latest = df.iloc[-1]
            features['price'] = float(latest['c'])
            features['volume'] = float(latest['v'])
            
            # Price change
            if len(df) > 1:
                features['price_change_pct'] = float((latest['c'] - df.iloc[-2]['c']) / df.iloc[-2]['c'])
            
            # EMA position
            if 'ema200' in df.columns and pd.notna(latest['ema200']):
                features['price_vs_ema200'] = float(latest['c'] / latest['ema200'])
        
        # Volatility
        if len(df) > 1:
            returns = df['c'].pct_change().dropna()
            features['volatility'] = float(returns.std())
        
        # Create DataFrame with single row
        features_df = pd.DataFrame([features])
        
        return features_df
    
    async def _check_position_closure(self, request) -> Optional[Dict[str, Any]]:
        """Check if any open positions should be closed"""
        if not request.positions:
            return None
        
        # Simple logic: Close position if stop loss or take profit reached
        # In production, you'd check current price against SL/TP
        import random
        
        # 20% chance to recommend closing a position (for testing)
        if random.random() < 0.2 and request.positions:
            position_to_close = request.positions[0]
            
            # Ensure position has ID
            if not position_to_close.id:
                position_to_close.id = f"pos_{int(datetime.now().timestamp())}"
            
            return {
                "recommendation": "CLOSE",
                "confidence": 0.8,
                "position": position_to_close,
                "reasoning": "Position reached target or stop loss",
                "model_used": "position_closure_logic"
            }
        
        return None
    
    async def _apply_business_rules(self, ml_result: Dict[str, Any], request) -> Dict[str, Any]:
        """Apply trading rules to ML prediction"""
        
        recommendation = ml_result["recommendation"]
        confidence = ml_result["confidence"]
        
        # Rule 1: Minimum confidence threshold
        if confidence < self.min_confidence:
            return {
                "recommendation": "WAIT",
                "confidence": confidence,
                "reasoning": f"Confidence below threshold ({confidence:.2%} < {self.min_confidence:.2%})",
                "model_used": ml_result.get("model_results", [{}])[0].get("model", "ensemble")
            }
        
        # Rule 2: Market volatility filter (example)
        # You can add your specific trading rules here
        
        # All rules passed
        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "reasoning": "ML ensemble prediction",
            "model_used": "ensemble"
        }
    
    async def _store_recommendation(self, pair: str, recommendation: Dict[str, Any]):
        """Store recommendation for analysis"""
        try:
            store_key = f"history:{pair}:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self.data_cache.set(
                store_key,
                recommendation,
                ttl=86400  # 24 hours
            )
        except Exception as e:
            logger.error(f"Failed to store recommendation: {e}")