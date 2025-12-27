"""
Main orchestrator for ML Pipeline
Trains and evaluates trading models
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collector import DataCollector
from src.feature_engineer import FeatureEngineer
from src.ensemble_model import TradingEnsembleModel
from src.reinforcement_learner import RLTrader

class MLPipeline:
    """Complete ML pipeline for trading predictions"""
    
    def __init__(self, config_path: str = "config/ml_config.yaml"):
        self.config = self._load_config(config_path)
        self.data_collector = DataCollector(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.ensemble_model = None
        self.rl_trader = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def run_pipeline(self, symbol: str = "SOL/USDT", timeframe: str = "5m"):
        """Run complete ML pipeline"""
        
        print("=" * 60)
        print("ML TRADING PIPELINE")
        print("=" * 60)
        
        # Step 1: Collect data
        print("\n1️⃣ COLLECTING DATA...")
        df = self.data_collector.collect_data(symbol, timeframe)
        print(f"   Collected {len(df)} candles for {symbol}")
        
        # Step 2: Feature engineering
        print("\n2️⃣ ENGINEERING FEATURES...")
        df_features = self.feature_engineer.create_all_features(df)
        
        # Step 3: Train ensemble model
        print("\n3️⃣ TRAINING ENSEMBLE MODEL...")
        self.ensemble_model = TradingEnsembleModel(self.config)
        
        # Prepare features and target
        X, y = self.ensemble_model.prepare_features(df_features)
        
        # Train model
        performance = self.ensemble_model.train(X, y)
        
        # Save model
        model_path = f"models/ensemble_model_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.joblib"
        self.ensemble_model.save(model_path)
        
        # Step 4: Train reinforcement learning agent
        print("\n4️⃣ TRAINING REINFORCEMENT LEARNING AGENT...")
        self.rl_trader = RLTrader(self.config)
        
        # Select best features for RL
        best_features = self.feature_engineer.select_best_features(
            df_features, y, 
            top_k=self.config['training']['feature_selection']['top_k_features']
        )
        
        # Prepare data for RL
        df_rl = df_features[best_features + ['close']].copy()
        
        # Train RL agent
        rl_performance = self.rl_trader.train(df_rl, best_features)
        
        # Save RL model
        rl_path = f"models/rl_model_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}"
        self.rl_trader.save(rl_path)
        
        # Step 5: Generate final report
        print("\n5️⃣ GENERATING REPORT...")
        self._generate_report(performance, rl_performance, model_path, rl_path)
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 60)
        
        return {
            'ensemble_model': self.ensemble_model,
            'rl_trader': self.rl_trader,
            'performance': performance,
            'rl_performance': rl_performance
        }
    
    def _generate_report(self, ensemble_perf: dict, rl_perf: dict, model_path: str, rl_path: str):
        """Generate performance report"""
        
        report = f"""
        ML TRADING PIPELINE REPORT
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        ===================================================
        
        ENSEMBLE MODEL PERFORMANCE:
        ---------------------------
        Accuracy: {ensemble_perf.get('ensemble', {}).get('accuracy', 0):.3f}
        Precision: {ensemble_perf.get('ensemble', {}).get('precision', 0):.3f}
        Recall: {ensemble_perf.get('ensemble', {}).get('recall', 0):.3f}
        F1 Score: {ensemble_perf.get('ensemble', {}).get('f1', 0):.3f}
        ROC AUC: {ensemble_perf.get('ensemble', {}).get('roc_auc', 0):.3f}
        
        REINFORCEMENT LEARNING PERFORMANCE:
        -----------------------------------
        Total Trades: {rl_perf.get('total_trades', 0)}
        Win Rate: {rl_perf.get('win_rate', 0):.1%}
        Total PnL: ${rl_perf.get('total_pnl', 0):.2f}
        Sharpe Ratio: {rl_perf.get('sharpe_ratio', 0):.2f}
        Max Drawdown: {rl_perf.get('max_drawdown', 0):.1%}
        
        MODEL FILES:
        ------------
        Ensemble Model: {model_path}
        RL Model: {rl_path}.zip
        
        NEXT STEPS:
        ----------
        1. Deploy models to AI Service
        2. Integrate with Trading Bot
        3. Monitor performance
        4. Retrain weekly
        
        ===================================================
        """
        
        # Save report
        report_path = f"reports/pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs('reports', exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"   Report saved to: {report_path}")
        print(report)
    
    def predict(self, candles: list) -> dict:
        """
        Make prediction using trained models
        
        Args:
            candles: List of OHLCV candles
            
        Returns:
            Dictionary with predictions from all models
        """
        
        if self.ensemble_model is None:
            raise ValueError("Models not trained. Run pipeline first.")
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Create features
        df_features = self.feature_engineer.create_all_features(df)
        
        # Get latest features
        latest_features = df_features.iloc[-1:].drop(
            columns=['timestamp'] if 'timestamp' in df_features.columns else []
        )
        
        # Get ensemble prediction
        ensemble_pred = self.ensemble_model.predict(latest_features)
        
        # Get RL prediction if available
        rl_pred = None
        if self.rl_trader:
            # Prepare observation for RL
            best_features = self.feature_engineer.select_best_features(
                df_features, pd.Series([0] * len(df_features)),  # Dummy target
                top_k=self.config['training']['feature_selection']['top_k_features']
            )
            
            # Get RL observation
            observation = self.rl_trader.env._get_observation() if hasattr(self.rl_trader, 'env') else None
            
            if observation is not None:
                action, confidence = self.rl_trader.predict(observation)
                action_map = {0: "WAIT", 1: "LONG", 2: "SHORT", 3: "CLOSE"}
                rl_pred = {
                    'action': action_map.get(action, "WAIT"),
                    'confidence': confidence,
                    'action_id': action
                }
        
        # Combine predictions
        final_prediction = ensemble_pred['final_recommendation']
        
        # Override with RL if confidence is high
        if rl_pred and rl_pred['confidence'] > 0.7:
            final_prediction = {
                'recommendation': rl_pred['action'],
                'confidence': rl_pred['confidence'],
                'reasoning': f"RL Agent recommendation (confidence: {rl_pred['confidence']:.1%})",
                'source': 'rl_agent'
            }
        else:
            final_prediction['source'] = 'ensemble_model'
        
        return {
            'ensemble_prediction': ensemble_pred,
            'rl_prediction': rl_pred,
            'final_prediction': final_prediction,
            'timestamp': datetime.now().isoformat(),
            'features_used': len(latest_features.columns)
        }


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Trading Pipeline")
    parser.add_argument("--symbol", default="SOL/USDT", help="Trading symbol")
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--predict", action="store_true", help="Make prediction")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLPipeline()
    
    if args.train:
        # Train models
        pipeline.run_pipeline(args.symbol, args.timeframe)
    
    elif args.predict:
        # Example prediction
        example_candles = [
            {
                'timestamp': '2024-01-15T10:00:00Z',
                'open': 95.50, 'high': 96.20, 'low': 94.80, 'close': 95.80, 'volume': 1500.0
            },
            {
                'timestamp': '2024-01-15T10:05:00Z',
                'open': 95.80, 'high': 96.50, 'low': 95.20, 'close': 96.00, 'volume': 1800.0
            }
        ]
        
        prediction = pipeline.predict(example_candles)
        print("\n📊 Prediction Results:")
        print(f"Recommendation: {prediction['final_prediction']['recommendation']}")
        print(f"Confidence: {prediction['final_prediction']['confidence']:.1%}")
        print(f"Source: {prediction['final_prediction']['source']}")
    
    else:
        print("Please specify --train or --predict")

if __name__ == "__main__":
    main()
