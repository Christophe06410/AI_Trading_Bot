AI TRADING SYSTEM - OPERATIONAL GUIDELINES
1. SYSTEM ARCHITECTURE
Three Independent Services:
ML Pipeline: Model training and feature engineering
AI Service: Real-time prediction API (FastAPI)
Trading Bot: Executes trades based on AI recommendations

Communication Flow:
text
Trading Data → ML Pipeline → Trained Models → AI Service → Trading Bot → Market


2. DEPENDENCY MANAGEMENT
Required Software:
Python 3.9+
Redis (for caching)
8GB RAM minimum, 16GB recommended
Stable internet connection

Installation Order:
Install Redis
Install ML Pipeline
Install AI Service
Install Trading Bot

3. MODEL TRAINING PROTOCOL
Before Training:
Ensure 30+ days of historical data per symbol
Verify exchange API keys are valid
Check disk space (>5GB free)

Training Command:
# Standard training (one symbol)
python src/main.py --train --symbol "SOL/USDT"

# Batch training (multiple symbols)
python src/main.py --train-all

# With custom parameters
python src/main.py --train --symbol "BTC/USDT" --timeframe "15m" --days 60

Training Schedule:
Daily: Retrain RL agent

Weekly: Retrain ensemble models

Monthly: Full pipeline retrain with extended history

4. API USAGE GUIDELINES
Rate Limits:
Maximum 60 requests/minute per API key
Batch endpoint: Maximum 10 symbols per request
Health checks don't count toward limits

Response Interpretation:
Confidence ≥ 0.70: Strong signal, consider trading
Confidence 0.60-0.70: Moderate signal, use with caution
Confidence < 0.60: Weak signal, wait for better opportunity
Recommendation "CLOSE": Close specified position immediately

5. TRADING PARAMETERS
# Recommended settings
max_position_size: 1000 USD  # Per trade
max_portfolio_exposure: 0.10  # 10% of total capital
max_symbols_trading: 3        # Simultaneous trades

Risk Management:
Stop loss: 3-5% per trade
Take profit: 6-10% per trade
Maximum daily loss: 5% of capital
Maximum weekly drawdown: 15%

6. MONITORING CHECKLIST
Daily Checks:
API service responsive (curl http://localhost:8000/health)
Model accuracy > 60%
Win rate > 55%
Redis cache hit rate > 80%
No critical alerts in logs

Weekly Reviews:
Performance metrics trending
Model drift detection results
Feature importance analysis
Error rate analysis