"""
Reinforcement Learning for optimal trading decisions
Learns when to enter/exit positions based on market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class TradingEnvironment(gym.Env):
    """Custom trading environment for reinforcement learning"""
    
    def __init__(self, df: pd.DataFrame, features: List[str], config: Dict):
        super(TradingEnvironment, self).__init__()
        
        self.df = df
        self.features = features
        self.config = config
        self.current_step = 0
        
        # Action space: 0=WAIT, 1=LONG, 2=SHORT, 3=CLOSE
        self.action_space = spaces.Discrete(4)
        
        # Observation space: features + portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(features) + 4,),  # Features + portfolio state
            dtype=np.float32
        )
        
        # Trading parameters
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.position = None  # None, LONG, or SHORT
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
        # Track performance
        self.trades = []
        self.portfolio_values = []
        
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trades = []
        self.portfolio_values = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        
        # Market features
        if self.current_step < len(self.df):
            market_features = self.df.iloc[self.current_step][self.features].values.astype(np.float32)
        else:
            market_features = np.zeros(len(self.features), dtype=np.float32)
        
        # Portfolio state
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            1.0 if self.position == 'LONG' else 0.0,  # Is LONG
            1.0 if self.position == 'SHORT' else 0.0,  # Is SHORT
            self.position_size if self.position else 0.0  # Position size
        ], dtype=np.float32)
        
        return np.concatenate([market_features, portfolio_state])
    
    def _get_current_price(self) -> float:
        """Get current price"""
        if self.current_step < len(self.df):
            return float(self.df.iloc[self.current_step]['close'])
        return 0.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        
        current_price = self._get_current_price()
        reward = 0.0
        done = False
        
        # Execute action
        if action == 0:  # WAIT
            reward = self._action_wait(current_price)
            
        elif action == 1:  # LONG
            reward = self._action_long(current_price)
            
        elif action == 2:  # SHORT
            reward = self._action_short(current_price)
            
        elif action == 3:  # CLOSE
            reward = self._action_close(current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.df) - 1:
            done = True
        
        # Update portfolio value
        portfolio_value = self._calculate_portfolio_value(current_price)
        self.portfolio_values.append(portfolio_value)
        
        # Get observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': portfolio_value,
            'step': self.current_step
        }
        
        return observation, reward, done, info
    
    def _action_wait(self, current_price: float) -> float:
        """Wait action"""
        reward = 0.0
        
        # Check if position hit stop loss or take profit
        if self.position:
            if self.position == 'LONG':
                if current_price <= self.stop_loss:
                    # Stop loss hit
                    pnl = (current_price - self.entry_price) * self.position_size
                    self.balance += pnl
                    self.position = None
                    reward = -1.0  # Penalty for stop loss
                    
                elif current_price >= self.take_profit:
                    # Take profit hit
                    pnl = (current_price - self.entry_price) * self.position_size
                    self.balance += pnl
                    self.position = None
                    reward = 1.0  # Reward for take profit
                    
            elif self.position == 'SHORT':
                if current_price >= self.stop_loss:
                    pnl = (self.entry_price - current_price) * self.position_size
                    self.balance += pnl
                    self.position = None
                    reward = -1.0
                    
                elif current_price <= self.take_profit:
                    pnl = (self.entry_price - current_price) * self.position_size
                    self.balance += pnl
                    self.position = None
                    reward = 1.0
        
        return reward
    
    def _action_long(self, current_price: float) -> float:
        """Enter LONG position"""
        
        # Close existing position if any
        close_reward = 0.0
        if self.position:
            close_reward = self._action_close(current_price)
        
        # Calculate position size (Kelly criterion simplified)
        position_value = self.balance * 0.1  # 10% of balance
        position_size = position_value / current_price
        
        # Enter LONG position
        self.position = 'LONG'
        self.position_size = position_size
        self.entry_price = current_price
        self.stop_loss = current_price * 0.97  # 3% stop loss
        self.take_profit = current_price * 1.06  # 6% take profit
        
        # Small penalty for entering position (encourages careful entries)
        return close_reward - 0.1
    
    def _action_short(self, current_price: float) -> float:
        """Enter SHORT position"""
        
        # Close existing position if any
        close_reward = 0.0
        if self.position:
            close_reward = self._action_close(current_price)
        
        # Calculate position size
        position_value = self.balance * 0.1  # 10% of balance
        position_size = position_value / current_price
        
        # Enter SHORT position
        self.position = 'SHORT'
        self.position_size = position_size
        self.entry_price = current_price
        self.stop_loss = current_price * 1.03  # 3% stop loss
        self.take_profit = current_price * 0.94  # 6% take profit
        
        return close_reward - 0.1
    
    def _action_close(self, current_price: float) -> float:
        """Close current position"""
        
        if not self.position:
            return 0.0  # No position to close
        
        # Calculate PnL
        if self.position == 'LONG':
            pnl = (current_price - self.entry_price) * self.position_size
        else:  # SHORT
            pnl = (self.entry_price - current_price) * self.position_size
        
        # Update balance
        self.balance += pnl
        
        # Record trade
        self.trades.append({
            'entry_price': self.entry_price,
            'exit_price': current_price,
            'position': self.position,
            'pnl': pnl,
            'step': self.current_step
        })
        
        # Reset position
        self.position = None
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
        # Reward proportional to PnL (normalized)
        reward = pnl / (self.entry_price * self.position_size) if self.entry_price > 0 else 0.0
        
        return reward
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        
        if not self.position:
            return self.balance
        
        if self.position == 'LONG':
            position_value = self.position_size * current_price
        else:  # SHORT
            # For SHORT, we owe the position
            position_value = self.position_size * (2 * self.entry_price - current_price)
        
        return self.balance + position_value
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate trading performance metrics"""
        
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        # Calculate metrics
        trades_df = pd.DataFrame(self.trades)
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        total_pnl = trades_df['pnl'].sum()
        
        # Sharpe ratio (simplified)
        returns = trades_df['pnl'] / (trades_df['entry_price'] * self.position_size)
        sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0.0
        
        # Max drawdown from portfolio values
        portfolio_series = pd.Series(self.portfolio_values)
        rolling_max = portfolio_series.expanding().max()
        drawdowns = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }


class RLTrader:
    """Reinforcement Learning trading agent"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.env = None
        
    def train(self, df: pd.DataFrame, features: List[str]):
        """Train RL agent"""
        
        print("🧠 Training Reinforcement Learning Agent...")
        
        # Create environment
        self.env = TradingEnvironment(df, features, self.config)
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Initialize PPO agent
        self.model = PPO(
            'MlpPolicy',
            vec_env,
            learning_rate=self.config['reinforcement_learning']['training']['learning_rate'],
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=self.config['reinforcement_learning']['training']['gamma'],
            gae_lambda=self.config['reinforcement_learning']['training']['gae_lambda'],
            verbose=1
        )
        
        # Train agent
        total_timesteps = self.config['reinforcement_learning']['training']['total_timesteps']
        self.model.learn(total_timesteps=total_timesteps)
        
        print("✅ RL Agent trained!")
        
        # Evaluate performance
        performance = self.env.get_performance_metrics()
        print(f"   Total Trades: {performance['total_trades']}")
        print(f"   Win Rate: {performance['win_rate']:.1%}")
        print(f"   Total PnL: ${performance['total_pnl']:.2f}")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        
        return performance
    
    def predict(self, observation: np.ndarray) -> Tuple[int, float]:
        """Predict action given observation"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        action, _ = self.model.predict(observation, deterministic=True)
        
        # Get action probabilities
        action_probs = self.model.policy.get_distribution(self.model.policy.obs_to_tensor(observation)[0]).distribution.probs
        confidence = float(action_probs[0, action].item())
        
        return int(action), confidence
    
    def save(self, path: str):
        """Save trained model"""
        if self.model:
            self.model.save(path)
            print(f"✅ RL model saved to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        self.model = PPO.load(path)
        print(f"✅ RL model loaded from {path}")
