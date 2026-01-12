import numpy as np
import pandas as pd

class PositionSizer:
    """
    Dynamic Position Sizing based on:
    1. Signal probability (confidence)
    2. Volatility (ATR)
    3. Kelly Criterion
    """
    
    def __init__(self, 
                 base_risk_pct=2.0,      # Base risk per trade (% of portfolio)
                 max_risk_pct=5.0,       # Maximum risk per trade
                 min_risk_pct=0.5,       # Minimum risk per trade
                 kelly_fraction=0.25):   # Fraction of Kelly to use (quarter-Kelly)
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.min_risk_pct = min_risk_pct
        self.kelly_fraction = kelly_fraction
    
    def calculate_kelly(self, win_rate, win_loss_ratio):
        """
        Kelly Criterion: f* = (bp - q) / b
        where b = win/loss ratio, p = win probability, q = 1 - p
        """
        if win_rate <= 0 or win_loss_ratio <= 0:
            return 0
        
        b = win_loss_ratio
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Use fractional Kelly for safety
        return max(0, kelly * self.kelly_fraction)
    
    def adjust_for_probability(self, base_size, probability, threshold=0.5):
        """
        Scale position size based on signal probability
        Higher probability = larger position
        """
        if probability <= threshold:
            return 0
        
        # Linear scaling: prob 0.5 -> 0.5x, prob 1.0 -> 1.5x
        confidence_multiplier = 0.5 + (probability - threshold) * 2
        return base_size * confidence_multiplier
    
    def adjust_for_volatility(self, base_size, current_atr_pct, avg_atr_pct):
        """
        Inverse volatility sizing:
        - High volatility -> smaller position
        - Low volatility -> larger position
        """
        if current_atr_pct <= 0 or avg_atr_pct <= 0:
            return base_size
        
        volatility_ratio = avg_atr_pct / current_atr_pct
        # Clamp between 0.5x and 2x
        volatility_multiplier = np.clip(volatility_ratio, 0.5, 2.0)
        
        return base_size * volatility_multiplier
    
    def calculate_position_size(self, 
                                 portfolio_value,
                                 probability,
                                 current_atr_pct,
                                 avg_atr_pct,
                                 historical_win_rate=None,
                                 win_loss_ratio=None,
                                 stop_loss_pct=1.5):
        """
        Calculate optimal position size combining all factors
        
        Returns:
            position_value: Dollar amount to invest
            risk_pct: Actual risk percentage used
        """
        # Start with base risk
        risk_pct = self.base_risk_pct
        
        # 1. Apply Kelly Criterion if historical data available
        if historical_win_rate is not None and win_loss_ratio is not None:
            kelly_pct = self.calculate_kelly(historical_win_rate, win_loss_ratio) * 100
            if kelly_pct > 0:
                risk_pct = min(risk_pct, kelly_pct)
        
        # 2. Adjust for signal probability
        risk_pct = self.adjust_for_probability(risk_pct, probability)
        
        # 3. Adjust for volatility
        risk_pct = self.adjust_for_volatility(risk_pct, current_atr_pct, avg_atr_pct)
        
        # 4. Clamp to min/max
        risk_pct = np.clip(risk_pct, self.min_risk_pct, self.max_risk_pct)
        
        # Calculate position size based on stop loss
        # If SL is X%, and we want to risk Y% of portfolio, position size = Y/X * portfolio
        position_pct = (risk_pct / stop_loss_pct) * 100
        position_pct = min(position_pct, 100)  # Can't exceed 100% of portfolio (no leverage)
        
        position_value = portfolio_value * (position_pct / 100)
        
        return position_value, risk_pct
    
    def calculate_batch_sizes(self, df, probabilities, portfolio_value, stop_loss_pct=1.5):
        """
        Calculate position sizes for a batch of signals
        """
        sizes = []
        risks = []
        
        avg_atr = df['atr_pct'].mean() if 'atr_pct' in df.columns else 1.0
        
        for i, prob in enumerate(probabilities):
            current_atr = df.iloc[i]['atr_pct'] if 'atr_pct' in df.columns else avg_atr
            
            size, risk = self.calculate_position_size(
                portfolio_value=portfolio_value,
                probability=prob,
                current_atr_pct=current_atr,
                avg_atr_pct=avg_atr,
                stop_loss_pct=stop_loss_pct
            )
            sizes.append(size)
            risks.append(risk)
        
        return np.array(sizes), np.array(risks)


class DrawdownProtection:
    """
    Maximum Drawdown Protection System
    """
    
    def __init__(self,
                 max_drawdown_pct=10.0,     # Stop trading if DD exceeds this
                 warning_drawdown_pct=5.0,  # Reduce position size at this level
                 recovery_threshold_pct=3.0, # Resume full trading when DD recovers to this
                 cooldown_periods=48):       # Periods to wait after max DD hit (48 = 12 hours on 15m)
        self.max_drawdown_pct = max_drawdown_pct
        self.warning_drawdown_pct = warning_drawdown_pct
        self.recovery_threshold_pct = recovery_threshold_pct
        self.cooldown_periods = cooldown_periods
        
        self.peak_value = None
        self.is_stopped = False
        self.cooldown_remaining = 0
    
    def update(self, current_value):
        """Update peak and check drawdown"""
        if self.peak_value is None:
            self.peak_value = current_value
            return 1.0, False  # Full size, not stopped
        
        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.is_stopped = False
        
        # Calculate current drawdown
        drawdown_pct = (self.peak_value - current_value) / self.peak_value * 100
        
        # Check if in cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return 0.0, True
        
        # Check max drawdown
        if drawdown_pct >= self.max_drawdown_pct:
            self.is_stopped = True
            self.cooldown_remaining = self.cooldown_periods
            return 0.0, True
        
        # Check warning level
        if drawdown_pct >= self.warning_drawdown_pct:
            # Reduce position size proportionally
            reduction = 1 - (drawdown_pct - self.warning_drawdown_pct) / (self.max_drawdown_pct - self.warning_drawdown_pct)
            return max(0.25, reduction), False  # Minimum 25% size
        
        return 1.0, False
    
    def reset(self):
        """Reset the protection system"""
        self.peak_value = None
        self.is_stopped = False
        self.cooldown_remaining = 0


class RiskManager:
    """
    Combined Risk Management System
    """
    
    def __init__(self, initial_portfolio=10000):
        self.position_sizer = PositionSizer()
        self.drawdown_protection = DrawdownProtection()
        self.portfolio_value = initial_portfolio
        self.initial_portfolio = initial_portfolio
        self.trade_history = []
        
    def get_position_size(self, probability, current_atr_pct, avg_atr_pct, stop_loss_pct=1.5):
        """Get position size with all protections applied"""
        
        # Check drawdown protection
        dd_multiplier, is_stopped = self.drawdown_protection.update(self.portfolio_value)
        
        if is_stopped:
            return 0.0, 0.0, "STOPPED"
        
        # Calculate base position size
        position_value, risk_pct = self.position_sizer.calculate_position_size(
            portfolio_value=self.portfolio_value,
            probability=probability,
            current_atr_pct=current_atr_pct,
            avg_atr_pct=avg_atr_pct,
            stop_loss_pct=stop_loss_pct
        )
        
        # Apply drawdown reduction
        position_value *= dd_multiplier
        
        status = "FULL" if dd_multiplier == 1.0 else f"REDUCED_{int(dd_multiplier*100)}%"
        
        return position_value, risk_pct, status
    
    def update_portfolio(self, pnl):
        """Update portfolio value after a trade"""
        self.portfolio_value += pnl
        self.trade_history.append({
            'pnl': pnl,
            'portfolio': self.portfolio_value
        })
    
    def get_stats(self):
        """Get risk management statistics"""
        if not self.trade_history:
            return {}
        
        pnls = [t['pnl'] for t in self.trade_history]
        portfolios = [t['portfolio'] for t in self.trade_history]
        
        # Calculate max drawdown
        peak = self.initial_portfolio
        max_dd = 0
        for p in portfolios:
            if p > peak:
                peak = p
            dd = (peak - p) / peak * 100
            max_dd = max(max_dd, dd)
        
        return {
            'total_return_pct': (self.portfolio_value - self.initial_portfolio) / self.initial_portfolio * 100,
            'max_drawdown_pct': max_dd,
            'total_trades': len(pnls),
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0,
            'avg_win': np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0,
            'avg_loss': np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0,
        }
