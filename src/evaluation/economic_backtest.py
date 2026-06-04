"""
💰 ECONOMIC BACKTEST MODULE
==============================
Demonstrates the ECONOMIC VALUE of anomaly detection.

Quant journal requirement: "Does detecting anomalies make money 
or reduce risk for a portfolio manager?"

Strategy:
  1. S&P 500 Buy-and-Hold (baseline)
  2. Anomaly-Hedged Strategy: Reduce equity exposure to 0% when 
     anomaly is detected, hold cash until signal clears
  3. Regime-Aware Hedged: Same as (2) but with regime-specific 
     confidence weighting

Metrics reported:
  - Annualized Return
  - Annualized Volatility
  - Sharpe Ratio (rf = risk-free rate)
  - Maximum Drawdown (MDD)
  - Calmar Ratio (Return / MDD)
  - Sortino Ratio
  - Dollar savings per $100K invested
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class EconomicBacktest:
    """Run economic backtests to demonstrate anomaly detection value.
    
    Parameters
    ----------
    returns : np.ndarray
        Daily returns of the asset (S&P 500).
    risk_free_rate : float
        Annual risk-free rate (default 0.04 = 4%).
    """
    
    def __init__(self, returns: np.ndarray, 
                 risk_free_rate: float = 0.04,
                 dates: pd.DatetimeIndex = None):
        self.returns = returns
        self.rf = risk_free_rate
        self.rf_daily = (1 + risk_free_rate) ** (1/252) - 1
        self.dates = dates
        self.results = {}
    
    def _compute_metrics(self, daily_returns: np.ndarray, name: str) -> dict:
        """Compute comprehensive portfolio metrics."""
        # Cumulative wealth
        wealth = np.cumprod(1 + daily_returns)
        total_return = wealth[-1] - 1
        n_days = len(daily_returns)
        n_years = n_days / 252
        
        # Annualized return
        ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        
        # Annualized volatility
        ann_vol = np.std(daily_returns) * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = daily_returns - self.rf_daily
        sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * np.sqrt(252)
        
        # Maximum drawdown
        cum_max = np.maximum.accumulate(wealth)
        drawdowns = (cum_max - wealth) / cum_max
        max_dd = np.max(drawdowns)
        
        # Calmar ratio
        calmar = ann_return / (max_dd + 1e-10)
        
        # Sortino ratio (downside deviation)
        downside = daily_returns[daily_returns < 0]
        downside_std = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-10
        sortino = (ann_return - self.rf) / (downside_std + 1e-10)
        
        # Win rate
        win_rate = np.mean(daily_returns > 0)
        
        # Dollar value: final wealth on $100K
        dollar_100k = 100000 * wealth[-1]
        
        return {
            'strategy': name,
            'total_return': float(total_return),
            'ann_return': float(ann_return),
            'ann_volatility': float(ann_vol),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'calmar_ratio': float(calmar),
            'sortino_ratio': float(sortino),
            'win_rate': float(win_rate),
            'dollar_100k': float(dollar_100k),
            'n_days': n_days,
            'wealth_curve': wealth,
        }
    
    def run_buy_and_hold(self) -> dict:
        """Baseline: fully invested in S&P 500."""
        result = self._compute_metrics(self.returns, 'Buy-and-Hold S&P 500')
        self.results['buy_hold'] = result
        return result
    
    def run_anomaly_hedged(self, anomaly_signals: np.ndarray,
                           hedge_pct: float = 1.0,
                           lookahead_days: int = 5,
                           name: str = 'Anomaly-Hedged') -> dict:
        """Strategy: reduce exposure when anomaly detected.
        
        Parameters
        ----------
        anomaly_signals : np.ndarray
            Binary anomaly predictions (1 = anomaly, 0 = normal).
        hedge_pct : float
            Fraction of portfolio to hedge (1.0 = go fully to cash).
        lookahead_days : int
            How many days to stay hedged after an anomaly signal.
        """
        min_len = min(len(self.returns), len(anomaly_signals))
        returns = self.returns[:min_len]
        signals = anomaly_signals[:min_len]
        
        # Expand signals: stay hedged for lookahead_days after each alert
        hedged = np.zeros(min_len)
        for i in range(min_len):
            if signals[i] == 1:
                end = min(i + lookahead_days, min_len)
                hedged[i:end] = 1
        
        # Adjusted returns: hedged portion earns risk-free rate
        adj_returns = np.where(
            hedged == 1,
            (1 - hedge_pct) * returns + hedge_pct * self.rf_daily,
            returns
        )
        
        result = self._compute_metrics(adj_returns, name)
        result['n_hedged_days'] = int(hedged.sum())
        result['pct_time_hedged'] = float(hedged.mean())
        self.results[name] = result
        return result
    
    def run_regime_aware_hedged(self, anomaly_signals: np.ndarray,
                                 regimes: np.ndarray,
                                 regime_confidence: np.ndarray = None,
                                 lookahead_days: int = 5) -> dict:
        """Regime-aware hedging: hedge proportional to regime risk.
        
        High vol regime + anomaly = 100% hedge
        Med vol regime + anomaly = 70% hedge
        Low vol regime + anomaly = 40% hedge
        """
        min_len = min(len(self.returns), len(anomaly_signals), len(regimes))
        returns = self.returns[:min_len]
        signals = anomaly_signals[:min_len]
        regimes_aligned = regimes[:min_len]
        
        # Regime-specific hedge ratios
        regime_hedge = {0: 0.4, 1: 0.7, 2: 1.0}  # Low, Med, High
        
        hedged = np.zeros(min_len)
        hedge_pcts = np.zeros(min_len)
        
        for i in range(min_len):
            if signals[i] == 1:
                end = min(i + lookahead_days, min_len)
                r = regimes_aligned[i]
                h = regime_hedge.get(r, 0.7)
                hedged[i:end] = 1
                hedge_pcts[i:end] = np.maximum(hedge_pcts[i:end], h)
        
        adj_returns = np.where(
            hedged == 1,
            (1 - hedge_pcts) * returns + hedge_pcts * self.rf_daily,
            returns
        )
        
        result = self._compute_metrics(adj_returns, 'RC-EME Regime-Aware Hedged')
        result['n_hedged_days'] = int(hedged.sum())
        result['pct_time_hedged'] = float(hedged.mean())
        self.results['regime_hedged'] = result
        return result
    
    def comparison_table(self) -> pd.DataFrame:
        """Generate formatted comparison table."""
        rows = []
        for key, result in self.results.items():
            rows.append({
                'Strategy': result['strategy'],
                'Ann. Return': f"{result['ann_return']:.2%}",
                'Ann. Vol': f"{result['ann_volatility']:.2%}",
                'Sharpe': f"{result['sharpe_ratio']:.3f}",
                'Max DD': f"{result['max_drawdown']:.2%}",
                'Calmar': f"{result['calmar_ratio']:.3f}",
                'Sortino': f"{result['sortino_ratio']:.3f}",
                '$100K Final': f"${result['dollar_100k']:,.0f}",
            })
        return pd.DataFrame(rows)
    
    def dollar_savings(self) -> dict:
        """Calculate dollar savings from hedging."""
        if 'buy_hold' not in self.results:
            self.run_buy_and_hold()
        
        bh = self.results['buy_hold']
        savings = {}
        
        for key, result in self.results.items():
            if key == 'buy_hold':
                continue
            
            # Savings from avoiding drawdowns
            dd_reduction = bh['max_drawdown'] - result['max_drawdown']
            dollar_dd_saved = 100000 * dd_reduction
            
            # Dollar difference in final wealth
            dollar_diff = result['dollar_100k'] - bh['dollar_100k']
            
            savings[result['strategy']] = {
                'drawdown_reduction': float(dd_reduction),
                'dollar_drawdown_saved': float(dollar_dd_saved),
                'dollar_wealth_difference': float(dollar_diff),
                'sharpe_improvement': float(result['sharpe_ratio'] - bh['sharpe_ratio']),
            }
        
        return savings


class GARCHBaseline:
    """GARCH(1,1) baseline for anomaly detection.
    
    Traditional econometric approach: detect anomalies when 
    realized volatility exceeds GARCH-predicted volatility 
    by a threshold.
    """
    
    def __init__(self):
        self.omega = None
        self.alpha = None
        self.beta = None
    
    def fit(self, returns: np.ndarray):
        """Fit GARCH(1,1) using maximum likelihood (simplified)."""
        T = len(returns)
        
        # Initialize with sample variance
        h = np.zeros(T)
        h[0] = np.var(returns)
        
        # Grid search for GARCH params (simplified MLE)
        best_ll = -np.inf
        best_params = (0.00001, 0.1, 0.8)
        
        for omega in [0.000001, 0.00001, 0.0001]:
            for alpha in [0.05, 0.1, 0.15, 0.2]:
                for beta in [0.7, 0.75, 0.8, 0.85, 0.9]:
                    if alpha + beta >= 1.0:
                        continue
                    
                    h[0] = omega / (1 - alpha - beta + 1e-10)
                    for t in range(1, T):
                        h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
                        h[t] = max(h[t], 1e-10)
                    
                    # Log-likelihood
                    ll = -0.5 * np.sum(np.log(2 * np.pi * h) + returns**2 / h)
                    
                    if ll > best_ll:
                        best_ll = ll
                        best_params = (omega, alpha, beta)
        
        self.omega, self.alpha, self.beta = best_params
        
        # Compute conditional variance series
        self.h = np.zeros(T)
        self.h[0] = self.omega / (1 - self.alpha - self.beta + 1e-10)
        for t in range(1, T):
            self.h[t] = self.omega + self.alpha * returns[t-1]**2 + self.beta * self.h[t-1]
            self.h[t] = max(self.h[t], 1e-10)
        
        self.cond_vol = np.sqrt(self.h)
        return self
    
    def predict_anomalies(self, returns: np.ndarray, 
                           threshold_std: float = 2.5) -> np.ndarray:
        """Detect anomalies where |return| > threshold * GARCH vol."""
        T = len(returns)
        h = np.zeros(T)
        h[0] = self.omega / (1 - self.alpha - self.beta + 1e-10)
        
        for t in range(1, T):
            h[t] = self.omega + self.alpha * returns[t-1]**2 + self.beta * h[t-1]
            h[t] = max(h[t], 1e-10)
        
        vol = np.sqrt(h)
        anomalies = (np.abs(returns) > threshold_std * vol).astype(int)
        return anomalies


class VaRAnalysis:
    """Value-at-Risk and CVaR analysis per regime.
    
    Shows that regime conditioning produces better-calibrated 
    tail risk estimates.
    """
    
    def __init__(self, returns: np.ndarray, regimes: np.ndarray):
        self.returns = returns
        self.regimes = regimes
    
    def historical_var(self, confidence: float = 0.95) -> dict:
        """Compute Historical VaR and CVaR per regime."""
        alpha = 1 - confidence
        regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
        results = {}
        
        # Global (unconditional)
        var_global = np.percentile(self.returns, alpha * 100)
        cvar_global = np.mean(self.returns[self.returns <= var_global])
        results['Global'] = {
            'VaR': float(var_global),
            'CVaR': float(cvar_global),
            'n_samples': len(self.returns),
        }
        
        # Per-regime (conditional)
        for r in np.unique(self.regimes):
            mask = self.regimes == r
            r_returns = self.returns[mask]
            rname = regime_names.get(r, str(r))
            
            var_r = np.percentile(r_returns, alpha * 100)
            cvar_r = np.mean(r_returns[r_returns <= var_r]) if np.any(r_returns <= var_r) else var_r
            
            results[rname] = {
                'VaR': float(var_r),
                'CVaR': float(cvar_r),
                'n_samples': int(mask.sum()),
            }
        
        return results
    
    def var_violation_analysis(self, confidence: float = 0.95) -> dict:
        """Check VaR violation rates across regimes.
        
        Under correct calibration, violations should equal (1-confidence).
        Regime-conditioned VaR should be better calibrated.
        """
        alpha = 1 - confidence
        
        # Global VaR violations
        var_global = np.percentile(self.returns, alpha * 100)
        violations_global = np.mean(self.returns < var_global)
        
        # Regime-conditioned VaR violations
        regime_names = {0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'}
        violations_conditioned = {}
        
        for r in np.unique(self.regimes):
            mask = self.regimes == r
            r_returns = self.returns[mask]
            var_r = np.percentile(r_returns, alpha * 100)
            violations_conditioned[regime_names.get(r, str(r))] = {
                'expected_rate': float(alpha),
                'actual_rate': float(np.mean(r_returns < var_r)),
                'calibration_error': float(abs(np.mean(r_returns < var_r) - alpha)),
            }
        
        return {
            'global': {
                'expected_rate': float(alpha),
                'actual_rate': float(violations_global),
                'calibration_error': float(abs(violations_global - alpha)),
            },
            'per_regime': violations_conditioned,
        }
    
    def summary_table(self, confidence: float = 0.95) -> pd.DataFrame:
        """Generate VaR/CVaR summary table for the paper."""
        var_results = self.historical_var(confidence)
        rows = []
        for regime, metrics in var_results.items():
            rows.append({
                'Regime': regime,
                f'VaR ({confidence:.0%})': f"{metrics['VaR']:.4f}",
                f'CVaR ({confidence:.0%})': f"{metrics['CVaR']:.4f}",
                'N': metrics['n_samples'],
            })
        return pd.DataFrame(rows)


if __name__ == '__main__':
    # Demo with project data
    X_win = np.load('artifacts/market_windows_10f.npy')
    y = np.load('artifacts/market_labels.npy')
    
    # Extract returns from close prices
    returns = np.diff(X_win[:, -1, 0]) / (np.abs(X_win[:-1, -1, 0]) + 1e-10)
    
    # Simulate anomaly signals (use actual model predictions later)
    np.random.seed(42)
    vol = np.std(X_win[:, :, 0], axis=1)
    q33, q66 = np.quantile(vol, [0.33, 0.66])
    regimes = np.zeros(len(vol), dtype=int)
    regimes[vol <= q33] = 0
    regimes[(vol > q33) & (vol <= q66)] = 1
    regimes[vol > q66] = 2
    
    # Backtest
    bt = EconomicBacktest(returns[:len(y)-1])
    bh = bt.run_buy_and_hold()
    hedged = bt.run_anomaly_hedged(y[1:], name='RC-EME Hedged')
    regime_hedged = bt.run_regime_aware_hedged(y[1:], regimes[1:])
    
    print("\nECONOMIC BACKTEST RESULTS")
    print("=" * 80)
    print(bt.comparison_table().to_string(index=False))
    
    savings = bt.dollar_savings()
    print("\nDOLLAR SAVINGS")
    for strategy, s in savings.items():
        print(f"  {strategy}:")
        print(f"    MDD reduction:     {s['drawdown_reduction']:.2%}")
        print(f"    $ saved (drawdown): ${s['dollar_drawdown_saved']:,.0f} per $100K")
        print(f"    Sharpe improvement: {s['sharpe_improvement']:+.3f}")
    
    # GARCH baseline
    print("\nGARCH(1,1) BASELINE")
    print("=" * 80)
    garch = GARCHBaseline()
    garch.fit(returns)
    print(f"  GARCH params: omega={garch.omega:.6f}, alpha={garch.alpha:.4f}, beta={garch.beta:.4f}")
    garch_anomalies = garch.predict_anomalies(returns)
    print(f"  GARCH anomalies detected: {garch_anomalies.sum()} ({garch_anomalies.mean()*100:.1f}%)")
    
    # VaR analysis
    print("\nVaR/CVaR ANALYSIS")
    print("=" * 80)
    var_analysis = VaRAnalysis(returns, regimes[1:])
    print(var_analysis.summary_table().to_string(index=False))
    
    violations = var_analysis.var_violation_analysis()
    print(f"\n  Global VaR calibration error: {violations['global']['calibration_error']:.4f}")
    for regime, v in violations['per_regime'].items():
        print(f"  {regime} calibration error: {v['calibration_error']:.4f}")
