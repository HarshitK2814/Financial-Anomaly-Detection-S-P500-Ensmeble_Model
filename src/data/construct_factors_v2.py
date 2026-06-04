"""
🏗️ EXTENDED FACTOR CONSTRUCTION V2 — 32 Economic Factors
==========================================================
Extends the original 24-factor set with 8 additional financially-
motivated factors targeting tail risk, liquidity, and persistence.

Input:  (N_samples, Window_Size, N_Features)  — raw OHLCV windows
Output: (N_samples, 32_Factors)               — interpretable factors

NEW FACTORS (8):
  • Volume_Shock         — Abnormal trading volume deviation
  • Garman_Klass_Vol     — Superior intraday volatility estimator
  • Parkinson_Vol        — Range-based volatility (illiquid periods)
  • Amihud_Illiquidity   — Price impact per unit volume
  • Realized_Skew        — Crash asymmetry within window
  • Max_Drawdown         — Worst-case loss within window
  • VIX_Proxy            — Implied volatility approximation (ATR/Close)
  • Hurst_Exponent       — Mean-reversion vs trending indicator

EXISTING FACTORS (8 base → 24 with lags):
  • Return_Trend, Momentum_Factor, Composite_Volatility
  • Latent_State_A/B/C, Tail_Kurtosis, Tail_Skew
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ──────────────────────────────────────────────────────────────
# Helper functions for new factors
# ──────────────────────────────────────────────────────────────

def _volume_shock(X: np.ndarray, vol_channel: int = 3,
                  lookback: int = 20) -> np.ndarray:
    """Abnormal volume relative to rolling mean.

    V_shock = (V_last - mean(V_lookback)) / mean(V_lookback)
    Captures institutional flow anomalies.
    """
    volumes = X[:, :, vol_channel]  # (N, T)
    last_vol = volumes[:, -1]
    mean_vol = np.mean(volumes[:, -lookback:], axis=1) + 1e-8
    return (last_vol - mean_vol) / mean_vol


def _garman_klass_volatility(X: np.ndarray,
                              high_ch: int = 1,
                              low_ch: int = 2,
                              close_ch: int = 0,
                              open_ch: int = 0) -> np.ndarray:
    """Garman-Klass volatility estimator (1980).

    GK = 0.5 * log(H/L)^2 - (2*ln2 - 1) * log(C/O)^2
    More efficient than close-to-close vol for intraday data.

    Note: When OHLC are available. Falls back to regular vol if
    only returns (channel 0) are present.
    """
    n_features = X.shape[2]
    if n_features < 4:
        # Fallback: use return-based realized vol
        return np.std(X[:, :, 0], axis=1)

    H = np.abs(X[:, :, high_ch]) + 1e-8
    L = np.abs(X[:, :, low_ch]) + 1e-8
    C = np.abs(X[:, :, close_ch]) + 1e-8
    O = np.abs(X[:, :, open_ch]) + 1e-8

    hl = np.log(H / L + 1e-8)
    co = np.log(C / O + 1e-8)

    gk = 0.5 * np.mean(hl ** 2, axis=1) - (2 * np.log(2) - 1) * np.mean(co ** 2, axis=1)
    return np.sqrt(np.maximum(gk, 0))


def _parkinson_volatility(X: np.ndarray,
                           high_ch: int = 1,
                           low_ch: int = 2) -> np.ndarray:
    """Parkinson range-based volatility estimator (1980).

    Parkinson = sqrt( 1/(4*N*ln2) * sum(log(H/L))^2 )
    Better than close-to-close vol when volume is low.
    """
    n_features = X.shape[2]
    if n_features < 3:
        return np.std(X[:, :, 0], axis=1)

    H = np.abs(X[:, :, high_ch]) + 1e-8
    L = np.abs(X[:, :, low_ch]) + 1e-8
    hl = np.log(H / L + 1e-8)
    N = X.shape[1]
    return np.sqrt(np.mean(hl ** 2, axis=1) / (4 * np.log(2)))


def _amihud_illiquidity(X: np.ndarray,
                         return_ch: int = 0,
                         vol_ch: int = 3) -> np.ndarray:
    """Amihud (2002) illiquidity ratio.

    ILLIQ = mean( |r_t| / V_t )
    High values = large price impact per unit volume = illiquid.
    """
    n_features = X.shape[2]
    returns = np.abs(X[:, :, return_ch])
    if n_features > 3:
        volume = np.abs(X[:, :, vol_ch]) + 1e-8
        illiq = np.mean(returns / volume, axis=1)
    else:
        # No volume channel: use return magnitude as proxy
        illiq = np.mean(returns, axis=1)
    return illiq


def _realized_skewness(X: np.ndarray, return_ch: int = 0) -> np.ndarray:
    """Realized skewness of the return distribution within each window.

    Negative skew = crash asymmetry (left tail heavier).
    """
    return skew(X[:, :, return_ch], axis=1)


def _max_drawdown(X: np.ndarray, return_ch: int = 0) -> np.ndarray:
    """Maximum drawdown within each window.

    MDD = max_{s<t} (P_s - P_t) / P_s
    Computed from cumulative returns.
    """
    returns = X[:, :, return_ch]
    # Reconstruct cumulative price path from returns
    cum_returns = np.cumsum(returns, axis=1)
    running_max = np.maximum.accumulate(cum_returns, axis=1)
    drawdown = running_max - cum_returns
    max_dd = np.max(drawdown, axis=1)
    return max_dd


def _vix_proxy(X: np.ndarray, return_ch: int = 0,
               atr_window: int = 20) -> np.ndarray:
    """VIX-like implied volatility proxy.

    Uses rolling Average True Range normalized by current level.
    VIX_proxy = ATR_20 / |mean(close)|
    """
    returns = X[:, :, return_ch]
    # ATR approximation: rolling std of returns * sqrt(252)
    rolling_atr = np.std(returns[:, -atr_window:], axis=1) * np.sqrt(252)
    mean_level = np.abs(np.mean(returns, axis=1)) + 1e-8
    return rolling_atr / mean_level


def _hurst_exponent(X: np.ndarray, return_ch: int = 0,
                     max_lag: int = 20) -> np.ndarray:
    """Simplified Hurst exponent via R/S analysis.

    H < 0.5: mean-reverting
    H ≈ 0.5: random walk
    H > 0.5: trending

    Uses simplified computation for efficiency.
    """
    returns = X[:, :, return_ch]
    N = returns.shape[0]
    hurst_values = np.full(N, 0.5)  # Default to random walk

    for i in range(N):
        series = returns[i]
        lags = range(2, min(max_lag, len(series) // 2))
        if len(list(lags)) < 3:
            continue

        rs_values = []
        lag_values = []
        for lag in lags:
            subseries = series[:lag]
            mean_s = np.mean(subseries)
            deviate = np.cumsum(subseries - mean_s)
            R = np.max(deviate) - np.min(deviate)
            S = np.std(subseries, ddof=1) + 1e-8
            rs_values.append(R / S)
            lag_values.append(lag)

        if len(rs_values) >= 3:
            log_lags = np.log(lag_values)
            log_rs = np.log(np.array(rs_values) + 1e-8)
            # Linear regression: H = slope of log(R/S) vs log(n)
            try:
                slope = np.polyfit(log_lags, log_rs, 1)[0]
                hurst_values[i] = np.clip(slope, 0.0, 1.0)
            except (np.linalg.LinAlgError, ValueError):
                pass

    return hurst_values


# ──────────────────────────────────────────────────────────────
# Main factor construction
# ──────────────────────────────────────────────────────────────

def construct_factors_v2(input_path: str, output_dir: str,
                          labels_path: str = None,
                          include_lags: bool = True,
                          lag_periods: list = None) -> pd.DataFrame:
    """Construct extended 32-factor set from raw windowed data.

    Parameters
    ----------
    input_path : str
        Path to numpy array of shape (N, T, F) — windowed data.
    output_dir : str
        Directory to save factor CSV and NPY files.
    labels_path : str, optional
        Path to labels array. If provided, aligns labels after lag.
    include_lags : bool
        Whether to add temporal lag features (default True).
    lag_periods : list
        Lag periods to use (default [1, 5]).

    Returns
    -------
    pd.DataFrame
        Factor dataframe with all 32 (or 96 with lags) features.
    """
    if lag_periods is None:
        lag_periods = [1, 5]

    print("🏗️ CONSTRUCTING EXTENDED FACTORS V2 (32 base features)")
    print("=" * 55)

    # Load raw data
    X = np.load(input_path)
    print(f"Input Shape: {X.shape}")

    df = pd.DataFrame()

    # ── ORIGINAL FACTORS (8 base) ──────────────────────────────

    # F0: Returns → Trend/Drift
    df['Return_Trend'] = np.mean(X[:, :, 0], axis=1)

    # F1: Momentum Proxy
    df['Momentum_Factor'] = np.mean(X[:, :, 1], axis=1) if X.shape[2] > 1 \
        else np.mean(X[:, :, 0], axis=1)

    # F2–F9: Volatility Proxies → Composite
    if X.shape[2] >= 10:
        vol_channels = X[:, :, [2, 3, 7, 8, 9]]
    else:
        vol_channels = X[:, :, :min(X.shape[2], 3)]
    df['Composite_Volatility'] = np.mean(np.mean(vol_channels, axis=2), axis=1)

    # Latent/Microstructure States
    for idx, name in enumerate(['Latent_State_A', 'Latent_State_B', 'Latent_State_C']):
        ch = 4 + idx
        if ch < X.shape[2]:
            df[name] = np.mean(X[:, :, ch], axis=1)
        else:
            df[name] = 0.0

    # Tail Risk
    df['Tail_Kurtosis'] = kurtosis(X[:, :, 0], axis=1)
    df['Tail_Skew'] = skew(X[:, :, 0], axis=1)

    print(f"  Original factors: {df.shape[1]}")

    # ── NEW FACTORS (8 additional) ─────────────────────────────

    # Volume Shock
    if X.shape[2] > 3:
        df['Volume_Shock'] = _volume_shock(X, vol_channel=3)
    else:
        df['Volume_Shock'] = np.max(np.abs(X[:, :, 0]), axis=1)
    print("  + Volume_Shock")

    # Garman-Klass Volatility
    df['Garman_Klass_Vol'] = _garman_klass_volatility(X)
    print("  + Garman_Klass_Vol")

    # Parkinson Volatility
    df['Parkinson_Vol'] = _parkinson_volatility(X)
    print("  + Parkinson_Vol")

    # Amihud Illiquidity
    df['Amihud_Illiquidity'] = _amihud_illiquidity(X)
    print("  + Amihud_Illiquidity")

    # Realized Skewness (separate from Tail_Skew for clarity)
    df['Realized_Skew'] = _realized_skewness(X)
    print("  + Realized_Skew")

    # Max Drawdown
    df['Max_Drawdown'] = _max_drawdown(X)
    print("  + Max_Drawdown")

    # VIX Proxy
    df['VIX_Proxy'] = _vix_proxy(X)
    print("  + VIX_Proxy")

    # Hurst Exponent
    df['Hurst_Exponent'] = _hurst_exponent(X)
    print("  + Hurst_Exponent")

    base_count = df.shape[1]
    print(f"\n  Total base factors: {base_count}")

    # ── TEMPORAL LAGS ──────────────────────────────────────────

    if include_lags:
        cols_to_lag = df.columns.tolist()
        for lag in lag_periods:
            for col in cols_to_lag:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

        # Drop NaN from lags
        n_before = len(df)
        df.dropna(inplace=True)
        n_after = len(df)
        print(f"  Dropped {n_before - n_after} samples from lags")

    print(f"\n  Final factor count: {df.shape[1]}")
    print(f"  Final sample count: {len(df)}")

    # ── HANDLE LABELS ──────────────────────────────────────────

    y = None
    if labels_path and os.path.exists(labels_path):
        y_full = np.load(labels_path)
        y = y_full[df.index]
        print(f"  Labels aligned: {y.sum()} anomalies / {len(y)} total")

    # ── SAVE ───────────────────────────────────────────────────

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'journal_factors_v2.csv')
    npy_path = os.path.join(output_dir, 'X_factors_v2.npy')
    df.to_csv(csv_path, index=False)
    np.save(npy_path, df.values)

    if y is not None:
        np.save(os.path.join(output_dir, 'y_factors_v2.npy'), y)

    print(f"\n💾 Saved: {csv_path}")
    print(f"💾 Saved: {npy_path}")

    # ── FACTOR SUMMARY TABLE ──────────────────────────────────

    factor_info = {
        'Return_Trend':         ('Trend',       'Medium-term price drift'),
        'Momentum_Factor':      ('Momentum',    'Directional persistence'),
        'Composite_Volatility': ('Volatility',  'Market instability composite'),
        'Latent_State_A':       ('Microstructure', 'Latent state channel A'),
        'Latent_State_B':       ('Microstructure', 'Latent state channel B'),
        'Latent_State_C':       ('Microstructure', 'Range-based liquidity proxy'),
        'Tail_Kurtosis':        ('Tail Risk',   'Excess kurtosis (fat tails)'),
        'Tail_Skew':            ('Tail Risk',   'Return skewness'),
        'Volume_Shock':         ('Liquidity',   'Abnormal volume deviation'),
        'Garman_Klass_Vol':     ('Volatility',  'Intraday vol estimator (GK 1980)'),
        'Parkinson_Vol':        ('Volatility',  'Range-based vol (Parkinson 1980)'),
        'Amihud_Illiquidity':   ('Liquidity',   'Price impact per unit volume'),
        'Realized_Skew':        ('Tail Risk',   'Within-window crash asymmetry'),
        'Max_Drawdown':         ('Tail Risk',   'Worst-case loss in window'),
        'VIX_Proxy':            ('Volatility',  'Implied vol approximation'),
        'Hurst_Exponent':       ('Persistence', 'Mean-reversion vs trending'),
    }

    print("\n📊 FACTOR TAXONOMY:")
    print(f"{'Factor':<25} {'Category':<15} {'Description'}")
    print("-" * 75)
    for name, (cat, desc) in factor_info.items():
        print(f"  {name:<23} {cat:<15} {desc}")

    return df


# ──────────────────────────────────────────────────────────────
# Factor metadata for downstream XAI modules
# ──────────────────────────────────────────────────────────────

FACTOR_METADATA = {
    'Return_Trend':         {'category': 'Trend',          'economic_meaning': 'Medium-term price drift'},
    'Momentum_Factor':      {'category': 'Momentum',       'economic_meaning': 'Directional persistence (RSI-like)'},
    'Composite_Volatility': {'category': 'Volatility',     'economic_meaning': 'Market instability composite'},
    'Latent_State_A':       {'category': 'Microstructure', 'economic_meaning': 'Latent state channel A'},
    'Latent_State_B':       {'category': 'Microstructure', 'economic_meaning': 'Latent state channel B'},
    'Latent_State_C':       {'category': 'Microstructure', 'economic_meaning': 'Range-based liquidity proxy'},
    'Tail_Kurtosis':        {'category': 'Tail Risk',      'economic_meaning': 'Excess kurtosis (fat tails)'},
    'Tail_Skew':            {'category': 'Tail Risk',      'economic_meaning': 'Return distribution skewness'},
    'Volume_Shock':         {'category': 'Liquidity',      'economic_meaning': 'Abnormal volume deviation'},
    'Garman_Klass_Vol':     {'category': 'Volatility',     'economic_meaning': 'Intraday vol estimator (Garman-Klass 1980)'},
    'Parkinson_Vol':        {'category': 'Volatility',     'economic_meaning': 'Range-based vol (Parkinson 1980)'},
    'Amihud_Illiquidity':   {'category': 'Liquidity',      'economic_meaning': 'Price impact per unit volume (Amihud 2002)'},
    'Realized_Skew':        {'category': 'Tail Risk',      'economic_meaning': 'Within-window crash asymmetry'},
    'Max_Drawdown':         {'category': 'Tail Risk',      'economic_meaning': 'Worst-case loss in window'},
    'VIX_Proxy':            {'category': 'Volatility',     'economic_meaning': 'Implied volatility approximation via ATR'},
    'Hurst_Exponent':       {'category': 'Persistence',    'economic_meaning': 'Mean-reversion (H<0.5) vs trending (H>0.5)'},
}

# Base factor names (before lags)
BASE_FACTOR_NAMES = list(FACTOR_METADATA.keys())


if __name__ == '__main__':
    construct_factors_v2(
        input_path='artifacts/market_windows_10f.npy',
        output_dir='artifacts',
        labels_path='artifacts/market_labels.npy',
    )
