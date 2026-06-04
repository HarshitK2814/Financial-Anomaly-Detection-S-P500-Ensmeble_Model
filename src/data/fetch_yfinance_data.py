"""
⬇️ FETCH 10-YEAR DATA (YFINANCE)
===============================
Downloads S&P 500 (^GSPC) data for the last 10 years.
Engineers 10 features to replace the 'Unknown' set.
Creates: artifacts/sp500_10y_windows.npy

Features (0-9):
0: Log Returns (Anchor)
1: RSI (Momentum)
2: Realized Volatility (20d)
3: Volume Change
4: High-Low Range (Intraday Vol)
5: MACD
6: Bollinger Band Position
7: Skewness (20d)
8: Kurtosis (20d)
9: Lagged Return (t-1)
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os

def calculate_technical_indicators(df):
    # Ensure Close is used
    close = df['Close']
    
    # 0. Log Returns
    df['Log_Ret'] = np.log(close / close.shift(1))
    
    # 1. RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'] / 100.0 # Normalize 0-1
    
    # 2. Realized Volatility (20d)
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    
    # 3. Volume Change (Log)
    df['Vol_Change'] = np.log(df['Volume'] / df['Volume'].shift(1))
    
    # 4. High-Low Range
    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    
    # 5. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    # 6. BB Position
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_Pos'] = (close - (sma20 - 2*std20)) / (4*std20)
    
    # 7. Skewness
    df['Skew_20'] = df['Log_Ret'].rolling(20).skew()
    
    # 8. Kurtosis
    df['Kurt_20'] = df['Log_Ret'].rolling(20).kurt()
    
    # 9. Lagged Return
    df['Lag_Ret'] = df['Log_Ret'].shift(1)
    
    return df[['Log_Ret', 'RSI', 'Vol_20', 'Vol_Change', 'HL_Range', 'MACD', 'BB_Pos', 'Skew_20', 'Kurt_20', 'Lag_Ret']]

def make_windows(data, window_size=128, stride=1):
    """
    Create sliding windows from a 2-D array.

    Parameters
    ----------
    stride : int
        Step size between consecutive windows.
        Use stride=1 for fully-overlapping windows (default, backwards-compat).
        Use stride=window_size for non-overlapping windows, which eliminates
        direct temporal overlap between train and test splits in CV.
        NOTE: Even with stride=1, a purge gap of >= window_size rows should
        be enforced between train and test in any CV splitter to prevent
        look-ahead leakage through overlapping windows.
    """
    windows = []
    # Drop rows with any NaN before windowing
    data = data[~np.isnan(data).any(axis=1)]

    for i in range(0, len(data) - window_size, stride):
        windows.append(data[i : i + window_size])
    return np.array(windows)

def main():
    print("⬇️ DOWNLOADING S&P 500 DATA (2021-2025)")
    print("=======================================")
    
    # 1. Download
    ticker = "^GSPC"
    print(f"Fetching {ticker}...")
    df = yf.download(ticker, start="2021-01-01", end="2025-12-31", interval="1d", progress=False)
    
    # Handle MultiIndex columns (yfinance > 0.2.0)
    # They look like ('Close', '^GSPC')
    if isinstance(df.columns, pd.MultiIndex):
        print("Flattening MultiIndex columns...")
        # If level 1 is ticker, just drop it
        df.columns = df.columns.get_level_values(0)
    
    print(f"Raw shape: {df.shape}")
    print("Columns:", df.columns)
    print(df.head())
    
    if len(df) == 0:
        print("❌ Error: Download failed.")
        return

    # 2. Process Features
    print("⚙️ Engineering Features...")
    feats = calculate_technical_indicators(df)
    
    # Fill/Drop NaNs
    feats.fillna(0, inplace=True) # First few rows will be nan, better to drop
    feats = feats.iloc[30:] # Drop warming up period
    
    print("Feature Stats:")
    print(feats.describe())
    
    # Clean Infs
    feats.replace([np.inf, -np.inf], 0, inplace=True)
    feats.fillna(0, inplace=True)
    
    # 3. Windowing — raw (un-normalized) features saved.
    # ⚠️  LEAKAGE FIX: Global Z-score normalization removed.
    # Computing mean/std over the full series and applying before windowing
    # constitutes look-ahead bias because statistics from future (test) rows
    # leak into past (train) windows.  Normalization MUST be performed inside
    # each CV fold: fit StandardScaler on X_train only, then transform X_test.
    print("🪟 Creating Windows (128 steps, stride=1)...")
    X_windows = make_windows(feats.values, window_size=128, stride=1)
    
    print(f"Final Tensor Shape: {X_windows.shape}")
    
    # 5. Save
    os.makedirs('artifacts', exist_ok=True)
    out_path = 'artifacts/sp500_5y_windows.npy'
    np.save(out_path, X_windows)
    print(f"\n💾 Saved to {out_path}")
    
    # Save Dates for reference
    dates = df.index[30:]
    # Align dates with windows (Date of the LAST step in window is the label date)
    valid_dates = dates[128:] 
    # Actually make_windows range is len - 128.
    # Window i: indices [i : i+128]. Last idx is i+127.
    # Corresponding date is dates[i+127].
    window_dates = dates[127 : 127 + len(X_windows)]
    
    pd.DataFrame(window_dates, columns=['Date']).to_csv('artifacts/sp500_5y_dates.csv', index=False)
    print("💾 Saved Dates to artifacts/sp500_5y_dates.csv")

if __name__ == '__main__':
    main()
