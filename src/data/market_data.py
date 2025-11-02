import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os

def label_anomalies(data, volatility_threshold=2, volume_threshold=2):
    """
    Label market anomalies using multiple criteria:
    - Volatility spikes
    - Volume anomalies
    - Price trend breakouts
    - RSI extremes
    """
    labels = np.zeros(len(data))
    
    # 1. Volatility-based anomalies
    returns_z = np.abs(stats.zscore(data['returns']))
    std_z = np.abs(stats.zscore(data['rolling_std_21d']))
    volatility_anomalies = (returns_z > volatility_threshold) | (std_z > volatility_threshold)
    
    # 2. Volume anomalies
    volume_z = np.abs(stats.zscore(data['volume_ratio']))
    volume_anomalies = volume_z > volume_threshold
    
    # 3. RSI extremes
    rsi_anomalies = (data['rsi'] < 30) | (data['rsi'] > 70)
    
    # 4. Trend breaks (when price crosses moving averages)
    price = data['close']
    ma21 = data['close_ma21']
    crosses_ma = ((price > ma21) & (price.shift(1) <= ma21.shift(1))) | \
                 ((price < ma21) & (price.shift(1) >= ma21.shift(1)))
    
    # Combine signals with OR condition
    labels = (
        volatility_anomalies |  # High volatility events
        volume_anomalies |      # Unusual volume
        (rsi_anomalies & crosses_ma)  # RSI extremes during MA crosses
    ).astype(int)
    
    return labels

def create_windowed_labels(base_labels, window_size=128, stride=5, anomaly_threshold=0.2):
    """
    Create window labels with more lenient threshold:
    Label window as anomalous if >20% of points are anomalous
    """
    windowed_labels = []
    for i in range(0, len(base_labels) - window_size + 1, stride):
        window_labels = base_labels[i:i + window_size]
        # More lenient threshold - 20% of points need to be anomalous
        windowed_labels.append(1 if window_labels.mean() > anomaly_threshold else 0)
    return np.array(windowed_labels)

def create_sliding_windows(data, window_size=128, stride=1):
    """Create sliding windows from time series data"""
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def fetch_sp500_data(start_date="2005-01-01", end_date="2025-01-01"):
    """Fetch S&P 500 data with technical indicators"""
    # Fetch S&P 500 data
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)
    # Fix MultiIndex columns
    sp500.columns = sp500.columns.get_level_values(0)
    print("Columns available:", sp500.columns.tolist())
    
    # Calculate technical indicators
    df = pd.DataFrame(index=sp500.index)
    
    # Price-based features
    df['returns'] = sp500['Close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])
    df['rolling_std_5d'] = df['returns'].rolling(window=5).std()
    df['rolling_std_21d'] = df['returns'].rolling(window=21).std()
    
    # Price levels and changes
    df['close'] = sp500['Close']
    df['close_ma5'] = df['close'].rolling(window=5).mean()
    df['close_ma21'] = df['close'].rolling(window=21).mean()
    
    # Volume features
    df['volume'] = sp500['Volume']
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma21'] = df['volume'].rolling(window=21).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma21']
    
    # Volatility
    df['true_range'] = calculate_true_range(sp500)
    df['atr'] = df['true_range'].rolling(window=14).mean()
    
    # Momentum
    df['rsi'] = calculate_rsi(sp500['Close'])
    
    # Drop NaN values and standardize
    df = df.dropna()
    
    # Standardize features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    
    return df_scaled

def calculate_true_range(data):
    """Calculate True Range for ATR"""
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if __name__ == "__main__":
    try:
        print("Fetching S&P 500 data...")
        data = fetch_sp500_data()
        print(f"Features available: {data.columns.tolist()}")
        
        # Generate point-wise labels
        print("\nGenerating anomaly labels...")
        base_labels = label_anomalies(
            data,
            volatility_threshold=2,  # More sensitive threshold
            volume_threshold=2
        )
        print(f"Point-wise anomalies: {base_labels.sum()}")
        
        # Create windows with more lenient labeling
        window_size = 128
        stride = 5
        
        print("\nCreating sliding windows...")
        X = create_sliding_windows(data.values, window_size=window_size, stride=stride)
        windowed_labels = create_windowed_labels(
            base_labels,
            window_size=window_size,
            stride=stride,
            anomaly_threshold=0.2  # Window is anomalous if 20% of points are anomalous
        )
        
        # Save processed data
        print("\nSaving processed data...")
        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        np.save(os.path.join(artifacts_dir, 'market_windows.npy'), X)
        np.save(os.path.join(artifacts_dir, 'market_labels.npy'), windowed_labels)
        
        print(f"\nProcessed market data shape: {X.shape}")
        print(f"Features per window: {data.shape[1]}")
        print(f"Window labels shape: {windowed_labels.shape}")
        print(f"Anomaly ratio: {windowed_labels.mean():.3f}")
        print(f"Number of anomalies: {windowed_labels.sum()}")
        
    except Exception as e:
        print(f"Error processing market data: {str(e)}")
        raise