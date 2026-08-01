import pandas as pd

def feature_engineering(input_path, output_path):
    
    # Load Cleaned Data
    df = pd.read_csv(input_path)
    
    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"])
    
    df = df.sort_values("Date").reset_index(drop=True)
    
    # 1. Daily Return
    df["Daily_Return"] = df["Close"].pct_change() 
    
    # 2. Lag_features
    df["Return_Lag_1"] = df["Daily_Return"].shift(1)
    df["Return_Lag_2"] = df["Daily_Return"].shift(2)
    df["Return_Lag_3"] = df["Daily_Return"].shift(3)
    
    # 3. Simple Moving Average
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 4. Exponential Moving Average
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    
    # 5. Volatility Features
    df['Volatility_20'] = df["Daily_Return"].rolling(window=20).std()
    
    # 6. Volume Change
    df["Volume_Change"] = df["Volume"].pct_change()
    
    # 7. Intraday Features
    df['High_Low_Range'] = ((df["High"] - df["Low"]) / df["Close"])
    
    df["Open_Close_Change"] = ((df["Close"] - df["Open"]) / df["Open"])
    
    
    # 8. Relative Strength Index
    delta = df['Close'].diff()
    
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    
    df["RSI_14"] = 100 - (100/ (1 + rs))
    
    
    # 9. MACD - Moving Average Convergence Divergence
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    
    df["MACD"] = ema_12 - ema_26
    
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    

    # save processed data 
    df.to_csv(output_path, index=False)
    
    return df
