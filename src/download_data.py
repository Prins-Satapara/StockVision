import yfinance as yf
import pandas as pd
from pathlib import Path


def download_stocks(save_path): 
    
    
    
    stocks = [
        "AAPL",          # Apple
        "MSFT",          # Microsoft
        "NVDA",          # NVIDIA
        "AMZN",          # Amazon
        "RELIANCE.NS",   # Reliance
        "TCS.NS",        # TCS
        "HDFCBANK.NS",   # HDFC Bank
        "INFY.NS"        # Infosys
    ]

    for stock in stocks:
        df = yf.download(
            stock,
            start="2015-01-01",
            end="2025-12-31"
        )

        df.columns = df.columns.get_level_values(0)

        df["Company"] = stock

        df.to_csv(f"{save_path}/{stock}.csv")
    
    return df