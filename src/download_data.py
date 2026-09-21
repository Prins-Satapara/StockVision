import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


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

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for stock in stocks:
        
        file_path = save_path / f"{stock}.csv"
        
        # Case 1 :- CSV file is already exists.....
        if file_path.exists():
            
            existing_df = pd.read_csv(file_path,  parse_dates=["Date"])
        
            last_date = existing_df["Date"].max()
            start_date = last_date + timedelta(days=1)
            
            print(f"\n{stock}")
            print(f"Existing data through: {last_date.date()}")
            
            new_df = yf.download(
                stock, 
                start=start_date.strftime("%Y-%m-%d"),
                end=datetime.today().strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False
            )
            
            if new_df.empty:
                print("No new data availbale.")
                continue
            
            new_df.columns = new_df.columns.get_level_values(0)
            
            new_df = new_df.reset_index()
            new_df["Company"] = stock
            
            updated_df = pd.concat(
                [existing_df, new_df],
                ignore_index=True
            )
            
            updated_df.drop_duplicates(
                subset=["Date"], 
                keep="last",
                inplace=True
            )
            
            updated_df.sort_values("Date", inplace=True)
            
            updated_df.to_csv(file_path, index=False)
            
            print(
                f"Added {len(new_df)} new rows."
            )

            print(
                f"Updated data through: "
                f"{updated_df['Date'].max().date()}"
            )
        
        # Case 2 : CSV file dosn't exist
        else:
            
            print(f"\n{stock}")
            print("CSV not found. Downloading historical data...")

            df = yf.download(
                stock,
                start="2015-01-01",
                end=datetime.today().strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False
            )
            
            if df.empty:
                print("No data found.")
                continue

            df.columns = df.columns.get_level_values(0)

            df = df.reset_index()

            df["Company"] = stock

            df.to_csv(file_path, index=False)

            print(f"Downloaded {len(df)} rows.")

            print(
                f"Data through: "
                f"{df['Date'].max().date()}"
            )