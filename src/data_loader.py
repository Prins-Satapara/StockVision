import pandas as pd 

from src.config import PROCESSED_DATA_DIR

def load_stock_data(stock):
    
    # create file path
    file_path = PROCESSED_DATA_DIR / f"{stock}_features.csv"
    
    # load
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    return df
