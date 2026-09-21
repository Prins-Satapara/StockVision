import numpy as np
import pandas as pd

def prepare_ml_data(input_path, output_path, horizon = "1D"):
    
    # load processed data
    df = pd.read_csv(input_path, parse_dates=["Date"])
    
    # sort chronological order
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Validate required features
    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Company",
        "Adj_Daily_Return"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )
        
    # Validate horizon
    valid_horizons = {
        "1D": 1,
        "5D": 5,
        "20D": 20
    }

    if horizon not in valid_horizons:
        raise ValueError(
            f"Invalid horizon: {horizon}. "
            f"Choose from {list(valid_horizons.keys())}"
        )

    days_ahead = valid_horizons[horizon]

    # Future close
    df["Future_Close"] = df["Close"].shift(-days_ahead)

    # Create target
    # 1 -> future price UP
    # 0 -> future price DOWN
    target_column = f"Target_{horizon}"

    df[target_column] = (
        df["Future_Close"] > df["Close"]
    ).astype("Int64")

    # Last N rows have no future price
    df.loc[
        df["Future_Close"].isna(),
        target_column
    ] = pd.NA


    # Convert infinity to NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove rows containing NaN
    df.dropna(inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True) 
    
    
    # Future_Close was only needed to create Target
    # It must NOT be available to the ML model
    df.drop(columns=["Future_Close"], inplace=True)
    

    
    # save the final results
    
    df.to_csv(output_path, index=False)
    
    return df