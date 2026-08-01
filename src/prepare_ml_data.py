import numpy as np
import pandas as pd

def prepare_ml_data(input_path, output_path):
    
    # load processed data
    df = pd.read_csv(input_path)
    
    
    # next day close
    df["Next_Close"] = df["Close"].shift(-1)
    
    
    # create target
        # 1 -> next day UP
        # 0 -> next day DOWN
    df["Target"] = (df['Next_Close'] > df["Close"]).astype('Int64')
    
    
    # Last row has no future price, so its target is unknown
    df.loc[df["Next_Close"].isna(), "Target"] = pd.NA


    # Convert infinity to NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove rows containing NaN
    df.dropna(inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True) 
    
    
    # Next_Close was only needed to create Target
    # It must NOT be available to the ML model
    df.drop(columns=["Next_Close"], inplace=True)
    

    
    # save the final results
    
    df.to_csv(output_path, index=False)
    
    return df