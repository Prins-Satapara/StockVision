import pandas as pd


def data_cleaning(myFile, output_path):

    # Load dataset
    df = pd.read_csv(myFile)

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"])

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Sort by Date
    df.sort_values("Date", inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    return df