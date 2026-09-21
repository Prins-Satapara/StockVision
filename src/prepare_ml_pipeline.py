from pathlib import Path

from src.config import STOCKS, PROCESSED_DATA_DIR, ML_READY_DATA_DIR

from src.prepare_ml_data import prepare_ml_data


HORIZONS = ["1D", "5D", "20D"]


def prepare_all_ml_data():

    ML_READY_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\nPreparing ML datasets...")

    for stock in STOCKS:

        input_path = (PROCESSED_DATA_DIR / f"{stock}_features.csv")

        if not input_path.exists():
            print(f"{stock}: feature file not found.")
            continue

        for horizon in HORIZONS:

            output_path = (ML_READY_DATA_DIR /f"{stock}_{horizon}_ml.csv")

            prepare_ml_data(input_path=input_path, output_path=output_path, horizon=horizon)

            print(f"✓ {stock} | {horizon}")

    print("✓ ML datasets prepared.")