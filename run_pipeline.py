import pandas as pd

from src.config import STOCKS, RAW_DATA_DIR, CLEANED_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from src.download_data import download_stocks
from src.clean_data import data_cleaning
from src.feature_engineering import feature_engineering
from src.prepare_ml_pipeline import prepare_all_ml_data
from src.prediction import predict_stock


HORIZONS = ["1D", "5D", "20D"]

def update_data():

    print("\n" + "=" * 70)
    print("STEP 1 — DOWNLOADING LATEST DATA")
    print("=" * 70)

    download_stocks(save_path=RAW_DATA_DIR)

    print("\n✓ Download completed.")


def clean_all_data():

    print("\n" + "=" * 70)
    print("STEP 2 — CLEANING DATA")
    print("=" * 70)

    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for stock in STOCKS:

        input_path = (RAW_DATA_DIR /f"{stock}.csv")

        output_path = (CLEANED_DATA_DIR /f"{stock}_cleaned.csv")

        if not input_path.exists():

            print(f"⚠ {stock}: raw file not found.")

            continue

        data_cleaning(input_path=input_path, output_path=output_path)

        print(f"✓ {stock}")


def engineer_all_features():

    print("\n" + "=" * 70)
    print("STEP 3 — FEATURE ENGINEERING")
    print("=" * 70)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for stock in STOCKS:

        input_path = (CLEANED_DATA_DIR /f"{stock}_cleaned.csv")

        output_path = (PROCESSED_DATA_DIR /f"{stock}_features.csv")

        if not input_path.exists():

            print(f"⚠ {stock}: cleaned file not found.")

            continue

        feature_engineering(input_path=input_path, output_path=output_path)

        print(f"✓ {stock}")


def prepare_ml_data():

    print("\n" + "=" * 70)
    print("STEP 4 — PREPARING ML DATA")
    print("=" * 70)

    prepare_all_ml_data()

    print("\n✓ ML data preparation completed.")


def generate_predictions():

    print("\n" + "=" * 70)
    print("STEP 5 — GENERATING PREDICTIONS")
    print("=" * 70)

    predictions = []

    for stock in STOCKS:

        for horizon in HORIZONS:

            try:

                result = predict_stock(stock=stock, horizon=horizon)

                predictions.append(result)

                print(
                    f"✓ {stock:<15} "
                    f"{horizon:<4} → "
                    f"{result['Prediction']:<5} "
                    f"{result['Probability']:.2f}%"
                )

            except Exception as e:

                print(
                    f"✗ {stock} | "
                    f"{horizon} | "
                    f"{e}"
                )

    predictions_df = pd.DataFrame(predictions)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (RESULTS_DIR / "latest_predictions.csv")

    predictions_df.to_csv(output_path, index=False
)

    print(
        f"\n✓ Predictions saved to:\n"
        f"{output_path}"
    )

    return predictions_df


def run_pipeline():

    print("\n")
    print("=" * 70)
    print("             STOCKVISION PIPELINE")
    print("=" * 70)

    update_data()

    clean_all_data()

    engineer_all_features()

    prepare_ml_data()

    predictions_df = generate_predictions()

    print("\n")
    print("=" * 70)
    print("             PIPELINE COMPLETED")
    print("=" * 70)

    print(
        f"\nTotal predictions: "
        f"{len(predictions_df)}"
    )

    return predictions_df


if __name__ == "__main__":

    run_pipeline()