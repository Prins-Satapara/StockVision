import argparse
import pandas as pd

from src.config import (
    STOCKS,
    RAW_DATA_DIR,
    CLEANED_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)

from src.download_data import download_stocks
from src.clean_data import data_cleaning
from src.feature_engineering import feature_engineering
from src.prepare_ml_pipeline import prepare_all_ml_data
from src.prediction import predict_stock


HORIZONS = ["1D", "5D", "20D"]

# STEP 1 — UPDATE / APPEND LATEST DATA

def update_data():
    
    print("\n" + "=" * 70)
    print("                    DOWNLOADING / UPDATING LATEST DATA")
    print("=" * 70)

    download_stocks(save_path=RAW_DATA_DIR)

    print("\n✓ Data update completed.")

# STEP 2 — CLEAN DATA

def clean_all_data():

    print("\n" + "=" * 70)
    print("                    CLEANING DATA")
    print("=" * 70)

    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for stock in STOCKS:

        input_path = RAW_DATA_DIR / f"{stock}.csv"
        output_path = CLEANED_DATA_DIR / f"{stock}_cleaned.csv"

        if not input_path.exists():
            print(f"⚠ {stock}: raw file not found.")
            continue

        data_cleaning(input_path=input_path, output_path=output_path)

        print(f"✓ {stock}")
        

# STEP 3 — FEATURE ENGINEERING

def engineer_all_features():

    print("\n" + "=" * 70)
    print("                    FEATURE ENGINEERING")
    print("=" * 70)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for stock in STOCKS:

        input_path = CLEANED_DATA_DIR / f"{stock}_cleaned.csv"
        output_path = PROCESSED_DATA_DIR / f"{stock}_features.csv"

        if not input_path.exists():
            print(f"⚠ {stock}: cleaned file not found.")
            continue

        feature_engineering(
            input_path=input_path,
            output_path=output_path
        )

        print(f"✓ {stock}")


# STEP 4 — PREPARE ML DATA (Only required when --retrain is used)

def prepare_ml_data():

    print("\n" + "=" * 70)
    print("                    PREPARING ML DATA")
    print("=" * 70)

    prepare_all_ml_data()

    print("\n✓ ML data preparation completed.")



# STEP 5 — TRAIN BASELINE MODELS (Only required when --retrain is used)


def train_models():

    print("\n" + "=" * 70)
    print("                    TRAINING BASELINE MODELS")
    print("=" * 70)

    from src.train_models import train_baseline_models

    results = train_baseline_models()

    print("\n✓ Baseline model training completed.")

    return results


# STEP 6 — HYPERPARAMETER TUNING (Only required when --retrain is used)

def tune_models():

    print("\n" + "=" * 70)
    print("                    HYPERPARAMETER TUNING")
    print("=" * 70)

    from src.tune_models import tune_models as run_tuning

    results = run_tuning(n_trials=15)

    print("\n✓ Hyperparameter tuning completed.")

    return results


# STEP 7 — FINAL MODEL SELECTION + TRAINING (Only required when --retrain is used)

def train_final_models():

    print("\n" + "=" * 70)
    print("                    FINAL MODEL SELECTION & TRAINING")
    print("=" * 70)

    from src.final_model import train_final_models as run_final_training

    results = run_final_training()

    print("\n✓ Final models trained and saved.")

    return results


# STEP 8 — GENERATE PREDICTIONS

def generate_predictions():

    print("\n" + "=" * 70)
    print("                    GENERATING PREDICTIONS")
    print("=" * 70)

    predictions = []

    for stock in STOCKS:

        for horizon in HORIZONS:

            try:

                result = predict_stock(
                    stock=stock,
                    horizon=horizon
                )

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

    output_path = RESULTS_DIR / "latest_predictions.csv"

    predictions_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\n✓ Predictions saved to:\n"
        f"{output_path}"
    )

    return predictions_df


# COMPLETE PIPELINE

def run_pipeline(retrain=False):

    print("\n")
    print("=" * 70)
    print("                    STOCKVISION PIPELINE")
    print("=" * 70)

    if retrain:

        print("\nMODE: FULL RETRAIN")
        print(
            "Data → Features → ML Data → "
            "Training → Tuning → Final Models → Predictions"
        )

    else:

        print("\nMODE: DATA UPDATE")
        print(
            "Data → Features → Existing Models → Predictions"
        )

    # ALWAYS UPDATE MARKET DATA

    update_data()

    # ALWAYS REFRESH CLEANED DATA

    clean_all_data()

    # ALWAYS REFRESH FEATURES

    engineer_all_features()

    # ONLY WITH --retrain

    if retrain:

        prepare_ml_data()

        train_models()

        tune_models()

        train_final_models()

    # GENERATE PREDICTIONS

    predictions_df = generate_predictions()

    # COMPLETION MESSAGE

    print("\n")
    print("=" * 70)
    print("                 PIPELINE COMPLETED")
    print("=" * 70)

    print(
        f"\nTotal predictions: "
        f"{len(predictions_df)}"
    )

    if retrain:

        print("\n✓ Full ML pipeline reproduced successfully.")
        print("✓ All final models were retrained.")

    else:

        print("\n✓ Latest market data updated.")
        print("✓ Existing final models were used.")
        print("✓ Models were NOT retrained.")

    return predictions_df



# COMMAND-LINE INTERFACE


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="StockVision automated ML pipeline")

    parser.add_argument(
        "--retrain",
        action="store_true",
        help=(
            "Run the complete ML pipeline including "
            "ML data preparation, baseline training, "
            "hyperparameter tuning and final model training."
        )
    )

    args = parser.parse_args()

    run_pipeline(
        retrain=args.retrain
    )