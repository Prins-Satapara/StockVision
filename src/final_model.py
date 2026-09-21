import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    STOCKS,
    FEATURES,
    ML_READY_DATA_DIR,
    FINAL_MODELS_DIR,
    RESULTS_DIR,
    MODELS_DIR,
)

from src.train_models import HORIZONS


# ============================================================
# Data helpers
# ============================================================

def load_ml_data(stock, horizon):
    """
    Load ML-ready data for one stock and prediction horizon.
    """

    if horizon not in HORIZONS:
        raise ValueError(
            f"Invalid horizon: {horizon}. Choose from {HORIZONS}"
        )

    file_path = ML_READY_DATA_DIR / f"{stock}_{horizon}_ml.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"ML-ready file not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        parse_dates=["Date"]
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    X = df[FEATURES].copy()
    y = df[f"Target_{horizon}"].astype(int).copy()

    return X, y


# ============================================================
# Result loading
# ============================================================

def _load_results():
    """
    Load baseline and tuning results produced by
    train_models.py and tune_models.py.
    """

    logistic_path = RESULTS_DIR / "logistic_results.csv"
    default_rf_path = RESULTS_DIR / "default_rf_results.csv"
    baseline_cv_path = RESULTS_DIR / "baseline_cv_results.csv"
    tuned_rf_path = RESULTS_DIR / "tuned_rf_results.csv"

    required_files = [
        logistic_path,
        default_rf_path,
        baseline_cv_path,
        tuned_rf_path,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required result files are missing:\n"
            + "\n".join(missing_files)
            + "\n\nRun Notebook 07 and Notebook 08 first."
        )

    logistic_results = pd.read_csv(logistic_path)
    default_rf_results = pd.read_csv(default_rf_path)
    baseline_cv_results = pd.read_csv(baseline_cv_path)
    tuned_rf_results = pd.read_csv(tuned_rf_path)

    return (
        logistic_results,
        default_rf_results,
        baseline_cv_results,
        tuned_rf_results,
    )


# ============================================================
# Model selection
# ============================================================

def select_best_models():
    """
    Compare Logistic Regression, Default Random Forest,
    and Tuned Random Forest.

    IMPORTANT:
    Final model selection is based ONLY on mean
    Time-Series CV accuracy.

    Test-set accuracy is retained for reporting but is
    never used to choose the final model.
    """

    (
        logistic_results,
        default_rf_results,
        baseline_cv_results,
        tuned_rf_results,
    ) = _load_results()

    # --------------------------------------------------------
    # Test-set metrics
    # --------------------------------------------------------

    comparison_df = (
        logistic_results[
            [
                "Horizon",
                "Stock",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
            ]
        ]
        .rename(
            columns={
                "Accuracy": "Logistic Test Accuracy",
                "Precision": "Logistic Test Precision",
                "Recall": "Logistic Test Recall",
                "F1-Score": "Logistic Test F1",
            }
        )
    )

    comparison_df = comparison_df.merge(
        default_rf_results[
            [
                "Horizon",
                "Stock",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
            ]
        ].rename(
            columns={
                "Accuracy": "Default RF Test Accuracy",
                "Precision": "Default RF Test Precision",
                "Recall": "Default RF Test Recall",
                "F1-Score": "Default RF Test F1",
            }
        ),
        on=["Horizon", "Stock"],
        how="inner",
    )

    comparison_df = comparison_df.merge(
        tuned_rf_results[
            [
                "Horizon",
                "Stock",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score",
            ]
        ].rename(
            columns={
                "Accuracy": "Tuned RF Test Accuracy",
                "Precision": "Tuned RF Test Precision",
                "Recall": "Tuned RF Test Recall",
                "F1_Score": "Tuned RF Test F1",
            }
        ),
        on=["Horizon", "Stock"],
        how="inner",
    )

    # --------------------------------------------------------
    # Time-Series CV metrics
    # --------------------------------------------------------

    comparison_df = comparison_df.merge(
        baseline_cv_results[
            [
                "Horizon",
                "Stock",
                "Logistic Regression CV",
                "Default Random Forest CV",
            ]
        ].rename(
            columns={
                "Logistic Regression CV": "Logistic CV",
                "Default Random Forest CV": "Default RF CV",
            }
        ),
        on=["Horizon", "Stock"],
        how="inner",
    )

    comparison_df = comparison_df.merge(
        tuned_rf_results[
            [
                "Horizon",
                "Stock",
                "Tuned_CV_Mean",
            ]
        ].rename(
            columns={
                "Tuned_CV_Mean": "Tuned RF CV",
            }
        ),
        on=["Horizon", "Stock"],
        how="inner",
    )

    # --------------------------------------------------------
    # Select using CV only
    # --------------------------------------------------------

    cv_model_columns = [
        "Logistic CV",
        "Default RF CV",
        "Tuned RF CV",
    ]

    comparison_df["Best Model"] = (
        comparison_df[cv_model_columns]
        .idxmax(axis=1)
        .map(
            {
                "Logistic CV": "Logistic Regression",
                "Default RF CV": "Default Random Forest",
                "Tuned RF CV": "Tuned Random Forest",
            }
        )
    )

    comparison_df["Best CV Accuracy"] = (
        comparison_df[cv_model_columns]
        .max(axis=1)
    )

    comparison_df = (
        comparison_df
        .sort_values(["Horizon", "Stock"])
        .reset_index(drop=True)
    )

    return comparison_df


# ============================================================
# Tuned parameter loading
# ============================================================

def load_best_params():
    """
    Load Optuna-selected Random Forest parameters.
    """

    params_path = MODELS_DIR / "best_rf_params.json"

    if not params_path.exists():
        raise FileNotFoundError(
            f"Best parameter file not found: {params_path}\n"
            "Run Notebook 08 first."
        )

    with open(
        params_path,
        "r",
        encoding="utf-8",
    ) as file:
        best_params = json.load(file)

    return best_params


# ============================================================
# Final model factory
# ============================================================

def create_final_model(model_name, tuned_params=None):
    """
    Create the selected final model.
    """

    if model_name == "Logistic Regression":

        return Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=42,
                        max_iter=1000,
                    ),
                ),
            ]
        )

    if model_name == "Default Random Forest":

        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "Tuned Random Forest":

        if tuned_params is None:
            raise ValueError(
                "Tuned parameters are required "
                "for Tuned Random Forest."
            )

        return RandomForestClassifier(
            **tuned_params,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# Final training pipeline
# ============================================================

def train_final_models():
    """
    Select and train the final 24 models.

    Model selection:
        Highest mean Time-Series CV accuracy.

    Final training:
        Selected model is retrained on ALL labeled data.

    Saved model format:
        models/final/{stock}_{horizon}_model.joblib
    """

    comparison_df = select_best_models()
    best_params = load_best_params()

    final_model_choice = {
        horizon: {}
        for horizon in HORIZONS
    }

    final_models = {
        horizon: {}
        for horizon in HORIZONS
    }

    FINAL_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Select and train every stock-horizon model
    # --------------------------------------------------------

    for horizon in HORIZONS:

        for stock in STOCKS:

            row = comparison_df[
                (comparison_df["Horizon"] == horizon)
                & (comparison_df["Stock"] == stock)
            ]

            if row.empty:
                raise ValueError(
                    f"No model selection result for "
                    f"{stock} | {horizon}"
                )

            model_name = row["Best Model"].iloc[0]

            final_model_choice[horizon][stock] = model_name

            tuned_params = None

            if model_name == "Tuned Random Forest":

                if (
                    horizon not in best_params
                    or stock not in best_params[horizon]
                ):
                    raise KeyError(
                        f"Missing tuned parameters for "
                        f"{stock} | {horizon}"
                    )

                tuned_params = best_params[horizon][stock]

            model = create_final_model(
                model_name=model_name,
                tuned_params=tuned_params,
            )

            X, y = load_ml_data(
                stock=stock,
                horizon=horizon,
            )

            # Train on all labeled data after model selection.
            model.fit(X, y)

            final_models[horizon][stock] = model

            model_path = (
                FINAL_MODELS_DIR
                / f"{stock}_{horizon}_model.joblib"
            )

            joblib.dump(
                model,
                model_path,
            )

            print(
                f"✓ {stock:<15} "
                f"{horizon:<4} → "
                f"{model_name}"
            )

    # --------------------------------------------------------
    # Save model-selection artifacts
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        RESULTS_DIR / "final_model_selection.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    choice_path = (
        RESULTS_DIR / "final_model_choice.json"
    )

    with open(
        choice_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_model_choice,
            file,
            indent=4,
        )

    print(
        "\n✓ Final models trained and saved."
    )

    return {
        "comparison_df": comparison_df,
        "final_model_choice": final_model_choice,
        "final_models": final_models,
    }
