import json

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.config import (
    STOCKS,
    FEATURES,
    ML_READY_DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
)


# ============================================================
# Configuration
# ============================================================

HORIZONS = ["1D", "5D", "20D"]

TARGET_COLUMNS = {
    "1D": "Target_1D",
    "5D": "Target_5D",
    "20D": "Target_20D",
}

N_SPLITS = 3
RANDOM_STATE = 42


# ============================================================
# Data helpers
# ============================================================

def load_ml_data(stock, horizon):
    """
    Load the prepared ML dataset for one stock and horizon.
    """

    if horizon not in HORIZONS:
        raise ValueError(
            f"Invalid horizon: {horizon}. "
            f"Choose from {HORIZONS}"
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

    return df


def chronological_split(X, y, test_size=0.20):
    """
    Chronological 80/20 train-test split.

    The final 20% is kept untouched and is used only
    for independent test-set evaluation.
    """

    split_idx = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()

    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


# ============================================================
# Model helpers
# ============================================================

def create_default_rf_model():
    """
    Create the baseline Random Forest model.
    """

    return RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def create_rf_from_params(params):
    """
    Create a Random Forest using Optuna-selected parameters.
    """

    return RandomForestClassifier(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def calculate_metrics(y_true, y_pred):
    """
    Calculate binary classification metrics.
    """

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1_Score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


# ============================================================
# Optuna objective
# ============================================================

def optimize_random_forest(
    X_train,
    y_train,
    n_trials=50,
):
    """
    Optimize Random Forest hyperparameters using Optuna.

    TimeSeriesSplit is applied only to the training data.
    The independent test set is never passed to Optuna.
    """

    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

        model = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1,
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=tscv,
            scoring="accuracy",
            n_jobs=1,
        )

        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    return study


# ============================================================
# Main tuning pipeline
# ============================================================

def tune_models(n_trials=50):
    """
    Tune Random Forest models for all stocks and horizons.

    Workflow for every stock-horizon combination:

        1. Load ML-ready data.
        2. Select features and horizon-specific target.
        3. Perform chronological 80/20 split.
        4. Run Optuna on the training portion.
        5. Use TimeSeriesSplit for CV.
        6. Train the tuned RF on the training portion.
        7. Evaluate tuned RF on the untouched test set.
        8. Evaluate default RF using the same split.
        9. Store best parameters and results.
       10. Save reusable result files.

    Returns:
        {
            "tuned_results": DataFrame,
            "default_rf_results": DataFrame,
            "best_params": dict
        }
    """

    if n_trials < 1:
        raise ValueError(
            "n_trials must be at least 1."
        )

    tuned_results = []
    default_rf_results = []

    # Nested structure:
    # best_params[horizon][stock] = {...}
    best_params = {
        horizon: {}
        for horizon in HORIZONS
    }

    # ========================================================
    # Loop through horizons and stocks
    # ========================================================

    for horizon in HORIZONS:

        target_column = TARGET_COLUMNS[horizon]

        for stock in STOCKS:

            print(
                f"\n{'=' * 60}\n"
                f"Tuning Random Forest: {stock} - {horizon}\n"
                f"{'=' * 60}"
            )

            # ------------------------------------------------
            # Load data
            # ------------------------------------------------

            df = load_ml_data(
                stock=stock,
                horizon=horizon,
            )

            missing_features = [
                feature
                for feature in FEATURES
                if feature not in df.columns
            ]

            if missing_features:
                raise ValueError(
                    f"{stock} - {horizon}: "
                    f"Missing features: {missing_features}"
                )

            if target_column not in df.columns:
                raise ValueError(
                    f"{stock} - {horizon}: "
                    f"Missing target column: {target_column}"
                )

            X = df[FEATURES].copy()
            y = df[target_column].astype(int).copy()

            # ------------------------------------------------
            # Chronological split
            # ------------------------------------------------

            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = chronological_split(
                X,
                y,
                test_size=0.20,
            )

            print(
                f"Training rows: {len(X_train)}"
            )
            print(
                f"Test rows: {len(X_test)}"
            )

            # ------------------------------------------------
            # Default Random Forest
            # ------------------------------------------------

            default_model = create_default_rf_model()

            default_model.fit(
                X_train,
                y_train,
            )

            default_pred = default_model.predict(
                X_test
            )

            default_metrics = calculate_metrics(
                y_test,
                default_pred,
            )

            # Default RF Time-Series CV
            default_tscv = TimeSeriesSplit(
                n_splits=N_SPLITS
            )

            default_cv_scores = cross_val_score(
                create_default_rf_model(),
                X_train,
                y_train,
                cv=default_tscv,
                scoring="accuracy",
                n_jobs=1,
            )

            default_rf_results.append(
                {
                    "Horizon": horizon,
                    "Stock": stock,
                    "Default_Accuracy": (
                        default_metrics["Accuracy"]
                    ),
                    "Default_Precision": (
                        default_metrics["Precision"]
                    ),
                    "Default_Recall": (
                        default_metrics["Recall"]
                    ),
                    "Default_F1_Score": (
                        default_metrics["F1_Score"]
                    ),
                    "Default_CV_Mean": (
                        default_cv_scores.mean()
                    ),
                    "Default_CV_Std": (
                        default_cv_scores.std()
                    ),
                }
            )

            # ------------------------------------------------
            # Optuna tuning
            # ------------------------------------------------

            study = optimize_random_forest(
                X_train=X_train,
                y_train=y_train,
                n_trials=n_trials,
            )

            params = study.best_params.copy()

            best_params[horizon][stock] = params

            best_cv_score = study.best_value

            print(
                f"Best CV accuracy: "
                f"{best_cv_score:.4f}"
            )
            print(
                f"Best parameters: {params}"
            )

            # ------------------------------------------------
            # Train tuned Random Forest
            # ------------------------------------------------

            tuned_model = create_rf_from_params(
                params
            )

            tuned_model.fit(
                X_train,
                y_train,
            )

            tuned_pred = tuned_model.predict(
                X_test
            )

            tuned_metrics = calculate_metrics(
                y_test,
                tuned_pred,
            )

            tuned_results.append(
                {
                    "Horizon": horizon,
                    "Stock": stock,
                    "Accuracy": tuned_metrics["Accuracy"],
                    "Precision": tuned_metrics["Precision"],
                    "Recall": tuned_metrics["Recall"],
                    "F1_Score": tuned_metrics["F1_Score"],
                    "Tuned_CV_Mean": best_cv_score,
                    "Tuned_CV_Std": 0.0,
                }
            )

            print(
                f"Test accuracy: "
                f"{tuned_metrics['Accuracy']:.4f}"
            )

            print(
                f"Completed: {stock} - {horizon}"
            )

    # ========================================================
    # Convert to DataFrames
    # ========================================================

    tuned_results_df = pd.DataFrame(
        tuned_results
    )

    default_rf_results_df = pd.DataFrame(
        default_rf_results
    )

    tuned_results_df = (
        tuned_results_df
        .sort_values(
            ["Horizon", "Stock"]
        )
        .reset_index(drop=True)
    )

    default_rf_results_df = (
        default_rf_results_df
        .sort_values(
            ["Horizon", "Stock"]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # Save results
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tuned_results_df.to_csv(
        RESULTS_DIR / "tuned_rf_results.csv",
        index=False,
    )

    default_rf_results_df.to_csv(
        RESULTS_DIR / "default_rf_results_tuning.csv",
        index=False,
    )

    with open(MODELS_DIR / "best_rf_params.json", "w", encoding="utf-8",) as f:
        json.dump(
            best_params,
            f,
            indent=4,
        )

    return {
        "tuned_results": tuned_results_df,
        "default_rf_results": default_rf_results_df,
        "best_params": best_params,
    }
