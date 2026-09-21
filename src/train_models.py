import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import (
    STOCKS,
    FEATURES,
    ML_READY_DATA_DIR,
    RESULTS_DIR,
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

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# Helper functions
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
    Split data chronologically.

    The first 80% is used for training and the final 20%
    is reserved as the untouched test set.
    """

    split_idx = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()

    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics for binary UP/DOWN prediction.
    """

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "F1-Score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
    }


def create_logistic_model():
    """
    Create the Logistic Regression pipeline.

    StandardScaler is included because Logistic Regression
    is sensitive to feature scale.
    """

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=1000
                )
            ),
        ]
    )


def create_default_rf_model():
    """
    Create the baseline Random Forest model.
    """

    return RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# ============================================================
# Main training pipeline
# ============================================================

def train_baseline_models():
    """
    Train and evaluate Logistic Regression and Default Random
    Forest models for every stock and prediction horizon.

    Workflow:
        1. Load ML-ready data
        2. Select features and horizon-specific target
        3. Chronological 80/20 split
        4. Train Logistic Regression
        5. Train Default Random Forest
        6. Evaluate both on untouched test data
        7. Perform Time-Series Cross-Validation on training data
        8. Save evaluation results
        9. Return all objects required by Notebook 07
    """

    logistic_models = {}
    rf_models = {}

    logistic_predictions = {}
    rf_predictions = {}

    X_train = {}
    X_test = {}
    y_train = {}
    y_test = {}

    logistic_results = []
    default_rf_results = []
    cv_rows = []

    # --------------------------------------------------------
    # Loop through all horizons and stocks
    # --------------------------------------------------------

    for horizon in HORIZONS:

        logistic_models[horizon] = {}
        rf_models[horizon] = {}

        logistic_predictions[horizon] = {}
        rf_predictions[horizon] = {}

        X_train[horizon] = {}
        X_test[horizon] = {}

        y_train[horizon] = {}
        y_test[horizon] = {}

        target_column = TARGET_COLUMNS[horizon]

        for stock in STOCKS:

            print(
                f"Training {stock} - {horizon}..."
            )

            # ------------------------------------------------
            # Load data
            # ------------------------------------------------

            df = load_ml_data(
                stock=stock,
                horizon=horizon
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

            # ------------------------------------------------
            # X and y
            # ------------------------------------------------

            X = df[FEATURES].copy()
            y = df[target_column].astype(int).copy()

            # ------------------------------------------------
            # Chronological 80/20 split
            # ------------------------------------------------

            (
                X_train[horizon][stock],
                X_test[horizon][stock],
                y_train[horizon][stock],
                y_test[horizon][stock],
            ) = chronological_split(
                X,
                y,
                test_size=0.20
            )

            X_train_stock = X_train[horizon][stock]
            X_test_stock = X_test[horizon][stock]
            y_train_stock = y_train[horizon][stock]
            y_test_stock = y_test[horizon][stock]

            # ------------------------------------------------
            # Logistic Regression
            # ------------------------------------------------

            logistic_model = create_logistic_model()

            logistic_model.fit(
                X_train_stock,
                y_train_stock
            )

            logistic_pred = logistic_model.predict(
                X_test_stock
            )

            logistic_models[horizon][stock] = logistic_model
            logistic_predictions[horizon][stock] = logistic_pred

            logistic_metrics = calculate_metrics(
                y_test_stock,
                logistic_pred
            )

            logistic_results.append(
                {
                    "Horizon": horizon,
                    "Stock": stock,
                    **logistic_metrics,
                }
            )

            # ------------------------------------------------
            # Default Random Forest
            # ------------------------------------------------

            rf_model = create_default_rf_model()

            rf_model.fit(
                X_train_stock,
                y_train_stock
            )

            rf_pred = rf_model.predict(
                X_test_stock
            )

            rf_models[horizon][stock] = rf_model
            rf_predictions[horizon][stock] = rf_pred

            rf_metrics = calculate_metrics(
                y_test_stock,
                rf_pred
            )

            default_rf_results.append(
                {
                    "Horizon": horizon,
                    "Stock": stock,
                    **rf_metrics,
                }
            )

            # ------------------------------------------------
            # Time-Series Cross-Validation
            # ------------------------------------------------

            tscv = TimeSeriesSplit(
                n_splits=N_SPLITS
            )

            logistic_cv_scores = cross_val_score(
                create_logistic_model(),
                X_train_stock,
                y_train_stock,
                cv=tscv,
                scoring="accuracy",
                n_jobs=-1
            )

            rf_cv_scores = cross_val_score(
                create_default_rf_model(),
                X_train_stock,
                y_train_stock,
                cv=tscv,
                scoring="accuracy",
                n_jobs=-1
            )

            cv_rows.append(
                {
                    "Horizon": horizon,
                    "Stock": stock,
                    "Logistic Regression CV": (
                        logistic_cv_scores.mean()
                    ),
                    "Default Random Forest CV": (
                        rf_cv_scores.mean()
                    ),
                    "Logistic Regression CV Std": (
                        logistic_cv_scores.std()
                    ),
                    "Default Random Forest CV Std": (
                        rf_cv_scores.std()
                    ),
                }
            )

            print(
                f"{stock} - {horizon} "
                "completed successfully."
            )

    # ========================================================
    # Convert results to DataFrames
    # ========================================================

    logistic_results_df = pd.DataFrame(
        logistic_results
    )

    default_rf_results_df = pd.DataFrame(
        default_rf_results
    )

    cv_results_df = pd.DataFrame(
        cv_rows
    )

    # Sort consistently
    logistic_results_df = (
        logistic_results_df
        .sort_values(["Horizon", "Stock"])
        .reset_index(drop=True)
    )

    default_rf_results_df = (
        default_rf_results_df
        .sort_values(["Horizon", "Stock"])
        .reset_index(drop=True)
    )

    cv_results_df = (
        cv_results_df
        .sort_values(["Horizon", "Stock"])
        .reset_index(drop=True)
    )

    # ========================================================
    # Save reusable results
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logistic_results_df.to_csv(
        RESULTS_DIR / "logistic_results.csv",
        index=False
    )

    default_rf_results_df.to_csv(
        RESULTS_DIR / "default_rf_results.csv",
        index=False
    )

    cv_results_df.to_csv(
        RESULTS_DIR / "baseline_cv_results.csv",
        index=False
    )

    # This comparison file is useful for Notebook 09.
    model_comparison_df = cv_results_df[
        [
            "Horizon",
            "Stock",
            "Logistic Regression CV",
            "Default Random Forest CV",
        ]
    ].copy()

    model_comparison_df.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False
    )

    # ========================================================
    # Return everything Notebook 07 needs
    # ========================================================

    return {
        "logistic_models": logistic_models,
        "rf_models": rf_models,

        "logistic_predictions": logistic_predictions,
        "rf_predictions": rf_predictions,

        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,

        "logistic_results": logistic_results_df,
        "default_rf_results": default_rf_results_df,
        "cv_results": cv_results_df,

        "model_comparison": model_comparison_df,
    }
