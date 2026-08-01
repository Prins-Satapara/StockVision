from pathlib import Path


# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Data Directories
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ML_READY_DATA_DIR = DATA_DIR / "ml_ready"


# Model Directories
MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODELS_DIR = MODELS_DIR / "final"


# Results Directory
RESULTS_DIR = PROJECT_ROOT / "results"


# Other Directories
IMAGES_DIR = PROJECT_ROOT / "images"
SRC_DIR = PROJECT_ROOT / "src"


# Stocks
STOCKS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS"
]

FEATURES = [
    "Daily_Return",
    "Return_Lag_1",
    "Return_Lag_2",
    "Return_Lag_3",
    "SMA_5",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "Volatility_20",
    "Volume_Change",
    "High_Low_Range",
    "Open_Close_Change",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist"
]