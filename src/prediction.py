import joblib

from src.config import FINAL_MODELS_DIR, FEATURES
from src.data_loader import load_stock_data

VALID_HORIZONS = ["1D", "5D", "20D"]

def load_model(stock, horizon = "1D"):
    
    if horizon not in VALID_HORIZONS:
        raise ValueError(
            f"Invalid horizon: {horizon}. "
            f"Choose from {VALID_HORIZONS}"
        )
    
    
    model_path = FINAL_MODELS_DIR / f"{stock}_{horizon}_model.joblib"
    model = joblib.load(model_path)
    
    return model

    
def predict_stock(stock, horizon = "1D"):
    
    # load data and model
    df = load_stock_data(stock=stock)
    model = load_model(stock=stock, horizon=horizon)
    
    
    # get the latest features    
    X_latest = df[FEATURES].iloc[[-1]]  
    
    
    # predictions and probabilities
    prediction = model.predict(X_latest)[0]
    
    probabilities = model.predict_proba(X_latest)[0]
    
    confidence = probabilities[prediction]
    
    
    # Get latest available date
    latest_date = df["Date"].iloc[-1]

    # Convert prediction into readable text
    direction = "UP" if prediction == 1 else "DOWN"
    
    
    return {
        "Stock": stock,
        "Horizon": horizon,
        "Date": latest_date.strftime("%Y-%m-%d"),
        "Prediction": direction,
        "Probability": round(float(confidence) * 100, 2)
    }
