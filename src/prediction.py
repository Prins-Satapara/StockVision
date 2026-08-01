import joblib

from src.config import FINAL_MODELS_DIR, FEATURES
from src.data_loader import load_stock_data

def load_model(stock):
    
    model_path = FINAL_MODELS_DIR / f"{stock}_model.joblib"
    model = joblib.load(model_path)
    
    return model


def predict_stock(stock):
    
    # load data and model
    df = load_stock_data(stock=stock)
    model = load_model(stock=stock)
    
    
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
        "Date": latest_date.strftime("%Y-%m-%d"),
        "Prediction": direction,
        "Probability": round(float(confidence) * 100, 2)
    }
