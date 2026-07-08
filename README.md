# StockVision

Exploratory data analysis and comparative study of historical stock performance across major US and Indian companies, using data pulled live from Yahoo Finance — with a machine learning extension (price prediction and direction classification) planned as the next phase.

![Normalized Comparison](images/normalized_comparison.png)

## Overview

StockVision pulls 11 years (2015–2025) of daily OHLCV data for 8 companies and walks through a full analysis pipeline: collection → cleaning → exploratory analysis → cross-market comparison. The next phase of the project (see [Roadmap](#roadmap)) extends this into feature engineering and machine learning — price prediction and direction classification with decision boundary visualization.

**Companies covered**

| 🇺🇸 US | 🇮🇳 India |
|---|---|
| Apple (AAPL) | Reliance Industries (RELIANCE.NS) |
| Microsoft (MSFT) | TCS (TCS.NS) |
| NVIDIA (NVDA) | HDFC Bank (HDFCBANK.NS) |
| Amazon (AMZN) | Infosys (INFY.NS) |

## Pipeline

| Stage | Notebook | What it does |
|---|---|---|
| 1. Data Collection | [`01_data_collection.ipynb`](notebooks/01_data_collection.ipynb) | Downloads daily OHLCV data via `yfinance` for all 8 tickers (2015-01-01 to 2025-12-31) into `data/raw/` |
| 2. Data Cleaning | [`02_data_cleaning.ipynb`](notebooks/02_data_cleaning.ipynb) | Parses dates, drops nulls/duplicates, sorts chronologically, writes to `data/cleaned/` |
| 3. EDA | [`03_EDA.ipynb`](notebooks/03_EDA.ipynb) | Statistical summaries, closing price & volume trends, price distribution, outlier detection, correlation/pair-plot analysis |
| 4. Comparative Analysis | [`04_Comparative_Analysis.ipynb`](notebooks/04_Comparative_Analysis.ipynb) | Normalized cross-company performance comparison (baseline = 100) across US and Indian markets |

## Sample Results

**Apple — closing price trend (2015–2025)**
![Closing Price Trend](images/closing_price_trend.png)

**Apple — price distribution**
![Price Distribution](images/price_distribution.png)

**All 8 companies — raw closing prices**
![Multi Company Closing Prices](images/multi_company_closing_prices.png)

### Key findings so far
- Apple's closing price shows a strong long-term upward trend, with visible drawdowns during market corrections followed by recovery.
- Price variables (Open/High/Low/Close) are very highly correlated; trading volume has a weaker, right-skewed relationship with price.
- Normalizing all 8 companies to a common baseline makes it possible to compare US vs. Indian equity growth directly, independent of currency and starting price.

## Repository Structure

```
StockVision/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_EDA.ipynb
│   └── 04_Comparative_Analysis.ipynb
├── src/
│   ├── config.py          # central path config (data/images/reports dirs)
│   ├── download_data.py   # yfinance download helper
│   └── clean_data.py      # cleaning pipeline
├── data/
│   ├── raw/                # downloaded CSVs
│   └── cleaned/             # cleaned CSVs
└── images/                  # exported charts used in this README
```

## Getting Started

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-username>/StockVision.git
cd StockVision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebooks in order
jupyter notebook notebooks/01_data_collection.ipynb
```

Run notebooks `01` → `04` in sequence — each stage depends on the output of the previous one.

## Tech Stack

- **Data source:** [yfinance](https://pypi.org/project/yfinance/) (Yahoo Finance API wrapper)
- **Data processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Environment:** Jupyter Notebook

## Roadmap

Data collection, cleaning, EDA, and comparative analysis are complete. The next phase extends this into a full machine learning pipeline:

- [x] Data collection pipeline (Yahoo Finance, 8 companies, 2015–2025)
- [x] Data cleaning pipeline
- [x] Exploratory data analysis
- [x] Comparative analysis (US vs. Indian markets)
- [ ] **Feature engineering** — technical indicators (SMA/EMA, RSI, MACD, Bollinger Bands), lagged prices, rolling volatility
- [ ] **Price prediction (regression)** — predict next-day closing price (Linear Regression, Random Forest, and beyond)
- [ ] **Direction classification** — predict next-day up/down movement, compared across multiple classifiers (Logistic Regression, KNN, SVM, Decision Tree, Random Forest)
- [ ] **Decision boundary visualization** — 2D projections (PCA) showing how each classifier partitions the feature space
- [ ] **Model evaluation** — RMSE/R² for regression, accuracy/precision/recall/F1/ROC-AUC for classification, feature importance analysis
- [ ] Walk-forward cross-validation & hyperparameter tuning
- [ ] Interactive dashboard (Streamlit/Plotly Dash) for exploring predictions live

## License

This project is for educational purposes.
