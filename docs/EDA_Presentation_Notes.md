# StockVision – EDA Presentation & Viva Notes

Main EDA notebooks: `03_EDA.ipynb` (Apple price EDA), `04_Comparative_Analysis.ipynb` (price comparison), **`04b_Statistical_EDA.ipynb` (new: statistical analysis of returns)**.

---

## 1. Suggested 5-minute flow

| Step | What to show | One-line message |
|------|--------------|------------------|
| 1 | Data quality table (Sec. 1) | Data is clean; 6 zero-volume days on Indian stocks are feed gaps. |
| 2 | Return vs volatility bars (Sec. 2) | US stocks: higher return *and* higher risk. NVIDIA is the extreme. |
| 3 | Histogram + Q-Q plot + Jarque-Bera (Sec. 3) | Returns are **not normal**; fat tails (extreme days 4–6× more common). |
| 4 | Rolling volatility (Sec. 4) | Risk is not constant; COVID spike in 2020. |
| 5 | Correlation heatmap (Sec. 5) | Same-market stocks move together (0.55 / 0.34); US vs India only 0.09. |
| 6 | Hypothesis tests (Sec. 6) | US vs India return and risk differ significantly; no weekday effect. |
| 7 | ADF + ACF (Sec. 7) | Prices non-stationary, returns stationary; volatility clusters, direction does not. |
| 8 | Target balance + feature correlation (Sec. 8) | Explains why the ML problem is hard and why accuracy alone can mislead. |

---

## 2. Numbers worth remembering

- 8 stocks, Jan 2015 – Sep 2026, about 2,900 trading days each.
- Excess kurtosis 3.7 – 9.6; days beyond ±3σ: 1.1 – 1.5 % (normal would give 0.27 %).
- NVIDIA: about 64 % annual return, 48 % annual volatility. HDFC Bank has the lowest volatility (22 %).
- Average correlation of daily returns: US–US 0.55, India–India 0.34, US–India 0.09. Highest pair: MSFT–AAPL / MSFT–AMZN 0.62, TCS–Infosys 0.61.
- Welch t-test US vs India mean return: p = 0.021. Levene test (variance): p ≈ 10⁻⁵⁹.
- One-sample t-test (mean return ≠ 0): significant for 6 of 8 stocks (not TCS, Infosys).
- ADF: price p-values 0.48 – 0.998 (non-stationary); return p-values ≈ 0 (stationary).
- Lag-1 autocorrelation: returns −0.05, squared returns 0.25.
- "Up" share of days: 1D 50–54 %, 20D up to 66 % (NVIDIA).
- Largest feature–target correlation (AAPL, 1D): 0.029.

---

## 3. Likely viva questions

**Why did you use returns instead of prices?**
Prices trend upward and have different scales (USD vs INR). Returns are scale-free, comparable and (as the ADF test shows) stationary, so statistical tests are valid on them.

**Why Adjusted Close?**
It corrects for stock splits and dividends, so returns reflect the true gain of an investor.

**What is a p-value?**
The probability of seeing a result at least this extreme if the null hypothesis were true. If p < 0.05 we reject the null hypothesis.

**What is kurtosis? What are "fat tails"?**
Kurtosis measures how heavy the tails are. A normal distribution has kurtosis 3 (excess kurtosis 0). Our stocks have excess kurtosis of 3.7–9.6, so extreme gains/losses are much more frequent than a normal model predicts.

**What does the Jarque-Bera test do?**
It tests normality using skewness and kurtosis. H0: the data is normal. p ≈ 0 for all stocks, so we reject normality.

**How do you read a Q-Q plot?**
It compares data quantiles with normal quantiles. Points on the straight line mean normal data; our points curve away at both ends, which means heavier tails.

**If returns are not normal, why do you still use a t-test?**
With about 2,900 observations the Central Limit Theorem makes the sample mean approximately normal, so the t-test is reliable. For the weekday test I used the non-parametric Kruskal-Wallis test.

**Why Welch's t-test and not the normal t-test?**
Welch does not assume equal variances, and the Levene test shows the variances of the two markets are very different.

**What does the Levene test do?**
It tests whether two or more groups have the same variance. H0: equal variance. We reject it, so the US portfolio has a significantly different (higher) risk.

**What is the Bonferroni correction?**
When you run many tests, some will look significant by chance. Bonferroni divides alpha by the number of tests (0.05 / 8 = 0.00625). After this correction no weekday effect remains.

**What is stationarity and why does it matter?**
A stationary series has constant mean and variance over time. Most statistical methods assume it. The ADF test (H0: non-stationary) shows prices are non-stationary and returns are stationary.

**What is autocorrelation (ACF)? What is volatility clustering?**
ACF is the correlation of a series with its own past. Returns have almost none (direction not predictable), but squared returns have strong positive autocorrelation: big moves follow big moves, which is volatility clustering.

**Why correlate returns and not prices?**
Two trending prices look highly correlated even if unrelated (spurious correlation). Returns remove the trend.

**Why are US and Indian stocks so weakly correlated?**
Different economies and sectors, and different trading hours (India closes before the US opens, so US news affects Indian stocks the next day). This also gives a diversification benefit.

**Why multiply by √252 to annualise volatility?**
Variance of independent daily returns adds over time (252 days), so standard deviation scales with √252. Mean return scales linearly (× 252).

**How does EDA help the machine-learning part?**
It shows (1) next-day direction is almost unpredictable, so modest accuracy is expected; (2) the "Up" class dominates at long horizons, so accuracy must be compared with the "always Up" baseline and precision/recall/F1 are needed; (3) several features (SMA, EMA, Adj Close) are almost identical (multicollinearity); (4) weekday is not a useful feature.

**What is multicollinearity?**
Predictors that are highly correlated with each other and carry the same information (for example SMA_20 and EMA_20, correlation above 0.99).

**Does "statistically significant" mean it is a good investment?**
No. It only means the average return is unlikely to be exactly zero in this sample. It says nothing about future returns, and past performance is not a guarantee.

---

## 4. Limitations (say them before the examiner does)

- Only 8 large, successful companies: **survivorship bias**; results do not generalise to all stocks.
- US returns are in USD and Indian returns in INR (no currency conversion).
- The US-vs-India portfolios use equal weights and only 4 stocks each; stocks inside a market are correlated, so the test assumes more independence than really exists.
- Correlation does not imply causation.
- The period (2015–2026) includes the COVID-19 crash, which strongly affects volatility results.
