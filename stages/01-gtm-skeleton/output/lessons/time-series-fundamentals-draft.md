# Time Series Fundamentals

## Learning Objectives

1. Decompose a time series into trend, seasonality, and residual components using additive decomposition
2. Detect non-stationarity in a time series using the Augmented Dickey-Fuller test
3. Generate lag features and rolling window statistics from temporal data
4. Apply differencing to transform a non-stationary series into a stationary one
5. Evaluate autocorrelation structure to determine appropriate lag depth for feature engineering

---

## Beat 1: Hook

Standard ML assumes rows are independent. Time series data violates this assumption by definition — each observation is correlated with its neighbors. Ignoring temporal ordering doesn't just reduce accuracy; it produces models that learn from leakage and fail in production. This lesson covers the mechanics that break and the patterns that work when your data has a time axis.

---

## Beat 2: Concept

**Temporal ordering and autocorrelation.** A time series is a sequence where position encodes information. Autocorrelation measures how much the present predicts the past — if autocorrelation is high, your data is not independent and your standard train/test split is wrong.

**Decomposition mechanics.** Any time series can be expressed as `y(t) = trend + seasonality + residual` (additive) or `y(t) = trend × seasonality × residual` (multiplicative). Trend is the long-term direction. Seasonality is a repeating cycle at a fixed period. Residual is everything left — this is where your model operates.

**Stationarity.** A stationary series has constant mean and variance over time. Most classical models (ARIMA, exponential smoothing) require stationarity as a precondition. Non-stationary data is detected via the Augmented Dickey-Fuller test and fixed via differencing: `y'(t) = y(t) - y(t-1)`.

**Feature engineering for time.** Lag features shift the target backward: `lag_1 = y(t-1)`. Rolling statistics aggregate a window: `rolling_mean_7 = mean(y(t-7), ..., y(t-1))`. These transforms temporal structure into tabular features that any model can consume.

---

## Beat 3: Demonstration

Generate a synthetic time series with known trend and seasonality, decompose it, plot autocorrelation, and run an ADF test. Show that differencing collapses a trending series into a stationary one. All output is printed to terminal — no browser dependency.

**Exercise hooks:**
- *Easy:* Modify the seasonality amplitude and observe how the decomposition changes
- *Medium:* Add a second seasonal period and attempt decomposition — document what breaks
- *Hard:* Implement rolling-window stationarity tests that detect where in the series the mean shifts

---

## Beat 4: Use It

**GTM Redirect:** Pipeline velocity forecasting — Zone 1 (Data Foundation) / Zone 2 (Signal Processing). [CITATION NEEDED — concept: pipeline velocity time series forecasting in GTM topic map]

Build lag features and rolling statistics on a synthetic quarterly revenue dataset that mimics CRM pipeline data. The same feature engineering pattern applies when transforming historical deal close data into inputs for pipeline prediction: create lag features from prior quarters, rolling averages for trend capture, and detect whether your revenue stream is stationary (it almost certainly is not).

**Exercise hooks:**
- *Easy:* Generate lag features at depths 1–4 on the revenue dataset
- *Medium:* Add rolling mean and rolling standard deviation features; compare ADF p-values before and after differencing
- *Hard:* Build a feature matrix from the pipeline dataset that includes lag features, rolling statistics, and differenced targets — print the correlation matrix to identify multicollinearity

---

## Beat 5: Ship It

Implement a stationarity detection and differencing pipeline that accepts any time series array, runs an ADF test, applies differencing iteratively until stationary, and logs the number of differencing operations required. This is the same preprocessing step required before feeding any pipeline or revenue data into a forecasting model in production.

**Exercise hooks:**
- *Easy:* Wrap the ADF test in a function that returns a boolean and the p-value
- *Medium:* Build the iterative differencing loop with a max iteration cutoff and print each step
- *Hard:* Add rolling-window stationarity detection that flags specific time ranges where the series becomes non-stationary — simulates detecting a market shift in pipeline data

---

## Beat 6: Extend It

Classical decomposition and differencing are baseline mechanics. From here:

- **ARIMA family:** Autoregression (AR) uses lagged values as predictors. Moving average (MA) uses lagged errors. ARIMA(p,d,q) combines both with differencing. The `p`, `d`, `q` parameters map directly to autocorrelation structure, differencing order, and error correlation.
- **Prophet:** Additive regression with Fourier terms for seasonality and piecewise linear trends. Useful when your pipeline data has multiple seasonal cycles (weekly + quarterly).
- **Sequence models:** LSTMs and Transformers handle long-range dependencies that classical methods miss. The bridge is: lag features are manual attention. Sequence models learn attention automatically.

The feature engineering patterns from this lesson — lags, rolling windows, stationarity enforcement — are the same regardless of which model family consumes them.