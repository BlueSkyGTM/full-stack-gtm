## Ship It

Walk-forward validation is the evaluation framework that prevents temporal leakage in production. Instead of a single train/test split, the model is retrained at each step on all available data up to time *t*, then predicts *t+1*. This simulates exactly what happens in deployment: each week, you have one more observation, you retrain, and you forecast the next period.

The comparison between walk-forward MAE and the naive baseline (predict last observed value) tells you whether your feature engineering adds value. If the model cannot beat "predict last week's pipeline," the lag features are not capturing signal beyond what pure persistence provides.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

n_weeks = 104
dates = pd.date_range(start='2022-01-03', periods=n_weeks, freq='W')
trend = np.linspace(50000, 180000, n_weeks)
seasonality = 15000 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
noise = np.random.normal(0, 8000, n_weeks)
pipeline = trend + seasonality + noise

df = pd.DataFrame({'pipeline_value': pipeline}, index=dates)
df['lag_1'] = df['pipeline_value'].shift(1)
df['lag_4'] = df['pipeline_value'].shift(4)
df['rolling_mean_4'] = df['pipeline_value'].shift(1).rolling(4).mean()
df['rolling_std_4'] = df['pipeline_value'].shift(1).rolling(4).std()
df = df.dropna()

features = ['lag_1', 'lag_4', 'rolling_mean_4', 'rolling_std_4']
target = 'pipeline_value'

initial_train = 60
step = 1

wf_predictions = []
wf_actuals = []
wf_dates = []
naive_predictions = []

for i in range(initial_train, len(df)):
    train_slice = df.iloc[:i]
    test_row = df.iloc[i]

    model = LinearRegression()
    model.fit(train_slice[features], train_slice[target])
    pred = model.predict(test_row[features].values.reshape(1, -1))[0]

    naive_pred = train_slice[target].iloc[-1]

    wf_predictions.append(pred)
    wf_actuals.append(test_row[target])
    wf_dates.append(df.index[i])
    naive_predictions.append(naive_pred)

wf_mae = mean_absolute_error(wf_actuals, wf_predictions)
naive_mae = mean_absolute_error(wf_actuals, naive_predictions)

wf_errors = [abs(a - p) for a, p in zip(wf_actuals, wf_predictions)]
naive_errors = [abs(a - p) for a, p in zip(wf_actuals, naive_predictions)]

print("=== Walk-Forward Validation Results ===")
print(f"Total forecasts: {len(wf_predictions)}")
print(f"Training window: expanding from {initial_train} to {initial_train + len(wf_predictions) - 1}")
print(f"\nModel MAE:  ${wf_mae:,.0f}")
print(f"Naive MAE:  ${naive_mae:,.0f}")
improvement = (1 - wf_mae / naive_mae) * 100
print(f"Improvement over naive: {improvement:+.1f}%")
print(f"\nModel error as % of mean actual: {wf_mae / np.mean(wf_actuals) * 100:.1f}%")

results = pd.DataFrame({
    'actual': wf_actuals,
    'model_pred': wf_predictions,
    'naive_pred': naive_predictions,
    'model_error': wf_errors,
    'naive_error': naive_errors,
}, index=pd.DatetimeIndex(wf_dates))

print("\n=== Last 10 Forecast Steps ===")
print(results.tail(10).round(0))

expanding_errors = []
window_sizes = list(range(initial_train, initial_train + 20))
for w in window_sizes:
    if w >= len(df):
        break
    train_slice = df.iloc[:w]
    test_row = df.iloc[w]
    m = LinearRegression()
    m.fit(train_slice[features], train_slice[target])
    p = m.predict(test_row[features].values.reshape(1, -1))[0]
    expanding_errors.append(abs(test_row[target] - p))

print(f"\n=== Error Trend (first 20 steps) ===")
print(f"First 5 steps avg error:  ${np.mean(expanding_errors[:5]):,.0f}")
print(f"Last 5 steps avg error:   ${np.mean(expanding_errors[-5:]):,.0f}")
```

The walk-forward output shows whether the model's error is stable as the training window grows. If early-step errors are high and later-step errors decrease, the model benefits from more data — it is learning genuine structure. If errors are flat or increasing, the feature set is not capturing the underlying pattern, and you need richer features (longer lags, seasonal indicators, external regressors).

The improvement percentage over the naive baseline is the single most important metric in this entire lesson. If it is near zero or negative, your engineered features add no value beyond persistence. In that case, the issue is rarely the model — it is that your features do not encode the right temporal structure. Go back to the ACF analysis and check whether your lag depths match the autocorrelation peaks.

##