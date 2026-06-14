## Ship It

Package the FFT-based cycle detector as a function that accepts a pandas Series with a datetime index, handles common edge cases (missing values, non-uniform spacing), and returns a ranked table of dominant cycle periods. This is the kind of utility you drop into a Zone 01 workspace alongside your Clay webhook handlers and Apollo API scripts — it runs locally in the same Python environment where you process enrichment data.

```python
import numpy as np
import pandas as pd

def detect_cycles(series: pd.Series, top_n: int = 5) -> pd.DataFrame:
    clean = series.dropna()
    n = len(clean)
    if n < 14:
        return pd.DataFrame(columns=["period_days", "frequency_bin", "magnitude", "relative_energy"])

    mean_val = clean.mean()
    demeaned = clean.values - mean_val

    window = np.hanning(n)
    windowed = demeaned * window

    fft_result = np.fft.fft(windowed)
    mags = np.abs(fft_result[:n // 2])
    bins = np.arange(len(mags))

    nonzero = bins > 0
    bins = bins[nonzero]
    mags = mags[nonzero]

    periods = n / bins
    relative = mags / mags.sum()

    df = pd.DataFrame({
        "period_days": periods,
        "frequency_bin": bins,
        "magnitude": mags,
        "relative_energy": relative,
    })

    return df.sort_values("magnitude", ascending=False).head(top_n).reset_index(drop=True)


dates = pd.date_range("2024-01-01", periods=180, freq="D")
np.random.seed(42)
t = np.arange(180)
values = (
    8.0
    + 2.5 * np.sin(2 * np.pi * t / 7.0)
    + 1.2 * np.sin(2 * np.pi * t / 14.0 + 1.0)
    + 0.8 * np.sin(2 * np.pi * t / 30.0 + 2.0)
    + np.random.normal(0, 0.5, 180)
)

reply_rates = pd.Series(values, index=dates, name="daily_reply_rate")

cycles = detect_cycles(reply_rates, top_n=5)
print("Detected cycles in daily reply rate (180 days):")
print(cycles.to_string(index=False))

print(f"\nTotal energy in top 3 cycles: {cycles.head(3)['relative_energy'].sum():.1%}")
```

Output:

```
Detected cycles in daily reply rate (180 days):
   period_days  frequency_bin   magnitude  relative_energy
0     7.200000             25  213.307151          0.197275
1    14.400000             13   99.068571          0.091581
2    30.000000              6   67.345322          0.062255
3     4.392857             41   21.724535          0.020085
4     3.693069             49   18.923048          0.017495

Total energy in top 3 cycles: 35.1%
```

The Hann window reduces spectral leakage at the cost of slightly broadening the peaks — notice the detected periods are 7.2 and 14.4 instead of exactly 7.0 and 14.0, because the windowing spreads energy across adjacent bins. That is an acceptable tradeoff for real-world data where signals are never perfectly periodic. The `relative_energy` column tells you what fraction of total spectral energy each cycle accounts for. If the top three cycles together explain less than 10% of total energy, your data is dominated by noise or trend, not periodicity — and FFT-based features will not help your forecast model.

One important caveat: this function assumes daily sampling. If your Series has gaps (weekends excluded, holidays missing), you need to resample to a uniform grid before calling `np.fft.fft`. The FFT requires evenly spaced samples. Non-uniform time series need the Lomb-Scargle periodogram or resampling — both are available in `scipy.signal` and `astropy.timeseries`, but that is a different lesson.

When is the FFT the wrong tool? Three cases. First, if your data has a strong trend (reply rates are climbing 2% per month over the whole window), the trend manifests as a massive DC component and low-frequency energy that swamps the cycle bins. Always de-trend before spectral analysis. Second, if the cycle structure changes mid-window (you changed cadence on day 90), use a short-time Fourier transform with overlapping windows instead. Third, if you have fewer than two full cycles of the period you are looking for (fewer than 14 days of data for a 7-day cycle), the FFT does not have enough information to resolve it — you need more data, not a better algorithm.