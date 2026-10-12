"""Step 4: test the trained model on the test year (data it has never seen).

For every test day, a 24-hour forecast is made at midnight and compared with the actual demand.
The test is done twice:
    perfect_weather   the model gets the observed weather of the forecast hours (best case)
    noisy_weather     the model gets forecast-like weather (realistic case, see src/noise.py)

Saves in reports/<location>/<run>/:
    test_metrics.csv     scores in kW per household, one column per test
    test_forecasts.csv   every forecast hour next to the actual value (for the run's own weather setting)
    test_week_*.png, test_calibration.png, test_error_by_hour.png
"""
import numpy as np
import pandas as pd

from src.features import demand_to_kw, load_features, window_starts
from src.metrics import compute_metrics
from src.model import forecast, load_trained_model, run_folder_name
from src.plots import plot_calibration, plot_error_by_hour_ahead, plot_forecast, sample_week
from src.utils import load_config

# Configure the case
cfg = load_config()
run = run_folder_name(cfg)
report_folder = f"reports/{cfg['location']}/{run}"

# Get data, trained model, and prediction settings
data = load_features(cfg["location"])
df, y, scaler = data["df"], data["y"], data["scaler"]
model, run_cfg = load_trained_model(cfg, data)
quantiles = run_cfg["quantiles"]
n_out = run_cfg["output_hours"]
starts = window_starts(df, "test", run_cfg, midnight_only=True)
forecast_times = df.index[(starts[:, None] + np.arange(n_out)).ravel()]

# Simple baselines to compare with: demand at the same hour yesterday, and one week ago
demand_kw = pd.Series(demand_to_kw(y, scaler), index=df.index)
target = demand_to_kw(y[(starts[:, None] + np.arange(n_out))], scaler)
baselines = {}
for days, name in [(1, "yesterday"), (7, "last_week")]:
    guess = demand_kw.reindex(forecast_times - pd.Timedelta(days=days)).to_numpy().reshape(target.shape)
    baselines[f"mae_baseline_{name}"] = float(np.nanmean(np.abs(guess - target)))

# Test with perfect and with noisy future weather
results, predictions = {}, {}
for setting in ["perfect", "noisy"]:
    preds, _ = forecast(model, data, starts, run_cfg, noisy_weather=(setting == "noisy"), seed=run_cfg["seed"])
    predictions[setting] = demand_to_kw(preds, scaler)
    results[f"{setting}_weather"] = {**compute_metrics(target, predictions[setting], quantiles), **baselines}
table = pd.DataFrame(results)
table.to_csv(f"{report_folder}/test_metrics.csv", index_label="metric")

# Table of all forecasts and figures, for the weather setting the model was trained with
preds = predictions[run_cfg["future_weather"]]
forecasts = pd.DataFrame({
    "timestamp": forecast_times,
    "forecast_made_at": np.repeat(df.index[starts], n_out),
    "hours_ahead": np.tile(np.arange(1, n_out + 1), len(starts)),
    "actual_kw": target.ravel(),
})
for i, q in enumerate(quantiles):
    forecasts[f"q{q}_kw"] = preds[:, :, i].ravel()
forecasts.to_csv(f"{report_folder}/test_forecasts.csv", index=False)

# Plot the performance
year = df.index[starts[0]].year
for week_start in [f"{year}-01-08", f"{year}-07-08"]:
    plot_forecast(sample_week(forecasts, week_start), quantiles,
                  f"Day-ahead forecasts, week of {week_start} ({run_cfg['future_weather']} weather)",
                  f"{report_folder}/test_week_{week_start}.png")
plot_calibration(results[f"{run_cfg['future_weather']}_weather"], quantiles, f"{report_folder}/test_calibration.png")
plot_error_by_hour_ahead(np.mean(np.abs(preds[:, :, quantiles.index(0.5)] - target), axis=0),
                         f"{report_folder}/test_error_by_hour.png")

print(f"Test results for {cfg['location']} / {run} ({len(starts)} days):")
print(table.round(4).to_string())
print(f"Saved to {report_folder}")
