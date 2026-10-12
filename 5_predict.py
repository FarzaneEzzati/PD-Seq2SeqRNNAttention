"""Step 5: forecast 24 hours.

The model needs the past 48 hours and the time + weather of the 24 forecast hours.
Set predict_start in config.yaml (e.g. "2024-07-15"), or leave it null to forecast the last full day
in the data. The observed weather of that day stands in for a weather forecast.
Saves reports/<location>/<run>/forecast_<date>.csv and .png
"""
import numpy as np
import pandas as pd
import torch

from src.features import demand_to_kw, load_features
from src.model import load_trained_model, run_folder_name
from src.plots import plot_forecast
from src.utils import load_config

cfg = load_config()
run = run_folder_name(cfg)
report_folder = f"reports/{cfg['location']}/{run}"

data = load_features(cfg["location"])
df, y, scaler = data["df"], data["y"], data["scaler"]
model, run_cfg = load_trained_model(cfg, data)
quantiles, n_in, n_out = run_cfg["quantiles"], run_cfg["input_hours"], run_cfg["output_hours"]

# Row where the forecast starts
if cfg["predict_start"] is None:
    start = (df.index[-1] - pd.Timedelta(hours=n_out - 1)).normalize()   # last full day
else:
    start = pd.Timestamp(cfg["predict_start"])
row = df.index.get_loc(start)

# The window (past + forecast hours) must be complete, without missing hours
window = df.index[row - n_in:row + n_out]
if row < n_in or len(window) < n_in + n_out or window[-1] - window[0] != pd.Timedelta(hours=n_in + n_out - 1):
    raise ValueError(f"Not enough complete data around {start} to make a forecast.")

# Forecast
past = torch.from_numpy(data["X"][None, row - n_in:row])
last_y = torch.from_numpy(y[row - 1:row])
future = torch.from_numpy(data["X_future"][None, row:row + n_out])
with torch.no_grad():
    preds = demand_to_kw(model(past, last_y, future).numpy()[0], scaler)

table = pd.DataFrame({"timestamp": pd.date_range(start, periods=n_out, freq="h")})
for i, q in enumerate(quantiles):
    table[f"q{q}_kw"] = preds[:, i]
table["actual_kw"] = demand_to_kw(y[row:row + n_out], scaler)

name = start.strftime("%Y-%m-%d_%H%M")
table.to_csv(f"{report_folder}/forecast_{name}.csv", index=False)
plot_forecast(table, quantiles, f"Forecast from {start:%Y-%m-%d %H:%M}", f"{report_folder}/forecast_{name}.png")
print(table.to_string(index=False, float_format="%.3f"))
print(f"MAE of the median: {np.mean(np.abs(table[f'q0.5_kw'] - table['actual_kw'])):.3f} kW")
print(f"Saved to {report_folder}/forecast_{name}.csv")
