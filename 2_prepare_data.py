"""Step 2: clean the raw data and build the table the model learns from.

Saves:
    data/interim/ercot_load.csv               cleaned hourly load, all regions (MW)
    data/interim/weather_<location>.csv       cleaned hourly weather
    data/processed/<location>/features.csv    model inputs (scaled) with a `split` column
    data/processed/<location>/scaler.pkl      scaler fitted on the training years only
"""
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import ercot, weather
from src.features import COLUMNS_TO_SCALE, add_split, build_table
from src.utils import load_config

cfg = load_config()
location = cfg["location"]
out_folder = f"data/processed/{location}"
os.makedirs("data/interim", exist_ok=True)
os.makedirs(out_folder, exist_ok=True)

print("Cleaning ERCOT load...")
load = ercot.read_load(max_gap_hours=cfg["max_gap_hours"])
load.to_csv("data/interim/ercot_load.csv")

print("Cleaning weather...")
raw_weather = pd.read_csv(f"data/raw/weather/{location}.csv", dtype=str)
clean = weather.clean_weather(raw_weather, max_gap_hours=cfg["max_gap_hours"])
clean.to_csv(f"data/interim/weather_{location}.csv")

print("Building features...")
df = build_table(load, clean, cfg["loc"], cfg["start_date"])
df = add_split(df, cfg)

# Fit the scaler on the training years only, then apply it to all rows
scaler = StandardScaler().fit(df.loc[df["split"] == "train", COLUMNS_TO_SCALE])
df[COLUMNS_TO_SCALE] = scaler.transform(df[COLUMNS_TO_SCALE])

df.to_csv(f"{out_folder}/features.csv")
joblib.dump(scaler, f"{out_folder}/scaler.pkl")
print(f"Saved {len(df):,} hours and {df.shape[1] - 1} features to {out_folder}")
print("Hours per split:", df["split"].value_counts().to_dict())
