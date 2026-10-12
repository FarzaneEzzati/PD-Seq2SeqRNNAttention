"""Build the model table from load + weather, split it by date, and cut it into windows."""
import joblib
import numpy as np
import pandas as pd

from src.weather import CONDITIONS

# These columns are standardised (mean 0, std 1). The others are already between -1 and 1.
COLUMNS_TO_SCALE = ["Temperature", "Wind", "Pressure", "Precipitation", "Demand"]


def build_table(load, weather, loc, start_date):
    """Join load and weather on the hour and add the model inputs."""
    df = weather.join(load[[loc["ercot_region"]]], how="inner")
    df = df[df.index >= pd.Timestamp(start_date)]

    # Demand per vulnerable household in kW
    zone_mw = df.pop(loc["ercot_region"])
    df["Demand"] = 1000 * (1 - loc["demand_downscale_rate"]) * zone_mw / loc["n_households"]

    # One column per weather condition group (always the same columns, in the same order)
    for name in CONDITIONS:
        df[f"Condition_{name}"] = (df["Condition"] == name).astype(float)
    df = df.drop(columns="Condition")

    # Calendar features as sine/cosine so that e.g. hour 23 is close to hour 0
    t = df.index
    days_in_year = np.where(t.is_leap_year, 366, 365)
    df["weekend"] = (t.dayofweek >= 5).astype(float)
    add_cycle(df, "month", t.month - 1, 12)
    add_cycle(df, "day_of_year", t.dayofyear - 1, days_in_year)
    add_cycle(df, "day_of_week", t.dayofweek, 7)
    add_cycle(df, "hour", t.hour, 24)
    add_cycle(df, "week", t.isocalendar().week.to_numpy() - 1, 52.14)
    return df


def add_cycle(df, name, value, period):
    angle = 2 * np.pi * np.asarray(value, dtype=float) / period
    df[f"sin_{name}"] = np.sin(angle)
    df[f"cos_{name}"] = np.cos(angle)


def add_split(df, cfg):
    """Label each hour as train, val, test, or future (after end_date)."""
    t = df.index
    df["split"] = "future"
    df.loc[t < pd.Timestamp(cfg["end_date"]) + pd.Timedelta(days=1), "split"] = "test"
    df.loc[t < pd.Timestamp(cfg["test_start"]), "split"] = "val"
    df.loc[t < pd.Timestamp(cfg["val_start"]), "split"] = "train"
    return df


def load_features(location):
    """Read the table made by 2_prepare_data.py.

    Returns a dictionary with:
        df              the full table (with the `split` column)
        X               all features, used for the past hours (encoder)
        X_future        all features except Demand, used for the forecast hours (decoder)
        future_columns  names of the X_future columns
        y               demand (scaled)
        scaler          the scaler from 2_prepare_data.py
    """
    folder = f"data/processed/{location}"
    df = pd.read_csv(f"{folder}/features.csv", index_col="timestamp", parse_dates=True)
    future_columns = [c for c in df.columns if c not in ("split", "Demand")]
    return {
        "df": df,
        "X": df.drop(columns="split").to_numpy(dtype=np.float32),
        "X_future": df[future_columns].to_numpy(dtype=np.float32),
        "future_columns": future_columns,
        "y": df["Demand"].to_numpy(dtype=np.float32),
        "scaler": joblib.load(f"{folder}/scaler.pkl"),
    }


def demand_to_kw(values, scaler):
    """Undo the scaling of the Demand column."""
    i = COLUMNS_TO_SCALE.index("Demand")
    return values * scaler.scale_[i] + scaler.mean_[i]


# ---------------------------------------------------------------- windows
# A window starts at a forecast hour p:
#   past    = the input_hours rows before p (all features)
#   future  = time and weather of the output_hours rows from p on
#   targets = demand of the output_hours rows from p on

def window_starts(df, split, cfg, midnight_only=False):
    """All forecast hours whose targets fall inside `split` and whose window has no gaps."""
    n_in, n_out = cfg["input_hours"], cfg["output_hours"]
    p = np.arange(n_in, len(df) - n_out + 1)
    split_label = df["split"].to_numpy()
    p = p[(split_label[p] == split) & (split_label[p + n_out - 1] == split)]

    # The window must cover consecutive hours (no rows missing in between)
    hours_spanned = (df.index[p + n_out - 1] - df.index[p - n_in]) / pd.Timedelta(hours=1)
    p = p[np.asarray(hours_spanned) == n_in + n_out - 1]

    if midnight_only:
        p = p[df.index[p].hour == 0]
    return p


def get_windows(data, starts, cfg):
    """Cut the data into windows. Returns:
        past      features of the input hours        [windows, input_hours, features]
        last_y    last known demand                  [windows]
        future    features of the forecast hours     [windows, output_hours, features - 1]
                  (time + weather, without demand)
        target    demand of the forecast hours       [windows, output_hours]
    """
    starts = np.asarray(starts)
    input_rows = starts[:, None] + np.arange(-cfg["input_hours"], 0)
    target_rows = starts[:, None] + np.arange(cfg["output_hours"])
    y = data["y"]
    return data["X"][input_rows], y[starts - 1], data["X_future"][target_rows], y[target_rows]
